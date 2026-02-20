import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, a, b, w, n):
        super().__init__()
        stuff = [nn.Linear(a, w), nn.Tanh()]
        for _ in range(n - 1):
            stuff.append(nn.Linear(w, w))
            stuff.append(nn.Tanh())
        stuff.append(nn.Linear(w, b))
        self.net = nn.Sequential(*stuff)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def pick(d, k, n):
    d = Path(d)
    hits = sorted(d.glob(f"poisson_K{k}_N{n}_S*.npz"))
    if not hits:
        raise FileNotFoundError("missing")
    return hits[0]


def load(p, idx):
    d = np.load(p)
    x = d["x"]
    y = d["y"]
    f = d["f"][idx]
    u = d["u"][idx]
    return x, y, f, u


def mk(x, y):
    xx, yy = np.meshgrid(x, y, indexing="ij")
    pts = np.stack([xx, yy], axis=-1)
    return pts, xx, yy


def mask(n):
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    b = (ii == 0) | (ii == n - 1) | (jj == 0) | (jj == n - 1)
    inn = ~b
    return inn.reshape(-1), b.reshape(-1)


def r2(a, b):
    return torch.linalg.norm(a - b) / torch.linalg.norm(b)


def vec(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def keep(model, arr):
    arr.append(vec(model).cpu().numpy().astype(np.float32))


def lp(model, pts, f, mask_in, mask_bc):
    u = model(pts)
    g = torch.autograd.grad(u.sum(), pts, create_graph=True)[0]
    d2x = torch.autograd.grad(g[:, 0].sum(), pts, create_graph=True)[0][:, 0]
    d2y = torch.autograd.grad(g[:, 1].sum(), pts, create_graph=True)[0][:, 1]
    lap = d2x + d2y
    res = -lap[:, None] - f
    lf = (res[mask_in] ** 2).mean()
    lb = (u[mask_bc] ** 2).mean()
    return lf + lb, lf, lb, u


def ld(model, pts, u_true):
    u = model(pts)
    l = ((u - u_true) ** 2).mean()
    return l, u


def do_adam(model, loss_fn, opt, steps, cb=None):
    out = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        l = loss_fn()
        l.backward()
        opt.step()
        out.append(l.item())
        if cb is not None:
            cb(i)
    return out


def do_lb(model, loss_fn, opt, steps, cb=None):
    out = []
    for i in range(steps):
        def closure():
            opt.zero_grad(set_to_none=True)
            l = loss_fn()
            l.backward()
            return l
        l = opt.step(closure)
        out.append(l.item())
        if cb is not None:
            cb(i)
    return out


def draw_l(hist, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(hist)
    ax.set_xlabel("iter")
    ax.set_ylabel("loss")
    ax.set_title("loss")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_p(x, y, f, u_true, u_pred, out_path, title):
    ext = (x.min(), x.max(), y.min(), y.max())
    err = u_pred - u_true
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    im0 = axes[0].imshow(f, origin="lower", extent=ext, cmap="viridis")
    axes[0].set_title("f")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(u_true, origin="lower", extent=ext, cmap="viridis")
    axes[1].set_title("u true")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = axes[2].imshow(u_pred, origin="lower", extent=ext, cmap="viridis")
    axes[2].set_title("u pred")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    im3 = axes[3].imshow(err, origin="lower", extent=ext, cmap="coolwarm")
    axes[3].set_title("err")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def get():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="task1_output/data")
    p.add_argument("--out-dir", type=str, default="task2_output")
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--hidden-layers", type=int, default=4)
    p.add_argument("--hidden-width", type=int, default=128)
    p.add_argument("--adam-steps", type=int, default=5000)
    p.add_argument("--lbfgs-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save-trajectory", action="store_true")
    p.add_argument("--trajectory-interval", type=int, default=50)
    p.add_argument("--trajectory-include-lbfgs", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def dev(flag):
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def one(kind, model, pts, f, u_true, mask_in, mask_bc, args, out_dir, shape, x, y):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(pts.device)

    if kind == "pinn":
        def loss_fn():
            l, _, _, _ = lp(model, pts, f, mask_in, mask_bc)
            return l
    elif kind == "data":
        def loss_fn():
            l, _ = ld(model, pts, u_true)
            return l
    else:
        raise ValueError("bad")

    snaps = []
    if args.save_trajectory:
        keep(model, snaps)

        def on_a(i):
            if args.trajectory_interval > 0 and (i + 1) % args.trajectory_interval == 0:
                keep(model, snaps)

        if args.trajectory_include_lbfgs:
            def on_b(i):
                if args.trajectory_interval > 0 and (i + 1) % args.trajectory_interval == 0:
                    keep(model, snaps)
        else:
            on_b = None
    else:
        on_a = None
        on_b = None

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist = do_adam(model, loss_fn, opt, args.adam_steps, cb=on_a)

    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, line_search_fn="strong_wolfe")
    hist += do_lb(model, loss_fn, opt, args.lbfgs_steps, cb=on_b)

    if args.save_trajectory:
        keep(model, snaps)

    model.eval()
    if kind == "pinn":
        _, lf, lb, u_pred = lp(model, pts, f, mask_in, mask_bc)
        lf = lf.item()
        lb = lb.item()
        loss = None
    else:
        loss, u_pred = ld(model, pts, u_true)
        loss = loss.item()
        lf = None
        lb = None

    with torch.no_grad():
        r = r2(u_pred, u_true).item()

    u_pred_grid = u_pred.reshape(shape).detach().cpu().numpy()
    u_true_grid = u_true.reshape(shape).detach().cpu().numpy()
    f_grid = f.reshape(shape).detach().cpu().numpy()

    met = {
        "method": kind,
        "adam_steps": args.adam_steps,
        "lbfgs_steps": args.lbfgs_steps,
        "hidden_layers": args.hidden_layers,
        "hidden_width": args.hidden_width,
        "sample_idx": args.sample_idx,
        "relative_l2": r,
    }
    if args.save_trajectory:
        met["save_trajectory"] = True
        met["trajectory_interval"] = args.trajectory_interval
        met["trajectory_include_lbfgs"] = args.trajectory_include_lbfgs
        met["trajectory_snapshots"] = len(snaps)
    if kind == "pinn":
        met["loss_f"] = lf
        met["loss_bc"] = lb
    else:
        met["loss"] = loss

    with open(out_dir / "metrics.json", "w") as fptr:
        json.dump(met, fptr, indent=2)

    state = {"state_dict": model.state_dict(), "config": {"hidden_layers": args.hidden_layers, "hidden_width": args.hidden_width}}
    torch.save(state, out_dir / "model.pt")

    if args.save_trajectory:
        if snaps:
            arr = np.stack(snaps, axis=0)
        else:
            arr = np.empty((0, 0), dtype=np.float32)
        np.savez_compressed(out_dir / "trajectory.npz", states=arr)

    np.savez_compressed(
        out_dir / "predictions.npz",
        x=np.array(x, dtype=np.float32),
        y=np.array(y, dtype=np.float32),
        f=f_grid,
        u_true=u_true_grid,
        u_pred=u_pred_grid,
        loss=np.array(hist, dtype=np.float32),
    )

    draw_l(hist, out_dir / "loss_curve.png")
    draw_p(x, y, f_grid, u_true_grid, u_pred_grid, out_dir / "prediction.png", title=f"{kind.upper()} prediction")

    return met


def go():
    args = get()
    seed(args.seed)
    d = dev(args.device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for k in args.k_values:
        path = pick(args.data_dir, k, args.n)
        x, y, f, u = load(path, args.sample_idx)
        n = len(x)

        pts, _, _ = mk(x, y)
        pts = pts.reshape(-1, 2)
        m_in, m_bc = mask(n)

        p0 = torch.from_numpy(pts).float().to(d)
        f0 = torch.from_numpy(f.reshape(-1, 1)).float().to(d)
        u0 = torch.from_numpy(u.reshape(-1, 1)).float().to(d)

        m0 = torch.from_numpy(m_in).to(d)
        m1 = torch.from_numpy(m_bc).to(d)

        for kind in ["data", "pinn"]:
            model = Net(2, 1, args.hidden_width, args.hidden_layers)
            if kind == "pinn":
                p_use = p0.clone().requires_grad_(True)
            else:
                p_use = p0
            case = out / f"K{k}" / kind
            one(kind, model, p_use, f0, u0, m0, m1, args, case, (n, n), x, y)


if __name__ == "__main__":
    go()
