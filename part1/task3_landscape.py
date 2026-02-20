import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Net(nn.Module):
    def __init__(self, a, b, w, n):
        super().__init__()
        stuff = [nn.Linear(a, w), nn.Tanh()]
        for _ in range(n - 1):
            stuff.append(nn.Linear(w, w))
            stuff.append(nn.Tanh())
        stuff.append(nn.Linear(w, b))
        self.net = nn.Sequential(*stuff)

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
    return np.stack([xx, yy], axis=-1)


def pick_idx(n, sub_in, sub_bc, seed):
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    bc = (ii == 0) | (ii == n - 1) | (jj == 0) | (jj == n - 1)
    inn = ~bc
    inn_idx = np.where(inn.reshape(-1))[0]
    bc_idx = np.where(bc.reshape(-1))[0]
    rng = np.random.default_rng(seed)
    if sub_in > 0:
        sub_in = min(sub_in, inn_idx.size)
        inn_idx = rng.choice(inn_idx, size=sub_in, replace=False)
    if sub_bc > 0:
        sub_bc = min(sub_bc, bc_idx.size)
        bc_idx = rng.choice(bc_idx, size=sub_bc, replace=False)
    return inn_idx, bc_idx


def vec(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def unvec(arr, model, dev):
    out = []
    off = 0
    for p in model.parameters():
        cnt = p.numel()
        chunk = arr[off:off + cnt]
        if chunk.size != cnt:
            raise ValueError("bad")
        t = torch.from_numpy(chunk.reshape(p.shape)).to(dev).type_as(p)
        out.append(t)
        off += cnt
    if off != arr.size:
        raise ValueError("bad")
    return out


def norm1(d, w, mode):
    eps = 1e-12
    if mode == "none":
        return d
    if mode == "layer":
        wn = w.norm()
        dn = d.norm()
        if wn > 0 and dn > 0:
            d = d * (wn / (dn + eps))
        return d
    if mode == "filter":
        if w.dim() <= 1:
            wn = w.norm()
            dn = d.norm()
            if wn > 0 and dn > 0:
                d = d * (wn / (dn + eps))
            return d
        for i in range(w.shape[0]):
            wn = w[i].norm()
            dn = d[i].norm()
            if wn > 0 and dn > 0:
                d[i] = d[i] * (wn / (dn + eps))
        return d
    raise ValueError("bad")


def norml(dirs, model, mode):
    if mode == "none":
        return dirs
    return [norm1(d, w, mode) for d, w in zip(dirs, model.parameters())]


def rand_d(model, seed, mode):
    torch.manual_seed(seed)
    out = []
    for p in model.parameters():
        d = torch.randn_like(p)
        d = norm1(d, p, mode)
        out.append(d)
    return out


def load_tr(case):
    path = case / "trajectory.npz"
    if not path.exists():
        raise FileNotFoundError("missing")
    d = np.load(path)
    if "states" not in d:
        raise ValueError("bad")
    states = d["states"]
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("bad")
    return states


def pca_d(states):
    c = states.astype(np.float64)
    c -= c.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    if vt.shape[0] < 2:
        raise ValueError("bad")
    return vt[0], vt[1]


def set_p(model, base, d1, d2, a, b):
    with torch.no_grad():
        for p, bb, x, y in zip(model.parameters(), base, d1, d2):
            p.copy_(bb + a * x + b * y)


def lp(model, x0, f1, x1):
    u = model(x0)
    g = torch.autograd.grad(u.sum(), x0, create_graph=True)[0]
    d2x = torch.autograd.grad(g[:, 0].sum(), x0, create_graph=True)[0][:, 0]
    d2y = torch.autograd.grad(g[:, 1].sum(), x0, create_graph=True)[0][:, 1]
    res = -(d2x + d2y)[:, None] - f1
    lf = (res ** 2).mean()
    u_bc = model(x1)
    lb = (u_bc ** 2).mean()
    return lf + lb


def ld(model, pts, u_true):
    u = model(pts)
    return ((u - u_true) ** 2).mean()


def save_c(a, b, grid, out_path, title, log_scale):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if log_scale:
        vals = np.log10(np.clip(grid, 1e-30, None))
    else:
        vals = grid
    aa, bb = np.meshgrid(a, b, indexing="ij")
    c = ax.contourf(aa, bb, vals, levels=40, cmap="viridis")
    ax.scatter([0], [0], c="red", s=25, marker="x")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_title(title)
    fig.colorbar(c, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_s(a, b, grid, out_path, title, log_scale):
    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111, projection="3d")
    aa, bb = np.meshgrid(a, b, indexing="ij")
    if log_scale:
        vals = np.log10(np.clip(grid, 1e-30, None))
    else:
        vals = grid
    ax.plot_surface(aa, bb, vals, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_zlabel("log10(loss)" if log_scale else "loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_m(case, dev):
    with open(case / "metrics.json") as f:
        met = json.load(f)
    path = case / "model.pt"
    if not path.exists():
        raise FileNotFoundError("missing")
    model = Net(2, 1, met["hidden_width"], met["hidden_layers"]).to(dev)
    state = torch.load(path, map_location=dev)
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(sd)
    model.eval()
    return model, met


def get():
    p = argparse.ArgumentParser()
    p.add_argument("--case-root", type=str, default="task2_output_gpu_baseline")
    p.add_argument("--data-dir", type=str, default="task1_output/data")
    p.add_argument("--out-dir", type=str, default="task3_output")
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--methods", type=str, nargs="+", default=["data", "pinn"], choices=["data", "pinn"])
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--sample-idx", type=int, default=None)
    p.add_argument("--grid-size", type=int, default=41)
    p.add_argument("--alpha-range", type=float, default=1.0)
    p.add_argument("--beta-range", type=float, default=1.0)
    p.add_argument("--normalize", type=str, default="filter", choices=["filter", "layer", "none"])
    p.add_argument("--direction-source", type=str, default="pca", choices=["pca", "random"])
    p.add_argument("--normalize-pca", action="store_true")
    p.add_argument("--subsample-interior", type=int, default=-1)
    p.add_argument("--subsample-boundary", type=int, default=-1)
    p.add_argument("--log-scale", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def dev(flag):
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def go():
    args = get()
    seed(args.seed)
    d = dev(args.device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for k in args.k_values:
        ds = pick(args.data_dir, k, args.n)
        x, y, f, u = load(ds, args.sample_idx or 0)
        n = len(x)
        pts = mk(x, y).reshape(-1, 2)

        inn_idx, bc_idx = pick_idx(n, args.subsample_interior, args.subsample_boundary, args.seed)

        p0 = torch.from_numpy(pts).float().to(d)
        f0 = torch.from_numpy(f.reshape(-1, 1)).float().to(d)
        u0 = torch.from_numpy(u.reshape(-1, 1)).float().to(d)

        x0 = p0[inn_idx].clone().requires_grad_(True)
        f1 = f0[inn_idx]
        x1 = p0[bc_idx]

        for kind in args.methods:
            case = Path(args.case_root) / f"K{k}" / kind
            model, met = load_m(case, d)
            idx = args.sample_idx if args.sample_idx is not None else met.get("sample_idx", 0)

            if idx != (args.sample_idx or 0):
                x, y, f, u = load(ds, idx)
                pts = mk(x, y).reshape(-1, 2)
                p0 = torch.from_numpy(pts).float().to(d)
                f0 = torch.from_numpy(f.reshape(-1, 1)).float().to(d)
                u0 = torch.from_numpy(u.reshape(-1, 1)).float().to(d)
                x0 = p0[inn_idx].clone().requires_grad_(True)
                f1 = f0[inn_idx]
                x1 = p0[bc_idx]

            base = [p.detach().clone() for p in model.parameters()]
            if args.direction_source == "pca":
                states = load_tr(case)
                v1, v2 = pca_d(states)
                d1 = unvec(v1, model, d)
                d2 = unvec(v2, model, d)
                if args.normalize_pca:
                    d1 = norml(d1, model, args.normalize)
                    d2 = norml(d2, model, args.normalize)
            else:
                d1 = rand_d(model, args.seed, args.normalize)
                d2 = rand_d(model, args.seed + 1, args.normalize)

            alphas = np.linspace(-args.alpha_range, args.alpha_range, args.grid_size)
            betas = np.linspace(-args.beta_range, args.beta_range, args.grid_size)
            grid = np.zeros((args.grid_size, args.grid_size), dtype=np.float64)

            for i, a in enumerate(alphas):
                for j, b in enumerate(betas):
                    set_p(model, base, d1, d2, a, b)
                    if kind == "pinn":
                        loss = lp(model, x0, f1, x1)
                        grid[i, j] = loss.item()
                    else:
                        with torch.no_grad():
                            loss = ld(model, p0, u0)
                            grid[i, j] = loss.item()

            set_p(model, base, d1, d2, 0.0, 0.0)

            out_dir = out / f"K{k}" / kind
            out_dir.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(out_dir / "landscape.npz", alphas=alphas, betas=betas, loss=grid)

            title = f"K={k} {kind.upper()} landscape"
            save_c(alphas, betas, grid, out_dir / "contour.png", title, args.log_scale)
            save_s(alphas, betas, grid, out_dir / "surface.png", title, args.log_scale)


if __name__ == "__main__":
    go()
