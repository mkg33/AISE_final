import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class D2(Dataset):
    def __init__(self, path, times, use_grid=True, time_indices=None, pair_mode="all2all", input_time_index=0, time_input="t_out"):
        data = np.load(path)
        self.data = data.astype(np.float32)
        self.u0 = self.data[:, 0]
        self.times = np.array(times, dtype=np.float32)
        self.num_times = len(self.times)
        self.use_grid = use_grid
        self.x = np.linspace(0.0, 1.0, self.u0.shape[-1], dtype=np.float32)
        self.pair_mode = pair_mode
        self.input_time_index = input_time_index
        self.time_input = time_input

        if time_indices is None:
            time_indices = list(range(len(times)))
        self.time_indices = list(time_indices)

        pairs = []
        t_idx = sorted(set(self.time_indices))
        if self.pair_mode == "all2all":
            for tr in range(self.data.shape[0]):
                for i, t_in in enumerate(t_idx):
                    for t_out in t_idx[i + 1:]:
                        pairs.append((tr, t_in, t_out))
        elif self.pair_mode in {"fixed", "u0"}:
            t_in = 0 if self.pair_mode == "u0" else self.input_time_index
            for tr in range(self.data.shape[0]):
                for t_out in t_idx:
                    if t_out < t_in:
                        continue
                    pairs.append((tr, t_in, t_out))
        else:
            raise ValueError("bad")

        self.pairs = pairs
        self.num_trajectories = self.data.shape[0]
        self.num_pairs = len(self.pairs)
        self.pairs_per_trajectory = self.num_pairs // self.num_trajectories if self.num_trajectories else 0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        tr, t_in_idx, t_out_idx = self.pairs[i]
        u_in = self.data[tr, t_in_idx]
        u_out = self.data[tr, t_out_idx]
        t_in = self.times[t_in_idx]
        t_out = self.times[t_out_idx]
        dt = t_out - t_in
        if self.time_input == "t_out":
            t_val = t_out
        elif self.time_input == "dt":
            t_val = dt
        else:
            raise ValueError("bad")
        t_arr = np.full_like(self.x, t_val)
        if self.use_grid:
            inp = np.stack([u_in, self.x, t_arr], axis=-1)
        else:
            inp = np.stack([u_in, t_arr], axis=-1)
        out = u_out[:, None]
        return torch.from_numpy(inp), torch.from_numpy(out)


class Spec(nn.Module):
    def __init__(self, a, b, m):
        super().__init__()
        self.a = a
        self.b = b
        self.m = m
        s = 1.0 / (a * b)
        self.w = nn.Parameter(s * torch.randn(a, b, m, dtype=torch.cfloat))

    def forward(self, x):
        b, _, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(b, self.b, x_ft.size(-1), device=x.device, dtype=torch.cfloat)
        m = min(self.m, x_ft.size(-1))
        out_ft[:, :, :m] = torch.einsum("bix,iox->box", x_ft[:, :, :m], self.w[:, :, :m])
        x = torch.fft.irfft(out_ft, n=n, dim=-1)
        return x


class Film(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, w), nn.GELU(), nn.Linear(w, 2 * w))

    def forward(self, t):
        return self.net(t)


class Net(nn.Module):
    def __init__(self, in_ch, modes, width, depth):
        super().__init__()
        self.fc0 = nn.Linear(in_ch, width)
        self.convs = nn.ModuleList([Spec(width, width, modes) for _ in range(depth)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, kernel_size=1) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.InstanceNorm1d(width, affine=False) for _ in range(depth)])
        self.films = nn.ModuleList([Film(width) for _ in range(depth)])
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, 1)
        self.act = nn.GELU()

    def forward(self, x):
        t = x[:, 0, -1].unsqueeze(-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        for c, w, n, f in zip(self.convs, self.ws, self.norms, self.films):
            y = c(x) + w(x)
            y = n(y)
            gb = f(t)
            g, b = gb.chunk(2, dim=1)
            y = y * (1.0 + g[:, :, None]) + b[:, :, None]
            x = self.act(y)
        x = x.permute(0, 2, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def r2(pred, targ):
    e = torch.linalg.norm(pred - targ, dim=(1, 2))
    d = torch.linalg.norm(targ, dim=(1, 2)) + 1e-12
    return e / d


def r2s(pred, targ):
    e = torch.sum((pred - targ) ** 2, dim=(1, 2))
    d = torch.sum(targ ** 2, dim=(1, 2)) + 1e-12
    return e / d


def pick_idx(times, eval_times, tol=1e-6):
    out = []
    seen = set()
    for t in eval_times:
        m = None
        for i, t_ref in enumerate(times):
            if abs(t_ref - t) <= tol:
                m = i
                break
        if m is None:
            raise ValueError("bad")
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def test(model, loader, dev):
    model.eval()
    tot_r = 0.0
    tot_m = 0.0
    cnt = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev)
            p = model(x)
            mse = torch.mean((p - y) ** 2).item()
            rel = r2(p, y)
            tot_r += rel.sum().item()
            tot_m += mse * x.size(0)
            cnt += x.size(0)
    return tot_m / cnt, tot_r / cnt


def eval_times(model, data_path, times, use_grid, batch_size, dev, eval_indices=None, input_time_index=0, time_input="t_out"):
    out = {}
    if eval_indices is None:
        eval_indices = list(range(len(times)))
    for t_idx in eval_indices:
        t_val = times[t_idx]
        ds = D2(data_path, times, use_grid=use_grid, time_indices=[t_idx], pair_mode="fixed", input_time_index=input_time_index, time_input=time_input)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        test_mse, test_rel = test(model, loader, dev)
        out[str(t_val)] = {"test_mse": test_mse, "test_rel_l2": test_rel}
    return out


def train(model, train_loader, val_loader, dev, epochs, lr, wd):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    hist = {"train_mse": [], "val_mse": [], "val_rel_l2": []}
    best_state = None
    best_val = float("inf")
    for _ in range(epochs):
        model.train()
        total = 0.0
        cnt = 0
        for x, y in train_loader:
            x = x.to(dev)
            y = y.to(dev)
            opt.zero_grad(set_to_none=True)
            p = model(x)
            l = torch.mean((p - y) ** 2)
            l.backward()
            opt.step()
            total += l.item() * x.size(0)
            cnt += x.size(0)
        train_mse = total / cnt
        val_mse, val_rel = test(model, val_loader, dev)
        hist["train_mse"].append(train_mse)
        hist["val_mse"].append(val_mse)
        hist["val_rel_l2"].append(val_rel)
        if val_rel < best_val:
            best_val = val_rel
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return hist, best_state, best_val


def load_pre(model_dir, dev):
    with open(model_dir / "metrics.json") as f:
        met = json.load(f)
    model = Net(in_ch=met["in_channels"], modes=met["modes"], width=met["width"], depth=met["depth"]).to(dev)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=dev))
    return model, met


def get():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=str(base / "FNO_data"))
    p.add_argument("--pretrained-dir", type=str, default=str(base / "task3_output"))
    p.add_argument("--out-dir", type=str, default=str(base / "task4_output"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--finetune-epochs", type=int, default=300)
    p.add_argument("--scratch-epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--eval-times", type=float, nargs="+", default=[1.0])
    p.add_argument("--eval-all-times", action="store_true")
    p.add_argument("--train-scratch", action="store_true")
    p.add_argument("--time-input", type=str, default=None, choices=["t_out", "dt"])
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

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    e_t = times if args.eval_all_times else args.eval_times
    eval_idx = pick_idx(times, e_t)
    pre_dir = Path(args.pretrained_dir)
    model, met = load_pre(pre_dir, d)
    use_grid = met["in_channels"] == 3

    time_input = args.time_input or met.get("time_input", "t_out")
    if met.get("time_input") is not None and met["time_input"] != time_input:
        pass
    test_unknown = data_dir / "data_test_unknown_128.npy"
    train_unknown = data_dir / "data_finetune_train_unknown_128.npy"
    val_unknown = data_dir / "data_finetune_val_unknown_128.npy"
    test_unknown_data = np.load(test_unknown)

    res = {
        "task": "part2_task4_finetune_unknown",
        "pretrained_dir": str(pre_dir),
        "in_channels": met["in_channels"],
        "modes": met["modes"],
        "width": met["width"],
        "depth": met["depth"],
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "finetune_epochs": args.finetune_epochs,
        "scratch_epochs": args.scratch_epochs,
        "pair_mode": "all2all",
        "num_times": len(times),
        "test_trajectories": int(test_unknown_data.shape[0]),
        "expected_pairs_per_trajectory": len(times) * (len(times) - 1) // 2,
        "metric_definition": "mean(||pred-true||_2 / ||true||_2)",
        "time_input": time_input,
        "eval_times": [times[i] for i in eval_idx],
        "eval_input_time": times[0],
        "zero_shot": {},
        "finetune": {},
    }

    model.eval()
    zt = eval_times(model, test_unknown, times, use_grid, args.batch_size, d, eval_indices=eval_idx, input_time_index=0, time_input=time_input)
    res["zero_shot"]["time_errors"] = zt
    res["zero_shot"]["test_rel_l2_t1"] = zt.get("1.0", {}).get("test_rel_l2")

    train_data = D2(train_unknown, times, use_grid=use_grid, pair_mode="all2all", time_input=time_input)
    val_data = D2(val_unknown, times, use_grid=use_grid, pair_mode="all2all", time_input=time_input)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    ft_model = Net(in_ch=met["in_channels"], modes=met["modes"], width=met["width"], depth=met["depth"]).to(d)
    ft_model.load_state_dict(model.state_dict())

    ft_hist, ft_best_state, ft_best_val = train(ft_model, train_loader, val_loader, d, args.finetune_epochs, args.lr, args.weight_decay)
    if ft_best_state is not None:
        ft_model.load_state_dict(ft_best_state)

    ft_times = eval_times(ft_model, test_unknown, times, use_grid, args.batch_size, d, eval_indices=eval_idx, input_time_index=0, time_input=time_input)
    res["finetune"]["time_errors"] = ft_times
    res["finetune"]["best_val_rel_l2"] = ft_best_val
    res["finetune"]["test_rel_l2_t1"] = ft_times.get("1.0", {}).get("test_rel_l2")
    res["finetune"]["train_trajectories"] = train_data.num_trajectories
    res["finetune"]["val_trajectories"] = val_data.num_trajectories
    res["finetune"]["train_pairs"] = train_data.num_pairs
    res["finetune"]["val_pairs"] = val_data.num_pairs
    res["finetune"]["pairs_per_trajectory"] = train_data.pairs_per_trajectory

    torch.save(ft_model.state_dict(), out_dir / "model_finetuned.pt")
    np.savez_compressed(
        out_dir / "finetune_history.npz",
        train_mse=np.array(ft_hist["train_mse"], dtype=np.float32),
        val_mse=np.array(ft_hist["val_mse"], dtype=np.float32),
        val_rel_l2=np.array(ft_hist["val_rel_l2"], dtype=np.float32),
    )

    if args.train_scratch:
        sc_model = Net(in_ch=met["in_channels"], modes=met["modes"], width=met["width"], depth=met["depth"]).to(d)
        sc_hist, sc_best_state, sc_best_val = train(sc_model, train_loader, val_loader, d, args.scratch_epochs, args.lr, args.weight_decay)
        if sc_best_state is not None:
            sc_model.load_state_dict(sc_best_state)
        sc_times = eval_times(sc_model, test_unknown, times, use_grid, args.batch_size, d, eval_indices=eval_idx, input_time_index=0, time_input=time_input)
        res["scratch"] = {
            "time_errors": sc_times,
            "best_val_rel_l2": sc_best_val,
            "test_rel_l2_t1": sc_times.get("1.0", {}).get("test_rel_l2"),
            "train_trajectories": train_data.num_trajectories,
            "val_trajectories": val_data.num_trajectories,
            "train_pairs": train_data.num_pairs,
            "val_pairs": val_data.num_pairs,
            "pairs_per_trajectory": train_data.pairs_per_trajectory,
        }
        torch.save(sc_model.state_dict(), out_dir / "model_scratch.pt")
        np.savez_compressed(
            out_dir / "scratch_history.npz",
            train_mse=np.array(sc_hist["train_mse"], dtype=np.float32),
            val_mse=np.array(sc_hist["val_mse"], dtype=np.float32),
            val_rel_l2=np.array(sc_hist["val_rel_l2"], dtype=np.float32),
        )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)

    if res["zero_shot"]["test_rel_l2_t1"] is not None:
        if res["finetune"]["test_rel_l2_t1"] is not None:
            pass

if __name__ == "__main__":
    go()
