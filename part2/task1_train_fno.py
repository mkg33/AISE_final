import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class D1(Dataset):
    def __init__(self, path, use_grid=True):
        data = np.load(path)
        self.u0 = data[:, 0].astype(np.float32)
        self.u1 = data[:, -1].astype(np.float32)
        self.use_grid = use_grid
        self.x = np.linspace(0.0, 1.0, self.u0.shape[-1], dtype=np.float32)

    def __len__(self):
        return self.u0.shape[0]

    def __getitem__(self, i):
        u0 = self.u0[i]
        if self.use_grid:
            inp = np.stack([u0, self.x], axis=-1)
        else:
            inp = u0[:, None]
        out = self.u1[i][:, None]
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


class Net(nn.Module):
    def __init__(self, in_ch, modes, width, depth):
        super().__init__()
        self.fc0 = nn.Linear(in_ch, width)
        self.convs = nn.ModuleList([Spec(width, width, modes) for _ in range(depth)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, kernel_size=1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        for c, w in zip(self.convs, self.ws):
            x = c(x) + w(x)
            x = self.act(x)
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


def test(net, loader, dev):
    net.eval()
    tot_r = 0.0
    tot_m = 0.0
    cnt = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev)
            p = net(x)
            mse = torch.mean((p - y) ** 2).item()
            rel = r2(p, y)
            tot_r += rel.sum().item()
            tot_m += mse * x.size(0)
            cnt += x.size(0)
    return tot_m / cnt, tot_r / cnt


def train(net, trl, val, dev, epochs, lr, wd):
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    h = {"train_mse": [], "val_mse": [], "val_rel_l2": []}
    best = None
    bestv = float("inf")
    for _ in range(epochs):
        net.train()
        total = 0.0
        cnt = 0
        for x, y in trl:
            x = x.to(dev)
            y = y.to(dev)
            opt.zero_grad(set_to_none=True)
            p = net(x)
            l = torch.mean((p - y) ** 2)
            l.backward()
            opt.step()
            total += l.item() * x.size(0)
            cnt += x.size(0)
        train_mse = total / cnt
        val_mse, val_rel = test(net, val, dev)
        h["train_mse"].append(train_mse)
        h["val_mse"].append(val_mse)
        h["val_rel_l2"].append(val_rel)
        if val_rel < bestv:
            bestv = val_rel
            best = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    return h, best, bestv


def get():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=str(base / "FNO_data"))
    p.add_argument("--out-dir", type=str, default=str(base / "task1_output"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--no-grid", action="store_true")
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

    data = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    use_grid = not args.no_grid
    in_ch = 2 if use_grid else 1

    tr = D1(data / "data_train_128.npy", use_grid=use_grid)
    va = D1(data / "data_val_128.npy", use_grid=use_grid)
    te = D1(data / "data_test_128.npy", use_grid=use_grid)

    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    val = DataLoader(va, batch_size=args.batch_size, shuffle=False)
    tel = DataLoader(te, batch_size=args.batch_size, shuffle=False)

    net = Net(in_ch=in_ch, modes=args.modes, width=args.width, depth=args.depth).to(d)

    h, best, bestv = train(net, trl, val, d, args.epochs, args.lr, args.weight_decay)

    if best is not None:
        net.load_state_dict(best)

    tm, trr = test(net, tel, d)

    torch.save(net.state_dict(), out / "model.pt")
    np.savez_compressed(
        out / "history.npz",
        train_mse=np.array(h["train_mse"], dtype=np.float32),
        val_mse=np.array(h["val_mse"], dtype=np.float32),
        val_rel_l2=np.array(h["val_rel_l2"], dtype=np.float32),
    )

    met = {
        "task": "part2_task1_one2one",
        "train_samples": len(tr),
        "val_samples": len(va),
        "test_samples": len(te),
        "in_channels": in_ch,
        "modes": args.modes,
        "width": args.width,
        "depth": args.depth,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "best_val_rel_l2": bestv,
        "test_mse": tm,
        "test_rel_l2": trr,
        "metric_definition": "mean(||pred-true||_2 / ||true||_2)",
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(met, f, indent=2)

    

if __name__ == "__main__":
    go()
