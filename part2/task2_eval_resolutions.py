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


def r2(pred, targ):
    e = torch.linalg.norm(pred - targ, dim=(1, 2))
    d = torch.linalg.norm(targ, dim=(1, 2)) + 1e-12
    return e / d


def r2s(pred, targ):
    e = torch.sum((pred - targ) ** 2, dim=(1, 2))
    d = torch.sum(targ ** 2, dim=(1, 2)) + 1e-12
    return e / d


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


def get():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=str(base / "FNO_data"))
    p.add_argument("--model-dir", type=str, default=str(base / "task1_output"))
    p.add_argument("--out-dir", type=str, default=str(base / "task2_output"))
    p.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 96, 128])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def dev(flag):
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def go():
    args = get()
    d = dev(args.device)

    data = Path(args.data_dir)
    m = Path(args.model_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(m / "metrics.json") as f:
        met = json.load(f)

    ch = met["in_channels"]
    use_g = ch == 2

    net = Net(in_ch=ch, modes=met["modes"], width=met["width"], depth=met["depth"]).to(d)
    net.load_state_dict(torch.load(m / "model.pt", map_location=d))

    res = {
        "model_dir": str(m),
        "in_channels": ch,
        "modes": met["modes"],
        "width": met["width"],
        "depth": met["depth"],
        "batch_size": args.batch_size,
        "metric_definition": "mean(||pred-true||_2 / ||true||_2)",
        "resolutions": {},
    }

    for r in args.resolutions:
        path = data / f"data_test_{r}.npy"
        if not path.exists():
            raise FileNotFoundError("missing")
        ds = D1(path, use_grid=use_g)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        tm, trr = test(net, loader, d)
        res["resolutions"][str(r)] = {"test_mse": tm, "test_rel_l2": trr, "samples": len(ds)}
        
    with open(out / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    go()
