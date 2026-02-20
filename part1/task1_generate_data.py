import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def mk(n):
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    return x, y


def base(k, x, y, r):
    i = np.arange(1, k + 1)
    j = np.arange(1, k + 1)
    sa = np.sin(np.pi * i[:, None] * x[None, :])
    sb = np.sin(np.pi * j[:, None] * y[None, :])
    ii, jj = np.meshgrid(i, j, indexing="ij")
    c1 = (ii**2 + jj**2) ** r
    c2 = (ii**2 + jj**2) ** (r - 1.0)
    return sa, sb, c1, c2


def make(k, n, r, num, rng):
    x, y = mk(n)
    sa, sb, c1, c2 = base(k, x, y, r)
    f = np.empty((num, n, n), dtype=np.float32)
    u = np.empty((num, n, n), dtype=np.float32)
    aa = np.empty((num, k, k), dtype=np.float32)
    s1 = np.pi / (k**2)
    s2 = 1.0 / (np.pi * (k**2))
    for t in range(num):
        coeff = rng.normal(size=(k, k))
        w1 = coeff * c1
        w2 = coeff * c2
        f[t] = (s1 * np.einsum("ij,ix,jy->xy", w1, sa, sb)).astype(np.float32, copy=False)
        u[t] = (s2 * np.einsum("ij,ix,jy->xy", w2, sa, sb)).astype(np.float32, copy=False)
        aa[t] = coeff.astype(np.float32, copy=False)
    return x, y, f, u, aa


def save(out, k, n, r, x, y, f, u, aa):
    out.mkdir(parents=True, exist_ok=True)
    num = f.shape[0]
    out = out / f"poisson_K{k}_N{n}_S{num}.npz"
    np.savez_compressed(
        out,
        x=x,
        y=y,
        f=f,
        u=u,
        a=aa,
        r=np.array(r, dtype=np.float32),
        K=np.array(k, dtype=np.int32),
    )
    return out


def draw(out, k, x, y, f, u, num_plot):
    out.mkdir(parents=True, exist_ok=True)
    num_plot = min(num_plot, f.shape[0])
    fig, axes = plt.subplots(num_plot, 2, figsize=(8, 3 * num_plot), constrained_layout=True)
    axes = np.atleast_2d(axes)
    x0, x1 = float(np.min(x)), float(np.max(x))
    y0, y1 = float(np.min(y)), float(np.max(y))
    if x0 == x1:
        x0, x1 = 0.0, float(f.shape[1] - 1)
    if y0 == y1:
        y0, y1 = 0.0, float(f.shape[2] - 1)
    ext = (x0, x1, y0, y1)
    for i in range(num_plot):
        ax1, ax2 = axes[i]
        im1 = ax1.imshow(f[i], origin="lower", extent=ext, cmap="viridis")
        ax1.set_title(f"K={k} f {i + 1}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        im2 = ax2.imshow(u[i], origin="lower", extent=ext, cmap="viridis")
        ax2.set_title(f"K={k} u {i + 1}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    out = out / f"examples_K{k}_N{len(x)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def get():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--num-plot", type=int, default=3)
    p.add_argument("--r", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="task1_output")
    return p.parse_args()


def go():
    a = get()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir)
    d0 = out / "data"
    p0 = out / "plots"
    for k in a.k_values:
        x, y, f, u, aa = make(k, a.n, a.r, a.num_samples, rng)
        save(d0, k, a.n, a.r, x, y, f, u, aa)
        draw(p0, k, x, y, f, u, a.num_plot)


if __name__ == "__main__":
    go()
