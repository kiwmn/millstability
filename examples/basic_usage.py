"""Plot the two 2DOF up-milling cases in Table 2 of Ding et al. (2010)."""

import math
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import millstability


def main():
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA-enabled PyTorch installation and NVIDIA GPU are required.")

    physics = dict(
        N=2, Kt=6.0e8, Kn=2.0e8,
        w0x=922.0 * 2 * math.pi, w0y=922.0 * 2 * math.pi,
        zetax=0.011, zetay=0.011, m_tx=0.03993, m_ty=0.03993,
        up_or_down=1, m=40, device_id=0,
    )
    grid = dict(stx=400, sty=200, w_st=0.0, w_fi=0.01, o_st=5000.0, o_fi=25000.0)
    speeds = np.linspace(grid["o_st"], grid["o_fi"], grid["stx"] + 1)
    depths = np.linspace(grid["w_st"], grid["w_fi"], grid["sty"] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    for ax, immersion in zip(axes, (0.1, 0.05)):
        print(f"Computing 2DOF case a/D={immersion:g}...", flush=True)
        # Synchronize both ends to measure the complete CPU/GPU computation.
        torch.cuda.synchronize(physics["device_id"])
        started = perf_counter()
        ei = millstability.milling_stability_ei_cuda(**physics, **grid, aD=immersion)
        torch.cuda.synchronize(physics["device_id"])
        elapsed = perf_counter() - started
        print(f"a/D={immersion:g}: {elapsed:.2f} s ({ei.numel():,} points)", flush=True)
        # EI axes are [speed, depth]; contour expects [vertical axis, horizontal axis].
        ax.contour(speeds, depths, ei.cpu().numpy().T, levels=[1.0], colors="black", linewidths=1.0)
        ax.set_title(rf"$a/D = {immersion:g}$")
        ax.set_xlabel(r"Spindle speed $\Omega$ (rpm)")
        ax.set_ylabel(r"Axial depth $w$ (m)")
        ax.set_xlim(grid["o_st"], grid["o_fi"])
        ax.set_ylim(grid["w_st"], grid["w_fi"])
        ax.set_xticks(np.arange(5000, 25001, 5000))
        ax.set_yticks(np.linspace(0, 0.01, 11))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4), useMathText=True)
        ax.tick_params(direction="in", top=True, right=True)

    output = Path(__file__).with_name("two_dof_stability.png")
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
