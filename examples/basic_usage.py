"""Compute milling stability on a process grid and for physical parameter rows."""

import math

import torch
import millstability


def main():
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA-enabled PyTorch installation and NVIDIA GPU are required.")

    device_id = 0
    physics = dict(
        Kt=6.0e8, Kn=2.0e8,
        w0x=922.0 * 2 * math.pi, w0y=922.0 * 2 * math.pi,
        zetax=0.011, zetay=0.011, m_tx=0.03993, m_ty=0.03993,
    )
    model = dict(N=2, aD=0.05, up_or_down=1, m=40, device_id=device_id)

    # Rows are spindle speeds in rpm; columns are axial cutting depths in metres.
    ei = millstability.milling_stability_ei_cuda(
        **physics, **model, stx=4, sty=3,
        w_st=0.0, w_fi=0.01, o_st=5000.0, o_fi=25000.0,
    )
    print("Spindle speeds (rpm):", torch.linspace(5000.0, 25000.0, 5))
    print("Cutting depths (m):", torch.linspace(0.0, 0.01, 4))
    print("Grid EI:\n", ei.cpu())
    print("Unstable points (EI > 1):\n", (ei > 1.0).cpu())

    # Columns: Kt, Kn, w0x, w0y, zetax, zetay, m_tx, m_ty.
    parameters = torch.tensor([
        [physics[key] for key in ("Kt", "Kn", "w0x", "w0y", "zetax", "zetay", "m_tx", "m_ty")],
        [5.4e8, 1.8e8, 890.0 * 2 * math.pi, 955.0 * 2 * math.pi,
         0.009, 0.014, 0.038, 0.042],
    ], dtype=torch.float32, device=f"cuda:{device_id}")
    ei = millstability.multi_parameter_single_point_ei(
        parameters, **model, w=0.004, o=15000.0,
    )
    print("EI per parameter row:", ei.cpu())


if __name__ == "__main__":
    main()
