from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="millstability",
    version="0.1.0",
    description="CUDA-accelerated milling stability analysis for PyTorch",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/kiwmn/millstability",
    license="MIT",
    python_requires=">=3.10",
    install_requires=["torch>=2.8"],
    packages=[],
    ext_modules=[
        CUDAExtension(
            name="millstability",
            sources=[
                str(Path("src") / source)
                for source in (
                    "bindings.cpp", "ei.cpp", "eigensolver.cpp", "ei_reduction.cu",
                    "matrices.cu", "grid_kernel.cu", "column_kernel.cu",
                    "parameter_kernel.cu", "parameter_column.cu",
                )
            ],
            include_dirs=[str(ROOT / "include")],
            libraries=["cudart", "cublas", "cusolver"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
