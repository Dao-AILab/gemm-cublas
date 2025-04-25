# Copyright (c) 2025, Tri Dao.
import ast
import re
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch import __version__ as torch_version_raw
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

PACKAGE_NAME = "gemm_cublas"

if torch_version_raw >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False
# Currently we can't compile with py_limited_api
py_limited_api = False

this_dir = Path(__file__).parent


def get_extensions():
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-fdiagnostics-color=always",
        ] + ["-DPy_LIMITED_API=0x03090000"] if py_limited_api else [],  # min CPython version 3.9
    }
    extensions_dir = this_dir / "csrc"

    ext_modules = [
        CUDAExtension(
            name=f"{PACKAGE_NAME}._C",
            sources=[f"{extensions_dir}/gemm_cublas.cpp"],
            libraries=["cublas"],
            extra_compile_args=extra_compile_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


def get_package_version():
    with open(this_dir / PACKAGE_NAME / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    package_version = ast.literal_eval(version_match.group(1))

    # custom Torch builds often violate semver, examples:
    # NGC: 2.6.0a0+df5bbc09d1.nv24.11
    # nightly torch: 2.6.0.dev20241117

    # ignore local build suffixes
    torch_version = torch_version_raw.split("+")[0].replace(".", "")
    # signal nightly builds
    if "dev" in torch_version:
        torch_version = torch_version[: torch_version.find("dev") + 3]

    fully_qualified_version = f"{package_version}+torch{torch_version}"
    print(f"Fully qualified version: {fully_qualified_version}")
    return fully_qualified_version


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="GEMM using cuBLAS, with a PyTorch interface",
    # Just 1 file, don't need ninja
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
