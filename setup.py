from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="lltm_cpp",
    packages=["lltm_cpp", "lltm_cpp-stubs"],
    include_package_data=True,
    ext_modules=[
        cpp_extension.CUDAExtension(  # type: ignore
            "lltm_cpp",
            [
                "lltm_cpp/lltm_cuda.cpp",
                "lltm_cpp/lltm.cpp",
                "lltm_cpp/lltm_cpu.cpp",
                "lltm_cpp/lltm_cuda_kernel.cu",
            ],
            include_dirs=["lltm_cpp"],
        ),
    ],
    setup_requires=["pybind11-stubgen"],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    package_data={"lltm_cpp-stubs": ["*.pyi"]},
)
