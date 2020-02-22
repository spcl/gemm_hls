Scalable matrix matrix multiplication on FPGA
=============================================

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3559536.svg)](https://doi.org/10.5281/zenodo.3559536)

This repository includes a pure Vivado HLS implementation of matrix-matrix
multiplication (A\*B=C) for Xilinx FPGAs, using Xilinx SDx to instantiate memory
and PCIe controllers and interface with the host. 

Experiments run on a [VCU1525](https://www.xilinx.com/products/boards-and-kits/vcu1525-a.html)
achieved 462 GFLOP/s, 301 GFLOP/s and 132 GFLOP/s for half, single, and double
precision, respectively, with routing across the three SLRs being the primary
bottleneck preventing further scaling. The code is not device-specific, and can
be configured for any Xilinx FPGA supported by the SDAccel environment. 

The implementation uses a systolic array approach, where linearly connected
processing elements compute distinct contributions to the outer product of tiles
of the output matrix. 

For a detailed description of the optimization techniques used here, we refer to
[this article](https://arxiv.org/abs/1805.08288). We also gave [a tutorial on
HLS](https://spcl.inf.ethz.ch/Teaching/2019-sc/) for HPC at SC'19, SC'18, PPoPP'18, and
at ETH Zurich. 

The compute kernel is in `kernel/Compute.cpp`, and the modules accessing memory
are in `kernel/Memory.cpp`.

Downloading the code
--------------------

This project uses the open source Vivado HLS extension library
[hlslib](https://github.com/definelicht/hlslib) for simulation, vectorization,
finding Xilinx tools, host-side integration and more.

Since hlslib is included as a submodule, make sure you clone with `--recursive`
or grab it after cloning with:

```
git submodule update --init 
```

Prerequisites
-------------

To build and run kernels in hardware, Xilinx
[Vitis](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html) or SDAccel
must be installed and available on the PATH (tested with versions 2018.2 and 2019.2).

Configuration and running
-------------------------

This project is configured and built using CMake. Most parameters must be set at
configuration-time, as they are used to specialize the hardware.

An example of configuring and building the kernel and executing it in hardware
is shown below (starting from the source directory):

```cpp
mkdir build
cd build
cmake ../ -DMM_DATA_TYPE=float -DMM_SIZE_N=8192 -DMM_SIZE_M=8192 -DMM_SIZE_P=8192 -DMM_PARALLELISM_N=32 -DMM_PARALLELISM_M=8 -DMM_MEMORY_TILE_SIZE_N=512 -DMM_MEMORY_TILE_SIZE_M=512
make
make synthesis
make compile_hardware 
make link_hardware
./RunHardware.exe hw
```

Matrix sizes use the convention that `A: NxK`, `B: KxM`, and `C: NxM`. For
dynamic matrix sizes, see below.

Per default the build targets the
[VCU1525](https://www.xilinx.com/products/boards-and-kits/vcu1525-a.html)
acceleration board, but this can be configured using the `MM_DSA_NAME` CMake
parameter.

The implementation is not restricted to use multiplication and addition as
operators. To use other operators, for example addition and minimum to implement
the [distance
product](https://en.wikipedia.org/wiki/Min-plus_matrix_multiplication), specify
them using the `MM_MAP_OP` and `MM_REDUCE_OP` CMake parameters, respectively. To
see which operators are pre-implemented, and examples of how to implement new
operators,  see `hlslib/include/hlslib/Operators.h`.

Dynamic sizes
-------------

By default, all matrix sizes are fixed in hardware at compile time to save
hardware resources. If dynamic/runtime sizes are required, set
the `-DMM_DYNAMIC_SIZES=ON` option in CMake. This will require executables to
take the matrix sizes as input. 

Selecting tile sizes
--------------------

See our [publication at FPGA'20](https://spcl.inf.ethz.ch/Publications/.pdf/gemm-fpga.pdf) [1] on how to choose tile sizes for optimal fast memory and compute utilization.

Parallel performance
--------------------

The amount of parallelism in the code is determined by the `MM_PARALLELISM_N`
and `MM_PARALLELISM_M` configuration variables. The former determines the number
of processing element instantiated, and the latter regulates the vector
width/granularity of each processing element. `MM_PARALLELISM_M` should be set
to a maximum of 8 to avoid performance and routing issues.

The expected performance in Op/s (FLOP/s in the case of floating point types) of
a given configuration can be computed as:

`2 * MM_PARALLELISM_N * MM_PARALLELISM_M * Frequency`

In practice, `MM_PARALLELISM_N` buffered values of A are applied to
`MM_PARALLELISM_M` values of B. 

Bugs
----

If you experience bugs, or have suggestions for improvements, please use the
issue tracker to report them.

References
----------

[1] Johannes de Fine Licht, Grzegorz Kwasniewski, and Torsten Hoefler, _"Flexible Communication Avoiding Matrix Multiplication on FPGA with High-Level Synthesis"_, in Proceedings of 28th ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA'20), 2020.
