Scalable matrix matrix multiplication on FPGA
=============================================

This repository includes a pure Vivado HLS implementation of matrix-matrix
multiplication (A\*B=C) for Xilinx FPGAs, using Xilinx SDx to instantiate memory
and PCIe controllers and interface with the host. The code was developed at the
[Scalable Parallel Computing Lab](https://spcl.inf.ethz.ch/) at ETH Zurich. 

Experiments run on a [TUL KU115](http://www.tul.com.tw/ProductsFPGA.html)
achieved 263 GFLOP/s, 184 GFLOP/s and 81 GFLOP/s for half, single, and double
precision, respectively, with routing across the two SLRs being the primary
bottleneck preventing further scaling. The code is not device-specific, and can
be configured for any Xilinx FPGA supported by the SDAccel environment. 

The implementation uses a systolic array approach, where linearly connected
processing elements buffer distinct elements of the A-matrix, and B is streamed
through all elements. Both rows and columns are tiled to allow arbitrarily large
matrices. 

For a detailed description of the optimization techniques used here, we refer to
[this article](https://arxiv.org/abs/1805.08288).

The compute kernel is in `kernel/MatrixMatrix.cpp`, and the modules accessing
memory are in `kernel/Memory.cpp`.

Downloading the code
--------------------

This project uses the open source Vivado HLS extension library
[hlslib](https://github.com/definelicht/hlslib) for simulation, vectorization,
finding Xilinx tools, host-side integration and more.

Since hlslib is included as a submodule, make sure you initialize it before
building the code:

```
git submodule update --init 
```

Prerequisites
-------------

To build and run kernels in hardware, [Xilinx
SDAccel](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/sdaccel-development-environment.html)
must be installed and available on the PATH (tested with version 2017.1 and
2017.2).

Configuration and running
-------------------------

This project is configured and built using CMake. Most parameters must be set at
configuration-time, as they are used to specialize the hardware.

An example of configuring and building the kernel and executing it in hardware
is shown below (starting from the source directory):

```cpp
mkdir build
cd build
cmake ../ -DMM_DATA_TYPE=float -DMM_SIZE_N=8192 -DMM_SIZE_M=8192 -DMM_SIZE_P=8192 -DMM_TILE_SIZE_N=32 -DMM_KERNEL_WIDTH=8 -DMM_TILE_SIZE_P=2048
make
make synthesis
make build_kernel
./RunHardware
```

Per default the build targets the [TUL
KU115](http://www.tul.com.tw/ProductsFPGA.html) board, but this can be
configured using the `MM_DSA_NAME` CMake parameter.

The implementation is not restricted to use multiplication and addition as
operators. To use other operators, for example addition and minimum to implement
the [distance
product](https://en.wikipedia.org/wiki/Min-plus_matrix_multiplication), specify
them using the `MM_MAP_OP` and `MM_REDUCE_OP` CMake parameters, respectively. To
see which operators are pre-implemented, and examples of how to implement new
operators,  see `hlslib/include/hlslib/Operators.h`.

Parallel performance
--------------------

The amount of parallelism in the code is determined by the `MM_TILE_SIZE_N` and
`MM_KERNEL_WIDTH` configuration variables. The former determines how many values
of A are buffered and applies to every value of B streamed in, and the latter is
the vectorization factor. While vectorization consumes bandwidth, the tile size
doesn't, thus allowing the kernel to scale arbitrarily with logic and fast
memory on the chip.

The expected performance in Op/s (FLOP/s in the case of floating point types) of
a given configuration can be computed as:

`2 * TileSizeN * KernelWidth * Frequency`

I.e., Every cycle, `TileSizeN` buffered values of A are applied to `KernelWidth`
streamed in values of B. 

Bugs
----

If you experience bugs, or have suggestions for improvements, please use the
issue tracker to report them.
