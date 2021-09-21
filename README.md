Scalable matrix matrix multiplication on FPGA
=============================================

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3952084.svg)](https://doi.org/10.5281/zenodo.3952084)

This repository includes a pure Vitis HLS implementation of matrix-matrix multiplication (A\*B=C) for Xilinx FPGAs, using Xilinx Vitis to instantiate memory and PCIe controllers and interface with the host. 

Experiments run on a [VCU1525](https://www.xilinx.com/products/boards-and-kits/vcu1525-a.html) achieved 462 GFLOP/s, 301 GFLOP/s and 132 GFLOP/s for half, single, and double precision, respectively, with routing across the three SLRs being the primary bottleneck preventing further scaling. The code is not device-specific, and can be configured for any Xilinx FPGA supported by the Xilinx OpenCL runtime.  Kernels have also been verified to execute on TUL KU115, Alveo U250, and Alveo U280 boards with similar results.

The implementation uses a systolic array approach, where linearly connected processing elements compute distinct contributions to the outer product of tiles of the output matrix. 

The approach used to implement this kernel was presented at [FPGA'20](https://spcl.inf.ethz.ch/Publications/.pdf/gemm-fpga.pdf) [1].  For a general description of the optimization techniques that we apply, we refer to our article on [HLS transformations](https://spcl.inf.ethz.ch/Publications/.pdf/hls-transformations.pdf) [2].  We also gave [a tutorial on HLS](https://spcl.inf.ethz.ch/Teaching/hls-tutorial/) for HPC at SC'21, ISC'21, SC'20, HiPEAC'20, SC'19, SC'18, and PPoPP'18.

Downloading the code
--------------------

This project uses the open source Vivado HLS extension library [hlslib](https://github.com/definelicht/hlslib) [3] for simulation, vectorization, finding Xilinx tools, host-side integration and more.

Since hlslib is included as a submodule, make sure you clone with `--recursive` or grab it after cloning with:

```
git submodule update --init 
```

Prerequisites
-------------

To build and run kernels in hardware, Xilinx [Vitis](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html) must be installed and available on the PATH (tested on Alveo U250 and Alveo U280 with version 2021.1).

Configuration and running
-------------------------

This project is configured and built using CMake. Most parameters must be set at configuration-time, as they are used to specialize the hardware.

An example of configuring and building the kernel and executing it in hardware is shown below (starting from the source directory):

```bash
mkdir build
cd build
cmake ../ -DMM_DATA_TYPE=float -DMM_PARALLELISM_N=32 -DMM_PARALLELISM_M=8 -DMM_MEMORY_TILE_SIZE_N=512 -DMM_MEMORY_TILE_SIZE_M=512
make
make hw
./RunHardware.exe 1024 1024 1024 hw
```

Matrix sizes use the convention that `A: NxK`, `B: KxM`, and `C: NxM`.

Per default the build targets the Alveo U250 acceleration board, but this can be configured using the `MM_PLATFORM` CMake parameter.

The implementation is not restricted to use multiplication and addition as operators. To use other operators, for example addition and minimum to implement the [distance product](https://en.wikipedia.org/wiki/Min-plus_matrix_multiplication), specify them using the `MM_MAP_OP` and `MM_REDUCE_OP` CMake parameters, respectively. To see which operators are pre-implemented, and examples of how to implement new operators,  see `hlslib/include/hlslib/xilinx/Operators.h`.

Selecting tile sizes
--------------------

See our [publication at FPGA'20](https://spcl.inf.ethz.ch/Publications/.pdf/gemm-fpga.pdf) [1] on how to choose tile sizes for optimal fast memory and compute utilization.

Parallel performance
--------------------

The amount of parallelism in the code is determined by the `MM_PARALLELISM_N` and `MM_PARALLELISM_M` configuration variables. The former determines the number of processing element instantiated, and the latter regulates the vector width/granularity of each processing element.  `MM_PARALLELISM_M` should be set to a maximum of 64 bytes / `sizeof(<your operand>)` (i.e., 8 for `float` or `int`, 4 for `double` or `long`, 16 for 16-bit `int`, etc.) to avoid performance and routing issues.

The expected performance in Op/s (FLOP/s in the case of floating point types) of a given configuration can be computed as:

`2 * MM_PARALLELISM_N * MM_PARALLELISM_M * Frequency`

In practice, `MM_PARALLELISM_N` buffered values of A are applied to `MM_PARALLELISM_M` values of B. 

Bugs
----

If you experience bugs, or have suggestions for improvements, please use the issue tracker to report them.

Publication
-----------

If this code has been useful to your research, please consider citing us:

**BibTeX:**
```
@inproceedings{mmm_hls,
  title={Flexible Communication Avoiding Matrix Multiplication on FPGA with High-Level Synthesis},
  author={de~Fine~Licht, Johannes and Kwasniewski, Grzegorz and Hoefler, Torsten},
  booktitle={The 2020 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA'20)},
  year={2020}
}
```

**Plain text:**
```
Johannes de Fine Licht, Grzegorz Kwasniewski, and Torsten Hoefler. "Flexible Communication Avoiding Matrix Multiplication on FPGA with High-Level Synthesis." In Proceedings of the 2020 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA'20).
```

References
----------

[1] Johannes de Fine Licht, Grzegorz Kwasniewski, and Torsten Hoefler, _"Flexible Communication Avoiding Matrix Multiplication on FPGA with High-Level Synthesis"_, in Proceedings of 28th ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA'20), 2020.

[2] Johannes de Fine Licht, Maciej Besta, Simon Meierhans, and Torsten Hoefler. _"Transformations of High-Level Synthesis Codes for High-Performance Computing."_ IEEE Transactions on Parallel and Distributed Systems (TPDS), Vol. 32, Issue 5, 2021.

[3] Johannes de Fine Licht, and Torsten Hoefler. _"hlslib: Software Engineering for Hardware Design."_, presented at the Fifth International Workshop on
Heterogeneous High-performance Reconfigurable Computing (H2RC'19).
