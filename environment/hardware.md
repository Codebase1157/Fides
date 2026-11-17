# Hardware and software environment

All experiments in the manuscript were run on a single workstation with the
following configuration. Any environment that provides a comparable CUDA
toolchain and >= 20 GB of GPU memory should reproduce the numbers within
noise; cross-GPU drift of 0.1 to 0.3 points in ACC is normal.

## Hardware

| Component | Specification |
|-----------|---------------|
| CPU       | AMD EPYC 7542 (32-core) |
| RAM       | 256 GB DDR4 |
| GPU       | 1x NVIDIA GeForce RTX 4090 (24 GB) |
| Storage   | 2 TB NVMe SSD |
| OS        | Ubuntu 22.04.3 LTS |

## Software

| Component | Version |
|-----------|---------|
| CUDA      | 12.1    |
| cuDNN     | 8.9.2   |
| Python    | 3.10.13 |
| PyTorch   | 2.1.2   |
| nvcc      | 12.1.r12.1/compiler.32688072_0 |
| GCC       | 11.4.0  |

## Determinism

We set deterministic seeds through `src/utils/seed.py` and run with
`torch.use_deterministic_algorithms(True)` wherever the ops permit. A small
amount of non-determinism remains in `scaled_dot_product_attention`; repeats
are reported with their standard deviation over ten runs.
