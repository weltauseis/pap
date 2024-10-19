This repository contains my work for the 2023-2024 Parallel Architectures class taught at the university of Bordeaux. The simulations are implemented inside of [EasyPAP](https://gforgeron.gitlab.io/easypap/), a parallel programming framework written by my teachers specifically for this class.

The relevant code is in `kernel/c/sandPile.c`.

Below is the original README of EasyPAP.

---

[EasyPAP](https://gforgeron.gitlab.io/easypap) aims at providing students with an easy-to-use programming environment
to learn parallel programming.

The idea is to parallelize sequential computations on 2D matrices over multicore and GPU platforms. At each iteration,
the current matrix can be displayed, allowing to visually check the correctness of the computation method.

Multiple variants can easily been developed (e.g. sequential, tiled, omp_for, omp_task, ocl, mpi) and compared.

Most of the parameters can be specified as command line arguments, when running the program :
* size of the 2D matrices or image file to be loaded
* kernel to use (e.g. pixellize)
* variant to use (e.g. omp_task)
* maximum number of iterations to perform
* interactive mode / performance mode
* monitoring mode
* and much more!

Please read the [Getting Started Guide](https://gforgeron.gitlab.io/easypap/doc/Getting_Started.pdf) before you start!
