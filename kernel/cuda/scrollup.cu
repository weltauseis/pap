#include "cppdefs.h"
#include "cuda_kernels.cuh"

EXTERN __global__ void scrollup_kernel_cuda(unsigned* image, unsigned* alt_image, unsigned DIM) {
	unsigned i = get_i();
	unsigned j = get_j();

	unsigned i2 = (i < DIM - 1) ? i + 1 : 0;

	next_img(i, j) = cur_img(i2, j);
}