#ifndef EASYPAP_CUDA_KERNELS_CUH
#define EASYPAP_CUDA_KERNELS_CUH

#define get_i() (blockIdx.y * blockDim.y + threadIdx.y)
#define get_j() (blockIdx.x * blockDim.x + threadIdx.x)
#define get_index() ((get_i()) * DIM + (get_j()))

#define cur_img(i, j) image[(i) * DIM + (j)]
#define next_img(i, j) alt_image[(i) * DIM + (j)]

#endif  // EASYPAP_CUDA_KERNELS_CUH
