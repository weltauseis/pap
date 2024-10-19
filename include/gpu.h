#ifndef EASYPAP_GPU_H
#define EASYPAP_GPU_H

#include "hooks.h"

#ifdef ENABLE_OPENCL ////////// OPENCL //////////
#include "ocl.h"

#define GPU_CAN_BE_USED 1

#define DEFAULT_GPU_VARIANT DEFAULT_OCL_VARIANT

#define easypap_number_of_gpus() easypap_number_of_gpus_ocl ()

#define gpu_init(b1, b2) ocl_init (b1, b2)

#define gpu_build_program(b) ocl_build_program (b)

#define gpu_alloc_buffers() ocl_alloc_buffers ()

#define gpu_send_data() ocl_send_data ()

#define gpu_retrieve_data() ocl_retrieve_data ()

#define gpu_map_textures(id) ocl_map_textures (id)

#define gpu_update_texture() ocl_update_texture ()

static inline void gpu_establish_bindings (void)
{
  the_compute = bind_it (kernel_name, "invoke", variant_name, 0);
  if (the_compute == NULL) {
    the_compute = ocl_invoke_kernel_generic;
    PRINT_DEBUG ('c', "Using generic [%s] OpenCL kernel launcher\n",
                 "ocl_compute");
  }
}

#elif defined(ENABLE_CUDA) ////////// CUDA //////////
#include "nvidia_cuda.h"

#define GPU_CAN_BE_USED 1

#define DEFAULT_GPU_VARIANT DEFAULT_CUDA_VARIANT

#define easypap_number_of_gpus() easypap_number_of_gpus_cuda ()

#define gpu_init(b1, b2) cuda_init (b1, b2)

#define gpu_build_program(b) cuda_build_program (b)

#define gpu_alloc_buffers() cuda_alloc_buffers ()

#define gpu_send_data() cuda_send_data ()

#define gpu_retrieve_data() cuda_retrieve_data ()

#define gpu_map_textures(id) cuda_map_textures (id)

#define gpu_update_texture() cuda_update_texture ()

static inline void gpu_establish_bindings (void)
{
  the_compute = bind_it (kernel_name, "invoke", variant_name, 0);
  if (the_compute == NULL) {
    the_compute = cuda_compute;
    PRINT_DEBUG ('c', "Using generic [%s] CUDA kernel launcher\n",
                 "cuda_compute");
  }
  the_cuda_kernel = bind_it (kernel_name, "kernel", variant_name, 1);
  the_cuda_kernel_finish = bind_it (kernel_name, "finish", variant_name, 0);
}

#else ////////// NO GPU //////////

#define GPU_CAN_BE_USED 0

#define DEFAULT_GPU_VARIANT NULL

#define easypap_number_of_gpus() 0

#define gpu_init(b1, b2)

#define gpu_build_program(b)

#define gpu_alloc_buffers()

#define gpu_send_data()

#define gpu_retrieve_data()

#define gpu_map_textures(id)

#define gpu_update_texture()

#define gpu_establish_bindings()


#endif

#endif // EASYPAP_GPU_H