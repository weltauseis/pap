#include <cstdio>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include "constants.h"
#include "global.h"
#include "img_data.h"
#include "hooks.h"
#include "cppdefs.h"

GLuint texid;
cudaGraphicsResource_t texResource;
cudaSurfaceObject_t surfaceObject;
uint32_t *gpu_image, *gpu_alt_image;

dim3 grid, block;

int nb_gpu;
int nb_threads_per_block;

static void cuda_show_devices(void) {
  int nb_devices;
  cudaGetDeviceCount(&nb_devices);

  printf("Number of devices: %d\n", nb_devices);

  for (int i = 0; i < nb_devices; i++) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, i);

    printf("Device %d: [%s]\n", i, props.name);
    printf("\tGlobal memory: %lu GB\n", props.totalGlobalMem / 1024 / 1024 / 1024);
    printf("\tConstant memory: %lu KB\n", props.totalConstMem / 1024);
    printf("\tShared memory per block: %lu KB\n", props.sharedMemPerBlock / 1024);
    printf("\tRegisters per block: %d\n", props.regsPerBlock);
    printf("\tThreads per block: %d\n", props.maxThreadsPerBlock);
    printf("\tNumber of SM: %d\n", props.multiProcessorCount);
    printf("\tFrequency: %d MHz\n", props.clockRate / 1000);
  }
}

EXTERN unsigned easypap_number_of_gpus_cuda (void)
{
  return (gpu_used ? 1 : 0);
}

EXTERN void cuda_send_data (void){
  cudaMemcpy(gpu_image, image, DIM * DIM * sizeof(uint32_t), cudaMemcpyHostToDevice);
}

EXTERN void cuda_retrieve_data (void){
  cudaMemcpy(image, gpu_image, DIM * DIM * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}


EXTERN void cuda_init(int show_config, int silent) {

  if (show_config){
    cuda_show_devices();
    exit(0);
  }

  // get number of devices
  cudaGetDeviceCount(&nb_gpu);

  int id_gpu = 0;
  char *str = getenv("DEVICE");
  if (str != NULL)
    id_gpu = atoi(str);

  if (id_gpu >= nb_gpu) {
    fprintf(stderr, "Error: GPU %d does not exist\n", id_gpu);
    exit(EXIT_FAILURE);
  }

  cudaSetDevice(id_gpu);

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, id_gpu);
  nb_threads_per_block = props.maxThreadsPerBlock;

  if (TILE_W * TILE_H > nb_threads_per_block) {
    fprintf(stderr, "Error: TILE_SIZE > %d\n", nb_threads_per_block);
    exit(EXIT_FAILURE);
  }

  printf("Using GPU %d: [%s]\n", id_gpu, props.name);

}

EXTERN void cuda_alloc_buffers (void){
  cudaMalloc((void **)&gpu_image, DIM * DIM * sizeof(uint32_t));
  cudaMalloc((void **)&gpu_alt_image, DIM * DIM * sizeof(uint32_t));
}

EXTERN void cuda_build_program (int list_variants){
    if (list_variants){
      printf("currently no variants for cuda\n");
      exit(0);
    }

    grid.x = DIM / TILE_W;
    grid.y = DIM / TILE_H;
    grid.z = 1;

    block.x = TILE_W;
    block.y = TILE_H;
    block.z = 1;

    printf("Tile size: %dx%d\n", TILE_W, TILE_H);
}


EXTERN unsigned cuda_compute(unsigned nb_iter) {

  // call kernel
  for (int i = 0; i < nb_iter; i++) {

    the_cuda_kernel<<<grid, block>>>(gpu_image, gpu_alt_image, DIM);
    cudaDeviceSynchronize();
    if (the_cuda_kernel_finish != NULL){
      the_cuda_kernel_finish<<<1,1>>>(DIM);
      cudaDeviceSynchronize();
		}

    // swap images
    uint32_t *tmp = gpu_image;
    gpu_image = gpu_alt_image;
    gpu_alt_image = tmp;
  }

  return 0;
}

EXTERN void cuda_map_textures(unsigned tex) {
  texid = tex;
  // register texture with CUDA
  cudaGraphicsGLRegisterImage(&texResource, texid, GL_TEXTURE_2D,
                              cudaGraphicsRegisterFlagsSurfaceLoadStore);

  cudaGraphicsMapResources(1, &texResource);

  // Récupération d'un cudaArray représentant la texture
  cudaArray_t array;
  cudaGraphicsSubResourceGetMappedArray(&array, texResource, 0, 0);

  // Création d'un objet de surface à partir du cudaArray
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = array;

  cudaCreateSurfaceObject(&surfaceObject, &resDesc);
  cudaGraphicsMapResources(1, &texResource);
  // note : we should call somewhere
  // cudaGraphicsUnmapResources(1, &texResource)

}

__global__ void cuda_update_texture(cudaSurfaceObject_t target, unsigned *image,
                                    unsigned dim) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dim && y < dim) {

    unsigned pixel = image[y * dim + x];
    // pixel to uchar4
    uchar4 data = make_uchar4(pixel >> 24, (pixel >> 16) & 255,
                              (pixel >> 8) & 255, pixel & 255);
    surf2Dwrite(data, target, x * sizeof(uchar4), y);
  }
}

EXTERN void cuda_update_texture(void) {

  //cudaGraphicsMapResources(1, &texResource);

  cuda_update_texture<<<grid, block>>>(surfaceObject, gpu_image, DIM);
  cudaDeviceSynchronize();

  //cudaGraphicsUnmapResources(1, &texResource);
}
