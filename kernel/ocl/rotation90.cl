#include "kernel/ocl/common.cl"


__kernel void rotation90_ocl (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  out [(DIM - x - 1) * DIM + y] = in [y * DIM + x];
}

