//
//  MetalFunctions.metal
//  Birdbrain
//
//  Created by Jorden Hill on 10/9/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

// USEFUL PROPERTIES THAT COMPARE OPENCL TO METAL:
// get_global_id = thread_position_in_grid
// get_local_id = thread_position_in_threadgroup
// get_global_size = thread_execution_width
// get_local_size = threads_per_threadgroup
// get_group_id = threadgroup_position_in_grid
// get_num_groups = threadgroups_per_grid

#define WARP_SIZE 32
#include <metal_stdlib>
using namespace metal;


typedef struct
{
  int m, n;
} MetalMatrixDim;


typedef struct
{
  ushort m;
} MetalVectorDim;

kernel void sigmoid(device float *x [[ buffer(0) ]],
                    device float *y [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]])
{
  y[id] = 1.0 / (1.0 + exp(-x[id]));
}

kernel void tanh(device float *x [[buffer(0)]],
                 device float *y [[buffer(1)]],
                 uint id [[thread_position_in_grid]])
{
  y[id] = tanh(x[id]);
}

kernel void relu(device float *x [[buffer(0)]],
                 device float *y [[buffer(1)]],
                 uint id [[thread_position_in_grid]])
{
  y[id] = fmax(x[id], 0.0);
}

kernel void sigmoid_prime(device float *x[[buffer(0)]],
                          device float *y [[buffer(1)]],
                          uint id [[thread_position_in_grid]])
{
  y[id] = (1.0 / (1.0 + exp(-x[id]))) * (1.0 - (1.0 / (1.0 + exp(-x[id]))));
}

kernel void tanh_prime(device float *x [[buffer(0)]],
                       device float *y [[buffer(1)]],
                       uint id [[thread_position_in_grid]])
{
  y[id] = 1 - pow(tanh(x[id]), 2);
}

kernel void relu_prime(device float *x [[buffer(0)]],
                       device float *y [[buffer(1)]],
                       uint id [[thread_position_in_grid]])
{
  y[id] = x[id] <= 0.0 ? 0.0 : 1.0;
}

kernel void add(device float *x [[buffer(0)]],
                device float *y [[buffer(1)]],
                device float *z [[buffer(2)]],
                constant MetalVectorDim& dim [[ buffer(3) ]],
                uint id [[thread_position_in_grid]])
{
  ushort m = dim.m;
  if (id >= m) {
    return;
  }
  
  z[id] = x[id] + y[id];
}

kernel void scalar_add(device float *A [[buffer(0)]],
                       constant float *c [[buffer(1)]],
                       device float *y [[buffer(2)]],
                       uint id[[thread_position_in_grid]])
{
  y[id] = A[id] + c[0];
}

kernel void subtract(device float *x [[buffer(0)]],
                     device float *y [[buffer(1)]],
                     device float *z [[buffer(2)]],
                     constant MetalVectorDim& dim [[ buffer(3) ]],
                     uint id [[thread_position_in_grid]])
{
  ushort m = dim.m;
  if (id >= m) {
    return;
  }
  
  z[id] = x[id] - y[id];
}

kernel void scalar_subtract(device float *A [[buffer(0)]],
                            constant float *c [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            uint id[[thread_position_in_grid]])
{
  y[id] = A[id] - c[0];
}

kernel void multiply(device float *x [[buffer(0)]],
                     device float *y [[buffer(1)]],
                     device float *z [[buffer(2)]],
                     constant MetalVectorDim& dim [[ buffer(3) ]],
                     uint id [[thread_position_in_grid]])
{
  ushort m = dim.m;
  if (id >= m) {
    return;
  }
  
  z[id] = x[id] * y[id];
}

kernel void scalar_multiply(device float *A [[buffer(0)]],
                            constant float *c [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            uint id[[thread_position_in_grid]])
{
  y[id] = A[id] * *c;
}

kernel void divide(device float *x [[buffer(0)]],
                   device float *y [[buffer(1)]],
                   device float *z [[buffer(2)]],
                   constant MetalVectorDim& dim [[ buffer(3) ]],
                   uint id [[thread_position_in_grid]])
{
  ushort m = dim.m;
  if (id >= m) {
    return;
  }
  z[id] = x[id] / y[id];
}

kernel void scalar_divide(device float *A [[buffer(0)]],
                          constant float *c [[buffer(1)]],
                          device float *y [[buffer(2)]],
                          uint id[[thread_position_in_grid]])
{
  y[id] = A[id] / c[0];
}
 
kernel void matrixvector_multiply(device float *A [[buffer(0)]],
                                  device float *x [[buffer(1)]],
                                  constant MetalMatrixDim& dims [[ buffer(2) ]],
                                  device float *y [[buffer (3)]],
                                  threadgroup float *partialDotProduct [[threadgroup(4)]],
                                  uint global_id [[thread_position_in_grid]],
                                  uint group_id [[threadgroup_position_in_grid]],
                                  uint num_groups [[threadgroups_per_grid]],
                                  uint local_size [[threads_per_threadgroup]],
                                  uint local_id [[thread_position_in_threadgroup]])
{
  int m = (int) dims.m;
  int n = (int) dims.n;
  
  for (int j = group_id; j < m; j += num_groups) {
    const device float* row = (const device float*)(A + j * n);

    float sum = 0.0f;
    for (int k = local_id; k < n; k += local_size) {
      sum += row[k] * x[k];
    }
  
    partialDotProduct[local_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint id = local_id & (WARP_SIZE - 1);
    
    float warpResult = 0.0f;
    if (local_id < (local_size / 2)) {
      volatile threadgroup float* p = partialDotProduct + 2 * local_id - id;
      p[0] += p[32];
      p[0] += p[16];
      p[0] += p[8];
      p[0] += p[4];
      p[0] += p[2];
      p[0] += p[1];
      warpResult = p[0];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (id == 0) {
      partialDotProduct[local_id / WARP_SIZE] = warpResult;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint size = local_size / (2 * WARP_SIZE);
    
    if (local_id < size / 2) {
      volatile threadgroup float* p = partialDotProduct + local_id;
      if (size >= 8)
        p[0] += p[4];
      if (size >= 4)
        p[0] += p[2];
      if (size >= 2)
        p[0] += p[1];
    }
    
    if (local_id == 0) {
        y[j] += partialDotProduct[0];
    }
  }
}

// TODO: Write kernel for getting sum of a vector

kernel void sum(device float *x [[buffer(0)]],
                device float *y [[buffer (1)]],
                uint id [[thread_position_in_grid]],
                uint gid [[threadgroup_position_in_grid]])
{
  ;
}

kernel void exp(device float *x [[buffer(0)]],
                device float *y [[buffer(1)]],
                uint id [[thread_position_in_grid]])
{
  y[id] = exp(x[id]);
}

kernel void square(device float *x [[buffer(0)]],
                   device float *y [[buffer(1)]],
                   uint id [[thread_position_in_grid]])
{
  y[id] = x[id] * x[id];
}

kernel void neg(device float *x [[buffer(0)]],
                device float *y [[buffer(1)]],
                uint id [[thread_position_in_grid]])
{
  y[id] = -x[id];
}
