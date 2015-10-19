//
//  MetalFunctions.metal
//  Birdbrain
//
//  Created by Jorden Hill on 10/9/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void sigmoid(device float *x [[ buffer(0) ]], device float *y [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {
    y[id] = 1.0 / (1.0 + exp(-x[id]));
}

kernel void tanh(device float *x [[buffer(0)]], device float *y [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
    y[id] = tanh(x[id]);
}

kernel void relu(device float *x [[buffer(0)]], device float *y [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
    y[id] = x[id] < 0.0 ? 0.0 : x[id];
}

kernel void sigmoid_prime(device float *x[[buffer(0)]], device float *y [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    y[id] = (1.0 / (1.0 + exp(-x[id]))) * (1.0 - (1.0 / (1.0 + exp(-x[id]))));
}

kernel void tanh_prime(device float *x [[buffer(0)]], device float *y [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
    y[id] = 1 - pow(tanh(x[id]), 2);
}

kernel void relu_prime(device float *x [[buffer(0)]], device float *y [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    y[id] = x[id] <= 0.0 ? 0.0 : 1.0;
}

kernel void add(device float *x [[buffer(0)]], device float *y [[buffer(1)]], device float
                *z [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    z[id] = x[id] + y[id];
}

kernel void subtract(device float *x [[buffer(0)]], device float *y [[buffer(1)]],
                     device float *z [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    z[id] = x[id] - y[id];
}

kernel void multiply(device float *x [[buffer(0)]], device float *y [[buffer(1)]],
                     device float *z [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    z[id] = x[id] * y[id];
}

kernel void scalar_multiply(device float *A [[buffer(0)]], constant float *c [[buffer(1)]],
                            device float *y [[buffer(2)]], uint id[[thread_position_in_grid]]) {
    y[id] = A[id] * c[0];
}