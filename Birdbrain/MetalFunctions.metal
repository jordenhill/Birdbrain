//
//  MetalFunctions.metal
//  Birdbrain
//
//  Created by Jorden Hill on 10/9/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void sigmoid(const device float *x [[ buffer(0) ]],
                    device float *y [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {
    y[id] = 1.0 / (1.0 + exp(-x[id]));
}

kernel void tanh(const device float *x [[buffer(0)]], device float *y [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
    y[id] = tanh(x[id]);
}

kernel void relu(const device float *x [[buffer(0)]], device float *y [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
    y[id] = x[id] < 0.0 ? 0.0 : x[id];
}

kernel void sigmoid_prime(const device float *x[[buffer(0)]], device float *y [[buffer(1)]],
                         uint id [[thread_position_in_grid]]) {
    y[id] = (1.0 / (1.0 + exp(-x[id]))) * (1.0 - (1.0 / (1.0 + exp(-x[id]))));
}

kernel void tanh_prime(const device float *x [[buffer(0)]], device float *y [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
    y[id] = 1 - pow(tanh(x[id]), 2);
}

kernel void relu_prime(const device float *x [[buffer(0)]], device float *y [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    y[id] = x[id] <= 0.0 ? 0.0 : 1.0;
}

kernel void add(const device float *x [[buffer(0)]], device float *y [[buffer(1)]], device float
                *z [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    z[id] = x[id] + y[id];
}

kernel void subtract(const device float *x [[buffer(0)]], device float *y [[buffer(1)]], device float
                *z [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    z[id] = x[id] - y[id];
}

kernel void multiply(const device float *x [[buffer(0)]], device float *y [[buffer(1)]], device float
                *z [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    z[id] = x[id] * y[id];
}