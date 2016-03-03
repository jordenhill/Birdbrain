//
//  LSTMNetwork.swift
//  Birdbrain
//
//  Created by Jorden Hill on 12/3/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

import Foundation

public class LSTMNetwork {
  var wgx: [Float]
  var wix: [Float]
  var wfx: [Float]
  var wox: [Float]
  var wgh: [Float]
  var wfh: [Float]
  var wih: [Float]
  var woh: [Float]
  var inputDim: Int
  var memCellCount: Int
  var GPU: MetalDevice!
  var useMetal: Bool
  
  public init(inputDim: Int, useMetal: Bool, memCellCount: Int) {
    self.inputDim = inputDim
    self.memCellCount = memCellCount
    self.useMetal = useMetal
    wgx = (1...inputDim * memCellCount).map{_ in initRand(inputDim)}
    wix = (1...inputDim * memCellCount).map{_ in initRand(inputDim)}
    wfx = (1...inputDim * memCellCount).map{_ in initRand(inputDim)}
    wox = (1...inputDim * memCellCount).map{_ in initRand(inputDim)}
    wgh = (1...memCellCount * memCellCount).map{_ in initRand(inputDim)}
    wfh = (1...memCellCount * memCellCount).map{_ in initRand(inputDim)}
    wih = (1...memCellCount * memCellCount).map{_ in initRand(inputDim)}
    woh = (1...memCellCount * memCellCount).map{_ in initRand(inputDim)}
    
    if (useMetal) {
      GPU = MetalDevice()
    }
  }
  
  public func feedforward(input: [[Float]])
    -> ([[Float]], [[Float]], [[Float]]) {
      
      if (useMetal) {
        return GPUCompute(input)
      } else {
        return CPUCompute(input)
      }
  }
  
  private func GPUCompute(input: [[Float]]) -> ([[Float]], [[Float]], [[Float]]) {
    let T = input.count
    let start: [Float] = (1...memCellCount).map{_ in 0.0}
    var g: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    var f: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    var i: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    var o: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    var s: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    var h: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    var p: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    
    g[0] = GPU.tanh(GPU.add(GPU.mvMul(wgx, m: memCellCount, n: inputDim, vector: input[0]),
      y: GPU.mvMul(wgh, m: memCellCount, n: memCellCount, vector: start)))
    i[0] = GPU.sigmoid(GPU.add(GPU.mvMul(wix, m: memCellCount, n: inputDim, vector: input[0]),
      y: GPU.mvMul(wih, m: memCellCount, n: memCellCount, vector: start)))
    f[0] = GPU.sigmoid(GPU.add(GPU.mvMul(wfx, m: memCellCount, n: inputDim, vector: input[0]),
      y: GPU.mvMul(wfh, m: memCellCount, n: memCellCount, vector: start)))
    o[0] = GPU.sigmoid(GPU.add(GPU.mvMul(wox, m: memCellCount, n: inputDim, vector: input[0]),
      y: GPU.mvMul(woh, m: memCellCount, n: memCellCount, vector: start)))
    s[0] = add(GPU.mul(g[0], y: i[0]), y: mul(s[0], y: f[0]))
    h[0] = mul(s[0], y: o[0])
    p[0] = softmax(s[0])
    
    for t in 1..<T {
      g[t] = GPU.tanh(GPU.add(GPU.mvMul(wgx, m: memCellCount, n: inputDim, vector: input[t]),
        y: GPU.mvMul(wgh, m: memCellCount, n: memCellCount, vector: h[t - 1])))
      i[t] = GPU.sigmoid(add(GPU.mvMul(wix, m: memCellCount, n: inputDim, vector: input[t]),
        y: GPU.mvMul(wih, m: memCellCount, n: memCellCount, vector: h[t - 1])))
      f[t] = GPU.sigmoid(add(GPU.mvMul(wfx, m: memCellCount, n: inputDim, vector: input[t]),
        y: GPU.mvMul(wfh, m: memCellCount, n: memCellCount, vector: h[t - 1])))
      o[t] = GPU.sigmoid(add(GPU.mvMul(wox, m: memCellCount, n: inputDim, vector: input[t]),
        y: GPU.mvMul(woh, m: memCellCount, n: memCellCount, vector: h[t - 1])))
      s[t] = GPU.add(GPU.mul(g[t], y: i[t]), y: mul(s[t - 1], y: f[t]))
      h[t] = GPU.mul(s[t], y: o[t])
      p[t] = softmax(s[t])
    }
    return (s, h, p)
  }
  
  private func CPUCompute(input: [[Float]]) -> ([[Float]], [[Float]], [[Float]]) {
    let T = input.count;
    let start: [Float] = (1...memCellCount).map{_ in 0.0}
    var g: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    var f: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    var i: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    var o: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    var s: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    var h: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    var p: [[Float]] = (1...T).map{_ in (1...memCellCount).map{_ in 0.0} }
    
    g[0] = tanh(add(mvMul(wgx, m: memCellCount, n: inputDim, x: input[0]),
      y: mvMul(wgh, m: memCellCount, n: memCellCount, x: start)))
    i[0] = sigmoid(add(mvMul(wix, m: memCellCount, n: inputDim, x: input[0]),
      y: mvMul(wih, m: memCellCount, n: memCellCount, x: start)))
    f[0] = sigmoid(add(mvMul(wfx, m: memCellCount, n: inputDim, x: input[0]),
      y: mvMul(wfh, m: memCellCount, n: memCellCount, x: start)))
    o[0] = sigmoid(add(mvMul(wox, m: memCellCount, n: inputDim, x: input[0]),
      y: mvMul(woh, m: memCellCount, n: memCellCount, x: start)))
    s[0] = add(mul(g[0], y: i[0]), y: mul(s[0], y: f[0]))
    h[0] = mul(s[0], y: o[0])
    p[0] = softmax(s[0])
    
    for t in 1..<T {
      g[t] = tanh(add(mvMul(wgx, m: memCellCount, n: inputDim, x: input[t]),
        y: mvMul(wgh, m: memCellCount, n: memCellCount, x: h[t - 1])))
      i[t] = sigmoid(add(mvMul(wix, m: memCellCount, n: inputDim, x: input[t]),
        y: mvMul(wih, m: memCellCount, n: memCellCount, x: h[t - 1])))
      f[t] = sigmoid(add(mvMul(wfx, m: memCellCount, n: inputDim, x: input[t]),
        y: mvMul(wfh, m: memCellCount, n: memCellCount, x: h[t - 1])))
      o[t] = sigmoid(add(mvMul(wox, m: memCellCount, n: inputDim, x: input[t]),
        y: mvMul(woh, m: memCellCount, n: memCellCount, x: h[t - 1])))
      s[t] = add(mul(g[t], y: i[t]), y: mul(s[t - 1], y: f[t]))
      h[t] = mul(s[t], y: o[t])
      p[t] = softmax(s[t])
    }
    return (s, h, p)
  }
  
  /**Calculates and returns the loss of the LSTM network.
   - Parameter input: The input to the network.
   - Parameter target: The target output, given as an array containing arrays of expected indexes.
   - numExamples: The number of examples used for training.
   - Returns: The loss of the network given the outputs and inputs.
   */
  public func calculateLoss(input: [[Float]], target: [[Int]], numExamples: Int) -> Float {
    let (_, _, p) = feedforward(input)
    var L = [Float]()
    var y = [[Float]](count: p.count, repeatedValue: [Float](count: p[0].count, repeatedValue: 0.0))
    
    for i in 0..<target.count {
      for k in target[i] {
        y[i][k] = 1.0
      }
    }
    
    for i in 0..<target.count {
      L.append(sum(mul(y[i], y: log(p[i]))))
    }
    
    return sum(L) / Float(numExamples)
  }
}
