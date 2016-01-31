//
//  RecurrentNeuralNetwork.swift
//  Birdbrain
//
//  Created by Jorden Hill on 12/1/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

import Foundation

///A Recurrent Neural Network class.
public class RecurrentNeuralNetwork {
  var inputDim: Int
  var hiddenDim: Int
  var whx: [Float]
  var why: [Float]
  var whh: [Float]
  var activationFunction: Int
  var bpttTruncate: Int
  var useMetal: Bool
  
  /**Initializer for Recurrent Neural Network
    - Parameter inputDim: Dimension of inputs to RNN.
    - Parameter hiddenDim: Dimension of hidden neurons in RNN.
  */
  public init(inputDim: Int, hiddenDim: Int, useMetal: Bool,
    activationFunction: Int, bpttTruncate: Int) {
    
    self.inputDim = inputDim
    self.hiddenDim = hiddenDim
    self.activationFunction = activationFunction
    self.useMetal = useMetal
    self.bpttTruncate = bpttTruncate
    let start = NSDate()
    whx = (1...hiddenDim * inputDim).map{_ in initRand(inputDim)}
    why = (1...inputDim * hiddenDim).map{_ in initRand(inputDim)}
    whh = (1...hiddenDim * hiddenDim).map{_ in initRand(hiddenDim)}
    print(NSDate().timeIntervalSinceDate(start))
  }
  
  //MARK: Feedforward
  
  /**Do a feedforward pass on the RNN.
   - Parameter input: Input to RNN.
   - Parameter useMetal: Indicate whether to use metal GPU functions.
   - Returns: An array tuple of the hidden states and outputs.
  */
  public func feedforward(input: [[Float]])
    -> ([[Float]], [[Float]]){
      
      if (useMetal) { //Use GPU
        return GPUCompute(input)
      } else { //Use CPU
        return CPUCompute(input)
      }
  }
  
  /**Do a feedforward pass and return the prediction.
   - Parameter input: Input array to RNN.
   - Returns: An array containing the predicted values
   */
  public func predict(input: [[Float]]) -> [Int] {
    let (o, _) = feedforward(input)
    return maxIndex(o)
  }
  
  public func calculateLoss(input: [[Float]], target: [[Float]], numExamples: Int) -> Float {
    var (o, _) = feedforward(input)
    var L = [Float]()
    
    for i in Range(start: 0, end: target.count) {
      L.append(sum(mul(target[i], y: log(o[i]))))
    }
    
    return sum(L) / Float(numExamples)
  }
  
  public func backprop(input: [[Float]], target: [[Float]], learningRate: Float) {
    let T = target.count
    var (o, s) = feedforward(input)
    var dwhx: [Float] = (1...whx.count).map {_ in 0.0}
    var dwhy: [Float] = (1...why.count).map {_ in 0.0}
    var dwhh: [Float] = (1...whh.count).map {_ in 0.0}
    var dhnext: [Float] = (1...s[0].count).map {_ in 0.0}
    var dhraw: [Float]
    
    for t in (0..<T).reverse() {
      var deltaO = o[t]
      
      for i in target[t] {
        let index = Int(i)
        deltaO[index] -= 1
      }
      
      dwhy = outer(deltaO, y: s[t])
      
      let dh = add(mvMul(dwhy, m: inputDim, n: hiddenDim, x: deltaO), y: dhnext)
      
      if (useMetal) {
        if (activationFunction == 1) {
          dhraw = mul(mtlSigmoidPrime(s[t]), y: dh)
        } else if (activationFunction == 2) {
          dhraw = mul(mtlTanhPrime(s[t]), y: dh)
        } else {
          dhraw = mul(mtlReluPrime(s[t]), y: dh)
        }
      } else {
        if (activationFunction == 1) {
          dhraw = mul(sigmoidPrime(s[t]), y: dh)
        } else if (activationFunction == 2) {
          dhraw = mul(tanhPrime(s[t]), y: dh)
        } else {
          dhraw = mul(reluPrime(s[t]), y: dh)
        }
      }
      
      dwhx = outer(dhraw, y: input[t])
      dwhh = outer(dhraw, y: s[t - 1])
      
      dhnext = mvMul(dwhh, m: hiddenDim, n: hiddenDim, x: dhraw)
    }
    
    //Update weights
    whx = sub(whx, y: mul(dwhx, c: learningRate))
    why = sub(why, y: mul(dwhy, c: learningRate))
    whh = sub(whh, y: mul(dwhh, c: learningRate))
  }
  
  //MARK: GPU Compute
  
  private func GPUCompute(input: [[Float]]) -> ([[Float]], [[Float]]) {
    let start: [Float] = (1...hiddenDim).map{_ in 0.0}
    let T = input.count;
    var layers: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    
    layers[0] = add(mvMul(whx, m: hiddenDim, n: inputDim, x: input[0]),
      y: mvMul(whh, m: hiddenDim, n: hiddenDim, x: start))
    
    for t in Range(start: 1, end: T) {
      layers[t] = add(mvMul(whx, m: hiddenDim, n: inputDim, x: input[t]),
        y: mvMul(whh, m: hiddenDim, n: hiddenDim, x: layers[t - 1]))
    }
    
    if (activationFunction == 1) {
      return GPUSigmoid(T, layers: layers)
    } else if (activationFunction == 2) {
      return GPUTanh(T, layers: layers)
    } else {
      return GPURelu(T, layers: layers)
    }
  }
  
  //MARK: GPU Activation Function Computation
  
  private func GPUSigmoid(T: Int, layers: [[Float]]) -> ([[Float]], [[Float]]) {
    var s: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    var o: [[Float]] = (1...T).map{_ in (1...inputDim).map{_ in 0.0}}
    
    for t in Range(start: 0, end: T) {
      s[t] = mtlSigmoid(layers[t])
      o[t] = mtlSoftmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))
    }
    
    return (s, o)
  }
  
  private func GPUTanh(T: Int, layers: [[Float]]) -> ([[Float]], [[Float]]) {
    var s: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    var o: [[Float]] = (1...T).map{_ in (1...inputDim).map{_ in 0.0}}
    
    for t in Range(start: 0, end: T) {
      s[t] = mtlTanh(layers[t])
      o[t] = mtlSoftmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))
    }
    
    return (s, o)
  }
  
  private func GPURelu(T: Int, layers: [[Float]]) -> ([[Float]], [[Float]]) {
    var s: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    var o: [[Float]] = (1...T).map{_ in (1...inputDim).map{_ in 0.0}}
    
    for t in Range(start: 0, end: T) {
      s[t] = mtlRelu(layers[t])
      o[t] = mtlSoftmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))

    }
    
    return (s, o)
  }
  
  //MARK: CPU Compute
  
  private func CPUCompute(input: [[Float]]) -> ([[Float]], [[Float]]) {
    let start: [Float] = (1...hiddenDim).map{_ in 0.0}
    let T = input.count;
    var layers: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    
    layers[0] = add(mvMul(whx, m: hiddenDim, n: inputDim, x: input[0]),
      y: mvMul(whh, m: hiddenDim, n: hiddenDim, x: start))
    
    for t in Range(start: 1, end: T) {
      layers[t] = add(mvMul(whx, m: hiddenDim, n: inputDim, x: input[t]),
        y: mvMul(whh, m: hiddenDim, n: hiddenDim, x: layers[t - 1]))
    }
    
    if (activationFunction == 1) {
      return CPUSigmoid(T, layers: layers)
    } else if (activationFunction == 2) {
      return CPUTanh(T, layers: layers)
    } else {
      return CPURelu(T, layers: layers)
    }
  }
  
  //MARK: CPU Actvation Function Computation
  
  private func CPUSigmoid(T: Int, layers: [[Float]]) -> ([[Float]], [[Float]]) {
    var s: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    var o: [[Float]] = (1...T).map{_ in (1...inputDim).map{_ in 0.0}}
    
    for t in Range(start: 0, end: T) {
      s[t] = sigmoid(layers[t])
      o[t] =  softmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))
    }
    
    return (s, o)
  }
  
  private func CPUTanh(T: Int, layers: [[Float]]) -> ([[Float]], [[Float]]) {
    var s: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    var o: [[Float]] = (1...T).map{_ in (1...inputDim).map{_ in 0.0}}
    
    for t in Range(start: 0, end: T) {
      s[t] = tanh(layers[t])
      o[t] =  softmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))
    }
    
    return (s, o)
  }
  
  private func CPURelu(T: Int, layers: [[Float]]) -> ([[Float]], [[Float]]) {
    var s: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    var o: [[Float]] = (1...T).map{_ in (1...inputDim).map{_ in 0.0}}
    
    for t in Range(start: 0, end: T) {
      s[t] = relu(layers[t])
      o[t] = softmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))
    }
    
    return (s, o)
  }
}
