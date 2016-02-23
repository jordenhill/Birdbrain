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
  var activationFunction: String
  var useMetal: Bool
  
  /**Initializer for Recurrent Neural Network
    - Parameter inputDim: Dimension of input vector to RNN.
    - Parameter hiddenDim: Dimension hidden vectors in RNN (number of input vectors).
  */
  public init(inputDim: Int, hiddenDim: Int, useMetal: Bool, activationFunction: String) {
    
    self.inputDim = inputDim
    self.hiddenDim = hiddenDim
    self.activationFunction = activationFunction
    self.useMetal = useMetal
    whx = (1...hiddenDim * inputDim).map{_ in initRand(inputDim)}
    why = (1...inputDim * hiddenDim).map{_ in initRand(inputDim)}
    whh = (1...hiddenDim * hiddenDim).map{_ in initRand(hiddenDim)}
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
    let (_, o) = feedforward(input)
    return maxIndex(o)
  }
  
  public func calculateLoss(input: [[Float]], target: [[Int]], numExamples: Int) -> Float {
    let (_, o) = feedforward(input)
    var L = [Float]()
    var y = [[Float]](count: o.count, repeatedValue: [Float](count: o[0].count, repeatedValue: 0.0))
    
    for i in 0..<target.count {
      for k in target[i] {
        y[i][k] = 1.0
      }
    }
    
    for i in 0..<target.count {
      L.append(sum(mul(y[i], y: log(o[i]))))
    }
    
    return sum(L) / Float(numExamples)
  }
  
  public func backprop(input: [[Float]], target: [[Int]], learningRate: Float) {
    let T = target.count
    var (s, o) = feedforward(input)
    var dwhx: [Float] = (1...whx.count).map {_ in 0.0}
    var dwhy: [Float] = (1...why.count).map {_ in 0.0}
    var dwhh: [Float] = (1...whh.count).map {_ in 0.0}
    var dhnext: [Float] = (1...s[0].count).map {_ in 0.0}
    var dhraw: [Float]
    
    var deltaO = o[0]
    
    for i in 0..<target.count {
      for k in target[i] {
        deltaO[k] -= 1
      }
    }
    
    dwhy = outer(deltaO, y: s[0])
    
    let dh = add(tmvMul(dwhy, m: inputDim, n: hiddenDim, x: deltaO), y: dhnext)
    
    if (useMetal) {
      if (activationFunction == "tangent") {
        dhraw = mul(mtlTanhPrime(s[0]), y: dh)
      } else if (activationFunction == "relu") {
        dhraw = mul(mtlReluPrime(s[0]), y: dh)
      } else {
        dhraw = mul(mtlSigmoidPrime(s[0]), y: dh)
      }
    } else {
      if (activationFunction == "tangent") {
        dhraw = mul(tanhPrime(s[0]), y: dh)
      } else if (activationFunction == "relu") {
        dhraw = mul(reluPrime(s[0]), y: dh)
      } else {
        dhraw = mul(sigmoidPrime(s[0]), y: dh)
      }
    }
    
    dwhx = outer(dhraw, y: input[0])
    dwhh = outer(dhraw, y: s[0])
    
    dhnext = mvMul(dwhh, m: hiddenDim, n: hiddenDim, x: dhraw)
    
    for t in (1..<T).reverse() {
      var deltaO = o[t]
      
      for i in target[t] {
        let index = Int(i)
        deltaO[index] -= 1
      }
      
      dwhy = add(dwhy, y: outer(deltaO, y: s[t]))
      
      let dh = add(tmvMul(dwhy, m: inputDim, n: hiddenDim, x: deltaO), y: dhnext)
      
      if (useMetal) {
        if (activationFunction == "tangent") {
          dhraw = mul(mtlTanhPrime(s[t]), y: dh)
        } else if (activationFunction == "relu") {
          dhraw = mul(mtlReluPrime(s[t]), y: dh)
        } else {
          dhraw = mul(mtlSigmoidPrime(s[t]), y: dh)
        }
      } else {
        if (activationFunction == "tangent") {
          dhraw = mul(tanhPrime(s[t]), y: dh)
        } else if (activationFunction == "relu") {
          dhraw = mul(reluPrime(s[t]), y: dh)
        } else {
          dhraw = mul(sigmoidPrime(s[t]), y: dh)
        }
      }
      
      dwhx = add(dwhx, y: outer(dhraw, y: input[t]))
      dwhh = add(dwhh, y: outer(dhraw, y: s[t - 1]))
      
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
    
    for t in 1 ..< T {
      layers[t] = add(mvMul(whx, m: hiddenDim, n: inputDim, x: input[t]),
        y: mvMul(whh, m: hiddenDim, n: hiddenDim, x: layers[t - 1]))
    }
    
    if (activationFunction == "tangent") {
      return GPUTanh(T, layers: layers)
    } else if (activationFunction == "relu") {
      return GPURelu(T, layers: layers)
    } else {
      return GPUSigmoid(T, layers: layers)
    }
  }
  
  //MARK: GPU Activation Function Computation
  
  private func GPUSigmoid(T: Int, layers: [[Float]]) -> ([[Float]], [[Float]]) {
    var s: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    var o: [[Float]] = (1...T).map{_ in (1...inputDim).map{_ in 0.0}}
    
    for t in 0 ..< T {
      s[t] = mtlSigmoid(layers[t])
      o[t] = mtlSoftmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))
    }
    
    return (s, o)
  }
  
  private func GPUTanh(T: Int, layers: [[Float]]) -> ([[Float]], [[Float]]) {
    var s: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    var o: [[Float]] = (1...T).map{_ in (1...inputDim).map{_ in 0.0}}
    
    for t in 0 ..< T {
      s[t] = mtlTanh(layers[t])
      o[t] = mtlSoftmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))
    }
    
    return (s, o)
  }
  
  private func GPURelu(T: Int, layers: [[Float]]) -> ([[Float]], [[Float]]) {
    var s: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    var o: [[Float]] = (1...T).map{_ in (1...inputDim).map{_ in 0.0}}
    
    for t in 0 ..< T {
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
    
    for t in 1..<T {
      layers[t] = add(mvMul(whx, m: hiddenDim, n: inputDim, x: input[t]),
        y: mvMul(whh, m: hiddenDim, n: hiddenDim, x: layers[t - 1]))
    }
    
    if (activationFunction == "tangent") {
      return CPUTanh(T, layers: layers)
    } else if (activationFunction == "relu") {
      return CPURelu(T, layers: layers)
    } else {
      return CPUSigmoid(T, layers: layers)
    }
  }
  
  //MARK: CPU Actvation Function Computation
  
  private func CPUSigmoid(T: Int, layers: [[Float]]) -> ([[Float]], [[Float]]) {
    var s: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    var o: [[Float]] = (1...T).map{_ in (1...inputDim).map{_ in 0.0}}
    
    for t in 0..<T {
      s[t] = sigmoid(layers[t])
      o[t] =  softmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))
    }
    
    return (s, o)
  }
  
  private func CPUTanh(T: Int, layers: [[Float]]) -> ([[Float]], [[Float]]) {
    var s: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    var o: [[Float]] = (1...T).map{_ in (1...inputDim).map{_ in 0.0}}
    
    for t in 0 ..< T {
      s[t] = tanh(layers[t])
      o[t] =  softmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))
    }
    
    return (s, o)
  }
  
  private func CPURelu(T: Int, layers: [[Float]]) -> ([[Float]], [[Float]]) {
    var s: [[Float]] = (1...T).map{_ in (1...hiddenDim).map{_ in 0.0}}
    var o: [[Float]] = (1...T).map{_ in (1...inputDim).map{_ in 0.0}}
    
    for t in 0 ..< T {
      s[t] = relu(layers[t])
      o[t] = softmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))
    }
    
    return (s, o)
  }
}
