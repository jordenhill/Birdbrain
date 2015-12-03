//
//  RecurrentNeuralNetwork.swift
//  Birdbrain
//
//  Created by Jorden Hill on 12/1/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

import Foundation

public class RecurrentNeuralNetwork {
  var inputDim: Int
  var hiddenDim: Int
  var bpttTruncate: Int
  var Wi: [Float]
  var V: [Float]
  var Wh: [Float]
    
  public init(inputDim: Int, hiddenDim: Int, bpttTruncate: Int) {
    self.inputDim = inputDim
    self.hiddenDim = hiddenDim
    self.bpttTruncate = bpttTruncate
    Wi = [Float](count: hiddenDim * inputDim, repeatedValue: initRand(inputDim))
    V = [Float](count: inputDim * hiddenDim, repeatedValue: initRand(hiddenDim))
    Wh = [Float](count: hiddenDim * hiddenDim, repeatedValue: initRand(hiddenDim))
  }
    
  public func feedforward(x: [[Float]], activationFunction: Int) -> ([[Float]], [[Float]]){
    let T = x.count;
    var s = [[Float]](count: T + 1, repeatedValue: [Float](count: hiddenDim, repeatedValue: 0.0))
    let start = [Float](count: hiddenDim, repeatedValue: 0.0)
    var o = [[Float]](count: T, repeatedValue: [Float](count: inputDim, repeatedValue: 0.0))
        
    switch (activationFunction) {
      case 1:
        s[0] = sigmoid(add(mvMul(Wi, m: Int32(inputDim), n:Int32(hiddenDim), x: x[0]),
          y: mvMul(Wh, m: Int32(hiddenDim), n: Int32(hiddenDim), x: start)))
        o[0] = softmax(mvMul(V, m: Int32(inputDim), n: Int32(hiddenDim), x: s[0]))
      
        for t in Range(start: 1, end: T) {
          s[t] = sigmoid(add(mvMul(Wi, m: Int32(inputDim), n:Int32(hiddenDim), x: x[t]),
            y: mvMul(Wh, m: Int32(hiddenDim), n: Int32(hiddenDim), x: s[t - 1])))
          o[t] = softmax(mvMul(V, m: Int32(inputDim), n: Int32(hiddenDim), x: s[t]))
        }

      case 2:
        s[0] = tanh(add(mvMul(Wi, m: Int32(inputDim), n:Int32(hiddenDim), x: x[0]),
          y: mvMul(Wh, m: Int32(hiddenDim), n: Int32(hiddenDim), x: start)))
        o[0] = softmax(mvMul(V, m: Int32(inputDim), n: Int32(hiddenDim), x: s[0]))
        
        for t in Range(start: 1, end: T) {
          s[t] = tanh(add(mvMul(Wi, m: Int32(inputDim), n:Int32(hiddenDim), x: x[t]),
            y: mvMul(Wh, m: Int32(hiddenDim), n: Int32(hiddenDim), x: s[t - 1])))
          o[t] = softmax(mvMul(V, m: Int32(inputDim), n: Int32(hiddenDim), x: s[t]))
        }
      case 3:
        s[0] = relu(add(mvMul(Wi, m: Int32(inputDim), n:Int32(hiddenDim), x: x[0]),
          y: mvMul(Wh, m: Int32(hiddenDim), n: Int32(hiddenDim), x: start)))
        o[0] = softmax(mvMul(V, m: Int32(inputDim), n: Int32(hiddenDim), x: s[0]))
        
        for t in Range(start: 1, end: T) {
          s[t] = relu(add(mvMul(Wi, m: Int32(inputDim), n:Int32(hiddenDim), x: x[t]),
            y: mvMul(Wh, m: Int32(hiddenDim), n: Int32(hiddenDim), x: s[t - 1])))
          o[t] = softmax(mvMul(V, m: Int32(inputDim), n: Int32(hiddenDim), x: s[t]))
        }
      default:
        print("Not a proper entry for activation function")
    }
    
    return (s, o)
  }
}