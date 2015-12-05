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
  
  /**Initializer for Recurrent Neural Network
    - Parameter inputDim: Dimension of inputs to RNN.
    - Parameter hiddenDim: Dimension of hidden neurons in RNN.
  */
  public init(inputDim: Int, hiddenDim: Int) {
    self.inputDim = inputDim
    self.hiddenDim = hiddenDim
    whx = (1...hiddenDim * inputDim).map{_ in initRand(inputDim)}
    why = (1...inputDim * hiddenDim).map{_ in initRand(inputDim)}
    whh = (1...hiddenDim * hiddenDim).map{_ in initRand(hiddenDim)}
  }
  
  //MARK: Feedforward
  
  /**Do a feedforward pass on the RNN.
   - Parameter input: Input to RNN.
   - Parameter useMetal: Indicate whether to use metal GPU functions.
   - Parameter activationFunction: Activation function to call (1 - Sigmoid, 2 - Tanh, 3 - ReLu)
   - Returns: An array tuple of the hidden states and outputs.
  */
  public func feedforward(input: [[Float]], useMetal: Bool, activationFunction: Int)
    -> ([[Float]], [[Float]]){
      let T = input.count;
      var s = [[Float]](count: T, repeatedValue: [Float](count: hiddenDim, repeatedValue: 0.0))
      let start = [Float](count: hiddenDim, repeatedValue: 0.0)
      var o = [[Float]](count: T, repeatedValue: [Float](count: inputDim, repeatedValue: 0.0))
     
      if (useMetal) { //Use GPU
        if (activationFunction == 1) { //Sigmoid
          s[0] = mtlSigmoid(mtlAdd(mtlMul(whx, y: input[0]), y: mtlMul(whh, y: start)))
          o[0] = mtlSoftmax(mtlMul(why, y: s[0]))
          
          for t in Range(start: 1, end: T) {
            s[t] = mtlSigmoid(mtlAdd(whx, y: mtlMul(whh, y: s[t - 1])))
            s[t] = mtlSoftmax(mtlMul(whx, y: s[t]))
          }
        }
        else if (activationFunction == 2) { //Hyperbolic Tangent
          s[0] = mtlTanh(mtlAdd(mtlMul(whx, y: input[0]), y: mtlMul(whh, y: start)))
          o[0] = mtlSoftmax(mtlMul(why, y: s[0]))
          
          for t in Range(start: 1, end: T) {
            s[t] = mtlTanh(mtlAdd(whx, y: mtlMul(whh, y: s[t - 1])))
            s[t] = mtlSoftmax(mtlMul(whx, y: s[t]))
          }
        }
        else if (activationFunction == 3) { //Rectified Linear
          s[0] = mtlRelu(mtlAdd(mtlMul(whx, y: input[0]), y: mtlMul(whh, y: start)))
          o[0] = mtlSoftmax(mtlMul(why, y: s[0]))
          
          for t in Range(start: 1, end: T) {
            s[t] = mtlRelu(mtlAdd(whx, y: mtlMul(whh, y: s[t - 1])))
            s[t] = mtlSoftmax(mtlMul(whx, y: s[t]))
          }
        }
        else {
          print("Not a proper entry for activation function")
        }
      }
      else { //Use CPU
        if (activationFunction == 1) { //Sigmoid
          
          s[0] = sigmoid(add(mvMul(whx, m: hiddenDim, n: inputDim, x: input[0]),
            y: mvMul(whh, m: hiddenDim, n: hiddenDim, x: start)))
          o[0] = softmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[0]))

          for t in Range(start: 1, end: T) {
            s[t] = sigmoid(add(mvMul(whx, m: hiddenDim, n: inputDim, x: input[t]),
              y: mvMul(whh, m: hiddenDim, n: hiddenDim, x: s[t - 1])))
            o[t] = softmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))
          }
        }
        else if (activationFunction == 2) { //Hyperbolic tangent
          s[0] = tanh(add(mvMul(whx, m: hiddenDim, n: inputDim, x: input[0]),
            y: mvMul(whh, m: hiddenDim, n: hiddenDim, x: start)))
          o[0] = softmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[0]))
        
          for t in Range(start: 1, end: T) {
            s[t] = tanh(add(mvMul(whx, m: hiddenDim, n: inputDim, x: input[t]),
              y: mvMul(whh, m: hiddenDim, n: hiddenDim, x: s[t - 1])))
            o[t] = softmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))
          }
      
        }
        else if (activationFunction == 3) {//Rectified Linear
          s[0] = relu(add(mvMul(whx, m: hiddenDim, n: inputDim, x: input[0]),
            y: mvMul(whh, m: hiddenDim, n: hiddenDim, x: start)))
          o[0] = softmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[0]))
        
          for t in Range(start: 1, end: T) {
            s[t] = relu(add(mvMul(whx, m: hiddenDim, n: inputDim, x: input[t]),
              y: mvMul(whh, m: hiddenDim, n: hiddenDim, x: s[t - 1])))
            o[t] = softmax(mvMul(why, m: inputDim, n: hiddenDim, x: s[t]))
          }
        }
        else {
          print("Not a proper entry for activation function")
        }
    }
    
    return (s, o)
  }
}