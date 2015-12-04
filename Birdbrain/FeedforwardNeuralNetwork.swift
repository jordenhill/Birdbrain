//
//  FeedForwardNeuralNetwork.swift
//  Birdbrain
//  A standard, simple feedforward neural network. Can use accelerate functions or metal GPU 
//  functions.
//  Created by Jorden Hill on 9/29/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

import Metal

///A Feedforward Neural Network class.
public class FeedfowardNeuralNetwork {
  var numLayers: Int
  var sizes: [Int]
  var biases = [[Float]]()
  var weights = [[Float]]()
  var useMetal: Bool
  var activationFunction: Int
    
  /**Constructor for Feedforward Neural Network.
    - Parameter sizes: The number of layers and size of each layer in the network.
    - Parameter useMetal: Indicate whether to use Metal functions or not.
    - Parameter activationFunction: Activation function to use 
      (1 - Sigmoid, 2, Hyperbolic Tangent, 3 - ReLu).
   */
  public init (sizes: [Int], useMetal: Bool, activationFunction: Int) {
    //Set initialization values.
    numLayers = sizes.count
    self.sizes = sizes
    self.useMetal = useMetal
    self.activationFunction = activationFunction
    
    //Initialize the biases
    for y in sizes[1..<sizes.endIndex] {
      var bias = [Float](count: y, repeatedValue: 0)
      bias = bias.map() { _ in 0.0 }
      biases.append(bias)
    }
        
    //Initialize the weights
    for (x, y) in zip(sizes[0..<sizes.endIndex - 1], sizes[1..<sizes.endIndex]) {
      var w = [Float](count: x*y, repeatedValue: 0)
      w = w.map() { _ in initRand(x)}
      weights.append(w)
    }
  }
    
  /**Return the weights of the network.
    - Returns: An array of the network's weight values.
  */
  public func getWeights() -> [[Float]] {
    return weights
  }
    
  /**Return the biases of the network.
    - Returns: An array of the network's bias values.
  */
  public func getBiases() -> [[Float]] {
    return biases
  }
  
  /**Get the number of layers in the network.
    - Returns: The number of layers in the network.
  */
  public func getNetworkSize() -> Int {
    return numLayers
  }
    
  /**Perform a forward propagation on the network using the given input.
   - Parameter input: The input to the network.
   - Returns: An array of the activations at each layer of the network.
  */
  public func feedforward(input: [Float]) -> [[Float]] {
    var a = [[Float]]()
    var activation = input
    var i = 1
        
    if (useMetal) { //Use Metal GPU functions
      for (b,w) in zip(biases,weights) {
        let n = Int32(sizes[i - 1])
        let m = Int32(sizes[i])
                    
        if (activationFunction == 1) { //Sigmoid
          a.append(sigmoid(add(mvMul(w, m: m, n: n, x: activation), y: b)))
        }
        else if (activationFunction == 2) { //Hyperbolic tangent
          a.append(tanh(add(mvMul(w, m: m, n: n, x: activation), y: b)))
        }
        else if (activationFunction == 3) { //Rectified Linear
          a.append(relu(add(mvMul(w, m: m, n: n, x: activation), y: b)))
        }
        else {
          print("No appropriate activation function entered.")
        }
                    
        i += 1
        
        activation = a[a.endIndex - 1]
      }
    }
    else { //Use Accelerate CPU functions.
      for (b,w) in zip(biases,weights) {
        let n = Int32(sizes[i - 1])
        let m = Int32(sizes[i])
        
        if (activationFunction == 1) { //Sigmoid
          a.append(mtlSigmoid(mtlAdd(mvMul(w, m: m, n: n, x: activation), y: b)))
        }
        else if (activationFunction == 2) { //Hyperbolic tangent
          a.append(mtlTanh(mtlAdd(mvMul(w, m: m, n: n, x: activation), y: b)))
        }
        else if (activationFunction == 3) { //Rectified Linear
          a.append(mtlRelu(mtlAdd(mvMul(w, m: m, n: n, x: activation), y: b)))
        }
        else {
          print("No appropriate activation function entered.")
        }
                    
        i += 1
        
        activation = a[a.endIndex - 1]
      }
    }

    return a
  }
    
  /**Perform a backward propagation on the network.
    - Parameter input: The input to the network.
    - Parameter target: Target output of the network.
    - Parameter learningRate: The learning rate of the network.
  */
  public func backpropagate(input: [Float], target: [Float], learningRate: Float) {
    var nablaB = [[Float]]()
    var nablaW = [[Float]]()
        
    if (useMetal) {
      (nablaB, nablaW) = backprop(input, target: target)
            
      for l in Range(start:0, end: numLayers - 1) {
        weights[l] = sub(weights[l], y: mul(nablaW[l], y: learningRate))
        biases[l] = sub(biases[l], y: mul(nablaB[l], y: learningRate))
      }
    }
    else {
      (nablaB, nablaW) = mtl_backprop(input, target: target)
            
      for l in Range(start:0, end: numLayers - 1) {
        weights[l] = mtlSub(weights[l], y: mul(nablaW[l], y: learningRate))
        biases[l] = mtlSub(biases[l], y: mul(nablaB[l], y: learningRate))
      }
    }
  }
  
  /**Backpropagation helper function.
    - Parameter input: Input to neural network.
    - Parameter target: Target output of the network.
    - Returns: The changes in the weights and biases of the network.
  */
  private func backprop(input: [Float], target: [Float]) -> ([[Float]], [[Float]]){
    //Build namblaW and namblaB
    var deltaW = [[Float]]()
    var deltaB = [[Float]]()
    var delta = [Float]()
    var i = 1
    var m: Int32
    var n: Int32
    var zVals = [[Float]]()
    var activations = [[Float]]()
    var activation = input
        
    for w in weights {
      deltaW.append([Float](count: w.count, repeatedValue: 0.0))
    }
    
    for b in biases {
      deltaB.append([Float](count: b.count, repeatedValue: 0.0))
    }
    
    //Compute a forward pass, hold z values and activations
    for (b, w) in zip(biases, weights) {
      m = Int32(sizes[i])
      n = Int32(sizes[i - 1])
      let z = add(mvMul(w, m: m, n: n, x: activation), y: b)
      
      zVals.append(z)
                
      if (activationFunction == 1) {
        activation = sigmoid(z)
      }
      else if (activationFunction == 2) {
        activation = tanh(z)
      }
      else {
        activation = relu(z)
      }
            
      i += 1
      
      activations.append(activation)
    }
        
    //Create delta for last layer based on output, do a backward pass
    if (activationFunction == 1) {
      delta = mul(costDerivative(activations[activations.endIndex - 1], y: target),
        y: sigmoidPrime(zVals[zVals.endIndex - 1]))
    }
    else if (activationFunction == 2) {
      delta = mul(costDerivative(activations[activations.endIndex - 1], y: target),
        y: tanhPrime(zVals[zVals.endIndex - 1]))
    }
    else {
      delta = mul(costDerivative(activations[activations.endIndex - 1], y: target),
        y: reluPrime(zVals[zVals.endIndex - 1]))
    }
        
    deltaB[deltaB.endIndex - 1] = delta
    deltaW[deltaW.endIndex - 1] = formMatrix(activations[activations.endIndex - 2], y: delta)
        
    for (var l = 2; l < numLayers; l++) {
      let z = zVals[zVals.endIndex - l]
      
      if (activationFunction == 1) {
        delta = mvMul(formMatrix(weights[weights.endIndex - l + 1], y: delta),
          m: Int32(sizes[l - 2]), n: Int32(sizes[l - 1]), x: sigmoidPrime(z))
      }
      else if (activationFunction == 2) {
        delta = mvMul(formMatrix(weights[weights.endIndex - l + 1], y: delta),
          m: Int32(sizes[l - 2]), n: Int32(sizes[l - 1]), x: tanhPrime(z))
      }
      else {
        delta = mvMul(formMatrix(weights[weights.endIndex - l + 1], y: delta),
          m: Int32(sizes[l - 2]), n: Int32(sizes[l - 1]), x: reluPrime(z))
      }
            
      deltaB[deltaB.endIndex - l] = delta
      deltaW[deltaW.endIndex - l] = formMatrix(delta, y: activations[activations.endIndex - l])
    }
        
    return (deltaB, deltaW)
  }
    
  private func mtl_backprop(input: [Float], target: [Float]) -> ([[Float]], [[Float]]){
    //Build namblaW and namblaB
    var nablaW = [[Float]]()
    var nablaB = [[Float]]()
    var delta = [Float]()
    var i = 1
    var m: Int32
    var n: Int32
    var zVals = [[Float]]()
    var activation = input
    var activations = [[Float]]()
        
    for w in weights {
      nablaW.append([Float](count: w.count, repeatedValue: 0.0))
    }
    
    for b in biases {
      nablaB.append([Float](count: b.count, repeatedValue: 0.0))
    }
    
    for (b, w) in zip(biases, weights) {
      m = Int32(sizes[i])
      n = Int32(sizes[i - 1])
      let z = mtlAdd(mvMul(w, m: m, n: n, x: activation), y: b)

      zVals.append(z)
                
      if (activationFunction == 1) {
        activation = mtlSigmoid(z)
      }
      else if (activationFunction == 2) {
        activation = mtlTanh(z)
      }
      else {
        activation = mtlRelu(z)
      }
            
      i += 1
      
      activations.append(activation)
    }
        
    //Create delta for last layer based on output, do a backward pass
    if (activationFunction == 1) {
      delta = mtlMul(costDerivative(activations[activations.endIndex - 1], y: target),
        y: mtlSigmoidPrime(zVals[zVals.endIndex - 1]))
    }
    else if (activationFunction == 2) {
      delta = mtlMul(costDerivative(activations[activations.endIndex - 1], y: target),
        y: mtlTanhPrime(zVals[zVals.endIndex - 1]))
    }
    else {
      delta = mtlMul(costDerivative(activations[activations.endIndex - 1], y: target),
        y: mtlReluPrime(zVals[zVals.endIndex - 1]))
    }
        
    nablaB[nablaB.endIndex - 1] = delta
    nablaW[nablaW.endIndex - 1] = formMatrix(activations[activations.endIndex - 2], y: delta)
        
    for (var l = 2; l < numLayers; l++) {
      let z = zVals[zVals.endIndex - l]
            
      if (activationFunction == 1) {
        delta = mvMul(formMatrix(weights[weights.endIndex - l + 1], y: delta),
          m: Int32(sizes[l - 2]), n: Int32(sizes[l - 1]), x: mtlSigmoidPrime(z))
      }
      else if (activationFunction == 2) {
        delta = mvMul(formMatrix(weights[weights.endIndex - l + 1], y: delta),
          m: Int32(sizes[l - 2]), n: Int32(sizes[l - 1]), x: mtlTanhPrime(z))
      }
      else {
        delta = mvMul(formMatrix(weights[weights.endIndex - l + 1], y: delta),
          m: Int32(sizes[l - 2]), n: Int32(sizes[l - 1]), x: mtlReluPrime(z))
      }
            
      nablaB[nablaB.endIndex - l] = delta
      nablaW[nablaW.endIndex - l] = formMatrix(delta, y: activations[activations.endIndex - l])
    }
        
    return (nablaB, nablaW)
  }
    
  /**Combine and average weights in two neural nets.
    - Parameter otherNet: Another feedfowrd neural network.
  */
  public func combine(otherNet: FeedfowardNeuralNetwork) {
    precondition((numLayers == otherNet.numLayers) && (sizes == otherNet.sizes),
      "Nets must have the same size.")
    
    var newWeights = [[Float]]()
    
    for (w1, w2) in zip(weights, otherNet.weights) {
        newWeights.append(div(add(w1, y: w2), c: 2.0))
    }
      
    weights = newWeights
  }
}