//
//  FeedForwardNeuralNetwork.swift
//  Birdbrain
//  A standard, simple feedforward neural network. Can use accelerate functions or metal GPU 
//  functions.
//  Created by Jorden Hill on 9/29/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

///A Feedforward Neural Network class.
public class FeedfowardNeuralNetwork {
  var numLayers: Int
  var sizes: [Int]
  var biases = [[Float]]()
  var weights = [[Float]]()
  var useMetal: Bool
  var activationFunction: String
    
  /**Constructor for Feedforward Neural Network.
    - Parameter sizes: The number of layers and size of each layer in the network.
    - Parameter useMetal: Indicate whether to use Metal functions or not.
    - Parameter activationFunction: Activation function to use 
      (1 - Sigmoid, 2, Hyperbolic Tangent, 3 - ReLu).
   */
  public init? (sizes: [Int], useMetal: Bool, activateFunction: String) {
    // Check that size and activationFunction are correct
    if (sizes.count < 2) {
      print("Size of network must be at least 2.")
      return nil
    }
        
    if ((activateFunction != "sigmoid") && (activateFunction != "tangent") &&
      (activateFunction != "relu")) {
      print("Must use sigmoid, tangent, or relu as activateFunction parameter")
      return nil
    }
    
    //Set initialization values.
    numLayers = sizes.count
    self.sizes = sizes
    self.useMetal = useMetal
    self.activationFunction = activateFunction
    
    //Initialize the biases
    for y in sizes[1..<sizes.endIndex] {
      let bias: [Float] = (1...y).map() { _ in 0.0 }
      biases.append(bias)
    }
        
    //Initialize the weights
    for (x, y) in zip(sizes[0..<sizes.endIndex - 1], sizes[1..<sizes.endIndex]) {
      let weight: [Float] = (1...x * y).map() { _ in initRand(x)}
      weights.append(weight)
    }
  }
  
  // MARK: Getters
    
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
  
  // MARK: Setters
  
  /** Assign weights to the network. Number of weights must be fit shape of weight matrix.
  */
  public func setWeights(newWeights: [[Float]]) {
    var canAssign = true
    
    // Quick check to make sure the weights are equal in size.
    for (weight, size) in zip(weights, sizes) {
      if (weight.count != size) {
        canAssign = false
      }
    }
    
    // If it checks out assign the weights
    if (canAssign) {
      weights = newWeights
    }
  }
  
  //MARK: Forward pass
    
  /**Perform a forward propagation on the network using the given input.
   - Parameter input: The input to the network.
   - Returns: An array of the activations at each layer of the network.
  */
  public func feedforward(input: [Float]) -> [[Float]] {
    if (useMetal) { //Use Metal GPU functions
      return GPUFeedforward(input)
    }
    else { //Use Accelerate CPU functions.
      return CPUFeedforward(input)
    }
  }
  
  
  /**Use Metal GPU functions to perform a forward pass.
   - Parameter input: The input to the network.
   - Returns: An array of the activations at each layer of the network.
  */
  private func GPUFeedforward(input: [Float]) -> [[Float]] {
    var activations = [[Float]]()
    var layerInputs = input
    var layer = 1

    for (b,w) in zip(biases,weights) {
      let n = sizes[layer - 1]
      let m = sizes[layer]
      
      if (activationFunction == "sigmoid") { //Sigmoid
        activations.append(mtlSigmoid(add(mvMul(w, m: m, n: n, x: layerInputs), y: b)))
      } else if (activationFunction == "tangent") { //Hyperbolic tangent
        activations.append(mtlTanh(add(mvMul(w, m: m, n: n, x: layerInputs), y: b)))
      } else if (activationFunction == "relu") { //Rectified Linear
        activations.append(mtlRelu(add(mvMul(w, m: m, n: n, x: layerInputs), y: b)))
      } else {
        print("No appropriate activation function entered.")
      }
      
      layer += 1
      
      layerInputs = activations[activations.endIndex - 1]
    }
    
    return activations
  }
  
  /**Use CPU functions to perform a forward pass.
   - Parameter input: The input to the network.
   - Returns: An array of the activations at each layer of the network.
  */
  private func CPUFeedforward(input: [Float]) -> [[Float]] {
    var activations = [[Float]]()
    var layerInputs = input
    var layer = 1
    
    for (b,w) in zip(biases,weights) {
      let n = sizes[layer - 1]
      let m = sizes[layer]
      
      if (activationFunction == "sigmoid") { //Sigmoid
        activations.append(sigmoid(add(mvMul(w, m: m, n: n, x: layerInputs), y: b)))
      } else if (activationFunction == "tangent") { //Hyperbolic tangent
        activations.append(tanh(add(mvMul(w, m: m, n: n, x: layerInputs), y: b)))
      } else if (activationFunction == "relu") { //Rectified Linear
        activations.append(relu(add(mvMul(w, m: m, n: n, x: layerInputs), y: b)))
      } else {
        print("No appropriate activation function entered.")
      }
      
      layer += 1
      
      layerInputs = activations[activations.endIndex - 1]
    }
    
    return activations
  }
  
  //MARK: Backward pass
  
  /**Perform a backward propagation on the network.
    - Parameter input: The input to the network.
    - Parameter target: Target output of the network.
    - Parameter learningRate: The learning rate of the network.
  */
  public func backpropagate(input: [Float], target: [Float], learningRate: Float) {
    var nablaB = [[Float]]()
    var nablaW = [[Float]]()
        
    if (useMetal) {
      (nablaB, nablaW) = mtlBackprop(input, target: target)
    } else {
      (nablaB, nablaW) = backprop(input, target: target)
    }
    
    for l in 0..<numLayers - 1 {
      weights[l] = sub(weights[l], y: mul(nablaW[l], c: learningRate))
      biases[l] = sub(biases[l], y: mul(nablaB[l], c: learningRate))
    }
  }
  
  /**Backpropagation helper function.
    - Parameter input: Input to neural network.
    - Parameter target: Target output of the network.
    - Returns: The changes in the weights and biases of the network.
  */
  private func backprop(input: [Float], target: [Float]) -> ([[Float]], [[Float]]){
    //Build namblaW and namblaB
    var nablaW = [[Float]]()
    var nablaB = [[Float]]()
    var delta = [Float]()
    var layer = 1
    var m: Int
    var n: Int
    var zVals = [[Float]]()
    var activations: [[Float]] = [input]
    var activation = input
    
    for w in weights {
      nablaW.append([Float](count: w.count, repeatedValue: 0.0))
    }
    
    for b in biases {
      nablaB.append([Float](count: b.count, repeatedValue: 0.0))
    }
    
    //Compute a forward pass, hold z values and activations
    for (b, w) in zip(biases, weights) {
      m = sizes[layer]
      n = sizes[layer - 1]
      let z = add(mvMul(w, m: m, n: n, x: activation), y: b)
      
      zVals.append(z)
      
      if (activationFunction == "sigmoid") {
        activation = sigmoid(z)
      } else if (activationFunction == "tangent") {
        activation = tanh(z)
      } else {
        activation = relu(z)
      }
      
      layer += 1
      
      activations.append(activation)
    }
    
    //Create delta for last layer based on output, do a backward pass
    if (activationFunction == "sigmoid") {
      delta = mul(costDerivative(activations[activations.endIndex - 1], y: target),
        y: sigmoidPrime(zVals[zVals.endIndex - 1]))
    } else if (activationFunction == "tangent") {
      delta = mul(costDerivative(activations[activations.endIndex - 1], y: target),
        y: tanhPrime(zVals[zVals.endIndex - 1]))
    } else {
      delta = mul(costDerivative(activations[activations.endIndex - 1], y: target),
        y: reluPrime(zVals[zVals.endIndex - 1]))
    }
    
    nablaB[nablaB.endIndex - 1] = delta
    nablaW[nablaW.endIndex - 1] = outer(activations[activations.endIndex - 2], y: delta)
    
    for l in 2 ..< numLayers {
      let z = zVals[zVals.endIndex - l]
      let partialDelta = mvMul(weights[weights.endIndex - l + 1], m: sizes[sizes.endIndex - l],
        n: sizes[sizes.endIndex - l + 1], x: delta)
      
      if (activationFunction == "sigmoid") {
        delta = mul(partialDelta, y: sigmoidPrime(z))
      } else if (activationFunction == "tangent") {
        delta = mul(partialDelta, y: tanhPrime(z))
      } else {
        delta = mul(partialDelta, y: reluPrime(z))
      }
      
      nablaB[nablaB.endIndex - l] = delta
      nablaW[nablaW.endIndex - l] = outer(delta, y: activations[activations.endIndex - l - 1])
    }
    
    return (nablaB, nablaW)
  }
  
  /**Metal version of backpropagation helper function.
   - Parameter input: Input to neural network.
   - Parameter target: Target output of the network.
   - Returns: The changes in the weights and biases of the network.
   */
  private func mtlBackprop(input: [Float], target: [Float]) -> ([[Float]], [[Float]]){
    //Build namblaW and namblaB
    var nablaW = [[Float]]()
    var nablaB = [[Float]]()
    var delta = [Float]()
    var layer = 1
    var m: Int
    var n: Int
    var zVals = [[Float]]()
    var layerInput = input
    var activations = [[Float]]()
        
    for w in weights {
      nablaW.append([Float](count: w.count, repeatedValue: 0.0))
    }
    
    for b in biases {
      nablaB.append([Float](count: b.count, repeatedValue: 0.0))
    }
    
    for (b, w) in zip(biases, weights) {
      m = sizes[layer]
      n = sizes[layer - 1]
      let z = add(mvMul(w, m: m, n: n, x: layerInput), y: b)

      zVals.append(z)
                
      if (activationFunction == "sigmoid") {
        layerInput = mtlSigmoid(z)
      } else if (activationFunction == "tangent") {
        layerInput = mtlTanh(z)
      } else {
        layerInput = mtlRelu(z)
      }
            
      layer += 1
      
      activations.append(layerInput)
    }
        
    //Create delta for last layer based on output, do a backward pass
    if (activationFunction == "sigmoid") {
      delta = mul(costDerivative(activations[activations.endIndex - 1], y: target),
        y: mtlSigmoidPrime(zVals[zVals.endIndex - 1]))
    } else if (activationFunction == "tangent") {
      delta = mul(costDerivative(activations[activations.endIndex - 1], y: target),
        y: mtlTanhPrime(zVals[zVals.endIndex - 1]))
    } else {
      delta = mul(costDerivative(activations[activations.endIndex - 1], y: target),
        y: mtlReluPrime(zVals[zVals.endIndex - 1]))
    }
        
    nablaB[nablaB.endIndex - 1] = delta
    nablaW[nablaW.endIndex - 1] = outer(activations[activations.endIndex - 2], y: delta)
        
    nablaB[nablaB.endIndex - 1] = delta
    nablaW[nablaW.endIndex - 1] = outer(activations[activations.endIndex - 2], y: delta)
    
    for l in 2 ..< numLayers {
      let z = zVals[zVals.endIndex - l]
      let partialDelta = mvMul(weights[weights.endIndex - l + 1], m: sizes[sizes.endIndex - l],
        n: sizes[sizes.endIndex - l + 1], x: delta)
      
      if (activationFunction == "sigmoid") {
        delta = mul(partialDelta, y: mtlSigmoidPrime(z))
      }
      else if (activationFunction == "tangent") {
        delta = mul(partialDelta, y: mtlTanhPrime(z))
      }
      else {
        delta = mul(partialDelta, y: mtlReluPrime(z))
      }
      
      nablaB[nablaB.endIndex - l] = delta
      nablaW[nablaW.endIndex - l] = outer(delta, y: activations[activations.endIndex - l - 1])
    }
        
    return (nablaB, nablaW)
  }
  
  //TODO: Add loss function
  
  /** Calculate the loss of the network
    - Parameter:
    - Returns:
  */
  public func getLoss(input: [Float], target: [Float]) -> Float {
    return 0.0
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