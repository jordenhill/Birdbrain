//
//  Neural Network.swift
//  Birdbrain
//
//  Created by Jorden Hill on 9/29/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

import Metal

public class FeedfowardNeuralNetwork {
    var numLayers: Int
    var sizes: [Int]
    var biases = [[Float]]()
    var weights = [[Float]]()
    var useMetal = Int()
    var activationFunction = Int()
    
    //Construct the network of the size stated in the array. The count of the array is the number of
    //layers, the values in the array are the number of neurons per layer.
    public init (sizes: [Int], useMetal: Int, activationFunction: Int) {
        numLayers = sizes.count
        self.sizes = sizes
        self.useMetal = useMetal
        self.activationFunction = activationFunction
        
        //Prepare the biases
        for y in sizes[1..<sizes.endIndex] {
            var bias = [Float](count: y, repeatedValue: 0)
            bias = bias.map() { _ in 0.0 }
            biases.append(bias)
        }
        
        //Prepare the weights
        for (x, y) in zip(sizes[0..<sizes.endIndex - 1], sizes[1..<sizes.endIndex]) {
            var w = [Float](count: x*y, repeatedValue: 0)
            w = w.map() { _ in rand_gauss()}
            weights.append(w)
        }
    }
    
    //Return the weights of the network.
    public func getWeights() -> [[Float]] {
        return weights
    }
    
    //Return the biases of the network.
    public func getBiases() -> [[Float]] {
        return biases
    }
    
    public func getNetworkSize() -> Int {
        return numLayers
    }
    
    //Perform a forward propagation on the network and return the result. The number indicates the
    //activation function (1: Sigmoid, 2: Tanh, 3: ReLU)
    public func feedforward(input: [Float]) -> [[Float]] {
        var a = [[Float]]()
        var activation = input
        var i = 1
        
            switch(useMetal) {
            case 0:
                for (b,w) in zip(biases,weights) {
                    let n = Int32(sizes[i - 1])
                    let m = Int32(sizes[i])
                    
                    if (activationFunction == 1) { //sigmoid
                        a.append(sigmoid(add(mvMul(w, m: m, n: n, x: activation), y: b)))
                    }
                    else if (activationFunction == 2) { //hyperbolic tangent
                        a.append(tanh(add(mvMul(w, m: m, n: n, x: activation), y: b)))
                    }
                    else if (activationFunction == 3) { //Rectified Linear
                        a.append(relu(add(mvMul(w, m: m, n: n, x: activation), y: b)))
                    }
                    else {
                        print("No appropriate activation function entered.")
                    }
                    
                    i++
                    activation = a[a.endIndex - 1]
                }
            case 1:
                for (b,w) in zip(biases,weights) {
                    let n = Int32(sizes[i - 1])
                    let m = Int32(sizes[i])
                    
                    if (activationFunction == 1) { //sigmoid
                        a.append(mtlSigmoid(mtlAdd(mvMul(w, m: m, n: n, x: activation), y: b)))
                    }
                    else if (activationFunction == 2) { //hyperbolic tangent
                        a.append(mtlTanh(mtlAdd(mvMul(w, m: m, n: n, x: activation), y: b)))
                    }
                    else if (activationFunction == 3) { //Rectified Linear
                        a.append(mtlRelu(mtlAdd(mvMul(w, m: m, n: n, x: activation), y: b)))
                    }
                    else {
                        print("No appropriate activation function entered.")
                    }
                    
                    i++
                    activation = a[a.endIndex - 1]
                }
            default:
                print("Error, must enter 0 or 1 for UseMetal")
            }
        
        return a
    }
    
    //Perform a backward propagation on the network.
    public func backpropagate(input: [Float], target: [Float], learningRate: Float) {
        //Compute a forward pass, hold z values and activations
        var activations = [[Float]]()
        var nablaB = [[Float]]()
        var nablaW = [[Float]]()
        
        if (useMetal == 0) {
            (nablaB, nablaW) = backprop(input, target: target)
            
            for l in Range(start:0, end: numLayers - 1) {
                weights[l] = sub(weights[l], B: scalMul(nablaW[l], y: learningRate))
                biases[l] = sub(biases[l], B: scalMul(nablaB[l], y: learningRate))
            }
        }
        else {
            (nablaB, nablaW) = mtl_backprop(input, target: target)
            
            for l in Range(start:0, end: numLayers - 1) {
                weights[l] = mtlSub(weights[l], y: scalMul(nablaW[l], y: learningRate))
                biases[l] = mtlSub(biases[l], y: scalMul(nablaB[l], y: learningRate))
            }
        }
    }
    
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
            
            i++
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
        deltaW[deltaW.endIndex - 1] = formMatrix(activations[activations.endIndex - 2], B: delta)
        
        for (var l = 2; l < numLayers; l++) {
            let z = zVals[zVals.endIndex - l]
            
            if (activationFunction == 1) {
                delta = mvMul(formMatrix(weights[weights.endIndex - l + 1], B: delta),
                    m: Int32(sizes[l - 2]), n: Int32(sizes[l - 1]), x: sigmoidPrime(z))
            }
            else if (activationFunction == 2) {
                delta = mvMul(formMatrix(weights[weights.endIndex - l + 1], B: delta),
                    m: Int32(sizes[l - 2]), n: Int32(sizes[l - 1]), x: tanhPrime(z))
            }
            else {
                delta = mvMul(formMatrix(weights[weights.endIndex - l + 1], B: delta),
                    m: Int32(sizes[l - 2]), n: Int32(sizes[l - 1]), x: reluPrime(z))
            }
            
            deltaB[deltaB.endIndex - l] = delta
            deltaW[deltaW.endIndex - l] = formMatrix(delta, B: activations[activations.endIndex - l])
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
            
            i++
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
        nablaW[nablaW.endIndex - 1] = formMatrix(activations[activations.endIndex - 2], B: delta)
        
        for (var l = 2; l < numLayers; l++) {
            let z = zVals[zVals.endIndex - l]
            
            if (activationFunction == 1) {
                delta = mvMul(formMatrix(weights[weights.endIndex - l + 1], B: delta),
                    m: Int32(sizes[l - 2]), n: Int32(sizes[l - 1]), x: mtlSigmoidPrime(z))
            }
            else if (activationFunction == 2) {
                delta = mvMul(formMatrix(weights[weights.endIndex - l + 1], B: delta),
                    m: Int32(sizes[l - 2]), n: Int32(sizes[l - 1]), x: mtlTanhPrime(z))
            }
            else {
                delta = mvMul(formMatrix(weights[weights.endIndex - l + 1], B: delta),
                    m: Int32(sizes[l - 2]), n: Int32(sizes[l - 1]), x: mtlReluPrime(z))
            }
            
            nablaB[nablaB.endIndex - l] = delta
            nablaW[nablaW.endIndex - l] = formMatrix(delta, B: activations[activations.endIndex - l])
        }
        
        return (nablaB, nablaW)
    }
    
    //Rectified Linear Unit (y = max(0,x))
    private func relu(x: [Float]) -> [Float] {
        let activation: [Float] = x.map({($0 < 0.0) ? 0.0 : $0})
        return activation
    }
    
    //Sigmoid function (1 / (1 + e^-x))
    private func sigmoid(x: [Float]) -> [Float] {
        let ones = [Float](count: x.count, repeatedValue: 1.0)
        let z: [Float] =  div(ones, y: (scalAdd(1.0, x: exp(neg(x)))))
        return z
    }
    
    //Sigmoid prime (sigmoid(x) * (1 - sigmoid(x))
    public func sigmoidPrime(x: [Float]) -> [Float] {
        return mul(sigmoid(x), y: scalAdd(1.0, x: neg(sigmoid(x))))
    }
    
    //Hyperbolic tangent prime (1 - tanh(x)^2)
    public func tanhPrime(x: [Float]) -> [Float] {
        return scalAdd(1, x: neg(square(tanh(x))))
    }
    
    //ReLU prime (x > 0 = 1, x <= 0 = 0)
    public func reluPrime(x: [Float]) -> [Float] {
        let val: [Float] = x.map({($0 <= 0.0) ? 0.0 : 1})
        return val
    }
    
    public func costDerivative(output: [Float], y: [Float]) -> [Float] {
        return sub(output, B: y)
    }
    
}