//: Playground - noun: a place where people can play

import Cocoa
import Accelerate
import Birdbrain
import Metal

var str = "Hello, playground"

struct Vector2 // matches float2 in Metal
{
    var x0: Float = 0.0
    var x1: Float = 0.0
}

let sizes = [1000,1000,10]
var x = [Float](count: sizes[0], repeatedValue: 0.0)
x = x.map({_ in rand_gauss()})
var ann = FeedfowardNeuralNetwork(sizes: sizes)

var start = NSDate()
ann.getWeights()
ann.feedforward(x, activationFunction: 3, useMetal: 0)
var end = NSDate().timeIntervalSinceDate(start)

var s = NSDate()
//ann.feedforward(x, activationFunction: 1, useMetal: 1)
