//: Playground - noun: a place where people can play

import Cocoa
import Accelerate
import Birdbrain
import Metal

var str = "Hello, playground"

let x = [[Float]](count: 10, repeatedValue: [Float](count: 100, repeatedValue: 1))
//x[0].count
var rnn = RecurrentNeuralNetwork(inputDim: 100, hiddenDim: 2500, activationFunction: 1)
let start = NSDate()
rnn.feedforward(x, useMetal: false)
print(NSDate().timeIntervalSinceDate(start))

//var lstm = LSTMNetwork(inputDim: 100, memCellCount: 5)
//let res = lstm.feedforward(x, useMetal: false, activationFunction: 1)