//: Playground - noun: a place where people can play

import Cocoa
import Accelerate
import Birdbrain
import Metal

var str = "Hello, playground"

let x = [[Float]](count: 10, repeatedValue: [Float](count: 100, repeatedValue: 1))
//x[0].count
//var rnn = RecurrentNeuralNetwork(inputDim: 8000, hiddenDim: 100)
//rnn.feedforward(x, useMetal: false, activationFunction: 1)

var lstm = LSTMNetwork(inputDim: 100, memCellCount: 2500)
lstm.feedforward(x, useMetal: false, activationFunction: 1)