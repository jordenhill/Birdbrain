//: Playground - noun: a place where people can play

import Cocoa
import Accelerate
import Birdbrain
import Metal

var str = "Hello, playground"

let x = [Float](count: 5, repeatedValue: 0.0)
let dim = 5
/*
[0,0,0,0,0,0
 0,0,0,0,0,0
 0,0,0,0,0,0
 0,0,0,0,0,0
 0,0,0,0,0,0]
*/
var s = [Float](count: (x.count + 1) * dim, repeatedValue: 0.0)
var t = [Float](count: 10, repeatedValue: 0.0)
for x in Range(start: 0, end: 10) {
    t[x] = initRand(3)
}
t
let q: Float = 0.5435
softmax(t).maxElement()