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
let t: [Float] = [1, 3, 2, 4]


func softmax(z: [Float]) -> [Float] {
    let x = sum(exp(z))
    let y = [Float](count: z.count, repeatedValue: x)
    return div(exp(z), y: y)
}

sum(softmax(t))