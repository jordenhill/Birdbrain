//: Playground - noun: a place where people can play

import UIKit

var str = "Hello, playground"

var b = [[Int]](count: 100, repeatedValue: [Int](count: 1, repeatedValue: 0))
b[99][0] = 1
let d: [[Float]] = (0..<10).map { _ in (0..<100).map {_ in Float(arc4random_uniform(2)) } }

b
d