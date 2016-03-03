//A test playground containing various bits of quick "research"

import Foundation

//Size of Ints
sizeof(Int)
sizeof(Int32)
sizeof(Int16)

//Get length of string

var string = "Hello"
string.characters.count

//Checking size 
sizeof(Int) * 100000000

//C Strings in Swift
var char = "a"
let p = "string"
var ccharoptional = char.cStringUsingEncoding(NSUTF8StringEncoding)![0]
var sum = 0
for a in 0..<p.characters.count {
  print(Int(String(p[p.startIndex.advancedBy(a)]).cStringUsingEncoding(NSUTF8StringEncoding)![0]))
  sum += Int(String(p[p.startIndex.advancedBy(a)]).cStringUsingEncoding(NSUTF8StringEncoding)![0])
  print(sum)
}
sum

//Messing around with sorting arrays and a struct
struct thing {
  var val = Int()
}

var array = [thing]()
for i in 0..<10 {
  array.append(thing(val: i))
}

var firstElement = array[0]
var sortedArray = array.dropFirst().sort() { a,b in return a.val < b.val}
sortedArray.insert(firstElement, atIndex: 0)
array = sortedArray

//
var nextRandom = 1
var layer1Size = 10000
var a = 0
var b = 0
var syn0 = [Float](count: 5000, repeatedValue: 0.0)
nextRandom = nextRandom * 25214903917 + 11
syn0[a * layer1Size + b] = ((Float(nextRandom & 0xFFFF) / Float(65536)) - 0.5) / Float(layer1Size);

//
var activateFunction = "sig"

if ((activateFunction != "sigmoid") && (activateFunction != "tangent") &&
  (activateFunction != "relu")) {
  activateFunction = "sigmoid"
}

activateFunction

//Array construction
var x = [Float](count: 10, repeatedValue: 1)
var y = [Float](x)