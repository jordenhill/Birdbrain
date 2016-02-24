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

var activateFunction = "sigmoid"

print((activateFunction != "sigmoid") && (activateFunction != "tangent") && (activateFunction != "relu"))