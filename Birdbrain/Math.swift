//  Math.swift
//  Birdbrain
// A collection of mathematical functions that are useful for the neural network. Written using the
// Accelerate framework to optimize speed.

import Accelerate

var spare: Float = 0
var spareReady: Bool = false

public func sum(x: [Float]) -> Float {
    var result: Float = 0.0
    vDSP_sve(x, 1, &result, vDSP_Length(x.count))
    
    return result
}

//Add a vector x to a vector y
public func add(x: [Float], y: [Float]) -> [Float] {
    var results = [Float](y)
    cblas_saxpy(Int32(x.count), 1.0, x, 1, &results, 1)
    
    return results
}

//Add a scalar c to a vector x
public func add(x: [Float], var c: Float) -> [Float] {
    var result = [Float](count : x.count, repeatedValue : 0.0)
    vDSP_vsadd(x, 1, &c, &result, 1, vDSP_Length(x.count))
    
    return result
}

public func sub(A: [Float], B: [Float]) -> [Float] {
    var results = [Float](A)
    vDSP_vsub(B, 1, A, 1, &results, 1, vDSP_Length(A.count))
    
    return results
}

//Add a scalar value c of type double elementwise to a vector x


//Multiply a vextor x by a vector y.
/*
     x        y      results
   [ 1      [ 1       [ 1
     2    *   2     =   4
     3 ]      3 ]       9 ]
*/
public func mul(x: [Float], y: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vDSP_vmul(x, 1, y, 1, &results, 1, vDSP_Length(x.count))
    
    return results
}

//Multiply a vector x by a scalar y.
public func mul(x: [Float], var y: Float) -> [Float] {
    var result = [Float](count : x.count, repeatedValue : 0.0)
    vDSP_vsmul(x, 1, &y, &result, 1, vDSP_Length(x.count))
    
    return result
}

//Multiply a matrix A of float values by a vector x of float values.
/*
        A          x     results
    [ 1 2 3      [ 1     [ 14
      4 5 6    *   2   =   32
      7 8 9 ]      3 ]     50 ]
*/
public func mvMul(A: [Float], m: Int32, n: Int32, x: [Float]) -> [Float] {
    var results = [Float](count: Int(m), repeatedValue: 0.0)
    
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1, A, n,
        x, 1, 0, &results, 1)
    
    return results
}

//Divide a vector x by a vector y
public func div(x: [Float], y: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvdivf(&results, x, y, [Int32(x.count)])
    
    return results
}

//Divide a vector x by a scalar y
public func div(x: [Float], y: Float) -> [Float] {
    let divisor = [Float](count: x.count, repeatedValue: y)
    var result = [Float](count: x.count, repeatedValue: 0.0)
    vvdivf(&result, x, divisor, [Int32(x.count)])
    
    return result
}

//Perform an elementwise exponentiation on a vector x
public func exp(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvexpf(&results, x, [Int32(x.count)])

    return results
}

//Hyperbolic tangent for a vector
public func tanh(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvtanhf(&results, x, [Int32(x.count)])
    
    return results
}

//Square function for each element
public func square(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vDSP_vsq(x, 1, &results, 1, vDSP_Length(x.count))
    
    return results
}

//Negate each element in vector
public func neg(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vDSP_vneg(x, 1, &results, 1, vDSP_Length(x.count))
    
    return results
}

//Transpose a matrix
public func trans(A: [Float], m: Int, n: Int) -> [Float] {
    var transMatrix = [Float](count: A.count, repeatedValue: 0.0)
    
    vDSP_mtrans(A, 1, &transMatrix, 1, vDSP_Length(m), vDSP_Length(n))
    return transMatrix
}

//Multiply two vectors (considered as matrices in function) and create a matrix
public func formMatrix(A: [Float], B: [Float]) -> [Float] {
    var result = [Float](count: A.count * B.count, repeatedValue: 0.0)
    
    vDSP_mmul(A, 1, B, 1, &result, 1, vDSP_Length(result.count / B.count), vDSP_Length(result.count / A.count), vDSP_Length(1))
    
    return result
}

//Random gaussian
public func rand_gauss() -> Float {
    if (spareReady) {
        spareReady = false;
        return spare;
    }
    else {
        var u: Float
        var v: Float
        var s: Float
        repeat {
            u = (Float)(2.0 * drand48() - 1.0);
            v = (Float)(2.0 * drand48() - 1.0);
            s = u * u + v * v;
        } while ( s >= 1.0 );
        
        let mul = (Float)(sqrt( (-2.0 * log(s)) / s));
        spare = v * mul
        spareReady = true
        return u * mul
    }
}

public func initRand(n: Int) -> Float {
    let ARC4RANDOM_MAX: Float = 0x100000000
    let b = sqrtf(Float(1 / (Float(n))))
    let range = b - (-b)
    return ((Float(arc4random()) / Float(ARC4RANDOM_MAX)) * range - b)
}

//Rectified Linear Unit (y = max(0,x))
public func relu(x: [Float]) -> [Float] {
    let activation: [Float] = x.map({($0 < 0.0) ? 0.0 : $0})
    return activation
}

//Sigmoid function (1 / (1 + e^-x))
public func sigmoid(x: [Float]) -> [Float] {
    let ones = [Float](count: x.count, repeatedValue: 1.0)
    let z: [Float] =  div(ones, y: (add(exp(neg(x)), c: 1.0)))
    return z
}

//Sigmoid prime (sigmoid(x) * (1 - sigmoid(x))
public func sigmoidPrime(x: [Float]) -> [Float] {
    return mul(sigmoid(x), y: add(neg(sigmoid(x)), c: 1.0))
}

//Hyperbolic tangent prime (1 - tanh(x)^2)
public func tanhPrime(x: [Float]) -> [Float] {
    return add(neg(square(tanh(x))), c: 1.0)
}

//ReLU prime (x > 0 = 1, x <= 0 = 0)
public func reluPrime(x: [Float]) -> [Float] {
    let activation: [Float] = x.map({($0 <= 0.0) ? 0.0 : 1})
    return activation
}

//Softmax function (e^z / sum(e^z))
public func softmax(z: [Float]) -> [Float] {
    return div(exp(z), y: sum(exp(z)))
}

//Get the cost derivative
public func costDerivative(output: [Float], y: [Float]) -> [Float] {
    return sub(output, B: y)
}