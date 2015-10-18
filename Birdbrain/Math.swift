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

public func add(x: [Float], y: [Float]) -> [Float] {
    var results = [Float](y)
    cblas_saxpy(Int32(x.count), 1.0, x, 1, &results, 1)
    
    return results
}

public func sub(A: [Float], B: [Float]) -> [Float] {
    var results = [Float](A)
    vDSP_vsub(B, 1, A, 1, &results, 1, vDSP_Length(A.count))
    
    return results
}

//Add a scalar value c of type double elementwise to a vector x
public func scalAdd(var c: Float, x: [Float]) -> [Float] {
    var result = [Float](count : x.count, repeatedValue : 0.0)
    vDSP_vsadd(x, 1, &c, &result, 1, vDSP_Length(x.count))
    
    return result
}

public func scalMul(x: [Float], var y: Float) -> [Float] {
    var result = [Float](count : x.count, repeatedValue : 0.0)
    vDSP_vsmul(x, 1, &y, &result, 1, vDSP_Length(x.count))
    
    return result
}

//Multiply a vextor x by a vector y.
public func mul(x: [Float], y: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vDSP_vmul(x, 1, y, 1, &results, 1, vDSP_Length(x.count))
    
    return results
}

//Multiply a matrix A of double values by a vector x of double values.
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

public func dot(x: [Float], y: [Float]) -> Float {
    precondition(x.count == y.count, "Vectors must have equal count")
    
    var result: Float = 0.0
    vDSP_dotpr(x, 1, y, 1, &result, vDSP_Length(x.count))
    
    return result
}

public func trans(A: [Float], m: Int, n: Int) -> [Float] {
    var transMatrix = [Float](count: A.count, repeatedValue: 0.0)
    
    vDSP_mtrans(A, 1, &transMatrix, 1, vDSP_Length(m), vDSP_Length(n))
    return transMatrix
}

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

public func randf(x: Int, y: Int) -> Float {
    let ARC4RANDOM_MAX: Float = 0x100000000
    let b = sqrtf(Float(6 / (Float(x) + Float(y))))
    let range = b + b
    return ((Float(arc4random()) / Float(ARC4RANDOM_MAX)) * range - b)
}