//  Math.swift
//  Birdbrain
// A collection of mathematical functions that are useful for the neural network. Written using the
// Accelerate framework to optimize speed.

import Accelerate

var spare: Float = 0
var spareReady: Bool = false

//MARK: Vector Sum

/** Compute the vector sum of a vector.
  - Parameter x: Vector.
  - Returns: A single precision vector sum.
*/
func sum(x: [Float]) -> Float {
  var result: Float = 0.0
  
  vDSP_sve(x, 1, &result, vDSP_Length(x.count))
  
  return result
}

//MARK: Addition

/** Add a vector to another vector.
  - Parameter x: Vector x.
  - Parameter y: Vector y.
  - Returns: A vector containing the sum of the two vectors.
*/
func add(x: [Float], y: [Float]) -> [Float] {
  precondition(x.count == y.count, "Vectors must have the same length.")
  var result = [Float](y)
  
  cblas_saxpy(Int32(x.count), 1.0, x, 1, &result, 1)
  
  return result
}

/** Add a scalar to a vector.
  - Parameter x: Vector x.
  - Parameter c: scalar c.
  - Returns: A vector containing the sum of the vector and scalar.
*/
func add(x: [Float], var c: Float) -> [Float] {
  var result = [Float](count : x.count, repeatedValue : 0.0)
  
  vDSP_vsadd(x, 1, &c, &result, 1, vDSP_Length(x.count))
  
  return result
}

//MARK: Subtraction

/** Subtract a vector y from a vector x.
  - Parameter x: Vector x.
  - Parameter y: Vector y.
  - Returns: A vector containing the difference of x and y.
*/
func sub(x: [Float], y: [Float]) -> [Float] {
  precondition(x.count == y.count, "Vectors must have the same length.")
  var results = [Float](x)
  
  vDSP_vsub(y, 1, x, 1, &results, 1, vDSP_Length(x.count))
    
  return results
}

/** Subtract a scalar c from a vector x.
  - Parameter x: Vector x.
  - Parameter c: Scalar c.
  - Returns: A vector containing the difference of the scalar and the vector.
*/
func sub(A: [Float], c: Float) -> [Float] {
  var result = [Float](A)
  let operand = [Float](count: A.count, repeatedValue: c)

  vDSP_vsub(operand, 1, A, 1, &result, 1, vDSP_Length(A.count))
  
  return result
}

//MARK: Multiplication

/** Multiply a vector x by a vector y.
  - Parameter x: Vector x.
  - Parameter y: Vector y.
  - Returns: The product of vector x and vector y.
*/
func mul(x: [Float], y: [Float]) -> [Float] {
  precondition(x.count == y.count, "Vectors must have the same length.")
  var results = [Float](count: x.count, repeatedValue: 0.0)
  
  vDSP_vmul(x, 1, y, 1, &results, 1, vDSP_Length(x.count))
    
  return results
}

/** Multiply a vector x by a scalar y.
  - Parameter x: Vector x.
  - Parameter c: scalar c.
  - Returns: The product of vector x multiplied elementwise by scalar c.
*/
func mul(x: [Float], var y: Float) -> [Float] {
  var result = [Float](count : x.count, repeatedValue : 0.0)
  
  vDSP_vsmul(x, 1, &y, &result, 1, vDSP_Length(x.count))
  
  return result
}

/**Multiply a matrix A by a vector x.
 - Parameters:
  - A: matrix A.
  - m: Number of rows in matrix A.
  - n: Number of columns in matrix A.
  - x: Vector x.
 - Returns: A vector product of matrix A and vector x.
*/
public func mvMul(A: [Float], m: Int, n: Int, x: [Float]) -> [Float] {
  precondition(Int(n) == x.count, "Number of columns in matrix A must equal length of vector x.")
  var results = [Float](count: Int(m), repeatedValue: 0.0)
    
  cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(m), Int32(n), 1, A, Int32(n), x, 1, 0, &results, 1)
    
  return results
}

//MARK: Division

/** Divide a vector x by a vector y.
  - Parameter x: Vector to be divided.
  - Parameter y: Divisor vector.
  - Returns: A vector result of x divided by y.
*/
func div(x: [Float], y: [Float]) -> [Float] {
  precondition(x.count == y.count, "Vectors must be of equal length.")
  var results = [Float](count: x.count, repeatedValue: 0.0)
  
  vvdivf(&results, x, y, [Int32(x.count)])
    
  return results
}

/** Divide a vector x by a scalar y
  - Parameter x: Vector x.
  - Parameter c: Scalar c.
  - Returns: A vector containing x dvidided elementwise by vector c.
*/
func div(x: [Float], c: Float) -> [Float] {
  let divisor = [Float](count: x.count, repeatedValue: c)
  var result = [Float](count: x.count, repeatedValue: 0.0)
  
  vvdivf(&result, x, divisor, [Int32(x.count)])
    
  return result
}

//MARK: Special elementwise functions

/** Perform an elementwise exponentiation on a vector.
  - Parameter x: Vector x.
  - Returns: A vector containing x exponentiated elementwise.
*/
func exp(x: [Float]) -> [Float] {
  var results = [Float](count: x.count, repeatedValue: 0.0)
  
  vvexpf(&results, x, [Int32(x.count)])

  return results
}

/** Square a vector elementwise.
  - Parameter x: Vector x.
  - Returns: A vector containing elementwise squares of vectir x.
*/
func square(x: [Float]) -> [Float] {
  var results = [Float](count: x.count, repeatedValue: 0.0)
  
  vDSP_vsq(x, 1, &results, 1, vDSP_Length(x.count))
  
  return results
}

/**Negate each element in a vector.
  - Parameter x: Vector x.
  - Returns: An elementwise negation of vector x.
*/
func neg(x: [Float]) -> [Float] {
  var results = [Float](count: x.count, repeatedValue: 0.0)
  
  vDSP_vneg(x, 1, &results, 1, vDSP_Length(x.count))
    
  return results
}

//MARK: Matrix manipulations

/**Transpose a matrix.
  - Parameter A: Matrix A.
  - Parameter m: Number of rows in A.
  - Parameter n: Number of columns in A.
  - Returns: A transposed matrix A with m columns and n rows.
*/
func trans(A: [Float], m: Int, n: Int) -> [Float] {
  var transMatrix = [Float](count: A.count, repeatedValue: 0.0)
    
  vDSP_mtrans(A, 1, &transMatrix, 1, vDSP_Length(m), vDSP_Length(n))
  
  return transMatrix
}

/**Multiply two vectors and create a matrix.
 - Parameter x: Vector x
 - Parameter y: Vector y
 - Returns: A matrix of x.count rows and y.count columns.
*/
public func formMatrix(x: [Float], y: [Float]) -> [Float] {
  var result = [Float](count: x.count * y.count, repeatedValue: Float())
  
  let start = NSDate()
  vDSP_mmul(x, 1, y, 1, &result, 1, vDSP_Length(result.count / y.count),
    vDSP_Length(result.count / x.count), vDSP_Length(1))
  print(NSDate().timeIntervalSinceDate(start))
  return result
}

//MARK: RNGs

//Random gaussian
/**Random gaussian generator.
  - Returns: A random gaussian.
*/
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

/**Normal distribution random for initializing weights.
  - Parameter n: Number of connections from previous layer.
  - Returns: A uniform distribution between the interval [-1/sqrt(n),1/sqrt(n)].
*/
func initRand(n: Int) -> Float {
  let ARC4RANDOM_MAX: Float = 0x100000000
  let b = sqrtf(Float(1 / (Float(n))))
  let range = b - (-b)
  
  return ((Float(arc4random()) / Float(ARC4RANDOM_MAX)) * range - b)
}

//MARK: Activation Functions

/**A Rectified Linear Unit activation function.
  - Parameter x: A vector x.
  - Returns: A vector z = max(0,x).
*/
func relu(x: [Float]) -> [Float] {
  let z: [Float] = x.map({($0 < 0.0) ? 0.0 : $0})
  
  return z
}

/** A sigmoidal activation function.
  - Parameter x: A vector x.
  - Returns: A vector z = (1 / (1 + e^-x)).
*/
func sigmoid(x: [Float]) -> [Float] {
  let ones = [Float](count: x.count, repeatedValue: 1.0)
  let z: [Float] =  div(ones, y: (add(exp(neg(x)), c: 1.0)))
  
  return z
}

/** Hyperbolic tangent activation function.
  - Parameter x: A vector x.
  - Returns: A vector z = tanh(x).
*/
func tanh(x: [Float]) -> [Float] {
  var results = [Float](count: x.count, repeatedValue: 0.0)
  
  vvtanhf(&results, x, [Int32(x.count)])
  
  return results
}

//MARK: Activation function derivatives

/** Sigmoid prime function.
  - Parameter x: A vector x.
  - Returns: A vector y = (sigmoid(x) * (1 - sigmoid(x)).
*/
func sigmoidPrime(x: [Float]) -> [Float] {
  return mul(sigmoid(x), y: add(neg(sigmoid(x)), c: 1.0))
}

/** Hyperbolic tangent prime function. 
  - Parameter x: A vector x.
  - Returns: A vector y = (1 - tanh(x)^2).
*/
func tanhPrime(x: [Float]) -> [Float] {
  return add(neg(square(tanh(x))), c: 1.0)
}

/** ReLU prime function.
  - Parameter x: A vector x.
  - Returns: A vector y = (x > 0 = 1, x <= 0 = 0).
*/
func reluPrime(x: [Float]) -> [Float] {
  let activation: [Float] = x.map({($0 <= 0.0) ? 0.0 : 1})
  
  return activation
}

//MARK: Softmax function

/** Softmax function. 
  - Parameter z: A vector z.
  - Returns: A vector y = (e^z / sum(e^z))
*/
func softmax(z: [Float]) -> [Float] {
  let x = exp(sub(z, c: z.maxElement()!))
  
  return div(x, c: sum(x))
}

//MARK: Cost Derivative

/** Cost derivative function.
  - Parameter output: Output of network.
  - Parameter y: Target output.
  - Returns A vector y' = (output - y).
*/
func costDerivative(output: [Float], y: [Float]) -> [Float] {
  return sub(output, y: y)
}