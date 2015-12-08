//
//  MetalMath.swift
//  Birdbrain
//
//  Created by Jorden Hill on 12/3/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

import Foundation

//MARK: Addition

/** Calls Metal to add a vector to another vector.
  - Parameter x: Vector x.
  - Parameter y: Vector y.
  - Returns: A vector containing the sum of the two vectors.
*/
func mtlAdd(x: [Float], y: [Float]) -> [Float] {
  return performVecVecFunction("add", vectorX: x, vectorY: y)
}

/** Calls metal to add a scalar to a vector.
  - Parameter x: Vector x.
  - Parameter c: scalar c.
  - Returns: A vector containing the sum of the vector and scalar.
*/
func mtlAdd(x: [Float], c: Float) -> [Float] {
  return performVecScalFunction("scalar_add", vector: x, scalar: c)
}

//MARK: Subtraction

/** Subtract a vector y from a vector x.
  - Parameter x: Vector x.
  - Parameter y: Vector y.
  - Returns: A vector containing the difference of x and y.
*/
func mtlSub(x: [Float], y: [Float]) -> [Float] {
  return performVecVecFunction("subtract", vectorX: x, vectorY: y)
}

/** Subtract a scalar c from a vector x.
  - Parameter x: Vector x.
  - Parameter c: Scalar c.
  - Returns: A vector containing the difference of the scalar and the vector.
*/
func mtlSub(x: [Float], c: Float) -> [Float] {
  return performVecScalFunction("scalar_subtract", vector: x, scalar: c)
}

//MARK: Multiplication

/** Multiply a vector x by a vector y.
  - Parameter x: Vector x.
  - Parameter y: Vector y.
  - Returns: The product of vector x and vector y.
*/
func mtlMul(x: [Float], y: [Float]) -> [Float] {
  return performVecVecFunction("multiply", vectorX: x, vectorY: y)
}

/** Multiply a vector x by a scalar y.
  - Parameter x: Vector x.
  - Parameter c: scalar c.
  - Returns: The product of vector x multiplied elementwise by scalar c.
*/
func mtlMul(v: [Float], c: Float) -> [Float] {
  return performVecScalFunction("scalar_multiply", vector: v, scalar: c)
}

/** Multiply a matrix by a vector.
  - Parameter A: An m by n matrix.
  - Parameter m: Number of rows in matrix A.
  - Parameter n: Number of columns in matrix A.
  - Parameter x: Vector x.
  - Returns: The matrix vector product of Ax.
*/
func mtlMvMul(A: [Float], m: Int, n: Int, x: [Float]) -> [Float] {
  return performMatVecFunction("matrixvector_multiply", matrix: A, m: m, n: n, vector: x)
}

//MARK: Division

/** Divide a vector x by a vector y.
  - Parameter x: Vector to be divided.
  - Parameter y: Divisor vector.
  - Returns: A vector result of x divided by y.
*/
func mtlDiv(x: [Float], y: [Float]) -> [Float] {
  return performVecVecFunction("divide", vectorX: x, vectorY: y)
}

/** Divide a vector x by a scalar y
  - Parameter x: Vector x.
  - Parameter c: Scalar c.
  - Returns: A vector containing x dvidided elementwise by vector c.
*/
func mtlDiv(x: [Float], c: Float) -> [Float] {
  return performVecScalFunction("scalar_divide", vector: x, scalar: c)
}

//MARK: Special Elementwise functions

/** Perform an elementwise exponentiation on a vector.
  - Parameter x: Vector x.
  - Returns: A vector containing x exponentiated elementwise.
*/
func mtlExp(x: [Float]) -> [Float] {
  return performElementwiseFunction("exp", vector: x)
}

/** Square a vector elementwise.
  - Parameter x: Vector x.
  - Returns: A vector containing elementwise squares of vectir x.
*/
func mtlSquare(x: [Float]) -> [Float] {
  return performElementwiseFunction("square", vector: x)
}

/** Negate each element in a vector.
  - Parameter x: Vector x.
  - Returns: An elementwise negation of vector x.
*/
func mtlNeg(x: [Float]) -> [Float] {
  return performElementwiseFunction("neg", vector: x)
}

//MARK: Activation Functions

/** A sigmoidal activation function.
  - Parameter x: A vector x.
  - Returns: A vector z = (1 / (1 + e^-x)).
*/
func mtlSigmoid(x: [Float]) -> [Float] {
  return performActivationFunction("sigmoid", input: x)
}

/** Hyperbolic tangent activation function.
  - Parameter x: A vector x.
  - Returns: A vector z = tanh(x).
*/
func mtlTanh(x: [Float]) -> [Float] {
  return performActivationFunction("tanh", input: x)
}

/** A Rectified Linear Unit activation function.
  - Parameter x: A vector x.
  - Returns: A vector z = max(0,x).
*/
func mtlRelu(x: [Float]) -> [Float] {
  return performActivationFunction("relu", input: x)
}

//MARK: Activation Function Primes

/** Sigmoid prime function.
  - Parameter x: A vector x.
  - Returns: A vector y = (sigmoid(x) * (1 - sigmoid(x)).
*/
func mtlSigmoidPrime(x: [Float]) -> [Float] {
  return performActivationFunction("sigmoid_prime", input: x)
}

/** Hyperbolic tangent prime function.
  - Parameter x: A vector x.
  - Returns: A vector y = (1 - tanh(x)^2).
*/
func mtlTanhPrime(x: [Float]) -> [Float] {
  return performActivationFunction("tanh_prime", input: x)
}

/** ReLU prime function.
  - Parameter x: A vector x.
  - Returns: A vector y = (x > 0 = 1, x <= 0 = 0).
*/
func mtlReluPrime(x: [Float]) -> [Float] {
  return performActivationFunction("relu_prime", input: x)
}

//MARK: SoftMax Function

/** Softmax function.
  - Parameter z: A vector z.
  - Returns: A vector y = (e^z / sum(e^z))
*/
func mtlSoftmax(z: [Float]) -> [Float] {
  let x = mtlExp(mtlSub(z, c: z.maxElement()!))
  return performVecScalFunction("scalar_divide", vector: x, scalar: sum(x))
}

//MARK: Cost Derivative