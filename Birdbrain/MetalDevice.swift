//  MetalDevice.swift
//  Birdbrain
//  A collection of swift functions that will send a message to perform the called metal function.

import Foundation
import Metal

class MetalDevice {
  private var device: MTLDevice
  private var commandQueue: MTLCommandQueue
  private var pipelineState: MTLComputePipelineState!
  private var commandBuffer: MTLCommandBuffer
  private var library: MTLLibrary
    
  init() {
    device = MTLCreateSystemDefaultDevice()!
    commandQueue = device.newCommandQueue()
    pipelineState = nil
    library = device.newDefaultLibrary()!
    commandBuffer = commandQueue.commandBuffer()
  }
  
  /** Perform a sigmoid nonlinearity function on the vector.
   - Parameter vector: The vector.
   - Returns: The result of applying the function to the vector.
   */
  func sigmoid(vector: [Float]) -> [Float] {
    let function = library.newFunctionWithName("sigmoid")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: vector.count, repeatedValue: 0.0)
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let inputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(vector.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: vector.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: vector.count * sizeof(Float))
    
    return output
  }

  /** Perform a hyperbolic tangent nonlinearity function on the vector.
   - Parameter vecotr: The vector.
   - Returns: The result of applying the function to the vector.
   */
  func tanh(vector: [Float]) -> [Float] {
    let function = library.newFunctionWithName("tanh")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: vector.count, repeatedValue: 0.0)
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let inputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(vector.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: vector.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: vector.count * sizeof(Float))
    
    return output
  }
  
  /** Perform a hyperbolic tangent nonlinearity function on the vector.
   - Parameter vector: The vector.
   - Returns: The result of applying the function to the vector.
   */
  func relu(vector: [Float]) -> [Float] {
    let function = library.newFunctionWithName("relu")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: vector.count, repeatedValue: 0.0)
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let inputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(vector.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: vector.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: vector.count * sizeof(Float))
    
    return output
  }
  
  /** Perform a sigmoid nonlinearity function on the vector.
   - Parameter vector: The vector.
   - Returns: The result of applying the function to the vector.
   */
  func sigmoidPrime(vector: [Float]) -> [Float] {
    let function = library.newFunctionWithName("sigmoid_prime")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: vector.count, repeatedValue: 0.0)
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let inputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(vector.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: vector.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: vector.count * sizeof(Float))
    
    return output
  }
  
  /** Perform a hyperbolic tangent nonlinearity function on the vector.
   - Parameter vecotr: The vector.
   - Returns: The result of applying the function to the vector.
   */
  func tanhPrime(vector: [Float]) -> [Float] {
    let function = library.newFunctionWithName("tanh_prime")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: vector.count, repeatedValue: 0.0)
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let inputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(vector.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: vector.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: vector.count * sizeof(Float))
    
    return output
  }
  
  /** Perform a hyperbolic tangent nonlinearity function on the vector.
   - Parameter vector: The vector.
   - Returns: The result of applying the function to the vector.
   */
  func reluPrime(vector: [Float]) -> [Float] {
    let function = library.newFunctionWithName("relu_prime")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: vector.count, repeatedValue: 0.0)
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let inputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(vector.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: vector.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: vector.count * sizeof(Float))
    
    return output
  }
  
  /** Perform elementwise negation of the vector
   - Parameter x: The vector.
   - Returns: The vector with every element negated.
   */
  func neg(vector: [Float]) -> [Float] {
    let function = library.newFunctionWithName("neg")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: vector.count, repeatedValue: 0.0)
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let inputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(vector.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: vector.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: vector.count * sizeof(Float))
    
    return output
  }
  
  /** Perform elementwise exponentiation of the vector
   - Parameter x: The vector.
   - Returns: The vector with every element exponentiated.
   */
  func exp(vector: [Float]) -> [Float] {
    let function = library.newFunctionWithName("exp")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: vector.count, repeatedValue: 0.0)
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let inputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(vector.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: vector.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: vector.count * sizeof(Float))
    
    return output
  }
  
  /** Perform elementwise square of the vector
   - Parameter x: The vector.
   - Returns: The vector with every element squared.
   */
  func square(vector: [Float]) -> [Float] {
    let function = library.newFunctionWithName("square")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: vector.count, repeatedValue: 0.0)
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let inputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(vector.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: vector.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: vector.count * sizeof(Float))
    
    return output
  }
  
  /** Add two vectors.
   - Parameter x: First vector.
   - Parameter y: Second vector.
   - Returns: The result of adding vectors x and y.
   */
  func add(x: [Float], y: [Float]) -> [Float] {
    let function = library.newFunctionWithName("add")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: x.count, repeatedValue: 0.0)
    let dim: [UInt16] = [UInt16(x.count)]
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let firstInputBuffer = device.newBufferWithBytes(x, length: x.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let secondInputBuffer = device.newBufferWithBytes(y, length: y.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let dimBuffer = device.newBufferWithBytes(dim, length: sizeof(UInt16), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(x.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(firstInputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(secondInputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 2)
    commandEncoder.setBuffer(dimBuffer, offset: 0, atIndex: 3)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(x.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: output.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: output.count * sizeof(Float))
    
    return output
  }
  
  /** Add a scalar value to a vector.
   - Parameter x: The vector.
   - Parameter c: The scalar.
   - Returns: The result of adding vectors x and y.
   */
  func scalAdd(x: [Float], c: Float) -> [Float] {
    let function = library.newFunctionWithName("scalar_add")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: x.count, repeatedValue: 0.0)
    let scalar: [Float] = [c]
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let firstInputBuffer = device.newBufferWithBytes(x, length: x.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let secondInputBuffer = device.newBufferWithBytes(scalar, length: sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(x.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(firstInputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(secondInputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 2)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(x.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: output.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: output.count * sizeof(Float))
    
    return output
  }
  
  /** Subtract two vectors.
   - Parameter x: First vector.
   - Parameter y: Second vector.
   - Returns: The result of subtracting vector y from vector x.
   */
  func sub(x: [Float], y: [Float]) -> [Float] {
    let function = library.newFunctionWithName("subtract")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: x.count, repeatedValue: 0.0)
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let firstInputBuffer = device.newBufferWithLength(x.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let secondInputBuffer = device.newBufferWithLength(y.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(x.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(firstInputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(secondInputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 2)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(x.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: output.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: output.count * sizeof(Float))
    
    return output
  }
  
  /** Multiply two vectors
   - Parameter x: First vector.
   - Parameter y: Second vector.
   - Returns: The result of multiplying vectors x and y.
   */
  func mul(x: [Float], y: [Float]) -> [Float] {
    let function = library.newFunctionWithName("multiply")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: x.count, repeatedValue: 0.0)
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let firstInputBuffer = device.newBufferWithLength(x.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let secondInputBuffer = device.newBufferWithLength(y.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(x.count * sizeof(Float), options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(firstInputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(secondInputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 2)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(x.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: output.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: output.count * sizeof(Float))
    
    return output
  }
  
  /** Divide two vectors
   - Parameter x: First vector.
   - Parameter y: Second vector.
   - Returns: The result of dividing vectors x by vector y.
   */
  func div(x: [Float], y: [Float]) -> [Float] {
    let function = library.newFunctionWithName("divide")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    var output = [Float](count: x.count, repeatedValue: 0.0)
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    
    let firstInputBuffer = device.newBufferWithLength(x.count * sizeof(Float), options: MTLResourceOptions.StorageModePrivate)
    let secondInputBuffer = device.newBufferWithLength(y.count * sizeof(Float), options: MTLResourceOptions.StorageModePrivate)
    let outputBuffer = device.newBufferWithLength(x.count * sizeof(Float), options: MTLResourceOptions.StorageModePrivate)
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(firstInputBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(secondInputBuffer, offset: 0, atIndex: 1)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 2)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(x.count))
    threadGroups.width = pow
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: output.count * sizeof(Float), freeWhenDone: false)
    print(data.length)
    data.getBytes(&output, length: output.count * sizeof(Float))
    
    return output
  }
  
  /** Multiply a matrix by a vector
   - Parameter matrix: The matrix.
   - Parameter m: Number of rows in matrix.
   - Parameter n: Number of columns in matrix.
   - Parameter vector: The vector.
   - Returns: The result of dividing vectors x by vector y.
   */
  func mvMul(matrix: [Float], m: Int, n: Int, vector: [Float]) -> [Float] {
    let function = library.newFunctionWithName("matrixvector_multiply")
    var threadGroupSize: MTLSize
    var threadGroups: MTLSize
    let dims: [UInt32] = [UInt32(m), UInt32(n)]
    var output = [Float]()
    
    do {
      pipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("Caught error when trying to create computePipelineState!")
    }
    let matrixBuffer = device.newBufferWithBytes(matrix, length: matrix.count * sizeof(Float), options: MTLResourceOptions.StorageModeManaged)
    let vectorBuffer = device.newBufferWithBytes(vector, length: vector.count * sizeof(Float), options: MTLResourceOptions.StorageModeManaged)
    let dimsBuffer = device.newBufferWithBytes(dims, length: dims.count * sizeof(UInt32), options: MTLResourceOptions.StorageModeManaged)
    let outputBuffer = device.newBufferWithLength(vector.count * sizeof(Float), options: MTLResourceOptions.StorageModeManaged)
    
    threadGroupSize = MTLSizeMake(128, 1, 1)
    threadGroups = MTLSizeMake(0, 1, 1)
    
    let pow = Int(nearestPower2(m))
    threadGroups.width = pow
    
    let commandEncoder = commandBuffer.computeCommandEncoder()
    commandEncoder.setBuffer(matrixBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(vectorBuffer, offset: 0, atIndex: 1)
    commandEncoder.setBuffer(dimsBuffer, offset: 0, atIndex: 2)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 3)
    commandEncoder.setThreadgroupMemoryLength(vector.count, atIndex: 4)
    commandEncoder.setComputePipelineState(pipelineState!)
    
    commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    commandEncoder.endEncoding()
  
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let data = NSData(bytesNoCopy: outputBuffer.contents(), length: vector.count * sizeof(Float), freeWhenDone: false)
    data.getBytes(&output, length: vector.count * sizeof(Float))
    
    return output
  }
  
  /** Find the nearest power of 2 for a number
   - Parameter num: the number to find the nearest power 2 of.
   - Returns: The nearest power 2 of num (30 -> 32, 200 -> 256).
   */
  private func nearestPower2(num: Int) -> Int {
    var n: Int = num > 0 ? num - 1 : 0;
      
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n += 1;
      
    return n;
  }
}