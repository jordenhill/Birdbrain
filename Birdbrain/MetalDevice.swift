//  MetalDevice.swift
//  Birdbrain
//  A collection of swift functions that will send a message to perform the called metal function.

import Foundation
import Metal

func initMetal() -> (MTLDevice, MTLLibrary, MTLCommandBuffer, MTLComputeCommandEncoder) {
  let device = MTLCreateSystemDefaultDevice()
  let commandQueue = device!.newCommandQueue()
  let library = device?.newDefaultLibrary()
  let commandBuffer = commandQueue.commandBuffer()
  let commandEncoder = commandBuffer.computeCommandEncoder()
        
  return (device!, library!, commandBuffer, commandEncoder)
}

//MARK: Vector Sum Function

func performElementwiseFunction(function_name: String, vector: [Float]) -> [Float] {
  let (device, library, commandBuffer, commandEncoder) = initMetal()
  let function = library.newFunctionWithName(function_name)
  let sigmoidPipelineDescriptor = MTLComputePipelineDescriptor()
  sigmoidPipelineDescriptor.computeFunction = function
  //let computePipelineErrors = NSErrorPointer()
  var computePipelineState:MTLComputePipelineState? = nil
  var output = [Float](count: vector.count, repeatedValue: 0.0)
  
  do {
    computePipelineState = try device.newComputePipelineStateWithFunction(function!)
  } catch {
    print("catching..")
  }
  
  let inputBuffer = device.newBufferWithLength(vector.count * sizeof(Float),
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
  let outputBuffer = device.newBufferWithLength(output.count * sizeof(Float),
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
  
  commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
  commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 1)
  commandEncoder.setComputePipelineState(computePipelineState!)
  
  let threadExecutionWidth = computePipelineState!.threadExecutionWidth
  
  let threadsPerGroup = MTLSize(width:threadExecutionWidth,height:1,depth:1)
  let numThreadgroups = MTLSize(width:(vector.count+threadExecutionWidth)/threadExecutionWidth,
    height: 1, depth: 1)
  computePipelineState?.threadExecutionWidth
  computePipelineState?.maxTotalThreadsPerThreadgroup
  
  commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
  commandEncoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
  
  let data = NSData(bytesNoCopy: outputBuffer.contents(), length: vector.count*sizeof(Float),
    freeWhenDone: false)
  
  data.getBytes(&output, length:vector.count * sizeof(Float))
  
  return output
}

//MARK: Activation Function

func performActivationFunction(function_name: String, input: [Float]) -> [Float] {
  let (device, library, commandBuffer, commandEncoder) = initMetal()
  let function = library.newFunctionWithName(function_name)
  let sigmoidPipelineDescriptor = MTLComputePipelineDescriptor()
  sigmoidPipelineDescriptor.computeFunction = function
  //let computePipelineErrors = NSErrorPointer()
  var computePipelineState:MTLComputePipelineState? = nil
  var output = [Float](count: input.count, repeatedValue: 0.0)
    
  do {
    computePipelineState = try device.newComputePipelineStateWithFunction(function!)
  } catch {
    print("catching..")
  }
        
  let inputBuffer = device.newBufferWithLength(input.count * sizeof(Float),
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
  let outputBuffer = device.newBufferWithLength(output.count * sizeof(Float),
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
  
  commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
  commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 1)
  commandEncoder.setComputePipelineState(computePipelineState!)
        
  let threadExecutionWidth = computePipelineState!.threadExecutionWidth
        
  let threadsPerGroup = MTLSize(width: threadExecutionWidth,height:1,depth:1)
  let numThreadgroups = MTLSize(width:(input.count+threadExecutionWidth)/threadExecutionWidth,
    height: 1, depth: 1)
  
  commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
  commandEncoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
        
  let data = NSData(bytesNoCopy: outputBuffer.contents(), length: input.count*sizeof(Float),
    freeWhenDone: false)
  
  data.getBytes(&output, length:input.count * sizeof(Float))
    
  return output
}

//MARK: Matrix-Vector Function

func performMatVecFunction(functionName: String, matrix: [Float], m: Int, n: Int,
  vector: [Float]) -> [Float] {
  
    let (device, library, commandBuffer, commandEncoder) = initMetal()
    let function = library.newFunctionWithName(functionName)
    let pipelineDescriptor = MTLComputePipelineDescriptor()
    pipelineDescriptor.computeFunction = function
    //let computePipelineErrors = NSErrorPointer()
    var computePipelineState:MTLComputePipelineState? = nil
    var outputVector: [Float] = (1...m).map{_ in 0.0}
    let dims: [UInt16] = [UInt16(m), UInt16(n)]
  
    do {
      computePipelineState = try device.newComputePipelineStateWithFunction(function!)
    } catch {
      print("catching..")
    }
  
    let matrixBuffer = device.newBufferWithLength(matrix.count * sizeof(Float),
      options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let vectorBuffer = device.newBufferWithLength(vector.count * sizeof(Float),
      options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let dimBuffer = device.newBufferWithLength(dims.count * sizeof(UInt16),
      options: MTLResourceOptions.CPUCacheModeDefaultCache)
    let outputBuffer = device.newBufferWithLength(outputVector.count * sizeof(Float),
      options: MTLResourceOptions.CPUCacheModeDefaultCache)

    commandEncoder.setBuffer(matrixBuffer, offset: 0, atIndex: 0)
    commandEncoder.setBuffer(vectorBuffer, offset: 0, atIndex: 1)
    commandEncoder.setBuffer(dimBuffer, offset: 0, atIndex: 2)
    commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 3)
    commandEncoder.setComputePipelineState(computePipelineState!)

    let threadsPerGroup = MTLSize(width: 32, height: 1, depth:1)
    let numThreadgroups = MTLSize(width:(256 + 31) / 32, height: 1, depth: 1)
  
    let time = NSDate()
    commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
    commandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    print(NSDate().timeIntervalSinceDate(time))

    let data = NSData(bytesNoCopy: outputBuffer.contents(),
      length: vector.count*sizeof(Float), freeWhenDone: false)
    
    data.getBytes(&outputVector, length: m * sizeof(Float))

    return outputVector
}

//MARK: Vector-Vector Function

func performVecVecFunction(functionName: String, vectorX: [Float], vectorY: [Float]) -> [Float] {
  let (device, library, commandBuffer, commandEncoder) = initMetal()
  let function = library.newFunctionWithName(functionName)
  let sigmoidPipelineDescriptor = MTLComputePipelineDescriptor()
  sigmoidPipelineDescriptor.computeFunction = function
  //let computePipelineErrors = NSErrorPointer()
  var computePipelineState:MTLComputePipelineState? = nil
  var outputVector: [Float] = (1...vectorX.count).map{_ in 0.0}
    
  do {
    computePipelineState = try device.newComputePipelineStateWithFunction(function!)
  } catch {
    print("catching..")
  }
    
  let vectorXBuffer = device.newBufferWithLength(vectorX.count * sizeof(Float),
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
  let vectorYBuffer = device.newBufferWithLength(vectorY.count * sizeof(Float),
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
  let outputBuffer = device.newBufferWithLength(outputVector.count * sizeof(Float),
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
  
  commandEncoder.setBuffer(vectorXBuffer, offset: 0, atIndex: 0)
  commandEncoder.setBuffer(vectorYBuffer, offset: 0, atIndex: 1)
  commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 2)
  commandEncoder.setComputePipelineState(computePipelineState!)
  
  //let threadExecutionWidth = computePipelineState!.threadExecutionWidth
  let threadsPerGroup = MTLSize(width: 32, height:1, depth:1)
  let numThreadgroups = MTLSize(width: (256 + 31) / 32, height: 1, depth: 1)
  
  commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
  commandEncoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
    
  let data = NSData(bytesNoCopy: outputBuffer.contents(),
    length: vectorX.count*sizeof(Float), freeWhenDone: false)
  
  data.getBytes(&outputVector, length:vectorX.count * sizeof(Float))
  
  return outputVector
}

//MARK: Vector-Scalar function

func performVecScalFunction(functionName: String, vector: [Float], scalar: Float) -> [Float] {
  let (device, library, commandBuffer, commandEncoder) = initMetal()
  let function = library.newFunctionWithName(functionName)
  let sigmoidPipelineDescriptor = MTLComputePipelineDescriptor()
  sigmoidPipelineDescriptor.computeFunction = function
  //let computePipelineErrors = NSErrorPointer()
  var computePipelineState:MTLComputePipelineState? = nil
  var outputVector = [Float](count: vector.count, repeatedValue: 0.0)
    
  do {
    computePipelineState = try device.newComputePipelineStateWithFunction(function!)
  } catch {
    print("catching..")
  }

  let vectorBuffer = device.newBufferWithLength(vector.count * sizeof(Float),
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
  let scalarBuffer = createScalarBuffer(scalar, device: device)
  let outputBuffer = device.newBufferWithLength(outputVector.count * sizeof(Float),
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
    
  commandEncoder.setBuffer(vectorBuffer, offset: 0, atIndex: 0)
  commandEncoder.setBuffer(scalarBuffer, offset: 0, atIndex: 1)
  commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 2)
  
  let threadExecutionWidth = computePipelineState!.threadExecutionWidth
  let threadsPerGroup = MTLSize(width:threadExecutionWidth,height:1,depth:1)
  let numThreadgroups = MTLSize(width:(vector.count + threadExecutionWidth) /
    threadExecutionWidth, height: 1, depth: 1)
  
  commandEncoder.setComputePipelineState(computePipelineState!)
  commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
  
  commandEncoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
    
  let data = NSData(bytesNoCopy: outputBuffer.contents(),
    length: vector.count*sizeof(Float), freeWhenDone: false)
  
  data.getBytes(&outputVector, length:vector.count * sizeof(Float))
  
  return outputVector
}


//MARK: Extra Methods

func createBuffer(var vector: [Float], device: MTLDevice) -> MTLBuffer {  
  //Create a new buffer with this byte length
  let buffer = device.newBufferWithBytes(&vector, length: vector.count * sizeof(Float),
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
  
  return buffer
}

func createDimBuffer(var dims: [UInt16], device: MTLDevice) -> MTLBuffer {

  return device.newBufferWithBytes(&dims, length: 2 * sizeof(UInt16),
    options: MTLResourceOptions.CPUCacheModeDefaultCache)

}

func createScalarBuffer(var scalar: Float, device: MTLDevice) -> MTLBuffer {
  let byteLength = sizeof(Float)
  
  return device.newBufferWithBytes(&scalar, length: byteLength,
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
}

func createScalarBuffer(var scalar: Int, device: MTLDevice) -> MTLBuffer {
  let byteLength = sizeof(Int)
  
  return device.newBufferWithBytes(&scalar, length: byteLength,
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
}