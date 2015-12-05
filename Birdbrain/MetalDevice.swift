//  MetalDevice.swift
//  Birdbrain
//  A collection of swift functions that will send a message to perform the called metal function.

import Foundation
import Metal

func initMetal() -> (MTLDevice, MTLLibrary, MTLCommandBuffer, MTLComputeCommandEncoder) {
  let device = MTLCreateSystemDefaultDevice()
  let commandQueue = device!.newCommandQueue()
  let library = device!.newDefaultLibrary()
  let commandBuffer = commandQueue.commandBuffer()
  let commandEncoder = commandBuffer.computeCommandEncoder()
        
  return (device!, library!, commandBuffer, commandEncoder)
}

//MARK: Vector Sum Function

func performElementwiseFunction(function_name: String, vector: [Float]) -> [Float] {
  let (device, library, commandBuffer, commandEncoder) = initMetal()
  let sigmoidFunction = library.newFunctionWithName(function_name)
  let sigmoidPipelineDescriptor = MTLComputePipelineDescriptor()
  sigmoidPipelineDescriptor.computeFunction = sigmoidFunction
  //let computePipelineErrors = NSErrorPointer()
  var computePipelineState:MTLComputePipelineState? = nil
  var output = [Float](count: vector.count, repeatedValue: 0.0)
  
  do {
    computePipelineState = try device.newComputePipelineStateWithFunction(sigmoidFunction!)
  } catch {
    print("catching..")
  }
  
  let inputBuffer = createBuffer(vector, device: device)
  let outputBuffer = createBuffer(output, device: device)
  
  commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
  commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 1)
  commandEncoder.setComputePipelineState(computePipelineState!)
  
  let threadExecutionWidth = computePipelineState!.threadExecutionWidth
  
  let threadsPerGroup = MTLSize(width:threadExecutionWidth,height:1,depth:1)
  let numThreadgroups = MTLSize(width:(vector.count+threadExecutionWidth)/threadExecutionWidth,
    height: 1, depth: 1)
  
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
  let sigmoidFunction = library.newFunctionWithName(function_name)
  let sigmoidPipelineDescriptor = MTLComputePipelineDescriptor()
  sigmoidPipelineDescriptor.computeFunction = sigmoidFunction
  //let computePipelineErrors = NSErrorPointer()
  var computePipelineState:MTLComputePipelineState? = nil
  var output = [Float](count: input.count, repeatedValue: 0.0)
    
  do {
    computePipelineState = try device.newComputePipelineStateWithFunction(sigmoidFunction!)
  } catch {
    print("catching..")
  }
        
  let inputBuffer = createBuffer(input, device: device)
  let outputBuffer = createBuffer(output, device: device)
  
  commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
  commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 1)
  commandEncoder.setComputePipelineState(computePipelineState!)
        
  let threadExecutionWidth = computePipelineState!.threadExecutionWidth
        
  let threadsPerGroup = MTLSize(width:threadExecutionWidth,height:1,depth:1)
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

//MARK: Vector-Vector Function

func performVecVecFunction(functionName: String, vectorX: [Float], vectorY: [Float]) -> [Float] {
  let (device, library, commandBuffer, commandEncoder) = initMetal()
  let sigmoidFunction = library.newFunctionWithName(functionName)
  let sigmoidPipelineDescriptor = MTLComputePipelineDescriptor()
  sigmoidPipelineDescriptor.computeFunction = sigmoidFunction
  //let computePipelineErrors = NSErrorPointer()
  var computePipelineState:MTLComputePipelineState? = nil
  var outputVector = [Float](count: vectorX.count, repeatedValue: 0.0)
    
  do {
    computePipelineState = try device.newComputePipelineStateWithFunction(sigmoidFunction!)
  } catch {
    print("catching..")
  }
    
  let vectorXBuffer = createBuffer(vectorX, device: device)
  let vectorYBuffer = createBuffer(vectorY, device: device)
  let outputBuffer = createBuffer(outputVector, device: device)
    
  commandEncoder.setBuffer(vectorXBuffer, offset: 0, atIndex: 0)
  commandEncoder.setBuffer(vectorYBuffer, offset: 0, atIndex: 1)
  commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 2)
  commandEncoder.setComputePipelineState(computePipelineState!)
  
  let threadExecutionWidth = computePipelineState!.threadExecutionWidth
  let threadsPerGroup = MTLSize(width:threadExecutionWidth,height:1,depth:1)
  let numThreadgroups = MTLSize(width:(vectorX.count + threadExecutionWidth) /
    threadExecutionWidth, height: 1, depth: 1)
  
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
  let sigmoidFunction = library.newFunctionWithName(functionName)
  let sigmoidPipelineDescriptor = MTLComputePipelineDescriptor()
  sigmoidPipelineDescriptor.computeFunction = sigmoidFunction
  //let computePipelineErrors = NSErrorPointer()
  var computePipelineState:MTLComputePipelineState? = nil
  var outputVector = [Float](count: vector.count, repeatedValue: 0.0)
    
  do {
    computePipelineState = try device.newComputePipelineStateWithFunction(sigmoidFunction!)
  } catch {
    print("catching..")
  }
  
  let vectorBuffer = createBuffer(vector, device: device)
  let scalarBuffer = createScalarBuffer(scalar, device: device)
  let outputBuffer = createBuffer(outputVector, device: device)
    
  commandEncoder.setBuffer(vectorBuffer, offset: 0, atIndex: 0)
  commandEncoder.setBuffer(scalarBuffer, offset: 0, atIndex: 1)
  commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 2)
  commandEncoder.setComputePipelineState(computePipelineState!)
    
  let threadExecutionWidth = computePipelineState!.threadExecutionWidth
  let threadsPerGroup = MTLSize(width:threadExecutionWidth,height:1,depth:1)
  let numThreadgroups = MTLSize(width:(vector.count + threadExecutionWidth) /
    threadExecutionWidth, height: 1, depth: 1)
  
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
  let byteLength = vector.count * sizeof(Float)
  
  return device.newBufferWithBytes(&vector, length: byteLength,
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
}

func createScalarBuffer(var scalar: Float, device: MTLDevice) -> MTLBuffer {
  let byteLength = sizeof(Float)
  
  return device.newBufferWithBytes(&scalar, length: byteLength,
    options: MTLResourceOptions.CPUCacheModeDefaultCache)
}