//
//  BirdbrainTests.swift
//  BirdbrainTests
//
//  Created by Jorden Hill on 9/29/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

import XCTest
import Foundation
import Birdbrain

class FeedForwardNetworkTest: XCTestCase {
  let size1 = [10, 100, 10]
  let size2 = [10]

  func testCPUSigmoidConstruction() {
    let testNet = FeedfowardNeuralNetwork(sizes: size1, useMetal: false, activateFunction: "sigmoid")
    XCTAssertNotNil(testNet, "Sigmoid not properly initalized, is nil")
  }
  
  func testCPUHyperbolicTangerntConstruction() {
    let testNet = FeedfowardNeuralNetwork(sizes: size1, useMetal: false, activateFunction: "tangent")
    XCTAssertNotNil(testNet, "Tangent not properly initalized, is nil")
  }
  
  func testCPURectifiedLinearConstruction() {
    let testNet = FeedfowardNeuralNetwork(sizes: size1, useMetal: false, activateFunction: "relu")
    XCTAssertNotNil(testNet, "Rectified linear not properly initalized, is nil")
  }
  
  func testFailCreateNetworkWrongsize() {
    let testNet = FeedfowardNeuralNetwork(sizes: size2, useMetal: false, activateFunction: "sigmoid")
    XCTAssertNil(testNet, "Network initialized but should be nil with net size less than 2")
  }
  
  func testFailCreateNetworkWrongFunction() {
    let testNet = FeedfowardNeuralNetwork(sizes: size1, useMetal: false, activateFunction: "Fail")
    XCTAssertNil(testNet, "Network initialized but should be nil with incorrect activateFunction")
  }
}
