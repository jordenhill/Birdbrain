//
//  BirdbrainTests.swift
//  BirdbrainTests
//
//  Created by Jorden Hill on 9/29/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

import XCTest
@testable import Birdbrain

class BirdbrainTests: XCTestCase {
    let netSize: [Int] = [2,3,1]
    let input: [Float] = [1.0, 0.0]
    
    func testCreations() {
        XCTAssertNotNil(testSigmoidCreation(), "Sigmoid pass")
        XCTAssertNotNil(testMetalSigmoidCreation(), "Metal sigmoid pass")
        XCTAssertNotNil(testTanCreation(), "Tan pass")
        //XCTAssertNotNil(testMetalSigmoidCreation(), "Metal tan pass")
        //XCTAssertNotNil(testReluCreation(), "ReLu pass")
        //XCTAssertNotNil(testMetalReluCreation(), "Metal ReLu pass")
    }
    
    func testFeedForward() {
        
    }
    
    func testSigmoidCreation() -> FeedfowardNeuralNetwork {
        let net = FeedfowardNeuralNetwork(sizes: netSize, useMetal: 0, activationFunction: 1)
        return net;
    }
    
    func testMetalSigmoidCreation() -> FeedfowardNeuralNetwork {
        let net = FeedfowardNeuralNetwork(sizes: netSize, useMetal: 1, activationFunction: 1)
        return net;
    }
    
    func testTanCreation() -> FeedfowardNeuralNetwork {
        let net = FeedfowardNeuralNetwork(sizes: netSize, useMetal: 0, activationFunction: 2)
        return net;
    }
    
    func testMetalTanCreation() -> FeedfowardNeuralNetwork {
        let net = FeedfowardNeuralNetwork(sizes: netSize, useMetal: 1, activationFunction: 2)
        return net;
    }
    
    
    func testReluCreation() -> FeedfowardNeuralNetwork {
        let net = FeedfowardNeuralNetwork(sizes: netSize, useMetal: 0, activationFunction: 3)
        return net;
    }
    
    func testMetalReluCreation() -> FeedfowardNeuralNetwork {
        let net = FeedfowardNeuralNetwork(sizes: netSize, useMetal: 1, activationFunction: 3)
        return net;
    }
}
