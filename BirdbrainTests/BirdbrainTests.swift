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
    
    override func setUp() {
        super.setUp()
        let netSize = [2,3,2]
        let net = FeedfowardNeuralNetwork(sizes: netSize, useMetal: 0, activationFunction: 1)
        // Put setup code here. This method is called before the invocation of each test method in the class.
        
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        
    }
    
    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measureBlock {
            
        }
    }
    
}
