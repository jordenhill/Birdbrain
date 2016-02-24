//
//  Word2VecTest.swift
//  Birdbrain
//
//  Created by Jorden Hill on 2/24/16.
//  Copyright © 2016 Jorden Hill. All rights reserved.
//

import XCTest
import Birdbrain

class Word2VecTest: XCTestCase {
    
  override func setUp() {
    super.setUp()
    var vcw = vocabWord()
    vcw.count = 5
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
            // Put the code you want to measure the time of here.
        }
    }
    
}
