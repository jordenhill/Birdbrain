//
//  BirdbrainMathTest.swift
//  Birdbrain
//
//  Created by Jorden Hill on 11/25/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

import XCTest
import Foundation
import Birdbrain

//Test primary functions in Math.swift

class BirdbrainMathTest: XCTestCase {
    let testVector1: [Float] = [1.0, 2.0]
    let testVector2: [Float] = [3.0, 4.0]
    let testMatrix1: [Float] = [1.0, 2.0,
                                3.0, 4.0]
    let testMatrix2: [Float] = [1.0,  2.0,
                                3.0,  4.0]
    
    func testVectorSum() {
        XCTAssertTrue(sum(testVector1) == 3.0, "Wrong vector sum")
    }
    
    func testVectorAddition() {
        XCTAssertTrue(add(testVector1, y: testVector2) == [4.0, 6.0], "Wrong vector + vector result")
    }
    
    func testVectorSubtraction() {
        XCTAssertTrue(sub(testVector1, y: testVector2) == [-2.0, -2.0], "Wrong vector - vector result")
    }
    
    func testScalarAddition() {
      XCTAssertTrue(add(testVector1, c: 1.0) == [2.0, 3.0], "Wrong scalar vector addition result")
    }
    
    func testScalarMultiplication() {
        XCTAssertTrue(mul(testVector1, c: 2.0) == [2.0, 4.0], "Wrong scalar vector multiplication result")
    }
    
    func testVectorVectorMultiplication() {
        XCTAssertTrue(mul(testVector1, y: testVector2) == [3.0, 8.0], "Wrong vector-vector multiplication result")
    }
    
    func testVectorMatricMultiplication() {
        XCTAssertTrue(mvMul(testMatrix1, m: 2, n: 2, x: testVector1) == [5, 11], "Wrong vector matrix multiplication result")
    }
    
    func testVectorVectorDivision() {
        XCTAssertTrue(div(testVector2, y: testVector1) == [3.0, 2.0], "Wrong vector vector division result")
    }
    
    func testExponentiation() {
        XCTAssertTrue(exp(testVector1) == [exp(1.0), exp(2.0)], "Wrong elementwise exponentiation result")
    }
    
    func testHyperbolicTangent() {
        XCTAssertTrue(tanh(testVector1) == [tanhf(1.0), tanhf(2.0)], "Wrong elementwise hyperbolic tangent result")
    }
    
    func testSquare() {
        XCTAssertTrue(square(testVector1) == [powf(1.0, 2.0), powf(2.0, 2.0)], "Wrong elementwise square result")
    }
    
    func testNegation() {
        XCTAssertTrue(neg(testVector1) == [-1.0, -2.0], "Wrong elementwise negation result")
    }
    
    func testMatrixFormation() {
        XCTAssertTrue(outer(testMatrix1, y: testMatrix2) == [1.0, 2.0,  3.0,  4.0,
                                                                  2.0, 4.0,  6.0,  8.0,
                                                                  3.0, 6.0,  9.0, 12.0,
                                                                  4.0, 8.0, 12.0, 16.0], "Wrong matrix formation result")
    }
}