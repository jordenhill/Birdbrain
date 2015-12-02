//
//  RecurrentNeuralNetwork.swift
//  Birdbrain
//
//  Created by Jorden Hill on 12/1/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

import Foundation

public class RecurrentNeuralNetwork {
    var inputDim: Int
    var hiddenDim: Int
    var bpttTruncate: Int
    var U: [Float]
    var V: [Float]
    var W: [Float]
    
    public init(inputDim: Int, hiddenDim: Int, bpttTruncate: Int) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.bpttTruncate = bpttTruncate
        U = [Float](count: hiddenDim * inputDim, repeatedValue: initRand(inputDim))
        V = [Float](count: inputDim * hiddenDim, repeatedValue: initRand(hiddenDim))
        W = [Float](count: hiddenDim * hiddenDim, repeatedValue: initRand(hiddenDim))
    }
    
    public func feedforward(x: [Float]) -> ([[Float]], [[Float]]){
        let T = x.count;
        var s = [[Float]](count: T + 1, repeatedValue: [Float](count: hiddenDim, repeatedValue: 0.0))
        let start = [Float](count: hiddenDim, repeatedValue: 0.0)
        var o = [[Float]](count: T, repeatedValue: [Float](count: inputDim, repeatedValue: 0.0))
        
        s[0] = add(mvMul(U, m: Int32(inputDim), n:Int32(hiddenDim), x: s[0]), y: mvMul(W, m: Int32(hiddenDim), n: Int32(hiddenDim), x: start))
        o[0] = softmax(mvMul(V, m: Int32(inputDim), n: Int32(hiddenDim), x: s[0]))
        
        for t in Range(start: 1, end: T) {
            s[t] = add(mvMul(U, m: Int32(inputDim), n:Int32(hiddenDim), x: s[t]), y: mvMul(W, m: Int32(hiddenDim), n: Int32(hiddenDim), x: s[t - 1]))
            o[t] = softmax(mvMul(V, m: Int32(inputDim), n: Int32(hiddenDim), x: s[t]))
        }
        
        return (s, o)
    }
}