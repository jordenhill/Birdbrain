//
//  Word2Vec.swift
//  Birdbrain
//
//  Created by Jorden Hill on 2/23/16.
//  Copyright © 2016 Jorden Hill. All rights reserved.
//

/*import Foundation

let maxString = 100
let expTableSize = 1000
let maxExp = 6
let maxSentenceLength = 1000
let maxCodeLength = 40
let vocabHashSize = 30000000;

var vocab = [vocabWord]()
var binary = 0, cbow = 1, debugMode = 2, window = 5, minCount = 5, numThreads = 12, minReduce = 1
var vocabHash = Int()
var vocabMaxSize = 1000
var vocabSize = 0
var layer1Size = 100
var trainWords = 0, wordCountActual = 0, iter = 5, fileSize = 0, classes = 0
var alpha: Float = 0.025, startingAlpha = Float(), sample: Float = 0.001
var syn0 = Float(), syn1 = Float(), syn1Neg = Float(), expTable = Float()

var hs = 0, negative = 5;
var tableSize = Int(1e8);
var table = [Int]();

/*
 char train_file[MAX_STRING], output_file[MAX_STRING];
 char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
 struct vocab_word *vocab;
 int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
 int *vocab_hash;
 long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
 long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
 real alpha = 0.025, starting_alpha, sample = 1e-3;
 real *syn0, *syn1, *syn1neg, *expTable;
 clock_t start;
*/

struct vocabWord {
  var cn: Int
  var point: Int16
  var word: String
  var code: String
  var codeLength: String
}

func initalizeUnigramTable() {
  var a: Int, i: Int
  var trainWordsPow: Double = 0
  let d1: Double, power = 0.75
  
  table = [Int](count: tableSize, repeatedValue: 0)
  
  for a in 0..<vocabSize {
    trainWordsPow += pow(vocab[a], power)
  }
}
  
func readWord(word: String, file: NSData) {
  
}

// Adds a word to the vocabulary
func addWordToVocab(word: String) -> Int {
  var hash: Int = word.characters.count + 1, length: Int = word.characters.count + 1;
  if (length > maxString) {
    length = maxString
  }
  
  vocab[vocabSize].word = word
  vocab[vocabSize].cn = 0;
  vocabSize++;
  
  //TODO: Determine if necessary
  
  /*// Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }*/
  
  hash = getWordHash(word);
  
  while (vocab_hash[hash] != -1) {
    hash = (hash + 1) % vocabHashSize;
  }
  
  vocabHash[hash] = vocabSize - 1;
  
  return vocabSize - 1;
}*/