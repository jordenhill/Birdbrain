//
//  Word2Vec.swift
//  Birdbrain
//
//  Created by Jorden Hill on 2/23/16.
//  Copyright © 2016 Jorden Hill. All rights reserved.
//

import Foundation

struct vocabWord {
  var cn: Int //The count of occurences of the word
  var point: [Int]
  var word: String //The word
  var code: [Int8]
  var codeLength: Int8
  
  init() {
    cn = 0
    point = [0]
    word = ""
    code = [0]
    codeLength = 0
  }
}

let maxString = 100
let expTableSize = 1000
let maxExp: Float = 6
let maxSentenceLength = 1000
let maxCodeLength = 40
let vocabHashSize = 30000000;

var vocab = [vocabWord]()
var binary = 0, cbow = 1, debugMode = 2, window = 5, minCount = 5, numThreads = 12, minReduce = 1
var vocabHash = [Int](count: vocabHashSize, repeatedValue: 0)

var vocabMaxSize = 1000
var vocabSize = 0
var layer1Size = 100
var trainWords = 0, wordCountActual = 0, iter = 5, fileSize = 0, classes = 0
var alpha: Float = 0.025, startingAlpha = Float(), sample: Float = 0.001
var syn0 = Float(), syn1 = Float(), syn1Neg = Float()
var expTable = [Float](count: expTableSize, repeatedValue: 0.0)

var hs = 0, negative = 5
var tableSize = 100000000
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

/** The main method of Word2Vec, runs the system and trains the model on the words.
 */
func run() {
  vocab = [vocabWord](count: vocabSize, repeatedValue: vocabWord())
  vocabHash = [Int](count: vocabHashSize, repeatedValue: 0)
  expTable = [Float](count: expTableSize + 1, repeatedValue: 0.0)
  
  for a in 0 ..< expTableSize {
    expTable[a] = expf(Float((a / expTableSize * 2 - 1)) * maxExp)
  }
  
  //Used Accelerate here because it could be a bit faster
  expTable = div(expTable, y: add(expTable, c: 1.0))
  
  trainModel()
}

func trainModel() {
  
}

func initalizeUnigramTable() {
  var i = 0
  var trainWordsPow: Double = 0
  var d1 = 0.75, power = 0.75
  
  table = [Int](count: tableSize, repeatedValue: 0)
  
  for a in 0..<vocabSize {
    trainWordsPow += pow(Double(vocab[a].cn), power)
  }
 
  d1 = pow(Double(vocab[i].cn), power) / trainWordsPow
 
  for a in 0..<tableSize {
    table[a] = i
    if (Double(a / tableSize) > d1) {
      i += 1
      d1 += pow(Double(vocab[i].cn), power) / trainWordsPow
    }
    if (i >= vocabSize) {
      i = vocabSize
    }
  }
}

/** Compute the hash value of a word.
  - Parameter: The word whose hash is to be found.
  - Returns: An Int value that is the hash of the word.
*/
func getWordHash(word: String) -> Int {
  var hash = 0
  
  for character in 0 ..< word.characters.count {
    hash = hash * 257 + Int(String(word[word.startIndex.advancedBy(character)]).cStringUsingEncoding(NSUTF8StringEncoding)![0])
  }
  
  return hash
}

//** Adds a word to the vocabulary
   
func addWordToVocab(word: String) -> Int {
  var hash: Int = word.characters.count + 1, length: Int = word.characters.count + 1;
  if (length > maxString) {
    length = maxString
  }
  
  vocab[vocabSize].word = word
  vocab[vocabSize].cn = 0;
  vocabSize += 1;
  
  //TODO: Determine if necessary
  
  /*// Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }*/
  
  hash = getWordHash(word);
  
  while (vocabHash[hash] != -1) {
    hash = (hash + 1) % vocabHashSize;
  }
  
  vocabHash[hash] = vocabSize - 1;
  
  return vocabSize - 1;
}

/** Find the position of a word in the vocabulary.
  - Parameter word: the word to be found.
  - Returns: -1 if the word was not found, otherwise returns the position of the word.
*/
func searchVocab(word: String) -> Int {
  var hash = getWordHash(word)
  
  while (true) {
    if (vocabHash[hash] == -1) {
      return -1
    }
    if (word == vocab[vocabHash[hash]].word) {
      return vocabHash[hash]
    }
    hash = (hash + 1) % vocabHashSize
  }
}

/** Sort the vocuabulary by frequency using their word count values.
*/
func sortVocab() {
  var size: Int
  
  //Sort array, keeping </s> as first element
  let firstVocabElement = vocab.removeAtIndex(0)
  vocab.sortInPlace() {word1, word2 in word1.cn < word2.cn}
  vocab.insert(firstVocabElement, atIndex: 0)
  
  for a in 0..<vocabHashSize {
    vocabHash[a] = -1
  }
  
  size = vocabSize
  trainWords = 0
  
  for b in 0..<size {
    if ((vocab[b].cn < minCount) && (b != 0)) {
      vocabSize -= 1
    } else {
      // recompute hash
      var hash = getWordHash(vocab[b].word);
      while (vocabHash[hash] != -1) {
        hash = (hash + 1) % vocabHashSize
      }
      vocabHash[hash] = b;
      trainWords += vocab[b].cn;
    }
  }
  
  shrinkVocab(&vocab)
  
  for c in 0..<vocabSize {
    vocab[c].code = [Int8](count: maxCodeLength, repeatedValue: 0)
    vocab[c].point = [Int](count: maxCodeLength, repeatedValue: 0)
  }
}

/** Drop the early words until the current vocabulary size matches the new vocabulary size.
  - Parameter vocab: The array containing the vocab words
 */
func shrinkVocab(inout vocab: [vocabWord]) {
  var i = 1
  
  while (vocab.count > vocabSize) {
    vocab.removeAtIndex(i)
    i += 1
  }
}

/** Create a binary huffman tree
*/
func createBinaryTree() {
  var code = [Int8](count: maxCodeLength, repeatedValue: 0)
  var point = [Int](count: maxCodeLength, repeatedValue: 0)
  var count = [Int](count: vocabSize * 2 + 1, repeatedValue: 0)
  var binary = [Int](count: vocabSize * 2 + 1, repeatedValue: 0)
  var parentNode = [Int](count: vocabSize * 2 + 1, repeatedValue: 0)
  var min1i: Int
  var min2i: Int
  
  for a in 0 ..< vocabSize {
    count[a] = vocab[a].cn
  }
  
  for b in 0 ..< vocabSize * 2 {
    count[b] = Int(1e15)
  }
  
  var pos1 = vocabSize - 1
  var pos2 = vocabSize
  
  for c in 0 ..< vocabSize {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1 -= 1
      } else {
        min1i = pos2;
        pos2 += 1
      }
    } else {
      min1i = pos2
      pos2 += 1
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1
        pos1 -= 1
      } else {
        min2i = pos2
        pos2 += 1
      }
    } else {
      min2i = pos2
      pos2 += 1
    }
    count[vocabSize + c] = count[min1i] + count[min2i]
    parentNode[min1i] = vocabSize + c
    parentNode[min2i] = vocabSize + c
    binary[min2i] = 1
  }
  
  // Now assign binary code to each vocabulary word
  for d in 0 ..< vocabSize {
    var b = d
    var i = 0
    
    while (true) {
      code[i] = Int8(binary[b])
      point[i] = b
      i += 1
      b = parentNode[b];
      if (b == vocabSize * 2 - 2) {
        break
      }
    }
    vocab[d].codeLength = Int8(i);
    vocab[d].point[0] = vocabSize - 2;
    
    for e in 0 ..< i {
      vocab[d].code[i - e - 1] = code[e];
      vocab[d].point[i - e] = point[e] - vocabSize;
    }
  }
}