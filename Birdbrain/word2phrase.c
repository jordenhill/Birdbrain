//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <Birdbrain-Bridging-Header.h>
#include <Word2Vec.h>

#define MAX_STRING 60

const int vocabHashSize = 500000000; // Maximum 500M entries in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  char *word;
};

char trainFile[MAX_STRING], outputFile[MAX_STRING];
struct vocab_word *vocab_wrd;
int debugMode = 2, minCount = 5, *vocabHash, minReduce = 1;
long long vocabMaxSize = 10000, vocabSize = 0;
long long trainWords = 0;
real threshold = 100;

unsigned long long next_random = 1;

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void _ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int _GetWordHash(char *word) {
  unsigned long long a, hash = 1;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocabHashSize;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int _SearchVocab(char *word) {
  unsigned int hash = _GetWordHash(word);
  while (1) {
    if (vocabHash[hash] == -1) return -1;
    if (!strcmp(word, vocab_wrd[vocabHash[hash]].word)) return vocabHash[hash];
    hash = (hash + 1) % vocabHashSize;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int _ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  _ReadWord(word, fin);
  if (feof(fin)) return -1;
  return _SearchVocab(word);
}

// Adds a word to the vocabulary
int _AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab_wrd[vocabSize].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab_wrd[vocabSize].word, word);
  vocab_wrd[vocabSize].cn = 0;
  vocabSize++;
  // Reallocate memory if needed
  if (vocabSize + 2 >= vocabMaxSize) {
    vocabMaxSize += 10000;
    vocab_wrd=(struct vocab_word *)realloc(vocab_wrd, vocabMaxSize * sizeof(struct vocab_word));
  }
  hash = _GetWordHash(word);
  while (vocabHash[hash] != -1) hash = (hash + 1) % vocabHashSize;
  vocabHash[hash]=vocabSize - 1;
  return vocabSize - 1;
}

// Used later for sorting by word counts
int _VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void _SortVocab() {
  int a;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab_wrd[1], vocabSize - 1, sizeof(struct vocab_word), _VocabCompare);
  for (a = 0; a < vocabHashSize; a++) vocabHash[a] = -1;
  for (a = 0; a < vocabSize; a++) {
    // Words occuring less than minCount times will be discarded from the vocab_wrd
    if (vocab_wrd[a].cn < minCount) {
      vocabSize--;
      free(vocab_wrd[vocabSize].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = _GetWordHash(vocab_wrd[a].word);
      while (vocabHash[hash] != -1) hash = (hash + 1) % vocabHashSize;
      vocabHash[hash] = a;
    }
  }
  vocab_wrd = (struct vocab_word *)realloc(vocab_wrd, vocabSize * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void _ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocabSize; a++) if (vocab_wrd[a].cn > minReduce) {
    vocab_wrd[b].cn = vocab_wrd[a].cn;
    vocab_wrd[b].word = vocab_wrd[a].word;
    b++;
  } else free(vocab_wrd[a].word);
  vocabSize = b;
  for (a = 0; a < vocabHashSize; a++) vocabHash[a] = -1;
  for (a = 0; a < vocabSize; a++) {
    // Hash will be re-computed, as it is not actual
    hash = _GetWordHash(vocab_wrd[a].word);
    while (vocabHash[hash] != -1) hash = (hash + 1) % vocabHashSize;
    vocabHash[hash] = a;
  }
  fflush(stdout);
  minReduce++;
}

void _LearnVocabFromTrainFile() {
  char word[MAX_STRING], last_word[MAX_STRING], bigram_word[MAX_STRING * 2];
  FILE *fin;
  long long a, i, start = 1;
  for (a = 0; a < vocabHashSize; a++) vocabHash[a] = -1;
  fin = fopen(trainFile, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocabSize = 0;
  _AddWordToVocab((char *)"</s>");
  while (1) {
    _ReadWord(word, fin);
    if (feof(fin)) break;
    if (!strcmp(word, "</s>")) {
      start = 1;
      continue;
    } else start = 0;
    trainWords++;
    if ((debugMode > 1) && (trainWords % 100000 == 0)) {
      printf("Words processed: %lldK     vocab size: %lldK  %c", trainWords / 1000, vocabSize / 1000, 13);
      fflush(stdout);
    }
    i = _SearchVocab(word);
    if (i == -1) {
      a = _AddWordToVocab(word);
      vocab_wrd[a].cn = 1;
    } else vocab_wrd[i].cn++;
    if (start) continue;
    sprintf(bigram_word, "%s_%s", last_word, word);
    bigram_word[MAX_STRING - 1] = 0;
    strcpy(last_word, word);
    i = _SearchVocab(bigram_word);
    if (i == -1) {
      a = _AddWordToVocab(bigram_word);
      vocab_wrd[a].cn = 1;
    } else vocab_wrd[i].cn++;
    if (vocabSize > vocabHashSize * 0.7) _ReduceVocab();
  }
  _SortVocab();
  if (debugMode > 0) {
    printf("\nVocab size (unigrams + bigrams): %lld\n", vocabSize);
    printf("Words in train file: %lld\n", trainWords);
  }
  fclose(fin);
}

void _TrainModel() {
  long long pa = 0, pb = 0, pab = 0, oov, i, li = -1, cn = 0;
  char word[MAX_STRING], last_word[MAX_STRING], bigram_word[MAX_STRING * 2];
  real score;
  FILE *fo, *fin;
  printf("Starting training using file %s\n", trainFile);
  _LearnVocabFromTrainFile();
  fin = fopen(trainFile, "rb");
  fo = fopen(outputFile, "wb");
  word[0] = 0;
  while (1) {
    strcpy(last_word, word);
    _ReadWord(word, fin);
    if (feof(fin)) break;
    if (!strcmp(word, "</s>")) {
      fprintf(fo, "\n");
      continue;
    }
    cn++;
    if ((debugMode > 1) && (cn % 100000 == 0)) {
      printf("Words written: %lldK%c", cn / 1000, 13);
      fflush(stdout);
    }
    oov = 0;
    i = _SearchVocab(word);
    if (i == -1) oov = 1; else pb = vocab_wrd[i].cn;
    if (li == -1) oov = 1;
    li = i;
    sprintf(bigram_word, "%s_%s", last_word, word);
    bigram_word[MAX_STRING - 1] = 0;
    i = _SearchVocab(bigram_word);
    if (i == -1) oov = 1; else pab = vocab_wrd[i].cn;
    if (pa < minCount) oov = 1;
    if (pb < minCount) oov = 1;
    if (oov) score = 0; else score = (pab - minCount) / (real)pa / (real)pb * (real)trainWords;
    if (score > threshold) {
      fprintf(fo, "_%s", word);
      pb = 0;
    } else fprintf(fo, " %s", word);
    pa = pb;
  }
  fclose(fo);
  fclose(fin);
}

int _ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int _main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD2PHRASE tool v0.1a\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters / phrases\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-threshold <float>\n");
    printf("\t\t The <float> value represents threshold for forming the phrases (higher means less phrases); default 100\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\nExamples:\n");
    printf("./word2phrase -train text.txt -output phrases.txt -threshold 100 -debug 2\n\n");
    return 0;
  }
  if ((i = _ArgPos((char *)"-train", argc, argv)) > 0) strcpy(trainFile, argv[i + 1]);
  if ((i = _ArgPos((char *)"-debug", argc, argv)) > 0) debugMode = atoi(argv[i + 1]);
  if ((i = _ArgPos((char *)"-output", argc, argv)) > 0) strcpy(outputFile, argv[i + 1]);
  if ((i = _ArgPos((char *)"-min-count", argc, argv)) > 0) minCount = atoi(argv[i + 1]);
  if ((i = _ArgPos((char *)"-threshold", argc, argv)) > 0) threshold = atof(argv[i + 1]);
  vocab_wrd = (struct vocab_word *)calloc(vocabMaxSize, sizeof(struct vocab_word));
  vocabHash = (int *)calloc(vocabHashSize, sizeof(int));
  _TrainModel();
  return 0;
}
