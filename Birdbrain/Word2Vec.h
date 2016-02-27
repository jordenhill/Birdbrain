//
//  Word2Vec.h
//  Birdbrain
//
//  Created by Jorden Hill on 2/25/16.
//  Copyright © 2016 Jorden Hill. All rights reserved.
//

#ifndef Word2Vec_h
#define Word2Vec_h

#include "stdio.h"

extern int min_count;
extern long long train_words;
extern int *vocab_hash;
extern long long vocab_max_size;
extern 

void InitUnigramTable();
void ReadWord(char *word, FILE *fin);
int GetWordHash(char *word);
int SearchVocab(char *word);
int ReadWordIndex(FILE *fin);
int AddWordToVocab(char *word);
int VocabCompare(const void *a, const void *b);
void SortVocab();
void ReduceVocab();
void CreateBinaryTree();
void LearnVocabFromTrainFile();
void SaveVocab();
void ReadVocab();
void InitNet();
void *TrainModelThread(void *id);
void TrainModel();
int ArgPos(char *str, int argx, char **argv);
int main(int argc, char **argv);
#endif
