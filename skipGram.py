from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


__authors__ = ['author1','author2','author3']
__emails__  = ['fatherchristmas@northpole.dk','toothfairy@blackforest.no','easterbunny@greenfield.de']

def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( l.lower().split() )
    return sentences

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


class SkipGram:
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5, alpha=3/4):
        self.sentences = sentences
        self.nEmbed= nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount
        self.word2vec_init()
        self.alpha = alpha
        
    def train(self,stepsize, epochs):
        raise NotImplementedError('implement it!')

    def save(self,path):
        raise NotImplementedError('implement it!')

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        raise NotImplementedError('implement it!')

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')
        
    def word2vec_init(self) :
        """     Creates 4 dictionnaries for sentences:
            word2vec : for each word, creates a random uniform array of size nEmbed
            word_count : counts the number of occurrences of each word
            context2vec : for each word, creates a random uniform array of size nEmbed
            id2context : assign an id as key for each for each word
                It also creates a list of frequencies, where the frequencies are the nb of occurrence 
                of a word raised to the power of alpha, divided by the sum of all those weights"""
        self.word2vec = {}
        self.word_count = {}
        self.context2vec = {}
        for sent in self.sentences :
            for word in sent :
                if word in self.word2vec.keys() :
                    self.word_count[word] +=1
                else :
                    self.word2vec[word] = np.random.rand(self.nEmbed)
                    self.context2vec[word] = np.random.rand(self.nEmbed)
                    self.word_count[word] = 1
                    
        self.id2context = { i : w for i,w in enumerate(self.word2vec.keys())}
        
        self.freq = np.array(list(self.word_count.values()))
        self.freq = np.power(self.freq, self.alpha) 
        self.freq /= self.freq.sum()
        
        self.voc_size = len(self.freq)     
        return 
    
    def negative_sampling(self):
        """ This function returns indexes of words picked following the distribution below :
            P (word[i]) = frequency(word[i])^alpha / sum of all frequencies raised to the power alpha """
        sample = np.random.choice(a=np.arange(self.voc_size),size=self.negativeRate, p= self.freq)
        return sample

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train(...)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = mSkipGram.load(opts.model)
        for a,b,_ in pairs:
            print(sg.similarity(a,b))


