from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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
        """ This function save the model, i.e : 
            - nEmbed
            - negativeRate,
            - winSize
            - minCount
            - alpha
            - word2vec
            - word_count
            - context2vec
            - id2context
            - freq
            - voc_size
        """
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
        """ This function loads the model, i.e :
            - nEmbed
            - negativeRate,
            - winSize
            - minCount
            - alpha
            - word2vec
            - word_count
            - context2vec
            - id2context
            - freq
            - voc_size
            """
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
        
        self.voc_size=0
        self.id2context={}
        self.freq =np.array([])
        
        for word,count in self.word_count.items() :
            self.id2context[self.voc_size] = word
            self.freq= np.append(self.freq, np.power(count, self.alpha))
            self.voc_size +=1
        
        self.freq /= self.freq.sum()
        
        return 
    
    def negative_sampling(self):
        """ This function returns indexes of words picked following the distribution below :
            P (word[i]) = frequency(word[i])^alpha / sum of all frequencies raised to the power alpha """
        sample = np.random.choice(a=np.arange(self.voc_size),size=self.negativeRate, p= self.freq)
        return sample
    
    def lossFunction(v_w, v_c, v_notc):                                       
          notc = sum([np.log(1/(1+np.exp(np.dot(v_w, v)))) for v in v_notc])    
          return np.log(1/(1+np.exp(-np.dot(v_w, v_c)))) + notc 
    
    def sentence2io(self, sentence) :
        """ This function takes as input a sentence, and returns zip of tuples of 2 lists :
            - the first list contains the center word
            - the second contains the context words of this center word
        """
        

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


