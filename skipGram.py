from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from scipy.special import expit
from sklearn.preprocessing import normalize

import datetime

__authors__ = ['Ali M\'rabeth','Rym Laabiyad','Amine Sentissi','Arnaud Cluzel']
__emails__  = ['ali.mrabeth@essec.edu','rym.laabiyad@essec.edu','amine.sentissi@essec.edu', 'arnaud.cluzel@essec.edu']

### Tokenization and pre-processing ###

def text2sentences(path):
    '''This method retrieves the entire training text, 
        removes punctuation and line breaks,
        and returns a list of lists. 
        Each element of the main list is a sentence, made of a list of words.
    '''
    t1 = datetime.datetime.now()
    print('Reading sentences ...')
    
    sentences = []
    count = 0
    nlp = spacy.load('en')
    tokenizer = Tokenizer(nlp.vocab)
    with open(path) as f:
        for l in f:
            sentences.append([str(t) for t in tokenizer(l) if (t.is_punct == False and (str(t) == '\n') == False)])
            count += 1
            if count == 10000:
                break
    print(len(sentences), 'sentences read in', datetime.datetime.now() - t1)
    return sentences

def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

class SkipGram:
    def __init__(self, sentences = '', nEmbed = 100, negativeRate = 5, winSize = 2, minCount = 5,
                 alpha = 3/4, word2vec = {}):
        self.sentences = sentences
        self.nEmbed= nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount
        self.alpha = alpha
        
        self.randomvector = np.random.rand(self.nEmbed)
        self.word2vec = word2vec
        
    def word2vec_init(self) :     
        print('Initializing word embeddings ...')
        
        '''Count the words in the corpus, and only keep those
            that show up more than minCount times
        '''
        self.word_count = {}
        self.word2id = {}
        i = 0
        
        for sent in self.sentences :
            for word in sent :
                if word in self.word_count.keys():
                    if self.word_count[word] > self.minCount and word not in self.word2id.keys() :
                        self.word2id[word] = i
                        i += 1
                    else :
                        self.word_count[word] +=1     
                else :
                    self.word_count[word] = 1
        print('Initial vocabulary size :', len(self.word_count), 
              ', Reduced size :', len(self.word2id))
            
        '''Give the reduced vocabulary (words that show up more than minCount times)
            an embedding as both a center word and a context word
        '''
        self.id_sentences = []
        for sent in self.sentences :
            s = []
            for word in sent : 
                if word in self.word2id.keys() :
                    s.append(self.word2id[word])
            self.id_sentences.append(s)
            
        self.center_vec = np.random.rand(len(self.word2id), self.nEmbed)
        self.context_vec= np.random.rand(len(self.word2id), self.nEmbed)
        
        self.freq =np.array([])
        '''I need something about frequency here ... '''
          

    def train(self, stepsize = 0.02, epochs = 10):
        self.word2vec_init()
        context_dict = self.make_context_dict()
                    

        for i in range(epochs):
            t1 = datetime.datetime.now()
            t2 = datetime.datetime.now()
            print('Epoch', str(i+1), '/', str(epochs))
            count = 0
            
            loss = 0
            
            for center_word, context in context_dict.items():
                v_center_word = self.center_vec[center_word]
                for context_id in context : 
                    v_context_word = self.context_vec[context_id]
                    
                    negative_sample = self.negative_sampling_rand()
                    v_neg = np.array([self.context_vec[n] for n in negative_sample])
                    
                    dot_vcenter_vcontext = np.vdot(v_center_word,v_context_word)
                    dot_vcenter_Vneg = np.dot(v_center_word, np.transpose(v_neg))
                        
                    loss += self.loss_function( dot_vcenter_vcontext, dot_vcenter_Vneg)
                    if loss == np.inf :
                        print('Early stopping due to skipping a local minimum (maybe a change of stepsize can help).')
                        print('The results from the previous epoch (', str(i), ') will be kept.')
                        break
                        
                    for j, dot in enumerate(dot_vcenter_Vneg) :
                        self.context_vec[negative_sample[j]]= v_neg[j] - stepsize * self.gradient_neg_word (v_center_word, dot)
                        
                    self.context_vec[context_id]  = v_context_word - stepsize * self.gradient_context_word(v_center_word, dot_vcenter_vcontext)
                    tmp = self.gradient_center_word(v_center_word, v_context_word, v_neg, dot_vcenter_vcontext, dot_vcenter_Vneg)
                    v_center_word -= stepsize * tmp
                
                if loss == np.inf :
                    break
                self.center_vec[center_word] = v_center_word
                
                count += 1
                if(count % 1000 == 0) : 
                    print(count, 'center words were processed in', datetime.datetime.now() - t2)
                    t2 = datetime.datetime.now()
                
            if loss == np.inf :
                break
            else :
                print("For epoch number", str(i+1), ", the loss is : ", str(loss))
                print('Time spent :', datetime.datetime.now() - t1)


    def save(self,path):
        """ This method saves the model, i.e : 
            - word2id dictionnary
            - id2center_vec N-D Array
            as one dataframe with columns : word, id, vec
        """
        df1 = pd.DataFrame(data = self.word2id.items(), columns = ['word', 'id'])
        df2 = pd.DataFrame(self.center_vec)
        model = pd.concat([df1.drop(['id'], axis = 1), df2], axis = 1) 
        model.to_csv(path, index = False)
        
    @staticmethod
    def load(path):
        """ This method loads the model, i.e :
            - word2vec dictionnary
            """
        model = pd.read_csv(path)
        word2vec = {}
        for index, row in model.iterrows():
            word2vec[row['word']] = row[1:]
        return SkipGram(word2vec = word2vec)
    
    def similarity(self, word1, word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        
        if word1 in self.word2vec.keys() :
            v_word1 = self.word2vec[word1]
        else :
            v_word1 = self.randomvector
            
        if word2 in self.word2vec.keys() :
            v_word2 = self.word2vec[word2]
        else : 
            v_word2 = self.randomvector 
            
        similarity = np.dot(v_word1, v_word2) / (np.linalg.norm(v_word1) * np.linalg.norm(v_word2))
        return (similarity + 1)/2
    
    ### Context related methods ### 
        
    def sentence2io(self, sentence) :
        """ This method takes as input a sentence, and returns list of tuples :
            - the first element is the center word
            - the second is a list of context words
        """
        L = len(sentence)
        res = []
        for index, word in enumerate(sentence):
            inf = index - self.winSize
            sup = index + self.winSize + 1
            
            context = [sentence[i] for i in range(inf, 
                           sup) if 0 <= i < L and i != index]
            res.append((word, context))
        return res
        
    def make_context_dict(self):
        ''' This method creates a dictionary containing all the worlds of the (reduced) vocabulary
            as keys, and their context as an array, as value.
            If a word is found more than once in the context of a center word,
            it's only appended once.
        '''
        t1 = datetime.datetime.now()
        print('Making a dictionary {word_id : [context_id]} ...')
        context_dict = {}
        for sent in self.id_sentences:
            for s in self.sentence2io(sent):
                if s[0] in context_dict.keys():
                    for e in s[1]:
                        if e not in context_dict[s[0]] :
                            context_dict[s[0]].append(e)
                else:
                    context_dict[s[0]] = s[1]
        
        print('Context_dict done in', datetime.datetime.now() - t1)
        return context_dict
        
    def negative_sampling_rand(self):
        return np.random.randint(low = 0, high = len(self.word2id), size = self.negativeRate)
    
    def negative_sampling(self):
        """ This method returns words picked following the distribution below :
            P (word[i]) = frequency(word[i])^alpha / sum of all frequencies raised to the power alpha """
        return np.random.choice(a = np.arange(len(self.word2id)), size = self.negativeRate, p = self.freq)
    
    ### Loss function to minimize ###
    
    def sigmoid (self, x) :
        return 1/ (1+ np.exp(-x))
    
    def loss_function(self, dot_vcenter_vcontext, dot_vcenter_Vneg):
        """This method is the loss function. 
        The arguments are either word or vectors"""
        neg =  sum([np.log(self.sigmoid(-dot_vcenter_vneg)) for dot_vcenter_vneg in dot_vcenter_Vneg])       
        res = (- np.log(self.sigmoid(dot_vcenter_vcontext)) - neg) 
        return res
    
     ### Optimization methods ###
    
    def gradient_center_word (self, center_word, context_word, negative_sample, dot_Vcenter_Vcontext, dot_Vcenter_Vneg) :
        """ This method is the derived loss function by the vector of the central word 
            The arguments should be vectors of words embedding."""
        res = (self.sigmoid( dot_Vcenter_Vcontext )-1) * context_word
        for i,neg in enumerate(negative_sample) :
            res += (self.sigmoid(dot_Vcenter_Vneg[i])) * neg
        return res 
    
    def gradient_context_word (self, center_word, dot_Vcenter_Vcontext): 
        """ This method is the derived loss function by the vector of the context word
        The arguments should be vectors of words embedding."""
        res = (self.sigmoid( dot_Vcenter_Vcontext )-1) * center_word
        return res
    
    def gradient_neg_word (self, center_word, dot_Vcenter_Vneg ) :
        """ This method is the derived loss function by the vector of one negative sampled word
        The arguments should be vectors of words embedding."""
        res =(self.sigmoid(dot_Vcenter_Vneg)) * center_word
        return res 
    
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:                
        sentences = text2sentences(opts.text)
        
        sg = SkipGram(sentences)
        sg.train(stepsize = 0.05, epochs = 5)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a,b,sim in pairs:
            print(a, b, sg.similarity(a,b))

