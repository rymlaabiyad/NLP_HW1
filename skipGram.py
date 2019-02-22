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

__authors__ = ['author1','author2','author3']
__emails__  = ['fatherchristmas@northpole.dk','toothfairy@blackforest.no','easterbunny@greenfield.de']

### Tokenization and pre-processing ###

def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( l.lower().split() )
    return sentences

def text2sentences_lemma(path):
    sentences = []
    lem = spacy.load('en')
    with open(path) as f:
        for l in f:
            l = lem(l)
            sentences.append( [word.lemma_ for word in l] )
    return sentences

def text2sentences_without_punctuation(path):
    '''This function retrieves the entire training text, 
        removes punctuation and line breaks,
        and returns a list of lists. 
        Each element of the main list is a sentence, made of a list of words.
    '''
    
    sentences = []
    nlp = spacy.load('en')
    tokenizer = Tokenizer(nlp.vocab)
    with open(path) as f:
        for l in f:
            sentences.append([str(t) for t in tokenizer(l) if (t.is_punct == False and (str(t) == '\n') == False)])
    return sentences

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, sentences = '', nEmbed=300, negativeRate=15, winSize = 2, minCount = 5,
                 alpha=3/4):
        self.sentences = sentences
        self.nEmbed= nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount
        self.alpha = alpha
        self.randomvector = np.random.rand(self.nEmbed)
        
    def word2vec_init(self) :
        """ This function creates the following variables :
            
        - word2id (dictionnary): for each word, we assign and id. This id will also be the index of the word embedding vector in the id2center_vec N-D array
        - id2center_vec : this N-D array contains the word embeddings. The first dimension is the word id, and the second is the embedding
        
        - context2id (dictionnary): for each word, we assign and id. This id will also be the index of the word embedding (when its a context word) vector in the id2context_vec N-D array
        - id2context_vec : this N-D array contains the word embeddings. The first dimension is the word id, and the second is the word embedding when the word is a context
        
        - voc_size : the number of different words in the  whole corpus
        
        - word_count (1-D array): this array contains the number of occurences of each word. The indexes match wit the ids in word2id and context2id
        - freq (1-D array): the frequencies are the nb of occurrences of a word raised to the power of alpha, divided by the sum of all those weights
        
        It also modifies sentences so instead of containing words, it contains their ids """
        
        self.word2id = {}
        
        self.context2id = {}
        
        self.voc_size=0 
        
        self.word_count = np.array([])
        
        for n_sent,sent in enumerate(self.sentences) :
            
            for n_word, word in enumerate(sent) :
                
                if word not in self.word2id.keys() :
                    
                    self.word2id[word] = self.voc_size
                    
                    self.context2id[word] = self.voc_size
                    
                    self.word_count = np.append(self.word_count, 1)
                    
                    sentences[n_sent][n_word] = self.voc_size
                    
                    self.voc_size +=1
                    
                else :
                    word_id = self.word2id[word]
                    
                    self.word_count[ word_id ] +=1
                    sentences[n_sent][n_word] = word_id
                    
                
                
                
        self.id2center_vec = np.random.rand(self.voc_size, self.nEmbed)
        self.id2context_vec= np.random.rand(self.voc_size, self.nEmbed)
        
        temp = np.power(self.word_count, self.alpha)
        self.freq = temp / temp.sum()
        del temp
        
        pass
    
    def train(self,stepsize = 0.01, epochs = 10):
        self.word2vec_init()
        print("The corpus has " + str(self.voc_size) + " different words")
        print("The corpus has " + str(len(self.sentences)) + " sentences")
        context_dic = self.make_context_dict()
        
        
        for i in range(epochs):
            t1 = datetime.datetime.now()
            t2 = datetime.datetime.now()
            print('Epoch', str(i+1), '/', str(epochs))
            loss = 0
            words_proceeded = 0
            
            for center_word, context in context_dic.items():
                
                for context_word in context : 
                    
                    v_center_word = self.id2center_vec[center_word]
                    v_context_word = self.id2context_vec[context_word]
                        
                    negative_sample = self.negative_sampling() 
                    v_neg = np.array([self.id2context_vec[n] for n in negative_sample])
                    
                    dot_vcenter_vcontext = np.vdot(v_center_word,v_context_word)
                    dot_vcenter_Vneg = np.dot(v_neg, v_center_word)
                    
                    loss += self.loss_function( dot_vcenter_vcontext, dot_vcenter_Vneg)
                    
                    self.id2center_vec[center_word] = v_center_word - stepsize * self.gradient_center_word ( v_center_word, v_context_word, v_neg, dot_vcenter_vcontext, dot_vcenter_Vneg )
                    
                    self.id2context_vec[context_word]  = v_context_word - stepsize * self.gradient_context_word (v_center_word, dot_vcenter_vcontext)
                    
                    for j, dot in enumerate(dot_vcenter_Vneg) :
                        self.id2context_vec[negative_sample[j]]= v_neg[j] - stepsize * self.gradient_neg_word ( v_center_word, dot )
                words_proceeded +=1
                if (words_proceeded % 1000 ==0 ): 
                    print(str(words_proceeded) + " words proceeded")
                    print(words_proceeded, 'center words were processed in', datetime.datetime.now() - t2)
                    t2 = datetime.datetime.now()
                        
            print("Epoch number " + str(i+1) + ", the loss is : " + str(loss))
            print('Time spent :', datetime.datetime.now() - t1)
    
    pass

    def save(self,path):
        """ This function save the model, i.e : 
            - word2id dictionnary
            - id2center_vec N-D Array
        """
        #model = pd.DataFrame.from_dict(data = self.word2vec)
        #model.to_csv(path, index = False)

    @staticmethod
    def load(self, path):
        """ This function loads the model, i.e :
            - word2id dictionnary
            - id2center_vec N-D Array
            """
        #model = pd.read_csv(path)
        #self.word2vec = model.to_dict(orient='list')
        pass
        
    def similarity(self,word1, word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        
        if word1 in self.word2id.keys() :
            v_word1 = self.id2center_vec[ self.word2id[word1] ]
        else :
            v_word1 = self.randomvector
            
        if word2 in self.word2id.keys() :
            v_word2 = self.id2center_vec[ self.word2id[word2] ]
        else : 
            v_word2 = self.randomvector 
            
        similarity =np.dot(v_word1, v_word2) / ( np.linalg.norm(v_word1) * np.linalg.norm(v_word2) )
        similarity = (similarity +1)/2
        return similarity
        
    def negative_sampling(self):
        """ This function returns words picked following the distribution below :
            P (word[i]) = frequency(word[i])^alpha / sum of all frequencies raised to the power alpha """
        #sample = np.random.choice(a=np.arange(self.voc_size),size=self.negativeRate, p= self.freq)
        sample = np.random.randint(low=0, high= self.voc_size, size = self.negativeRate)
        return sample
    
    ### Loss function to minimize ###
    
    def sigmoid (self, x) :
        return 1/ (1+ np.exp(-x))
    
    def loss_function(self, dot_vcenter_vcontext, dot_vcenter_Vneg):
        """This function is the loss function. 
        The arguments are either word or vectors"""
        neg =  sum([np.log(self.sigmoid(-dot_vcenter_vneg)) for dot_vcenter_vneg in dot_vcenter_Vneg])       
        res = (- np.log(self.sigmoid(dot_vcenter_vcontext)) - neg) 
        return res
    
     ### Optimization functions ###
    
    def gradient_center_word (self, center_word, context_word, negative_sample, dot_Vcenter_Vcontext, dot_Vcenter_Vneg) :
        """ This function is the derived loss function by the vector of the central word 
            The arguments should be vectors of words embedding."""
        res = (self.sigmoid( dot_Vcenter_Vcontext )-1) * context_word
        for i,neg in enumerate(negative_sample) :
            res += (self.sigmoid(dot_Vcenter_Vneg[i])) * neg
        return res 
    
    def gradient_context_word (self, center_word, dot_Vcenter_Vcontext): 
        """ This function is the derived loss function by the vector of the context word
        The arguments should be vectors of words embedding."""
        res = (self.sigmoid( dot_Vcenter_Vcontext )-1) * center_word
        return res
    
    def gradient_neg_word (self, center_word, dot_Vcenter_Vneg ) :
        """ This function is the derived loss function by the vector of one negative sampled word
        The arguments should be vectors of words embedding."""
        res =(self.sigmoid(dot_Vcenter_Vneg)) * center_word
        return res 
    
    ### Context related functions ### 
        
    def sentence2io(self, sentence) :
        """ This function takes as input a sentence, and returns list of tuples :
            - the first element is the center word
            - the second is a list of context words
        """
        sentence = [ word_id for i,word_id in enumerate(sentence) if self.word_count[word_id]>self.minCount ]
        L = len(sentence)
        res = []
        for index,word in enumerate(sentence):
            if self.word_count[word] > self.minCount :
                inf = index - self.winSize
                sup = index + self.winSize + 1
                context = []
                    
                context = [sentence[i] for i in range(inf, sup) if 0 <= i < L and i != index]
            
                res.append( (word, context) )
        return res
    
    def make_context_dict(self):
        ''' This method creates a dictionary containing all the worlds of the (reduced) vocabulary
            as keys, and their context as an array, as value.
            If a word is found more than once in the context of a center word,
            it's only appended once.
        '''
        context_dict = {}
        for sent in self.sentences:
            for tup in self.sentence2io(sent):
                if tup[0] in context_dict.keys():
                    for context_word in tup[1]:
                        if context_word not in context_dict[tup[0]] :
                            context_dict[tup[0]].append(context_word)
                else:
                    context_dict[tup[0]] = tup[1]
        return context_dict 

"""if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences_without_punctuation(opts.text)
        sg = SkipGram(sentences)
        
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            print(sg.similarity(a,b))
"""

training_data_path = '/Users/alimrabeth/Desktop/Master Data Sciences & Business Analytics/Data Sciences Elective courses/NLP/Projet 1/sentences.txt'
sentences = text2sentences_without_punctuation(training_data_path)

sg = SkipGram(sentences, nEmbed=300, negativeRate=15, winSize = 2)
sg.train()
