from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from scipy.special import expit
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

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
    def __init__(self, sentences = '', nEmbed=100, negativeRate=5, winSize = 5, minCount = 5,
                 alpha=3/4, word2vec = {}):
        self.sentences = sentences
        self.nEmbed= nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount
        self.alpha = alpha
        self.randomvector = np.random.rand(self.nEmbed)
        self.word2vec = word2vec
        
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
        
    def word2vec_init2(self) :     
        print('Initializing word embeddings ...')
        
        '''Count the words in the corpus, and only keep those
            that show up more than minCount times
        '''
        self.word_count = {}
        
        for sent in self.sentences :
            for word in sent :
                if word in self.word_count.keys() :
                    self.word_count[word] +=1      
                else :
                    self.word_count[word] = 1
        print('Initial vocabulary size :', len(self.word_count))
            
        '''Give the reduced vocabulary (words that show up more than minCount times)
            an embedding as both a center word and a context word
        '''
        for sent in self.sentences :
            for i, word in enumerate(sent) : 
                if word not in self.word2vec.keys() and self.word_count[word] > self.minCount :
                    self.word2vec[word] = np.random.rand(self.nEmbed)
        print('Reduced vocabulary size :', len(self.word2vec))
            
        self.context2id={}
        self.contextID2vec = {}
        self.freq =np.array([])
        count = 0
        
        for word in self.word2vec.keys() :
            self.context2id[word] = count
            self.contextID2vec[count] = np.random.rand(self.nEmbed)
            self.freq= np.append(self.freq, np.power(count, self.alpha))
            count +=1
        
        self.freq /= self.freq.sum()
    
    def train(self,stepsize = 0.01, epochs = 10):
        self.word2vec_init()
        print("The corpus has " + str(self.voc_size) + " different words")
        print("The corpus has " + str(len(self.sentences)) + " sentences")
        
        for i in range(epochs):
            t1 = datetime.datetime.now()
            print('Epoch', str(i+1), '/', str(epochs))
            
            count_sentences = 0 
            loss = 0
            for sent in self.sentences :
                count_sentences +=1
                for tup in self.sentence2io(sent) :
                    center_word = tup[0]
                    for context_word in tup[1] :
                        
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
                        
                        
                        
                if (count_sentences % 1000 ==0 ): print(str(count_sentences) + " sentences proceeded")
            print("For epoch number", str(i+1), ", the loss is : ", str(loss))
            print('Time spent :', datetime.datetime.now() - t1)
    
    pass

    def train2(self, stepsize = 0.05, epochs = 10):
        self.word2vec_init2()
        context_dict = self.transform_context(self.make_context_dict())
                    
        for i in range(epochs):
            t1 = datetime.datetime.now()
            t2 = datetime.datetime.now()
            print('Epoch', str(i+1), '/', str(epochs))
            loss = 0
            count = 0
            
            for center_word, context in context_dict.items():
                negative_sample = self.negative_sampling2()
                v_center_word = self.word2vec[center_word]
                for context_id in context : 
                    v_context_word = self.contextID2vec[context_id]
                    v_neg = np.array([self.contextID2vec[n] for n in negative_sample])
                    
                    dot_vcenter_vcontext = np.vdot(v_center_word,v_context_word)
                    dot_vcenter_Vneg = np.dot(v_center_word, np.transpose(v_neg))
                        
                    loss += self.loss_function( dot_vcenter_vcontext, dot_vcenter_Vneg)
                        
                    for j, dot in enumerate(dot_vcenter_Vneg) :
                        self.contextID2vec[negative_sample[j]]= v_neg[j] - stepsize * self.gradient_neg_word (v_center_word, dot)
                        
                    self.contextID2vec[context_id]  = v_context_word - stepsize * self.gradient_context_word(v_center_word, dot_vcenter_vcontext)
                    tmp = self.gradient_center_word(v_center_word, v_context_word, v_neg, dot_vcenter_vcontext, dot_vcenter_Vneg)
                    v_center_word -= stepsize * tmp
                
                self.word2vec[center_word] = v_center_word
                
                count += 1
                if(count % 1000 == 0) : 
                    print(count, 'center words were processed in', datetime.datetime.now() - t2)
                    t2 = datetime.datetime.now()
            print("For epoch number", str(i+1), ", the loss is : ", str(loss))
            print('Time spent :', datetime.datetime.now() - t1)

    def save(self,path):
        """ This method saves the model, i.e : 
            - word2id dictionnary
            - id2center_vec N-D Array
        """
        
        
    def save2(self,path):
        """ This method saves the model, i.e : 
            - word2vec dictionnary
        """
        model = pd.DataFrame.from_dict(data = self.word2vec)
        model.to_csv(path, index = False)

    @staticmethod
    def load(path):
        """ This method loads the model, i.e :
            - word2vec dictionnary
            """
        model = pd.read_csv(path)
        return SkipGram(word2vec = model.to_dict(orient='list'))
        
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
        #linear_kernel(features_TFIDF[index_source:index_source+1],features_TFIDF[index_target:index_target+1])[0][0]
        return similarity
    
    def similarity2(self,word1, word2):
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
        
    def negative_sampling(self):
        """ This function returns words picked following the distribution below :
            P (word[i]) = frequency(word[i])^alpha / sum of all frequencies raised to the power alpha """
        #sample = np.random.choice(a=np.arange(self.voc_size),size=self.negativeRate, p= self.freq)
        sample = np.random.randint(low=0, high= self.voc_size, size = self.negativeRate)
        return sample
    
    def negative_sampling2(self):
        """ This method returns words picked following the distribution below :
            P (word[i]) = frequency(word[i])^alpha / sum of all frequencies raised to the power alpha """
        return np.random.choice(a = np.arange(len(self.word2vec)), size = self.negativeRate, p = self.freq)
    
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
    
    ### Context related methods ### 
        
    def sentence2io(self, sentence) :
        """ This method takes as input a sentence, and returns list of tuples :
            - the first element is the center word
            - the second is a list of context words
        """
        L = len(sentence)
        res = []
        for index, word in enumerate(sentence):
            if self.word_count[word] > self.minCount :
                inf = index - self.winSize
                sup = index + self.winSize + 1
                    
                context = [sentence[i] for i in range(inf, 
                           sup) if 0 <= i < L and i != index and self.word_count[sentence[i]] > self.minCount]
            
                res.append((word, context))
        return res
    
    def make_context_dict(self):
        ''' This method creates a dictionary containing all the worlds of the (reduced) vocabulary
            as keys, and their context as an array, as value.
            If a word is found more than once in the context of a center word,
            it's only appended once.
        '''
        t1 = datetime.datetime.now()
        print('Making a dictionary {word : [context]} ...')
        context_dict = {}
        for sent in self.sentences:
            for s in self.sentence2io(sent):
                if s[0] in context_dict.keys():
                    for e in s[1]:
                        context_dict[s[0]].append(e)
                else:
                    context_dict[s[0]] = s[1]
        
        print('Context_dict done in', datetime.datetime.now() - t1)
        return context_dict    
    
    def transform_context(self, context_dict):
        t1 = datetime.datetime.now()
        print('Replacing context words with their ID ...')
        for key, value in context_dict.items():
            context_dict[key] = [self.context2id[v] for v in np.unique(value)]
        print('Done in', datetime.datetime.now() - t1)
        return context_dict
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences_without_punctuation(opts.text)
        sg = SkipGram(sentences)
        
        ### Testing the sentence2io() method : sentence -> (word, [context]) ###
        '''sg.word2vec_init()
        print(sg.sentence2io(['hello', 'world', 'test']))'''
        
        ### Testing the make_context_dict() method : {word : [context]} ###
        '''sg.word2vec_init()
        context_dict = sg.transform_context(sg.make_context_dict())
        count = 0
        for center, context in context_dict.items():
            if(count < 10):
                print(center, len(context))
                count += 1
            else:
                break'''
        
        ### Testing the entire training process ###
        sg.train(epochs = 3)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a,b,sim in pairs:
            print(a, b, sg.similarity2(a,b))

