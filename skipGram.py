from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from scipy.special import expit
from sklearn.preprocessing import normalize


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
        
    def train(self,stepsize = 0.05, epochs = 10):
        self.word2vec_init()
        print("The corpus has " + str(self.voc_size) + " different words")
        
        count_sentences = 0 
        for i in range(epochs):
            loss = 0
            for sent in self.sentences :
                count_sentences +=1
                for tup in self.sentence2io(sent) :
                    center_word = tup[0]
                    for context_word in tup[1] : 
                        v_center_word = self.word2vec[center_word]
                        v_context_word = self.context2vec[context_word]
                        negative_sample = self.negative_sampling() 
                        v_neg = np.array([self.context2vec[n] for n in negative_sample])
                        
                        loss += self.loss_function( v_center_word, v_context_word, v_neg)
                        
                        self.word2vec[center_word] = v_center_word - stepsize * self.gradient_center_word ( v_center_word, v_context_word, v_neg)
                        self.context2vec[context_word]  = v_context_word - stepsize * self.gradient_context_word (v_center_word, v_context_word)
                        
                        for j, v in enumerate(v_neg) :
                            self.context2vec[negative_sample[j]]= v - stepsize * self.gradient_neg_word ( v_center_word, v )
                if (count_sentences % 1000 ==0 ): print(str(count_sentences) + " sentences proceeded")
            print("Epoch number" + str(i) + ", the loss is : " + str(loss))
    
    def trainV2(self,stepsize = 0.05, epochs = 10):
        self.word2vec_initV2()
        print("The corpus has " + str(self.voc_size) + " different words")
        self.wordcount_and_sentences2id()
        print("The corpus has " + str(self.sentences_id.shape[0]) + " sentences")
        
        
        for i in range(epochs):
            print("Epoch n : "+str(i))
            count_sentences = 0 
            loss = 0
            for sent in self.sentences_id :
                count_sentences +=1
                for tup in self.sentence2io(sent) :
                    center_word = tup[0]
                    for context_word in tup[1] : 
                        v_center_word = self.id2center_vec[center_word]
                        v_context_word = self.id2context_vec[context_word]
                        negative_sample = self.negative_sampling() 
                        v_neg = np.array([self.id2context_vec[n] for n in negative_sample])
                        
                        loss += self.loss_function( v_center_word, v_context_word, v_neg)
                        
                        self.id2center_vec[center_word] = v_center_word - stepsize * self.gradient_center_word ( v_center_word, v_context_word, v_neg)
                        self.id2context_vec[context_word]  = v_context_word - stepsize * self.gradient_context_word (v_center_word, v_context_word)
                        
                        for j, v in enumerate(v_neg) :
                            self.id2context_vec[negative_sample[j]]= v - stepsize * self.gradient_neg_word ( v_center_word, v )
                if (count_sentences % 1000 ==0 ): print(str(count_sentences) + " sentences proceeded")
            print("Epoch number" + str(i) + ", the loss is : " + str(loss))
            
            
    def train2(self, stepsize = 0.05, epochs = 10):
        self.word2vec_init()
        context_dict = self.make_context_dict()
                    
        for i in range(epochs):
            loss = 0
            
            for center_word, context in context_dict.items():
                for context_word in context : 
                    v_center_word = self.word2vec[center_word]
                    v_context_word = self.context2vec[context_word]
                    negative_sample = self.negative_sampling() 
                    v_neg = np.array([self.context2vec[n] for n in negative_sample])
                        
                    loss += self.loss_function( v_center_word, v_context_word, v_neg)
                        
                    self.word2vec[center_word] = v_center_word - stepsize * self.gradient_center_word ( v_center_word, v_context_word, v_neg)
                    self.context2vec[context_word]  = v_context_word - stepsize * self.gradient_context_word (v_center_word, v_context_word)
                        
                    for j, v in enumerate(v_neg) :
                        self.context2vec[negative_sample[j]]= v - stepsize * self.gradient_neg_word ( v_center_word, v )
            print("Epoch number", str(i), ", the loss is : ", str(loss))

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
        model = pd.DataFrame.from_dict(data = self.word2vec)
        model.to_csv(path, index = False)

    @staticmethod
    def load(self, path):
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
        model = pd.read_csv(path)
        self.word2vec = model.to_dict(orient='list')
        pass
    
    def similarity(self,word1, word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        test = ( word2 in self.word2vec.keys() )
        if word1 in self.word2vec.keys() :
            if test == 1 :
                similarity = np.dot(self.word2vec[word1], self.word2vec[word2]) / ( np.linalg.norm(self.word2vec[word1]) * np.linalg.norm(self.word2vec[word2]) )
            else : 
                word2 = self.randomvector
                similarity = np.dot(self.word2vec[word1],word2) /  ( np.linalg.norm(self.word2vec[word1]) * np.linalg.norm(word2) )
        else :
            word1 = self.randomvector
            if test == 1 :
                similarity = np.dot(word1,self.word2vec[word2]) /  ( np.linalg.norm(word1) * np.linalg.norm(self.word2vec[word2]) )
            else : 
                word2 = self.randomvector
                similarity = np.dot(word1,word2) /  ( np.linalg.norm(word1) * np.linalg.norm(word2) )
        return similarity
        
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
    
    def word2vec_initV2(self) :
        """     Creates 4 dictionnaries for sentences:
            word2vec : for each word, creates a random uniform array of size nEmbed
            word_count : counts the number of occurrences of each word
            context2vec : for each word, creates a random uniform array of size nEmbed
            id2context : assign an id as key for each for each word
                It also creates a list of frequencies, where the frequencies are the nb of occurrence 
                of a word raised to the power of alpha, divided by the sum of all those weights"""
        self.word2id = {}
        self.id2center_vec =np.array([])
        
        self.context2id = {}
        self.id2context_vec=np.array([])
        
        self.voc_size=0 
        
        for sent in self.sentences :
            for word in sent :
                
                if word not in self.word2id.keys() :
                    
                    self.word2id[word] = self.voc_size
                    self.id2center_vec = np.append(self.id2center_vec,np.random.rand(self.nEmbed))
                    
                    self.context2id[word] = self.voc_size
                    self.id2context_vec = np.append(self.id2context_vec,np.random.rand(self.nEmbed))
                    
                    self.voc_size +=1
        
        pass
                    
    def wordcount_and_sentences2id (self) :
        """ This function creates 3 new objects variables :
            - word_count : an np.array where the indexes are the values of word2id, and the values are the nb of times this word appears in the corpus
            - freq : the word_count raised to the power self.alpha, and then divided by the sum to have a frequencies
            - sentences_id : we create a new list from sentences, where we replace each word by its id in word2id dictionnary
             """
        self.word_count = np.zeros(self.voc_size)
        self.sentences_id = np.array([])
        
        for sent in self.sentences :
            sent_id=np.array([])
            for word in sent :
                
                word_id = self.word2id[word]
                
                self.word_count[ word_id ] +=1
                sent_id = np.append(sent_id , word_id)
                
            self.sentences_id = np.append(self.sentences_id, sent_id) 
        
        temp = np.power(self.word_count, self.alpha)
        self.freq = temp / temp.sum()
        del temp
        
        del self.sentences
        pass
    
    def negative_sampling(self):
        """ This function returns words picked following the distribution below :
            P (word[i]) = frequency(word[i])^alpha / sum of all frequencies raised to the power alpha """
        sample = np.random.choice(a=np.arange(self.voc_size),size=self.negativeRate, p= self.freq)
        #res = [self.id2context[neg_sample] for neg_sample in sample ]
        #return res
        return sample
    
    ### Loss function to minimize ###
    
    def sigmoid (self, x) :
        return 1/ (1+ np.exp(-x))
    
    def loss_function(self, word, context, negative_sample):
        """This function is the loss function. 
        The arguments are either word or vectors"""
        if(isinstance(word, str)) :
            word = self.word2vec[word]
        if(isinstance(context, str)):
            context = self.context2vec[context]
        
        neg = sum([np.log(self.sigmoid(-np.vdot(word, v))) for v in negative_sample])
        res = (- np.log(self.sigmoid(np.vdot(word, context))) - neg) 
        return res
    
     ### Optimization functions ###
    
    def gradient_center_word (self, center_word, context_word, negative_sample) :
        """ This function is the derived loss function by the vector of the central word 
            The arguments should be vectors of words embedding."""
        res = (self.sigmoid( np.vdot(center_word, context_word) )-1) * context_word
        for n in negative_sample :
            res += (self.sigmoid(np.vdot(center_word, n))) * n
        return res 
    
    def gradient_context_word (self, center_word, context_word): 
        """ This function is the derived loss function by the vector of the context word
        The arguments should be vectors of words embedding."""
        res = (self.sigmoid( np.vdot(center_word, context_word) )-1) * center_word
        return res
    
    def gradient_neg_word (self, center_word, neg_word ) :
        """ This function is the derived loss function by the vector of one negative sampled word
        The arguments should be vectors of words embedding."""
        res =(self.sigmoid(np.vdot(center_word, neg_word))) * center_word
        return res 
    
    ### Context related functions ### 
        
    def sentence2io(self, sentence) :
        """ This function takes as input a sentence, and returns list of tuples :
            - the first element is the center word
            - the second is a list of context words
        """
        L = sentence.shape[0]
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
        ''' This function creates a dictionary containing all the worlds of the vocabulary
            as keys, and their context as an array, as value 
        '''
        context_dict = {}
        for sent in self.sentences:
            for s in self.sentence2io(sent):
                if s[0] in context_dict.keys():
                    for e in s[1]:
                        context_dict[s[0]].append(e)
                else:
                    context_dict[s[0]] = s[1]
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
        
        sg.train2()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            print(sg.similarity(a,b))
"""

training_data_path = '/Users/alimrabeth/Desktop/Master Data Sciences & Business Analytics/Data Sciences Elective courses/NLP/Projet 1/sentences.txt'
sentences = text2sentences_without_punctuation(training_data_path)

sg = SkipGram(sentences, nEmbed=100, negativeRate=5, winSize = 3)
sg.trainV2()
