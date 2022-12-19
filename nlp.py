from sklearn.metrics.pairwise import pairwise_distances

from nltk.tokenize import WhitespaceTokenizer, RegexpTokenizer

# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from gensim.models import KeyedVectors

import spacy

# import scipy.sparse as sp

import numpy as np

def preprocess():
    return

class Corpus:
    def __init__(self, texts=[], titles=[], target_ints=[],
                       target_idx=[], tokenized_texts=None):
        self.texts = texts
        self.tokenized_texts = tokenized_texts
        self.titles = titles # should be unique
        self.target_ints = target_ints # integers corresponding to category names
        self.target_idx = target_idx # actual category names as strings

    def add_text(self, text, title, target_name):
        if title in self.titles:
            raise ValueError('Titles should be unique: %s is already in the corpus' %(title))
        if target_name not in self.target_idx:
            self.target_idx.append(target_name)
        target_int = self.target_idx.index(target_name) 
        self.texts.append(text)
        self.titles.append(title)
        self.target_ints.append(target_int)

    def generate_vectors(self, vec_type="count", tokenize_type=None, scale_type=None, ngram_min=1, ngram_max=1, mfi=100, max_df=1.0, min_df=0.0):

        # if we want a special tokenization method, we specify it and use here
        if tokenize_type == 'whitespace':
            tokenizer = WhitespaceTokenizer().tokenize
        elif tokenize_type == 'words':
            # keep only words without digits:
            tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b').tokenize
        elif tokenize_type == 'words_naive':
            # keep alphanumeric
            tokenizer = RegexpTokenizer(r'\w+').tokenize
        elif "spacy" in tokenize_type:
            name = vec_type.split("spacy-")[1]
            tokenizer = lambda x: x # don't do anything
            nlp = spacy.load(name)
            nlp.max_length = 10**10 
        else:
            tokenizer = lambda x: x.split()
        
        if tokenize_type!=None:
            self.tokenized_texts = []
            for text in self.texts:
                tokens = tokenizer(text)
                self.tokenized_texts.append(tokens)

        if vec_type in ["count", "tfidf", "tf"]:
            self.params = {
                'max_features': mfi,
                'max_df': max_df, # some cutoffs
                'min_df': min_df, # some cutoffs
                'preprocessor': lambda x: x, # do this before adding to corpus, NOT HERE
                'ngram_range': (ngram_min, ngram_max), # if we want to use ngrams
                'analyzer': 'word', # we only ever want to use words as tokens and never characters
            }
            if vec_type == "tf":
                self.params['use_idf']=False # default is True
            self.params['tokenizer'] = lambda x: x
            
            if vec_type in ["tfidf", "tf"]:
                vectorizer = TfidfVectorizer(**self.params)
            elif vec_type in ["count"]:
                vectorizer = CountVectorizer(**self.params)

            self.X = vectorizer.fit_transform(self.tokenized_texts)
        elif "w2v" in vec_type:
            name = vec_type.split("w2v-")[1]
            vectors = KeyedVectors.load(f'models/{name}.model')
            self.X = []
            self.missed = []
            for text in self.tokenized_texts:
                vec = np.zeros(vectors.vector_size)
                N = len(text)
                missed_words = []
                for item in text:
                    try:
                        vec += vectors[item]
                    except:
                        missed_words.append(item)
                        pass
                self.missed.append(missed_words)
                self.X.append(np.array(vec)/N)
            self.X = np.array(self.X)
        elif "spacy" in vec_type:
            self.X = []
            for text in self.tokenized_texts:
                self.X.append(np.array(nlp(text).vector))
            self.X = np.array(self.X)

        try:
            self.X = self.X.toarray()
        except:
            pass
            
        if scale_type == "std_col":
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
        elif scale_type == "std_row":
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X.T).T
        elif scale_type == "minmax_row":
            scaler = MinMaxScaler()
            self.X = scaler.fit_transform(self.X.T).T
        elif scale_type == "minmax_col":
            scaler = MinMaxScaler()
            self.X = scaler.fit_transform(self.X)

        return self.X



def distance_matrix(corpus=None, X=None, metric='manhattan'):
    if not metric in ('manhattan', 'cityblock', 'euclidean', 'cosine', 'minmax'):
        raise ValueError('Unsupported distance metric: %s' %(metric))
    if corpus:     
        try:
            X = corpus.X
            try:
                X = X.toarray()
            except AttributeError:
                pass
        except AttributeError:
            ValueError('Your corpus does not seem to have been vectorized yet.')
    if metric == 'minmax':
        return pairwise_distances(X, metric=minmax)
    else:
        return pairwise_distances(X, metric=metric)

def minmax(x, y):
    mins, maxs = 0.0, 0.0
    for i in range(x.shape[0]):
        a, b = x[i], y[i]
        if a >= b:
            maxs += a
            mins += b
        else:
            maxs += b
            mins += a
    if maxs > 0.0:
        return 1.0 - (mins / maxs)
    return 0.0