from sklearn.base import BaseEstimator, TransformerMixin
import unidecode
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import RSLPStemmer
import pandas as pd

class PreRubrica(BaseEstimator, TransformerMixin):
    
    def __init__(self):

        return None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        XX = X.copy()
        XX.loc[:,'rubrica_pre'] = XX.nome_rubrica.map(lambda x: self._unidecode_text(x))
        XX.loc[:,'rubrica_pre'] = XX.rubrica_pre.map(lambda x: self._SubstPersonalizada(x))
        XX.loc[:,'rubrica_pre'] = XX.rubrica_pre.map(lambda x: self._Tokenize(x))
        XX.loc[:,'rubrica_pre'] = XX.rubrica_pre.map(lambda x: self._Stemming(x))
        XX.loc[:,'rubrica_pre'] = XX.rubrica_pre.map(lambda x: self._RemoveStopWords(x))
        XX.loc[:,'rubrica_pre'] = XX.rubrica_pre.map(lambda x: ' '.join(x))
        
        return(XX.drop(['nome_rubrica'], axis=1).rubrica_pre)
    
    def _unidecode_text(self, text):
        try:
            text = unidecode.unidecode(text)
        except:
            pass
        return text

    def _SubstPersonalizada(self, rubrica):
        rubrica= rubrica.replace('1/3', 'um terço')
        rubrica= rubrica.replace('2/3', 'dois terços')
        rubrica= rubrica.replace('A++','Ç')
        rubrica= rubrica.replace('Af', 'Ã')
        rubrica= rubrica.replace('A%0', 'É')
        rubrica= rubrica.replace('As', 'Ú')
        rubrica= rubrica.replace('A"', 'Ó')
        rubrica= rubrica.replace('.',' ')#apenas para numeros (usar regexp)
        rubrica= rubrica.replace('?', '')
        return rubrica
    
    #tokenize
    def _Tokenize(self, sentence):
        #tokenizer = RegexpTokenizer(r'\w+')
        tokenizer = RegexpTokenizer(r'[^\W_]+')
        sentence = sentence.lower()
        sentence = tokenizer.tokenize(sentence)
        return sentence
    #stemming
    def _Stemming(self, sentence):
        stemmer = RSLPStemmer()
        phrase = []
        for word in sentence:
            phrase.append(stemmer.stem(word.lower()))
        return phrase

    #remove stopwords
    def _RemoveStopWords(self, sentence):
        stopwords = nltk.corpus.stopwords.words('portuguese')
        phrase = []
        for word in sentence:
            if word not in stopwords:
                phrase.append(word)
        return phrase
    