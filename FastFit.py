from sklearn.base import BaseEstimator, TransformerMixin
import fasttext as ft
from itertools import zip_longest

class FastFit(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None
    
    def fit(self, X, y):
        lista = []
        for rubrica,label in zip_longest(X, y):
            lista.append('__label__{label} {rubrica}'.format(label=label, rubrica = rubrica))
        with open('train.txt', 'w') as arquivo:
            for rubrica in lista:
                arquivo.write(f'{rubrica}\n')
        self.model = ft.train_supervised(input='train.txt', lr=0.05, epoch=100, wordNgrams=2, bucket=200000, dim=50, loss='ova')

    def predict(self, X):
        lista = []
        for rubrica in X:
            lista.append(self.model.predict(rubrica, k=-1)[0][0][len('__label__'):])
        
        return lista