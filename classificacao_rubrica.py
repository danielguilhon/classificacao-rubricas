#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from data_utils.odbc import SQL
from data_utils import query as q
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import ydata_profiling as yp
from PreRubrica import PreRubrica
from sklearn.pipeline import Pipeline

import logging
#logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.DEBUG)
#%%
#preprocessamento

def extrai_label(linha):
    label = []
    if(linha.parcela_unica==1):
        label.append('PU')
    if(linha.pode_receber_com_parcela_unica==1):
        label.append('PRPU')
    if(linha.contabiliza_para_teto==1):
        label.append('CT')
    if(linha.abate_teto==1):
        label.append('AT')

    return '__'.join(label)

def print_stats(preds, target,  sep='-', sep_len=40):
    print('Accuracy = %.3f' % accuracy_score(target, preds))
    print('Recall = %.3f' % recall_score(target, preds, average='weighted'))
    print('Precision = %.3f' % precision_score(target, preds, average='weighted'))
    print(sep*sep_len)
    print('Classification report:')
    print(metrics.classification_report(target, preds))
    print(sep*sep_len)
    print('Confusion matrix')
    metrics.ConfusionMatrixDisplay.from_predictions(target, preds)
    #metrics.confusion_matrix(target, preds)

#%%
#carrega rubricas classificadas do banco
logging.info('#carrega rubricas classificadas do banco')
query = 'rubricas_mapeadas'
sql_server = SQL(service_conn = True, database='BDU_SEFIP', maiusc=False)
df_rubricas = sql_server.as_df(q.get_query(query),
                                     persist_file=query,
                                     query_if_exists=False)

df_rubricas.loc[:,'label'] = df_rubricas.apply(lambda x: extrai_label(x), axis=1)

#%%
#yp.ProfileReport(df_rubricas)


# %%

rubricas_train, rubricas_test = train_test_split(df_rubricas[['cod','nome_rubrica', 'label']], test_size=0.20, random_state=42, stratify=df_rubricas[['label']])

# %%
from FastFit import FastFit

model_ft = Pipeline(steps=[
            ("pre_processamento", PreRubrica()),
            ("clf", FastFit())
        ])

model_ft.fit(rubricas_train[['nome_rubrica']], rubricas_train.label.values)

preds_ft = model_ft.predict(rubricas_test[['nome_rubrica']])
print_stats(preds_ft, rubricas_test.label.values)

# %%
from sklearn.naive_bayes import ComplementNB
model_tfidf = Pipeline(steps=[
            ("pre_processamento", PreRubrica()),
            ('vect', TfidfVectorizer(ngram_range=(1,2))),
            ("clf", ComplementNB())
        ])

model_tfidf.fit(rubricas_train[['nome_rubrica']], rubricas_train.label.values)
preds_tfidf = model_tfidf.predict(rubricas_test[['nome_rubrica']])
print_stats(preds_tfidf, rubricas_test.label.values)
# %%
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

xgb_estimator = XGBClassifier(
            n_estimators=250,
            max_depth=6,
            eta=0.1,
            subsample=0.6,
            colsample_bytree=0.8,
            n_jobs=-1,
            objective='binary:logistic'
    )

model_xgb = Pipeline(
    [
        ('pre_processamento', PreRubrica()),
        ('vect',  TfidfVectorizer(ngram_range=(1,2))),
        ('clf', xgb_estimator)
    ]
)
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(rubricas_train.label.values)
label_encoded_y = label_encoder.transform(rubricas_train.label.values)

model_xgb.fit(rubricas_train[['nome_rubrica']], label_encoded_y)
preds_xgb = model_xgb.predict(rubricas_test[['nome_rubrica']])
print_stats(preds_xgb, label_encoder.transform(rubricas_test.label.values))

#otimizacao parametros
import numpy as np
parameter_grid = {
    "vect__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
    "vect__min_df": (1, 3, 5, 10),
    "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
    "vect__norm": ("l1", "l2"),
    "clf__alpha": np.logspace(-6, 6, 13),
}

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=model_tfidf,
    param_distributions=parameter_grid,
    n_iter=40,
    random_state=0,
    n_jobs=2,
    verbose=1,
)

random_search.fit(rubricas_train[['nome_rubrica']], rubricas_train.label.values)
from sklearn.metrics import make_scorer
recall_scorer = make_scorer(recall_score, average='weighted')
best_parameters = random_search.best_estimator_.get_params()
print('Best Params - TF-IDF')
for param_name in sorted(parameter_grid.keys()):
    print(f"{param_name}: {best_parameters[param_name]}")


from sklearn.model_selection import cross_val_score, StratifiedKFold
test_accuracy = random_search.score(rubricas_test[['nome_rubrica']], rubricas_test.label.values, scoring=recall_scorer)
print(
    "Accuracy of the best parameters using the inner CV of "
    f"the random search: {random_search.best_score_:.3f}"
)
print(f"Accuracy on test set: {test_accuracy:.3f}")
skf = StratifiedKFold()
res_ft = cross_val_score(model_ft ,df_rubricas[['nome_rubrica']], df_rubricas.label.values, cv = skf, scoring='recall')

res_tfidf = cross_val_score(model_tfidf ,df_rubricas[['nome_rubrica']], df_rubricas.label.values, cv = skf, error_score=-1)

label_encoder = label_encoder.fit(df_rubricas.label.values)
label_encoded_y = label_encoder.transform(df_rubricas.label.values)

res_xgboost = cross_val_score(model_xgb ,df_rubricas[['nome_rubrica']], label_encoded_y, cv = skf, error_score=-1)
'''
logging.info('#salva modelo')
model.save_model('modelo/rubricas.bin')
# %
# rodar o modelo em novas rubricas
logging.info('busca rubricas não mapeadas')
query = 'rubricas_nao_mapeadas'
sql_server = SQL(service_conn = True)
rubricas_nao_mapeadas = sql_server.as_df(q.get_query(query),
                                     persist_file=query,
                                     query_if_exists=True)
logging.info('aplica unidecode_text para possível converão UTF-8')
rubricas_nao_mapeadas.loc[:,'nome_rubrica'] = rubricas_nao_mapeadas.nome_rubrica.apply(lambda x: unidecode_text(x))
rubricas_nao_mapeadas.loc[:,'rubrica_pre'] = rubricas_nao_mapeadas.nome_rubrica.apply(lambda x: ' '.join(pre_processamento(x)))

rubricas_nao_mapeadas.loc[:,'previsao'] = rubricas_nao_mapeadas.rubrica_pre.apply(lambda x: model.predict(x, k=-1)[0][0][len('__label__'):])
#%
previsoes = []
for rubrica in rubricas_nao_mapeadas.itertuples():
    rubrica_prevista = model.predict(rubrica.rubrica_pre, k=-1)
    previsoes.append({'cod': int(rubrica.cod), 'label': rubrica_prevista[0][0][len('__label__'):]})
previsoes_df = pd.DataFrame(previsoes)

# %
rubricas_nao_mapeadas.to_excel('dados/previsao_mapeamento_rubricas.xlsx')

'''
# %%
