import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer, WordnetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer as pst
from nltk.corpus import wordnet
from tqdm import tqdm
from nltk.corpus import stopwords
# from sklearn.decomposition import PCA
# import xgboost as xgb
from sklearn.pipeline import make_pipeline
import random

testdata=pd.read_csv('test_data.csv')
mycorpus=pd.read_csv('mycorpus.csv')
classifiers=open('xgb_classifications.py', 'rb')
models=pickle.load(classifiers)
classifiers.close()
mystopwords=set(stopwords.words('english'))
word_comments=[tokenizer.tokenize(x) for x in testdata["comment_text"]]
word_comments=[[x if ps.stem(x) in mycorpus else [set([ps.stem(w) for w in wordnet.synsets(x)]).intsection(set(mycorpus))][0] for x in y] for y in word_comments]
word_comments=[[ps.stem(x.lower()[:200]) for x in y if x.isalpha() and x not in mystopwords] for y in word_comments]

possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

X=np.zeros((len(word_comments),len(mycorpus)))
for i in tqdm(range(len(word_comments)),desc='Fitting the feature matrix...'):
    for y in word_comments[i]:
        if y in mycorpus:
            X[i,wordfeatures.index(y)]=1

result=[]
for i in range(6):
    yy=models[i].predict(X)
    result.append(yy)
result=np.column_stack([result[i] for i in range(6)])
df=pd.DataFrame(result, columns=possible_labels)
print(df)
