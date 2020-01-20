import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import wordnet
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.pipeline import make_pipeline
import random
import pickle


tokenizer = RegexpTokenizer(r'\w+')
ps=PorterStemmer()
mystopwords=set(stopwords.words('english'))

traindata = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv/train.csv")
word_comments=[tokenizer.tokenize(x) for x in traindata["comment_text"]]
word_comments=[[ps.stem(x.lower()[:200]) for x in y if x.isalpha() and x not in mystopwords] for y in word_comments]


all_words=[]
for w in tqdm(word_comments):
    all_words+=w


freq=nltk.FreqDist(all_words)
wordfeatures=[w for w in freq.keys() if freq[w]>100]
X=np.zeros((len(word_comments),len(wordfeatures)))
for i in tqdm(range(len(word_comments)),desc='Fitting the feature matrix...'):
    for y in word_comments[i]:
        if y in wordfeatures:
            X[i,wordfeatures.index(y)]=1



possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
Y = traindata[possible_labels].values
Y=pd.DataFrame(Y, columns=possible_labels)
pca = PCA(n_components=800, random_state=42)
model=xgb.XGBClassifier(random_state=1, learning_rate=0.2, n_jobs=-1)
model1=make_pipeline(pca,model)
indx=random.sample(list(range(X.shape[0])),int(X.shape[0]*.9))

print('[Training model...]')
start=time.process_time()
modela=model1.fit(X[indx,:],Y.iloc[indx,0])
modelb=model1.fit(X[indx,:],Y.iloc[indx,1])
modelc=model1.fit(X[indx,:],Y.iloc[indx,2])
modeld=model1.fit(X[indx,:],Y.iloc[indx,3])
modele=model1.fit(X[indx,:],Y.iloc[indx,4])
modelf=model1.fit(X[indx,:],Y.iloc[indx,5])
print(f'[Finished Training in {time.process_time()-start} seconds.]')

indt=list(set(list((range(X.shape[0]))))-set(indx))
result=[]
models=[modela,modelb,modelc,modeld,modele,modelf]
for i in range(6):
    yy=models[i].predict(X[indt,:])
    result.append(np.mean([x==y for x,y in zip(yy,Y.iloc[indt,i])]))
print(f'Final Accuracy: {np.mean(result)}')
pd.to_csv(wordfeatures,'mycorpus.csv')

classifiers=open('xgb_classifications.py', 'wb')
pickle.dump(models,classifiers)
classifiers.close()
