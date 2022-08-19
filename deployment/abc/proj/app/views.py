from multiprocessing.sharedctypes import Value
import re
from io import StringIO
import json
import nltk
import gensim
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
import lightgbm as lgb
from sklearn.svm import SVC
from django.shortcuts import render
from django.shortcuts import HttpResponse
from django.http import HttpResponse
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier                                      
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from gensim.models.fasttext import load_facebook_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

global score

def index(request):
    context ={'a':'Hello World'}
    return render(request, 'index.html',context)  

def tokenization(text):
    tokens = re.split('W+',text)
    return tokens

def remove_stopwords(text):
    tokens = [token.lower() for token in text if not token in nltk_stopwords]
    return tokens

def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    stem_text = ' '.join(stem_text)
    return stem_text

def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    lemm_text = ' '.join(lemm_text)
    return lemm_text

def createfile():
    global f
    l=[]
    l.extend(X_train_vtc)
    l.extend(X_test_vtc)
    f= pd.DataFrame(l)
    f['labels']=Y
    f.to_csv('./media/AI_ready_data.csv', index=False)
    f='AI_ready_data.csv'


def tf_idf():
    global nltk_stopwords,porter_stemmer,wordnet_lemmatizer,X,Y,score,df,X_train, X_test, y_train, y_test,X_train_vtc,X_test_vtc
    X=X.apply(lambda x:tokenization(x))
    X=X.apply(lambda x:remove_stopwords(x))        
    X=X.apply(lambda x:stemming(x))
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(X) 
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
    X_train_vtc=X_train
    X_test_vtc=X_test
    createfile()

def count_vect():
    global nltk_stopwords,porter_stemmer,wordnet_lemmatizer,X,Y,score,df,X_train, X_test, y_train, y_test,X_train_vtc,X_test_vtc
    X=X.apply(lambda x:tokenization(x))
    X=X.apply(lambda x:remove_stopwords(x))        
    X=X.apply(lambda x:stemming(x))
    cv = CountVectorizer()
    x = cv.fit_transform(X,Y)
    X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size = 0.2)
    X_train_vtc=X_train
    X_test_vtc=X_test
    createfile()

def word2vec_customized():
    global nltk_stopwords,porter_stemmer,wordnet_lemmatizer,X,Y,score,df,X_train, X_test, y_train, y_test,X_train_vtc,X_test_vtc
    X_train_vect_avg = []
    X_test_vect_avg = []
    X=X.apply(lambda x:tokenization(x))
    X=X.apply(lambda x:remove_stopwords(x))        
    X=X.apply(lambda x:stemming(x))
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
    w2v_model = gensim.models.Word2Vec(X_train,vector_size=100,window=5,min_count=2)
    words = set(w2v_model.wv.index_to_key )
    X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_train])
    X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_test])
    X_train_vect_avg = []
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(100, dtype=float))
    X_test_vect_avg = []
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(100, dtype=float))
    X_train_vtc=X_train_vect_avg
    X_test_vtc=X_test_vect_avg
    createfile()
    y_train=y_train.values.ravel()
    X_train=X_train_vect_avg
    X_test=X_test_vect_avg

def Word2vec_google():
    global nltk_stopwords,porter_stemmer,wordnet_lemmatizer,X,Y,score,df,X_train, X_test, y_train, y_test,X_train_vtc,X_test_vtc
    X_train_vect_avg = []
    X_test_vect_avg = []
    X=X.apply(lambda x:tokenization(x))
    X=X.apply(lambda x:remove_stopwords(x))        
    X=X.apply(lambda x:stemming(x))
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
    filename = "C:/Users/nikhil/Desktop/GoogleNews-vectors-negative300.bin"
    model_g= KeyedVectors.load_word2vec_format(filename, binary=True)
    words =set(model_g.index_to_key)
    X_train_vect= np.array([np.array([model_g[i] for i in ls if i in words])
                         for ls in X_train])     
    X_test_vect = np.array([np.array([model_g[i] for i in ls if i in words])
                         for ls in X_test])                        
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(300, dtype=float))
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(300, dtype=float)) 
    X_train_vtc=X_train_vect_avg
    X_test_vtc=X_test_vect_avg
    createfile()
    y_train=y_train.values.ravel() 
    X_train=X_train_vect_avg
    X_test=X_test_vect_avg


def fasttext():
    global nltk_stopwords,porter_stemmer,wordnet_lemmatizer,X,Y,score,df,X_train, X_test, y_train, y_test,X_train_vtc,X_test_vtc
    X=X.apply(lambda x:tokenization(x))
    X=X.apply(lambda x:remove_stopwords(x))        
    X=X.apply(lambda x:stemming(x))
    X_train_vect_avg= []
    X_test_vect_avg = []
    X_train, X_test, y_train, y_test = train_test_split (X,Y)
    filename =r"C:/Users/nikhil/Desktop/cc.en.300.bin" 
    model_g= load_facebook_model(filename)
    words =set(model_g.wv.index_to_key)     
    X_train_vect= np.array([np.array([model_g.wv[i] for i in ls if i in words])
                         for ls in X_train])     
    X_test_vect = np.array([np.array([model_g.wv[i] for i in ls if i in words])
                         for ls in X_test]) 
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(300, dtype=float))
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(300, dtype=float))
    X_train_vtc=X_train_vect_avg
    X_test_vtc=X_test_vect_avg
    createfile()
    y_train=y_train.values.ravel() 
    X_train=X_train_vect_avg
    X_test=X_test_vect_avg

def glove_split():
    global glove
    global nltk_stopwords,porter_stemmer,wordnet_lemmatizer,X,Y,score,df,X_train, X_test, y_train, y_test,X_train_vtc,X_test_vtc
    glove={}
    X=X.apply(lambda x:tokenization(x))
    X=X.apply(lambda x:remove_stopwords(x))        
    X=X.apply(lambda x:stemming(x))
    total_vocabulary=set(word for text in X for word in text)
    with open(r"C:/Users/nikhil\Desktop/glove.twitter.27B.100.txt", 'rb') as f:
        for line in f:
                parts = line.split()
                word = parts[0].decode('utf-8')
                if word in total_vocabulary:
                    vector = np.array(parts[1:], dtype=np.float32)
                    glove[word] = vector
    class W2vVectorizer(object):
        def __init__(self, w2v):
            self.w2v = w2v
            if len(w2v) == 0:
                self.dimensions = 0
            else:
                self.dimensions = len(w2v[next(iter(glove))])
        def fit(self, X, Y):
            return self
        def transform(self, X):
            return np.array([
                np.mean([self.w2v[w] for w in words if w in self.w2v]
                    or [np.zeros(self.dimensions)], axis=0) for words in X])
    vectorizer = W2vVectorizer(glove)
    X_glove=vectorizer.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_glove, Y,test_size=0.2)
    X_train_vtc=X_train
    X_test_vtc=X_test
    createfile()

def decision_tree_classifier():
    rf_dt = DecisionTreeClassifier()
    rf_model_dt = rf_dt.fit(X_train, y_train)
    y_pred = rf_model_dt.predict(X_test)
    score=classification_report(y_test, y_pred,output_dict=True)
    return score

def Logistic_Regression_classifier():
    clf = LogisticRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score=classification_report(y_test,y_pred,output_dict=True)
    return score 

def Gradient_Boosting_Classifier():
    rf_lr = GradientBoostingClassifier()
    rf_model_lr = rf_lr.fit(X_train, y_train)
    y_pred = rf_model_lr.predict(X_test)
    score=classification_report(y_test, y_pred,output_dict=True)
    return score 

def KNeighbors_classifier():
    rf_kn =KNeighborsClassifier()
    rf_model_kn = rf_kn.fit(X_train, y_train)
    y_pred = rf_model_kn.predict(X_test)
    score=classification_report(y_test, y_pred,output_dict=True)
    return score

def SVC_classsifier():
    rf_svc=SVC()
    rf_model_svc= rf_svc.fit(X_train, y_train)
    y_pred = rf_model_svc.predict(X_test)
    score=classification_report(y_test, y_pred,output_dict=True)
    return score

def RandomForest_classifier():
    rf_rf=RandomForestClassifier()
    rf_model_rf= rf_rf.fit(X_train, y_train)
    y_pred = rf_model_rf.predict(X_test)
    score=classification_report(y_test, y_pred,output_dict=True)
    return score

def mnb_classifier():
    rf_mnb=MultinomialNB()
    rf_model_mnb= rf_mnb.fit(X_train, y_train)
    y_pred = rf_model_mnb.predict(X_test)
    score=classification_report(y_test, y_pred,output_dict=True)
    return score

def stacking_1():
    estimator_list = [
    ('knn',KNeighborsClassifier()),
    ('svm_rbf',SVC()),
    ('dt', DecisionTreeClassifier()),
    ('rf',RandomForestClassifier()),
    ('mlp',MultinomialNB()) ]
    stack_model = StackingClassifier(
    estimators=estimator_list, final_estimator=LogisticRegression())
    stack_model.fit(X_train, y_train)
    y_pred = stack_model.predict(X_test)
    score=classification_report(y_test, y_pred,output_dict=True)
    return score

def preprocess(request):
    global nltk_stopwords,porter_stemmer,wordnet_lemmatizer,X,Y,score,df
    score=''
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    porter_stemmer=PorterStemmer()
    wordnet_lemmatizer=WordNetLemmatizer()
    we = str(request.POST["1"])
    cl = str(request.POST["2"])
    sep= str(request.POST["3"])
    uploaded_file = request.FILES['uploaded_file'].read()
    uploaded_file=str(uploaded_file,'utf-8')
    uploaded_file=StringIO(uploaded_file)
    sep='	'
    df=pd.read_csv(uploaded_file,sep,header=None, names=['label','text'])
    df = df.fillna('')
    X=df['text']
    Y=df['label']
    if we=='TFIDF':
        tf_idf()
    elif we=='COUNTVECT':
        count_vect()
    elif we=="WORDTOVECCOS":
        word2vec_customized()
    elif we=='WORDTOVECGOG':
        Word2vec_google()
    elif we=='FAST-TEXT':
        fasttext()
    elif we=='GLOVE':
        glove_split()
    if cl=="RANDOM_CL":
        score=RandomForest_classifier()    
    elif cl=='DECISION_CL':
        score=decision_tree_classifier()
    elif cl=="KNN_CL":
        score=KNeighbors_classifier()
    elif cl=="LOG_CL":
        score=Logistic_Regression_classifier()
    elif cl=="SVC_CL":
        score=SVC_classsifier()
    elif cl=="MN_CL":
        score=mnb_classifier() 
    elif cl=="STACK_CL_1":
        score=stacking_1()           
    score=pd.DataFrame(score).transpose()
    score.to_csv("classification_report.csv")
    df = pd.read_csv(r'classification_report.csv')
    df.rename(columns = {'Unnamed: 0':'Labels'},inplace = True)
    df.rename(columns = {'f1-score':'f1_score'},inplace = True)
    json_records = df.reset_index().to_json(orient = 'records')
    data = []
    data = json.loads(json_records)
    context = {'d':data,'f':f}
    return render(request,'table.html',context)


