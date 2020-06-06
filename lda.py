import sys
print(sys.argv)

n_topics=int(sys.argv[1])
n_words=int(sys.argv[2])
days_offset=int(sys.argv[3])
merge_by_day=sys.argv[4] in ['True','true']

print(f'n_topics={n_topics}, n_words={n_words}, days_offset={days_offset}, merge_by_day={merge_by_day}')


import pandas as pd

df = pd.DataFrame()
for chunk in pd.read_csv('Noticias_Dataset_IDANAE_latin1.csv', encoding='latin1',sep=';', chunksize=1000, nrows=69000):
    df = pd.concat([df, chunk], ignore_index=True)
	

import datetime
# len(df['FECHA SCRAPING'].unique())
len(df['FECHA SCRAPING'].unique())

#Eliminar fechas 0 o nulas
df = df[df['FECHA SCRAPING'] != '0']
df = df[df['FECHA SCRAPING'].notna()]
df = df[df['CUERPO SCRAPEADO'].notna()]

df['Date']= pd.to_datetime(df['FECHA SCRAPING'],format="%d/%m/%Y")
df=df[(df['Date']<datetime.datetime(2020,1,1))]
df=df[(df['Date']>datetime.datetime(2015,1,1))]

# eliminar caracteres extraños
df['CUERPO SCRAPEADO'] = df['CUERPO SCRAPEADO'].str.replace('\x93','')
df['CUERPO SCRAPEADO'] = df['CUERPO SCRAPEADO'].str.replace('\x94','')
df['CUERPO SCRAPEADO'] = df['CUERPO SCRAPEADO'].str.replace('\n',' ')
# Remove Emails
df['CUERPO SCRAPEADO'] = df['CUERPO SCRAPEADO'].str.replace('\S*@\S*\s?', '')

# Remove new line characters
df['CUERPO SCRAPEADO'] = df['CUERPO SCRAPEADO'].str.replace('\s+', ' ')

# Remove distracting single quotes
df['CUERPO SCRAPEADO'] = df['CUERPO SCRAPEADO'].str.replace("\'", "")

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer
import gensim

#nltk.download('stopwords')  # Esto solo hay que hacerlo una vez cuando instalas nltk
#nltk.download('punkt')  # lo mismo

STOP_WORDS_SPANISH = stopwords.words('spanish')
#stemmer = SnowballStemmer('spanish')
print("empezando a tokenizar!")
#Tokenizar
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(df['CUERPO SCRAPEADO']))

#Para buscar palabras que van juntas

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
#trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
#trigram_mod = gensim.models.phrases.Phraser(trigram)

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

#Eliminar stopwords, hacer biagramas y lematizar
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in STOP_WORDS_SPANISH] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
	
	
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)


# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'es' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])

print("empezando a lematizar!")
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# crear diccionario o lista de palabras unicas: words_index
from collections import Counter

#words_index = []
#for tokens in noticias_tokens:
 #   for w in tokens:
  #      if w not in words_index:
   #         words_index.append(w)
#words_index= Counter(set([w for tokens in noticias_tokens for w in tokens])).most_common(500)

counter = Counter() 
for tokens in data_lemmatized:
    for w in tokens:    
        counter.update([w])
		
		
words_index=[t[0] for t in counter.most_common(n_words)]

import numpy as np
from scipy.sparse import csr_matrix
# def convert_to_bow(noticias, words_index):
    # n_noticias = len(noticias)
    # n_words = len(words_index)
    # X = np.zeros((n_noticias, n_words))  # crear matriz inicialmente a cero 
    # for i, noticia in enumerate(noticias):
        # for word in noticia:
            # if word in words_index:
                # j = words_index.index(word)
                # X[i,j] += 1
    # return X

def convert_to_bow_sparse(noticias, words_index):
    row = []
    col = []
    data = []
    for i, noticia in enumerate(noticias):
        for word in noticia:
            if word in words_index:
                j = words_index.index(word)
                row.append(i)
                col.append(j)
                data.append(1)
    return csr_matrix((data, (row, col)), shape=(len(noticias), len(words_index)))
    
print("empezando a crear bolsa de palabras!")   
    
X = convert_to_bow_sparse(data_lemmatized, words_index)


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

print("empezando a entrenar LDA!")
lda = LatentDirichletAllocation(n_jobs=10,n_components=n_topics, random_state=0)
lda.fit(X)

matriz_topics = lda.transform(X)
columna_nueva_fecha=np.array(df['FECHA SCRAPING'])
X_final = np.column_stack((columna_nueva_fecha,matriz_topics)) 


df_final=pd.DataFrame(X_final)
df_final.insert(0, 'id', df_final.index)
df_final.rename(columns={'Unnamed: 0':'ID', 0: 'FECHA'}, inplace=True)

output_name= f'MatrizTopics_LDA_index_fecha_{n_topics}_{n_words}_{days_offset}_{merge_by_day}.csv'
df_final.to_csv(output_name,sep=';', index=None)

#############################################################################
import pandas as pd

df=pd.read_csv(output_name, delimiter=';')

#tipo de dato
#type(df['FECHA'].iloc[0])
#formato de fecha
df['Date']= pd.to_datetime(df['FECHA'],format="%d/%m/%Y")
df = df.sort_values(by=['Date'])

df_bolsa=pd.read_csv('IBEX.csv', delimiter=',')

#tipo de dato
#type(df_bolsa['Date'].iloc[0])
#formato de fecha
df_bolsa['Date']=pd.to_datetime(df_bolsa['Date'])


#eliminamos columna id
df= df.iloc[:,1:]

#Predicción del dia siguiente

df['Date']= df['Date'] + pd.DateOffset(days_offset)

#Prediccion por dia completo
if merge_by_day : 
	df = df.groupby(by=['FECHA','Date']).mean()
	df = df.sort_values(by=['FECHA'])

# Join por fecha para unir los registros de la Bolsa con los registros de topics
df_merge = pd.merge(df_bolsa,df,on='Date')

df_merge.insert(0, 'y', df_merge.apply(lambda row: 1 if row['Open'] > row['Close'] else 0, axis=1))

y = df_merge['y'].values
X =df_merge.loc[:,'1':].values

print("empezando a entrenar modelos!")




#X, y = load_iris(return_X_y=True)
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# Historical Split (last records for testing)
p = 0.3
test_size = int(X.shape[0] * p)
train_size = X.shape[0] - test_size

X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]


from sklearn.model_selection import GridSearchCV
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, expon
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


models = [
    ("Dummy", DummyClassifier(strategy='most_frequent'),{}),
    ("kneighbors", KNeighborsClassifier(), {'n_neighbors': [5,10,15,25,35,50,100]}),
    # ("svc", SVC(kernel="linear", gamma="auto"), {'C': expon(loc=0.0001, scale=100)}), # https://stats.stackexchange.com/questions/43943/which-search-range-for-determining-svm-optimal-c-and-gamma-parameters
    ("decissiontree", DecisionTreeClassifier(), {'max_depth': [5,10,15,25,35,50,100]}),
    ("randomforest", RandomForestClassifier(), {'max_depth': [5,10,15,25,35,50,100], 'n_estimators': [10,30,100], 'max_features': [1,2,5,10]}),
    ("adaboost", AdaBoostClassifier(), {}),
    ("gaussiannb", GaussianNB(), {}),
    ("quadraticdiscriminant", QuadraticDiscriminantAnalysis(), {}),
]

n_iters = 100

results = []

for model_name, model, params_distrib in models:
    print(f"training {model_name}")
    search = GridSearchCV(model, params_distrib, cv=5, scoring='precision')
    search.fit(X_train, y_train)
    print(search.best_params_)
    print(search.best_score_)
    print(search.best_estimator_)
    y_pred = search.best_estimator_.predict(X_test)
    test_precision = precision_score(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    results.append({
        'model_name': model_name,
        'best_estimator': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'test_precision': precision_score(y_test, y_pred),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_confusion_matrix': confusion_matrix(y_test, y_pred),
        # 'cv_results': search.cv_results,
    })
	
	
df_results = pd.DataFrame(results)
output_name= f'Results_{n_topics}_{n_words}_{days_offset}_{merge_by_day}.csv'
df_results.to_csv(output_name,sep=';', index=None)


