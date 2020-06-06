#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df=pd.read_csv('MatrizTopics_LDA_index_fecha_1.csv', delimiter=';')

#tipo de dato
#type(df['FECHA'].iloc[0])
#formato de fecha
df['Date']= pd.to_datetime(df['FECHA'],format="%d/%m/%Y")
df = df.sort_values(by=['Date'])
df


# In[2]:


df_bolsa=pd.read_csv('IBEX.csv', delimiter=',')
df_bolsa

#tipo de dato
#type(df_bolsa['Date'].iloc[0])
#formato de fecha
df_bolsa['Date']=pd.to_datetime(df_bolsa['Date'])


# In[3]:


#Prediccion por dia completo
#Preparamos las columnas para agrupar
df= df.iloc[:,1:]
df


# In[4]:


#Predicción del dia siguiente

df['Date']= df['Date'] + pd.DateOffset(1)


# In[3]:


#Prediccion por dia completo
df = df.groupby(by=['FECHA','Date']).mean()
df = df.sort_values(by=['FECHA'])
df


# In[4]:


#Comprobación que la suma del porcentaje de importancia de cada topic supa 1 por columna.
df["sum"] = df[:].sum(axis=1)
df


# In[4]:


# Join por fecha para unir los registros de la Bolsa con los registros de topics
df_merge = pd.merge(df_bolsa,df,on='Date')
df_merge


# In[63]:


df_merge.insert(0, 'y', df_merge.apply(lambda row: 1 if row['Open'] > row['Close'] else 0, axis=1))

df_merge


# In[64]:


y = df_merge['y'].values
X =df_merge.loc[:,'1':].values

df_merge['y'].value_counts()

#505/(505+453)
15439/(15439+13946)


# In[66]:


#X, y = load_iris(return_X_y=True)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier

# Historical Split (last records for testing)
print(X.shape[0])
p = 0.3
test_size = int(X.shape[0] * p)
train_size = X.shape[0] - test_size
print(test_size, train_size)
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, X_train.shape[0]+X_test.shape[0])


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model = DummyClassifier(strategy='most_frequent')
y_pred = model.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print(precision_score(y_test, y_pred, pos_label=1))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[67]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


model=DecisionTreeClassifier(max_depth=100)
#model=RandomForestClassifier(max_depth=100, n_estimators=50, max_features=1)

y_pred = model.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
#print(1-((y_test != y_pred).sum()/ X_test.shape[0]))
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred, pos_label=1))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[10]:


pre=[0,0,0,0,1]
true=[0,0,0,1,1]
confusion_matrix(true,pre)


# In[68]:


import matplotlib.pyplot as plt

xs = [10,15,20,25,30,35,40,45,50,75,100,125,150]
ys = []
for max_depth in xs:
    model=DecisionTreeClassifier(max_depth=max_depth)
    y_pred = model.fit(X_train, y_train).predict(X_test)
    precision= precision_score(y_test, y_pred, pos_label=1)
    ys.append(precision)
    print(precision)
    
plt.scatter(xs,ys)


# ## Grid search Cross-Validation

# In[45]:


from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
#from sklearn.datasets import make_moons
#X, y = make_moons()

model=DecisionTreeClassifier()
param_grid = {'max_depth': [10,30,40,50,75,100,150]}
search = GridSearchCV(model, param_grid, cv=5, scoring='precision')
search.fit(X_train, y_train)


# In[46]:


print(search.cv_results_)
print(search.best_params_)
print(search.best_score_)
print(search.best_estimator_)
print(precision_score(y_test, search.best_estimator_.predict(X_test)))


# In[49]:


from sklearn.model_selection import cross_val_score
cross_val_score(DecisionTreeClassifier(max_depth=100), X_train, y_train, cv=5, scoring='precision')


# ## Randomized Search Cross-Validation

# In[50]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, expon


model=DecisionTreeClassifier()
distributions = {'max_depth': randint(low=5, high=150)}
# distributions = {'max_depth': uniform(loc=5, scale=150)}
search = RandomizedSearchCV(model, distributions, cv=5, scoring='precision', random_state=0, n_iter=10)
search.fit(X_train, y_train)


# In[51]:


print(search.cv_results_)
print(search.best_params_)
print(search.best_score_)
print(search.best_estimator_)
print(precision_score(y_test, search.best_estimator_.predict(X_test)))


# ## Model Selection
# 
# Esto estaría bien correrlo en tu pedazo de ordenador tocho, con n_iters = 1000 o algo asi

# In[70]:


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
    ("kneighbors", KNeighborsClassifier(), {'n_neighbors': randint(low=5, high=150)}),
    # ("svc", SVC(kernel="linear", gamma="auto"), {'C': expon(loc=0.0001, scale=100)}), # https://stats.stackexchange.com/questions/43943/which-search-range-for-determining-svm-optimal-c-and-gamma-parameters
    ("decissiontree", DecisionTreeClassifier(), {'max_depth': randint(low=5, high=150)}),
    ("randomforest", RandomForestClassifier(), {'max_depth': randint(low=5, high=150), 'n_estimators': randint(low=5, high=50), 'max_features': randint(low=1, high=2)}),
    ("adaboost", AdaBoostClassifier(), {}),
    ("gaussiannb", GaussianNB(), {}),
    ("quadraticdiscriminant", QuadraticDiscriminantAnalysis(), {}),
]

n_iters = 10

results = []

for model_name, model, params_distrib in models:
    print(f"training {model_name}")
    search = RandomizedSearchCV(model, params_distrib, cv=5, scoring='precision', random_state=0, n_iter=n_iters)
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


# In[73]:


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

n_iters = 1

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


# In[71]:


df_results = pd.DataFrame(results)
df_results

