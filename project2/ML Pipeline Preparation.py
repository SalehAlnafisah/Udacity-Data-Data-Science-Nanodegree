#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[5]:


# import libraries
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


nltk.download(['punkt', 'wordnet'])


# In[6]:


pd.options.display.max_columns = None


# In[7]:


# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table("disaster_messages", con=engine)


# In[8]:


X = df['message']
Y = df.iloc[:, 4:]


# In[9]:


print(df.shape)
print(Y.shape)


# In[10]:


df[:1]


# In[11]:


Y[:1]


# ### 2. Write a tokenization function to process your text data

# In[12]:


def tokenize(text):
    
    
    """ tokenize text"""
  
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[13]:


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X,Y)


# In[15]:


pipeline.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[16]:


y_pred = pipeline.predict(X_test)


# In[17]:


def test_model(y_test, y_pred):
    
    """
    Function to iterate through columns and call sklearn classification report on each.
    """
    for index, column in enumerate(y_test):
        print(column, classification_report(y_test[column], y_pred[:, index]))


# In[18]:


test_model(y_test, y_pred)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[19]:


pipeline.get_params()


# In[20]:


# specify parameters for grid search 
parameters = {
    'clf__estimator__n_estimators' : [50, 100]
}


# In[21]:


# create grid search object 
cv = GridSearchCV(pipeline, param_grid=parameters)

cv


# In[22]:


cv.fit(X_train, y_train)


# In[23]:


cv.best_params_


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[24]:


y_pred = cv.predict(X_test)


# In[25]:


test_model(y_test, y_pred)


# In[26]:


accuracy = (y_pred == y_test).mean()
accuracy


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# In[ ]:





# ### 9. Export your model as a pickle file

# In[27]:


import pickle
filename = 'model.pkl'
pickle.dump(cv, open(filename, 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




