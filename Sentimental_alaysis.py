#!/usr/bin/env python
# coding: utf-8

# # Advanced Sentiment Analysis Pipeline
# 1. Data Filtering: Filter data to include only tweets from May 2009.
# 2. Enhanced Data Visualization: Generate additional visualizations.
# 3. Advanced Data Preprocessing: Text cleaning and TF-IDF feature extraction with more parameters.
# 4. Advanced Model Training: Hyperparameter tuning using GridSearchCV.
# 5. Model Evaluation: Comprehensive evaluation metrices

# In[2]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[6]:


# Step 1: Data Filtering
# Load dataset and filter for May 2009
df = pd.read_csv(r"C:\Users\ajayg\Desktop\University\Advanced Project 1\training.1600000.processed.noemoticon.csv", encoding='ISO-8859-1', names=['sentiment', 'id', 'date', 'query', 'user', 'text'])
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df_may_2009 = df[(df['date'] >= '2009-05-01') & (df['date'] <= '2009-05-31')]
df.head()


# In[7]:


# Step 2: Enhanced Data Visualization
# Distribution of sentiments
plt.figure(figsize=(10, 6))
sns.countplot(data=df_may_2009, x='sentiment')
plt.title('Distribution of Sentiments in May 2009')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()


# In[8]:


# Time-series plot
plt.figure(figsize=(15, 8))
df_may_2009.set_index('date').resample('D').size().plot(label='Total', legend=True)
df_may_2009[df_may_2009['sentiment'] == 0].set_index('date').resample('D').size().plot(label='Negative', legend=True)
df_may_2009[df_may_2009['sentiment'] == 4].set_index('date').resample('D').size().plot(label='Positive', legend=True)
plt.title('Number of Tweets Over May 2009')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.show()


# In[9]:


# Step 3: Advanced Data Preprocessing
# Text cleaning
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.lower()
df_may_2009['clean_text'] = df_may_2009['text'].apply(clean_text)


# In[10]:


# TF-IDF Transformation
X_may = df_may_2009['clean_text']
y_may = df_may_2009['sentiment']
X_train_may, X_test_may, y_train_may, y_test_may = train_test_split(X_may, y_may, test_size=0.2, random_state=42)
advanced_tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3, max_df=0.9)
X_train_tfidf_may = advanced_tfidf_vectorizer.fit_transform(X_train_may)
X_test_tfidf_may = advanced_tfidf_vectorizer.transform(X_test_may)


# In[11]:


# Step 4: Advanced Model Training
# Hyperparameter tuning for Logistic Regression
logistic_clf = LogisticRegression(random_state=42)
param_grid_logistic = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
grid_search_logistic = GridSearchCV(logistic_clf, param_grid_logistic, cv=3, n_jobs=-1, verbose=1)
grid_search_logistic.fit(X_train_tfidf_may, y_train_may)
best_params_logistic = grid_search_logistic.best_params_


# In[12]:


# Hyperparameter tuning for Bernoulli Naive Bayes
bernoulli_clf = BernoulliNB()
param_grid_bernoulli = {'alpha': [0.5, 1, 2], 'fit_prior': [True, False]}
grid_search_bernoulli = GridSearchCV(bernoulli_clf, param_grid_bernoulli, cv=3, n_jobs=-1, verbose=1)
grid_search_bernoulli.fit(X_train_tfidf_may, y_train_may)
best_params_bernoulli = grid_search_bernoulli.best_params_


# In[13]:


# Training models with best hyperparameters
best_logistic_clf = LogisticRegression(**best_params_logistic, random_state=42)
best_bernoulli_clf = BernoulliNB(**best_params_bernoulli)
simple_mlp_clf = MLPClassifier(random_state=42, max_iter=5)  # Simple neural network
models = {'Optimized Logistic Regression': best_logistic_clf, 'Optimized Bernoulli Naive Bayes': best_bernoulli_clf, 'Simple Neural Networks': simple_mlp_clf}
performance = {}
for name, model in models.items():
    model.fit(X_train_tfidf_may, y_train_may)
    y_pred = model.predict(X_test_tfidf_may)
    accuracy = accuracy_score(y_test_may, y_pred)
    classification_rep = classification_report(y_test_may, y_pred, target_names=['Negative', 'Positive'])
    performance[name] = {'Accuracy': accuracy, 'Classification Report': classification_rep}


# In[14]:


# Step 5: Model Evaluation
# Print performance metrics
for name, metrics in performance.items():
    print(f"{name} - Accuracy: {metrics['Accuracy']}")
    print(f"Classification Report:\n{metrics['Classification Report']}")


# In[ ]:




