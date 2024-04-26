#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(color_codes = True)


# #Dataset :
# https://www.kaggle.com/datasets/abhia1999/chronic-kidney-disease

# In[2]:


df = pd.read_csv('new_model.csv')
df


# #Exploratory Data Analysis

# In[3]:


sns.countplot(data=df, x="Htn", hue="Class")


# In[4]:


sns.countplot(data=df, x="Rbc", hue="Class")


# In[5]:


sns.histplot(data=df, x="Bp", hue="Class", multiple="stack")


# #Data Preprocessing

# In[6]:


df.isnull().sum()


# In[7]:


#replace 0 value with NaN
df_copy = df.copy(deep = True) #deep = True -> Buat salinan indeks dan data dalam dataframe
df_copy[['Bp','Sg','Bu','Sc','Sod','Pot','Hemo','Wbcc','Rbcc']] = df_copy[['Bp','Sg','Bu','Sc','Sod','Pot','Hemo','Wbcc','Rbcc']].replace(0,np.NaN)

# Showing the Count of NANs
print(df_copy.isnull().sum())


# #Check if the class label is balanced or not

# In[8]:


sns.countplot(data=df, x="Class", hue="Class")
print(df.Class.value_counts())


# #Do Oversampling Minority Class to Balance the class label

# In[9]:


from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = df[(df['Class']==1)]
df_minority = df[(df['Class']==0)]
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 n_samples= 250, 
                                 random_state=0)  
                                                  
# Combine majority class with upsampled minority class
df2 = pd.concat([df_minority_upsampled, df_majority])


# In[10]:


sns.countplot(data=df2, x="Class", hue="Class")
print(df2.Class.value_counts())


# #Check the Outlier using Boxplot

# In[11]:


sns.boxplot(x=df2["Bp"])


# In[12]:


sns.boxplot(x=df2["Sg"])


# In[13]:


sns.boxplot(x=df2["Bu"])


# In[14]:


sns.boxplot(x=df2["Sc"])


# In[15]:


sns.boxplot(x=df2["Sod"])


# In[16]:


sns.boxplot(x=df2["Pot"])


# In[17]:


sns.boxplot(x=df2["Hemo"])


# In[18]:


sns.boxplot(x=df2["Wbcc"])


# In[19]:


sns.boxplot(x=df2["Rbcc"])


# #Remove Outlier using Z-Score

# In[20]:


import scipy.stats as stats
z = np.abs(stats.zscore(df2))
data_clean = df2[(z<3).all(axis=1)]
data_clean.shape


# #Heatmap Data Correlation

# In[21]:


sns.heatmap(data_clean.corr(), fmt='.2g')


# In[22]:


#Rbc attribute is irrlevant, so we have to remove it
data_clean2 = df.drop(columns=['Rbc'])


# In[23]:


corr = data_clean2[data_clean2.columns[1:]].corr()['Class'][:-1]
plt.plot(corr)
plt.xticks(rotation=90)
plt.show()


# #Machine Learning Model Building

# In[24]:


X = data_clean2.drop('Class', axis=1)
y = data_clean2['Class']


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=43)


# #Random Forest
# 

# In[26]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)


# In[27]:


from sklearn.metrics import accuracy_score
y_pred = rfc.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")


# In[28]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))


# In[29]:


from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(rfc.score(X_test, y_test)*100)
plt.title(all_sample_title, size = 15)


# #KNearest Neighbor

# In[30]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# In[31]:


y_pred = knn.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")


# 

# In[32]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))


# In[33]:


from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(knn.score(X_test, y_test)*100)
plt.title(all_sample_title, size = 15)


# #AdaBoost

# In[34]:


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(random_state=0)
ada.fit(X_train, y_train)


# In[35]:


y_pred = ada.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")


# In[36]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))


# In[37]:


from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(ada.score(X_test, y_test)*100)
plt.title(all_sample_title, size = 15)


# #Logistic Regression

# In[38]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 0)
lr.fit(X_train, y_train)


# In[39]:


y_pred = lr.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")


# In[40]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))


# In[41]:


from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(lr.score(X_test, y_test)*100)
plt.title(all_sample_title, size = 15)


# In[ ]:




