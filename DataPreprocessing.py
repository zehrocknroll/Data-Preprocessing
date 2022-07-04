# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 00:07:40 2022

@author: zehra
"""

#Kütüphanelerin İmport Edilmesi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Verinin İmport Edilmesi

dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


#Kayıp Data Konusu


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(X[:,1:3]) 
X[:,1:3] = imputer.transform(X[:,1:3])


#Değişkenlerin "OneHotEncoding" Metoduyla Düzeltilmesi 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],
                       remainder="passthrough")

X = np.array(ct.fit_transform(X,y))



#Değişkenlerin "LabelEncoding" Metoduyla Düzeltilmesi

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)


#Train ve Test Setleri

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                test_size=0.2,random_state=1)



#Feature Scaling (Özellik Ölçekleme) Standardization

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train[:,:3] = ss.fit_transform(X[:,:3])
X_test[:,:3] = ss.transform(X[:,:3])














































