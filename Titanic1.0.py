# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 11:46:51 2018

@author: hp
"""
#import libraries
import numpy as np 

import pandas as pd

#import datasets

train = pd.read_csv('train.csv')
train.head()
test = pd.read_csv('test.csv')
test.head()
# count null(missing No.)
train.isnull().sum() # Age= 177 , Cabin= 687 and Embarked = 2

# counting the survived people
train['Survived'].value_counts()            
train["Survived"].value_counts(normalize=True)

#count - Categorical Feature 
# 1- Sex- Male or Female
train["Survived"][train["Sex"]=='male'].value_counts()
train["Survived"][train["Sex"]=='female'].value_counts()
train["Survived"][train["Sex"]=='male'].value_counts(normalize = True)
train["Survived"][train["Sex"]=='female'].value_counts(normalize = True)

#count- Ordinal Feature
#PClass 1 , 2 , 3 with tablular view

train["Survived"][train["Pclass"]==1].value_counts()
train["Survived"][train["Pclass"]==2].value_counts()
train["Survived"][train["Pclass"]==3].value_counts()
pd.crosstab(train.Pclass,train.Survived,margins=True)

train["Survived"][train["Pclass"]==1].value_counts(normalize=True)
train["Survived"][train["Pclass"]==2].value_counts(normalize= True)
train["Survived"][train["Pclass"]==3].value_counts(normalize= True)

# Table for Pclass + Sex 
pd.crosstab([train.Sex,train.Survived],train.Pclass,margins=True)#Pclass1 female survival rate is maximum 91 /94
   
#Age+ Pclass
pd.crosstab([train.Age,train.Survived],train.Pclass,margins=True)#we can observe that no. Child increases with Pclass
#Age + Sex
pd.crosstab([train.Age,train.Survived],train.Sex,margins=True)#we can observe that no. female child survivor are more

#Initials finding
train['Initial']=0
for i in train:
    train['Initial']=train.Name.str.extract('([A-Za-z]+)\.')
    
test['Initial']=0
for i in test:
    test['Initial']=test.Name.str.extract('([A-Za-z]+)\.')
    

pd.crosstab(train.Initial,train.Sex)
# Now observe and change the Initials

train['Initial'].replace(['Mlle','Mme','Ms', 'Dr', 'Major','Capt', 'Sir', 'Don', 'Lady', 'Countess','Rev','Col','Jonkheer'],['Miss','Miss','Miss','Mr','Mr','Mr','Mr','Mr','Mrs','Mrs','Mr','Mr','Mr'],inplace= True)
train.groupby('Initial')['Age'].mean()
test['Initial'].replace(['Mlle','Mme','Ms', 'Dr', 'Major','Capt', 'Sir', 'Don', 'Lady', 'Countess','Rev','Col','Dona'],['Miss','Miss','Miss','Mr','Mr','Mr','Mr','Mr','Mrs','Mrs','Mr','Mr','Mrs'],inplace= True)


train.loc[(train.Age.isnull())&(train.Initial=='Mr'),'Age']=33
train.loc[(train.Age.isnull())&(train.Initial=='Mrs'),'Age']=36
train.loc[(train.Age.isnull())&(train.Initial=='Master'),'Age']=5
train.loc[(train.Age.isnull())&(train.Initial=='Miss'),'Age']=22
         
          
test.loc[(test.Age.isnull())&(test.Initial=='Mr'),'Age']=33
test.loc[(test.Age.isnull())&(test.Initial=='Mrs'),'Age']=36
test.loc[(test.Age.isnull())&(test.Initial=='Master'),'Age']=5
test.loc[(test.Age.isnull())&(test.Initial=='Miss'),'Age']=22
          

train.Age.isnull().any()

     
# Embarked - Categorisal
pd.crosstab([train.Embarked,train.Pclass],[train.Sex,train.Survived],margins=True)

#as Maximum passengers are from Port S
train['Embarked'].fillna('S',inplace= True )
train.Embarked.isnull().any()
test['Embarked'].fillna('S',inplace= True )
#  Sibsip Survival
pd.crosstab([train.SibSp],train.Survived)
#with Pclass
pd.crosstab(train.SibSp,train.Pclass)

#Parch- Pclass
pd.crosstab(train.Parch,train.Pclass)

# Age - Continuous feature
train["Age_Band"] = 0
train.loc[train['Age']<=16,'Age_Band']=0
train.loc[(train['Age']>16)&(train['Age']<=32),'Age_Band']=1
train.loc[(train['Age']>32)&(train['Age']<=48),'Age_Band']=2
train.loc[(train['Age']>48)&(train['Age']<=64),'Age_Band']=3
train.loc[train['Age']>64,'Age_Band']=4
train["Age_Band"]

test["Age_Band"] = 0
test.loc[test['Age']<=16,'Age_Band']=0
test.loc[(test['Age']>16)&(test['Age']<=32),'Age_Band']=1
test.loc[(test['Age']>32)&(test['Age']<=48),'Age_Band']=2
test.loc[(test['Age']>48)&(test['Age']<=64),'Age_Band']=3
test.loc[test['Age']>64,'Age_Band']=4


#Count for no. of passengers in Age_Band              
train['Age_Band'].value_counts()

#Family_Size and Alone
train['Family_Size']=0
train['Family_Size']=train['Parch']+train['SibSp']#family size
train['Alone']=0
train.loc[train.Family_Size==0,'Alone']=1
         
train["Survived"][train["Family_Size"]].value_counts()  

test['Family_Size']=0
test['Family_Size']=test['Parch']+test['SibSp']#family size
test['Alone']=0
test.loc[test.Family_Size==0,'Alone']=1
           

#fare_range
train['Fare_Range']=pd.qcut(train['Fare'],4)
train.groupby(['Fare_Range'])['Survived'].mean() 

#Fare_cat
train['Fare_cat']=0
train.loc[train['Fare']<=7.91,'Fare_cat']=0
train.loc[(train['Fare']>7.91)&(train['Fare']<=14.454),'Fare_cat']=1
train.loc[(train['Fare']>14.454)&(train['Fare']<=31),'Fare_cat']=2
train.loc[(train['Fare']>31)&(train['Fare']<=513),'Fare_cat']=3
train['Fare_cat']

test['Fare_cat']=0
test.loc[test['Fare']<=7.91,'Fare_cat']=0
test.loc[(test['Fare']>7.91)&(test['Fare']<=14.454),'Fare_cat']=1
test.loc[(test['Fare']>14.454)&(test['Fare']<=31),'Fare_cat']=2
test.loc[(test['Fare']>31)&(test['Fare']<=513),'Fare_cat']=3

          
#Converting String into Numeric
train['Sex'].replace(['male','female'],[0,1],inplace=True)
train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
train['Initial'].replace(['Mr','Mrs','Miss','Master'],[0,1,2,3],inplace=True)

test['Sex'].replace(['male','female'],[0,1],inplace=True)
test['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
test['Initial'].replace(['Mr','Mrs','Miss','Master'],[0,1,2,3],inplace=True)

#dropping features
train.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)

test.drop(['Name','Age','Ticket','Fare','Cabin'],axis=1,inplace=True)

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

Train,Test=train_test_split(train,test_size=0.2,random_state=0,stratify=train['Survived'])
train_X=Train[Train.columns[1:]]
train_Y=Train[Train.columns[:1]]
test_X=Test[Test.columns[1:]]
test_Y=Test[Test.columns[:1]]
X=train[train.columns[1:]]
Y=train['Survived']

#Classifier= 'rbf' -SVM

model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(train_X, train_Y)
prediction1= model.predict(test_X)
print('Accuracy for rbf SVM is ', metrics.accuracy_score(prediction1,test_Y))

#Creating prediction result CSV
Test_X = test.iloc[:,1:]
Test_X.shape
y_pred = model.predict(Test_X)
y_pred.shape
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
print(my_solution)
print(my_solution.shape)

my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])
         
