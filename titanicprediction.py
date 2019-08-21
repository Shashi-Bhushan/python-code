# data analysis
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline

# machine learning
from sklearn import model_selection
from sklearn.decomposition import PCA

# machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv('./dataset/titanic/train.csv')
test_df = pd.read_csv('./dataset/titanic/test.csv')

f,ax = plt.subplots(3,4,figsize=(20,16))
sns.countplot('Pclass',data=train_df,ax=ax[0,0])
sns.countplot('Sex',data=train_df,ax=ax[0,1])
sns.boxplot(x='Pclass',y='Age',data=train_df,ax=ax[0,2])
sns.countplot('SibSp',hue='Survived',data=train_df,ax=ax[0,3],palette='husl')
sns.distplot(train_df['Fare'].dropna(),ax=ax[2,0],kde=False,color='b')
sns.countplot('Embarked',data=train_df,ax=ax[2,2])

sns.countplot('Pclass',hue='Survived',data=train_df,ax=ax[1,0],palette='husl')
sns.countplot('Sex',hue='Survived',data=train_df,ax=ax[1,1],palette='husl')
sns.distplot(train_df[train_df['Survived']==0]['Age'].dropna(),ax=ax[1,2],kde=False,color='r',bins=5)
sns.distplot(train_df[train_df['Survived']==1]['Age'].dropna(),ax=ax[1,2],kde=False,color='g',bins=5)
sns.countplot('Parch',hue='Survived',data=train_df,ax=ax[1,3],palette='husl')
sns.swarmplot(x='Pclass',y='Fare',hue='Survived',data=train_df,palette='husl',ax=ax[2,1])
sns.countplot('Embarked',hue='Survived',data=train_df,ax=ax[2,3],palette='husl')

ax[0,0].set_title('Total Passengers by Class')
ax[0,1].set_title('Total Passengers by Gender')
ax[0,2].set_title('Age Box Plot By Class')
ax[0,3].set_title('Survival Rate by SibSp')
ax[1,0].set_title('Survival Rate by Class')
ax[1,1].set_title('Survival Rate by Gender')
ax[1,2].set_title('Survival Rate by Age')
ax[1,3].set_title('Survival Rate by Parch')
ax[2,0].set_title('Fare Distribution')
ax[2,1].set_title('Survival Rate by Fare and Pclass')
ax[2,2].set_title('Total Passengers by Embarked')
ax[2,3].set_title('Survival Rate by Embarked')

# we can now drop the cabin feature
train_df.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)
test_df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

# 19% Age is missing, lets replace it by mean since it seems important
train_df.Age.fillna(train_df.Age.mean(), inplace=True)
test_df.Age.fillna(test_df.Age.mean(), inplace=True)

genders = {"male": 0, "female": 1}
port = {"S": 0, "C": 1, "Q": 2}

train_df['Embarked'].fillna('S', inplace=True)

train_df['Fare'].fillna(0, inplace=True)
train_df['Fare'] = train_df['Fare'].astype(int)

train_df['Sex'] = train_df['Sex'].map(genders)
train_df['Sex'] = train_df['Sex'].astype(int)

train_df['Embarked'] = train_df['Embarked'].map(port)
df = pd.get_dummies(train_df['Embarked'], drop_first=True)
df.columns = ['Embarked_S', 'Embarked_C']

train_df.drop(['Embarked'], axis=1, inplace=True)
train_df = pd.concat([train_df, df], axis=1)

train_df['Age'] = train_df['Age'].astype(int)
train_df.loc[train_df['Age'] <= 11, 'Age'] = 0
train_df.loc[(train_df['Age'] > 11) & (train_df['Age'] <= 18), 'Age'] = 1
train_df.loc[(train_df['Age'] > 18) & (train_df['Age'] <= 22), 'Age'] = 2
train_df.loc[(train_df['Age'] > 22) & (train_df['Age'] <= 27), 'Age'] = 3
train_df.loc[(train_df['Age'] > 27) & (train_df['Age'] <= 33), 'Age'] = 4
train_df.loc[(train_df['Age'] > 33) & (train_df['Age'] <= 40), 'Age'] = 5
train_df.loc[(train_df['Age'] > 40) & (train_df['Age'] <= 66), 'Age'] = 6
train_df.loc[train_df['Age'] > 66, 'Age'] = 7

train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
train_df.drop(['Name'], axis=1, inplace=True)

df_title = pd.get_dummies(train_df['Title'], drop_first=True)
train_df.drop(['Title'], axis=1, inplace=True)
train_df = pd.concat([train_df, df_title], axis=1)

df = pd.get_dummies(train_df['Age'], drop_first=True)
train_df.drop(['Age'], axis=1, inplace=True)
train_df = pd.concat([train_df, df], axis=1)

train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = 0
train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1

train_df.loc[ train_df['Fare'] <= 7.91, 'Fare'] = 0
train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare'] = 1
train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare']   = 2
train_df.loc[ train_df['Fare'] > 31, 'Fare'] = 3
train_df['Fare'] = train_df['Fare'].astype(int)

df_fare = pd.get_dummies(train_df['Fare'], drop_first=True)
train_df.drop(['Fare'], axis=1, inplace=True)
train_df = pd.concat([train_df, df_fare], axis=1)

###############################


test_df['Embarked'].fillna('S', inplace=True)

test_df['Fare'].fillna(0, inplace=True)
test_df['Fare'] = test_df['Fare'].astype(int)

test_df['Sex'] = test_df['Sex'].map(genders)
test_df['Sex'] = test_df['Sex'].astype(int)

test_df['Embarked'] = test_df['Embarked'].map(port)
df = pd.get_dummies(test_df['Embarked'], drop_first=True)
df.columns = ['Embarked_S', 'Embarked_C']

test_df.drop(['Embarked'], axis=1, inplace=True)
test_df = pd.concat([test_df, df], axis=1)

test_df['Age'] = test_df['Age'].astype(int)
test_df.loc[test_df['Age'] <= 11, 'Age'] = 0
test_df.loc[(test_df['Age'] > 11) & (test_df['Age'] <= 18), 'Age'] = 1
test_df.loc[(test_df['Age'] > 18) & (test_df['Age'] <= 22), 'Age'] = 2
test_df.loc[(test_df['Age'] > 22) & (test_df['Age'] <= 27), 'Age'] = 3
test_df.loc[(test_df['Age'] > 27) & (test_df['Age'] <= 33), 'Age'] = 4
test_df.loc[(test_df['Age'] > 33) & (test_df['Age'] <= 40), 'Age'] = 5
test_df.loc[(test_df['Age'] > 40) & (test_df['Age'] <= 66), 'Age'] = 6
test_df.loc[test_df['Age'] > 66, 'Age'] = 7

test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')
test_df.drop(['Name'], axis=1, inplace=True)

df_title2 = pd.get_dummies(test_df['Title'], drop_first=True)
test_df.drop(['Title'], axis=1, inplace=True)
test_df = pd.concat([test_df, df_title2], axis=1)

df2 = pd.get_dummies(test_df['Age'], drop_first=True)
test_df.drop(['Age'], axis=1, inplace=True)
test_df = pd.concat([test_df, df2], axis=1)

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df['IsAlone'] = 0
test_df.loc[test_df['FamilySize'] == 1, 'IsAlone'] = 1

test_df.loc[ test_df['Fare'] <= 7.91, 'Fare'] = 0
test_df.loc[(test_df['Fare'] > 7.91) & (test_df['Fare'] <= 14.454), 'Fare'] = 1
test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare'] <= 31), 'Fare']   = 2
test_df.loc[ test_df['Fare'] > 31, 'Fare'] = 3
test_df['Fare'] = test_df['Fare'].astype(int)

df_fare2 = pd.get_dummies(test_df['Fare'], drop_first=True)
test_df.drop(['Fare'], axis=1, inplace=True)
test_df = pd.concat([test_df, df_fare2], axis=1)

X_train_df = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test_df  = test_df.drop('PassengerId', axis=1).copy()

pca = PCA(n_components=8)

X_train = pca.fit_transform(X_train_df)

X_test = pca.transform(X_test_df)

kfold = StratifiedKFold(n_splits=10)

# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_train, Y_train)

ada_best = gsadaDTC.best_estimator_


#ExtraTrees
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(X_train, Y_train)

ExtC_best = gsExtC.best_estimator_

# RFC Parameters tunning
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_


# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_


### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'],
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_

votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)

Y_prediction = pd.Series(votingC.predict(X_test), name="Survived")

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_prediction
    })
submission.to_csv('submission.csv', index=False)