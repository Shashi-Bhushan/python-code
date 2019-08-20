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

# machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
train_df.drop(['PassengerId', 'Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
test_df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

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
train_df.loc[train_df['Age'] > 66, 'Age'] = 6

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
test_df.loc[test_df['Age'] > 66, 'Age'] = 6

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop('PassengerId', axis=1).copy()

seed = 7
scoring = 'recall'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('RF', RandomForestClassifier(n_estimators=100)))
models.append(('NB', GaussianNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_prediction
    })
submission.to_csv('submission.csv', index=False)