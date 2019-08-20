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

for dataset in [train_df, test_df]:
    dataset['Embarked'].fillna('S', inplace=True)

    dataset['Fare'].fillna(0, inplace=True)
    dataset['Fare'] = dataset['Fare'].astype(int)

    dataset['Sex'] = dataset['Sex'].map(genders)
    dataset['Sex'] = dataset['Sex'].astype(int)

    dataset['Embarked'] = dataset['Embarked'].map(port)

    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[dataset['Age'] > 66, 'Age'] = 6

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop('PassengerId', axis=1).copy()

seed = 7
scoring = 'accuracy'

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

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_prediction
    })
submission.to_csv('submission.csv', index=False)