from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import time

if __name__ == '__main__':

    df = pd.read_pickle("dfReduced.pkl")
    print('Start')

    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:10], df.iloc[:, 10], test_size=0.265)

    estimators = [('rf', RandomForestClassifier(n_jobs=20)),
                  ('svc', SVC()),
                  ('dt', DecisionTreeClassifier())]

    sclf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=10000, n_jobs=20))

    params = {'rf__n_estimators': [10, 25, 50, 100],
              'rf__criterion': ['gini', 'entropy'],
              'rf__max_depth': [3, 4, 5],
              'rf__max_features': ['auto', 'sqrt', 'log2'],
              'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
              'svc__C': [1.0, 10.0, 25.0],
              'dt__criterion': ['gini', 'entropy'],
              'dt__splitter': ['best', 'random'],
              'dt__max_depth': [3, 4, 5],
              'dt__max_features': ['auto', 'sqrt', 'log2']
              }
    start = time.time()

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, verbose=10, n_jobs=20)
    grid.fit(X_train, y_train)

    end = time.time()

    print('Best classifier:')
    print(grid.best_estimator_)

    scores = cross_val_score(grid.best_estimator_, df.iloc[:, 0:10], df.iloc[:, 10], cv=5, n_jobs=5)
    print('Cross Validaton: ', scores.mean())

    print('Czas optymalizacji w sekundach: ' + str(end - start))