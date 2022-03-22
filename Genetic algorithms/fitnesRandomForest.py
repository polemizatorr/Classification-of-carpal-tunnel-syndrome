from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def RfcEvaluation(X_train, X_test, y_train, y_test, df_reduced, individual):
    criterion = ['gini', 'entropy']

    estimator = RandomForestClassifier(n_estimators=individual[0], max_depth=individual[1],
                                       min_samples_split=individual[2], min_samples_leaf=individual[3],
                                       max_leaf_nodes=individual[4], max_features=individual[5],
                                       criterion=criterion[individual[6] % 3], n_jobs=1)
    # print(estimator)

    estimator.fit(X_train, y_train)
    result = estimator.predict(X_test)

    ScoreCV = cross_val_score(estimator, df_reduced.iloc[:, 0:10], df_reduced.iloc[:, 10], cv=5, n_jobs=1)

    return ScoreCV.sum() / 5