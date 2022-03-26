import pandas as pd
from sklearn.model_selection import train_test_split

from deap import base
from deap import creator
from deap import tools

import multiprocessing
import random
import numpy as np
import time
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier


def RfcParametersFeatures(numberFeatures, icls):
    genome = list()
    # --- Random forest

    # n_estimators
    genome.append(random.randint(10, 100)) # 0
    # max_depth
    genome.append(random.randint(1, 10)) # 1
    # min_samples_split
    genome.append(random.randint(2, 100)) # 2
    # min_samples_leaf
    genome.append(random.randint(2, 100)) # 3
    # max_leaf_nodes
    genome.append(random.randint(2, 20)) # 4
    # max_features
    genome.append(random.uniform(0.01, 1)) # 5
    # criterion
    genome.append(random.randint(0, 1)) # 6

    # ---- Decision Tree

    # max_depth
    genome.append(random.randint(1, 10)) # 7
    # min_samples_split
    genome.append(random.randint(2, 100)) # 8
    # min_samples_leaf
    genome.append(random.randint(2, 100)) # 9
    # max_leaf_nodes
    genome.append(random.randint(2, 20)) # 10
    # max_features
    genome.append(random.uniform(0.01, 1)) # 11
    # criterion
    genome.append(random.randint(0, 1)) # 12

    # ---- Logistic Regression

    # C
    genome.append(random.uniform(0.01, 100.0)) # 13
    # solver
    genome.append(random.randint(0, 3)) # 14
    # fit_intercept
    genome.append(random.randint(0, 1)) # 15
    # class_weight
    genome.append(random.randint(0, 1)) # 16

    return icls(genome)


def Rfcmutation(individual):
    numberParamer = random.randint(0, len(individual) - 1)
    if numberParamer == 0:
        individual[0] = random.randint(50, 100)
    elif numberParamer == 1:
        individual[1] = random.uniform(0.1, 10)
    elif numberParamer == 2:
        individual[2] = random.randint(2, 100)
    elif numberParamer == 3:
        individual[3] = random.randint(2, 100)
    elif numberParamer == 4:
        individual[4] = random.randint(2, 20)
    elif numberParamer == 5:
        individual[5] = random.uniform(0.01, 1)
    elif numberParamer == 6:
        individual[6] = random.randint(0, 1)
    elif numberParamer == 7:
        individual[7] = random.uniform(0.1,10)
    elif numberParamer == 8:
        individual[8] = random.randint(2, 100)
    elif numberParamer == 9:
        individual[9] = random.randint(2, 100)
    elif numberParamer == 10:
        individual[10] = random.randint(2, 20)
    elif numberParamer == 11:
        individual[11] = random.uniform(0.01, 1)
    elif numberParamer == 12:
        individual[12] = random.randint(0, 1)
    elif numberParamer == 13:
        individual[13] = random.uniform(0.01, 100.0)
    elif numberParamer == 14:
        individual[14] = random.randint(0, 3)
    elif numberParamer == 15:
        individual[15] = random.randint(0, 1)
    elif numberParamer == 16:
        individual[16] = random.randint(0, 1)


def RfcEvaluation(X_train, X_test, y_train, y_test, df_reduced, individual):
    criterion = ['gini', 'entropy']

    rf = RandomForestClassifier(n_estimators=individual[0], max_depth=individual[1],
                                       min_samples_split=individual[2], min_samples_leaf=individual[3],
                                       max_leaf_nodes=individual[4], max_features=individual[5],
                                       criterion=criterion[individual[6] % 3], n_jobs=1)

    criterion = ['gini', 'entropy']

    dt = DecisionTreeClassifier(max_depth=individual[7], min_samples_split=individual[8],
                                       min_samples_leaf=individual[9],
                                       max_leaf_nodes=individual[10], max_features=individual[11],
                                       criterion=criterion[individual[12] % 3])

    solver_arr = ['newton-cg', 'lbfgs', 'liblinear', 'saga']
    fit_intercept_arr = [True, False]
    class_weight_arr = ['balanced', None]

    lr = LogisticRegression(C=individual[13],
                                   solver=solver_arr[individual[14]],
                                   dual=False,
                                   fit_intercept=fit_intercept_arr[individual[15]],
                                   class_weight=class_weight_arr[individual[16]],
                                   max_iter=10000)

    estimators = [
        ('rf', rf),
        ('svc', SVC(C=20.44427,
                    kernel='linear',
                    coef0=0.58,
                    shrinking=False,
                    decision_function_shape='ovo',
                    degree=3,
                    gamma='scale')),
        ('dt', dt)
    ]
    estimator = StackingClassifier(
        estimators=estimators, final_estimator=lr
    )

    # print(estimator)

    estimator.fit(X_train, y_train)
    result = estimator.predict(X_test)

    ScoreCV = cross_val_score(estimator, df_reduced.iloc[:, 0:10], df_reduced.iloc[:, 10], cv=5, n_jobs=1)

    return ScoreCV.sum() / 5




creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("mate", tools.cxUniform,indpb=0.7)
toolbox.register("select", tools.selTournament,tournsize=3)

if __name__ == '__main__':
    print('stacking')
    df = pd.read_pickle("dfReduced.pkl")

    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:10], df.iloc[:, 10], test_size=0.265)

    sizePopulation = 100
    probabilityMutation = 0.6
    probabilityCrossover = 0.8
    numberIteration = 100
    numberElitism = 1
    processes = 32
    numberOfAtributtes = 17

    # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # creator.create("Individual", list, fitness=creator.FitnessMax)
    # toolbox = base.Toolbox()

    # zrównoleglenie
    pool = multiprocessing.Pool(processes)
    toolbox.register("map", pool.map)

    toolbox.register('individual', RfcParametersFeatures, numberOfAtributtes, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", RfcEvaluation, X_train, X_test, y_train, y_test, df)

    # SELEKCJA
    # toolbox.register("select", tools.selTournament, tournsize=3)

    # KRZYŻOWANIE
    # toolbox.register("mate", tools.cxOnePoint)

    toolbox.register("mutate", Rfcmutation)

    pop = toolbox.population(n=sizePopulation)
    # fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(pop, fitnesses):
        ind.fitness = fit

    start = time.time()
    g = 0
    print('Genetic loop')
    while g < numberIteration:
        g = g + 1
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        listElitism = []
        for x in range(0, numberElitism):
            listElitism.append(tools.selBest(pop, 1)[0])
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < probabilityCrossover:
                toolbox.mate(child1, child2)
                child1.fitness = 0.0
                child2.fitness = 0.0
        for mutant in offspring:
            if random.random() < probabilityMutation:
                toolbox.mutate(mutant)
                mutant.fitness = 0.0
        invalid_ind = [ind for ind in offspring if ind.fitness == 0.0]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness = fit
        pop[:] = offspring + listElitism
        fits = [ind.fitness for ind in pop]

        best_ind = tools.selBest(pop, 1)[0]
        print('Epoka ' + str(g))
        print(best_ind)
        print(best_ind.fitness)
    end = time.time()
    print("Gen %s, Best individual is %s, %s in %s s" % (g, best_ind, best_ind.fitness, (end - start)))
