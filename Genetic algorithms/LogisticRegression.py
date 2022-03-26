import pandas as pd
from sklearn.model_selection import train_test_split
from fitnesRandomForest import *
from sklearn.linear_model import LogisticRegression

from deap import base
from deap import creator
from deap import tools

import multiprocessing
import random
import numpy as np
import time
import math
from sklearn.tree import DecisionTreeClassifier


def LRParametersFeatures(numberFeatures, icls):
    genome = list()
    # C
    genome.append(random.uniform(0.01, 100.0))
    # solver
    genome.append(random.randint(0, 3))
    # fit_intercept
    genome.append(random.randint(0, 1))
    # class_weight
    genome.append(random.randint(0, 1))


    return icls(genome)


def LREvaluation(X_train, X_test, y_train, y_test, df_reduced, individual):
    solver_arr = ['newton-cg', 'lbfgs', 'liblinear', 'saga']
    fit_intercept_arr = [True, False]
    class_weight_arr = ['balanced', None]

    estimator = LogisticRegression(C=individual[0],
                                   solver=solver_arr[individual[1]],
                                   dual=False,
                                   fit_intercept=fit_intercept_arr[individual[2]],
                                   class_weight=class_weight_arr[individual[3]],
                                   max_iter=10000)

    estimator.fit(X_train, y_train)
    result = estimator.predict(X_test)

    ScoreCV = cross_val_score(estimator, df_reduced.iloc[:, 0:10], df_reduced.iloc[:, 10], cv=5)

    return ScoreCV.sum() / 5


def LRmutation(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer == 0:
        individual[0] = random.uniform(0.01, 100.0)
    elif numberParamer == 1:
        individual[1] = random.randint(0, 3)
    elif numberParamer == 2:
        individual[2] = random.randint(0, 1)
    elif numberParamer == 3:
        individual[3] = random.randint(0, 1)



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("mate", tools.cxUniform,indpb=0.7)
toolbox.register("select", tools.selTournament,tournsize=3)

if __name__ == '__main__':
    print('Logistic Regression')
    df = pd.read_pickle("dfReduced.pkl")

    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:10], df.iloc[:, 10], test_size=0.265)

    sizePopulation = 500
    probabilityMutation = 0.6
    probabilityCrossover = 0.8
    numberIteration = 500
    numberElitism = 1
    processes = 32
    numberOfAtributtes = 7

    # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # creator.create("Individual", list, fitness=creator.FitnessMax)
    # toolbox = base.Toolbox()

    # zrównoleglenie
    pool = multiprocessing.Pool(processes)
    toolbox.register("map", pool.map)

    toolbox.register('individual', LRParametersFeatures, numberOfAtributtes, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", LREvaluation, X_train, X_test, y_train, y_test, df)

    # SELEKCJA
    # toolbox.register("select", tools.selTournament, tournsize=3)

    # KRZYŻOWANIE
    # toolbox.register("mate", tools.cxOnePoint)

    toolbox.register("mutate", LRmutation)

    pop = toolbox.population(n=sizePopulation)
    # fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(pop, fitnesses):
        ind.fitness = fit

    start = time.time()
    g = 0
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
