import pandas as pd
from sklearn.model_selection import train_test_split
from fitnesRandomForest import *

from deap import base
from deap import creator
from deap import tools

import multiprocessing
import random
import numpy as np
import time
import math
from sklearn.tree import DecisionTreeClassifier


def DTParametersFeatures(numberFeatures, icls):
    genome = list()
    # max_depth
    genome.append(random.randint(1 ,10))
    # min_samples_split
    genome.append(random.randint(2, 100))
    # min_samples_leaf
    genome.append(random.randint(2, 100))
    # max_leaf_nodes
    genome.append(random.randint(2, 20))
    # max_features
    genome.append(random.uniform(0.01, 1))
    # criterion
    genome.append(random.randint(0, 1))

#     for i in range(0,numberFeatures):
#         genome.append(random.randin(0, 2))

    return icls(genome)


def DTEvaluation(X_train, X_test, y_train, y_test, df_reduced, individual):
    #     print('evaluate')
    criterion = ['gini', 'entropy']

    estimator = DecisionTreeClassifier(max_depth=individual[0], min_samples_split=individual[1],
                                       min_samples_leaf=individual[2],
                                       max_leaf_nodes=individual[3], max_features=individual[4],
                                       criterion=criterion[individual[5] % 3])

    estimator.fit(X_train, y_train)
    result = estimator.predict(X_test)

    ScoreCV = cross_val_score(estimator, df_reduced.iloc[:, 0:10], df_reduced.iloc[:, 10], cv=5)

    return ScoreCV.sum() / 5


def DTmutation(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer == 0:
        individual[0]=random.randint(1 ,10)
    elif numberParamer == 1:
        individual[1]=random.randint(2, 100)
    elif numberParamer == 2:
        individual[2]=random.randint(2, 100)
    elif numberParamer == 3:
        individual[3] = random.randint(2, 20)
    elif numberParamer == 4:
        individual[4] = random.uniform(0.01, 1)
    elif numberParamer == 5:
        individual[5] = random.randint(0, 1)



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("mate", tools.cxUniform,indpb=0.7)
toolbox.register("select", tools.selTournament,tournsize=3)

if __name__ == '__main__':
    print('Decision tree')
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

    toolbox.register('individual', DTParametersFeatures, numberOfAtributtes, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", DTEvaluation, X_train, X_test, y_train, y_test, df)

    # SELEKCJA
    # toolbox.register("select", tools.selTournament, tournsize=3)

    # KRZYŻOWANIE
    # toolbox.register("mate", tools.cxOnePoint)

    toolbox.register("mutate", DTmutation)

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
