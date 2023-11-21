__author__ = "Nicky Patterson"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "patni711@student.otago.ac.nz"

from copy import deepcopy

import numpy as np
import random

agentName = "<No More Spin>"
trainingSchedule = [("random_agent.py", 225), ("self", 75)]
class Cleaner:

    def __init__(self, nPercepts, nActions, gridSize, maxTurns):
        self.nPercepts = nPercepts
        self.nActions = nActions
        self.gridSize = gridSize
        self.maxTurns = maxTurns
        self.chromosome = np.random.randint(low=-100, high=100, size=(4, 64))

    def AgentFunction(self, percepts):
        visual, energy, bin, fails = percepts

        visual = visual.flatten()
        chromo = np.array(visual)
        chromo = np.concatenate((chromo, [energy, bin, fails]))

        actions = []
        for j in range(4):
            weight = np.dot(self.chromosome[j][:63], chromo)
            bias = self.chromosome[j][-1]
            actions.append(weight + bias)

        return actions


def evalFitness(population):
    N = len(population)
    fitness = np.zeros(N)

    for n, cleaner in enumerate(population):
        #  cleaner.game_stats['cleaned'] - int, total number of dirt loads picked up
        #  cleaner.game_stats['emptied'] - int, total number of dirt loads emptied at a charge station
        #  cleaner.game_stats['active_turns'] - int, total number of turns the bot was active (non-zero energy)
        #  cleaner.game_stats['successful_actions'] - int, total number of successful actions performed during active
        #                                                  turns
        #  cleaner.game_stats['recharge_count'] - int, number of turns spent at a charging station
        #  cleaner.game_stats['recharge_energy'] - int, total energy gained from the charging station
        #  cleaner.game_stats['visits'] - int, total number of squares visited (visiting the same square twice counts
        #                                      as one visit)

        fitnessStat = (cleaner.game_stats['cleaned'] * 2) + (cleaner.game_stats['active_turns']) + (cleaner.game_stats[
            'visits']*cleaner.game_stats['visits']) + (cleaner.game_stats['recharge_energy'] / 4) + (cleaner.
                                                                        game_stats['emptied'] * 2)
        fitness[n] = fitnessStat

    return fitness


def newGeneration(old_population):
    N = len(old_population)
    fitness = evalFitness(old_population)

    old_pop_sorted = sorted(old_population, key=lambda cleaner: fitness[old_population.index(cleaner)], reverse=True)
    old_pop_superior = old_pop_sorted[:(len(old_pop_sorted)//2)]
    new_population = list()

    elitism_percentage = 0.1

    elitism_count = int(elitism_percentage * N)

    for n in range(elitism_count):
        new_cleaner = deepcopy(old_pop_sorted[n])
        new_population.append(new_cleaner)

    for n in range(elitism_count, N):
        parent1 = random.choice(old_pop_superior)
        parent2 = random.choice(old_pop_superior)
        new_cleaner = crossover(parent1, parent2)
        new_population.append(new_cleaner)
        mutation(new_cleaner)

    avg_fitness = np.mean(fitness)

    return new_population, avg_fitness


def crossover(parent1, parent2):
    new_cleaner = Cleaner(parent1.nPercepts, parent1.nActions, parent1.gridSize, parent1.maxTurns)
    crossover_point = np.random.randint(0, len(parent1.chromosome))
    new_cleaner.chromosome[:crossover_point] = parent1.chromosome[:crossover_point]
    new_cleaner.chromosome[crossover_point:] = parent2.chromosome[crossover_point:]

    return new_cleaner


def mutation(cleaner):
    mutation_rate = 0.1
    chromo_mut = random.randint(0, 3)
    for j in range(len(cleaner.chromosome[chromo_mut])):
        if random.uniform(0, 1) < mutation_rate:
            mutation_value = random.randint(-100, 100)
            cleaner.chromosome[chromo_mut][j] = mutation_value

    return cleaner

