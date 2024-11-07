import random, numpy as np
from solution_utils import evaluate_solution, generate_solution, validate_solution
POP_SIZE = 1000


class Route:
    def __init__(self, solution, fitness):
        self.solution = solution
        self.fitness = fitness

class TSP:
    def __init__(self, data):
        self.data = data
        self.population = [None]*POP_SIZE
        self.best_solution = None

    def generate_population(self):
        for _ in range(POP_SIZE):
            solution = generate_solution(self.data)
            fitness = evaluate_solution(self.data, solution)
            self.population[_] = Route(solution, fitness)

    def rulete_selection(self, fitness_sum):
        # prob set as 1/fitness - we minimize fitness but maximize prob
        total_prob = 0
        cross_prob = np.random.random()
        for individual in self.population:
            total_prob+=(1/individual.fitness)/fitness_sum
            if cross_prob <= total_prob:
                return individual
            
    # OX - Order Crossover
    def crossover(self, parent1, parent2):
        solution_size = len(parent1.solution)
        solution = [None]*solution_size
        start, end = sorted(np.random.choice(np.arange(solution_size), 2))
        solution[start:end] = parent1.solution[start:end]
        genes = [gene for gene in parent2.solution if gene not in solution]
        pos = 0
        for _ in range(solution_size):
            if solution[_] is None:
                solution[_] = genes[pos]
                pos+=1
        validate_solution(self.data, solution)
        return Route(solution, evaluate_solution(self.data, solution))

    def generational_succession(self):
        new_population = [None]*POP_SIZE
        fitness_sum = sum(1/individual.fitness for individual in self.population)
        for _ in range(POP_SIZE):
            parent1 = self.rulete_selection(fitness_sum)
            parent2 = self.rulete_selection(fitness_sum)
            child = self.crossover(parent1, parent2)
            new_population[_] = child
        return new_population

    def mutation(self):
        pass
    
    def TSP_run(self, generations):
        self.generate_population()
        self.population = sorted(self.population, key=lambda individual: individual.fitness)
        self.best_individual = self.population[0]
        # self.best_individual = sorted(self.population, key=lambda individual: individual.fitness)[0]

        print(self.best_individual.fitness)

        for generation in range(generations):
            # self.population = self.generational_succession()
            self.population = sorted(self.generational_succession(), key=lambda individual: individual.fitness)
            self.mutation()
            # sorted_population = sorted(self.population, key=lambda individual: individual.fitness)
            # if self.best_individual.fitness > sorted_population[0].fitness:
            #     self.best_individual = sorted_population[0]
            if self.best_individual.fitness > self.population[0].fitness:
                self.best_individual = self.population[0]
        
        return self.best_individual
