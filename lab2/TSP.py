import numpy as np
from solution_utils import evaluate_solution, generate_solution, validate_solution
POP_SIZE = 100


class Route:
    def __init__(self, solution, fitness):
        self.solution = solution
        self.fitness = fitness

class TSP:
    def __init__(self, data):
        self.data = data
        self.population = np.array([])
        self.best_solution = None

    def generate_population(self):
        for _ in range(POP_SIZE):
            solution = generate_solution(self.data)
            fitness = evaluate_solution(self.data, solution)
            self.population.append(Route(solution, fitness))

    def rulete_selection(self, fitness_sum):
        total_prob = 0
        cross_prob = np.random.random()
        for individual in self.population:
            total_prob+=(individual.fitness/fitness_sum)
            if cross_prob <= total_prob:
                return individual
            
    def crossover(self, parent1, parent2):
        key = np.random.randint(1, len(parent1.solution)-1)
        solution = parent1.solution.copy()[:key]
        for param in parent2.solution.copy()[:-1]:
            if param not in solution:
                solution.append(param)
        solution.append(parent2.solution.copy()[-1])
        validate_solution(self.data, solution)
        return Route(solution, evaluate_solution(self.data, solution))



    def generational_succession(self):
        new_population = np.array([])
        fitness_sum = sum(individual.fitness for individual in self.population)
        for _ in range(POP_SIZE):
            parent1 = self.rulete_selection(fitness_sum)
            parent2 = self.rulete_selection(fitness_sum)
            child = self.crossover(parent1, parent2)
            new_population.append(child)
        return new_population
                


    def mutation(self):
        pass
    
    def TSP_run(self, generations):
        self.generate_population()
        # self.population = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)
        # self.best_solution = self.population[0]
        self.best_individual = sorted(self.population, key=lambda individual: individual.fitness)[0]

        for generation in range(generations):
            self.population = self.generational_succession()
            self.mutation()
            sorted_population = sorted(self.population, key=lambda individual: individual.fitness)[0]
            if self.best_individual.fitness > sorted_population[0]:
                self.best_individual = sorted_population[0]
