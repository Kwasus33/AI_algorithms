import numpy as np
from solution_utils import evaluate_solution, generate_solution, validate_solution
POP_SIZE = 1000
MUTATION_PROB = 0.05
EPS = 1e-6


class Route:
    def __init__(self, solution, evaluation):
        self.solution = solution
        self.evaluation = evaluation
        self.fitness = (1/evaluation)**3 + EPS # fitness is (1/evaluation)**3 - lowest eval means best fitness value and ^2 quarantees bigger bias

class TSP:
    def __init__(self, data):
        self.data = data
        self.population = [None]*POP_SIZE
        self.best_solution = None

    def generate_population(self):
        for _ in range(POP_SIZE):
            solution = generate_solution(self.data)
            evaluation = evaluate_solution(self.data, solution)
            self.population[_] = Route(solution, evaluation)

    def roulette_selection(self, fitness_sum):
        total_prob = 0
        cross_prob = np.random.random()
        for individual in self.population:
            # for individuals of lowest evaluation fitness is highest so it's crossing probability
            total_prob+=(individual.fitness)/fitness_sum
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
        fitness_sum = sum(individual.fitness for individual in self.population)
        for _ in range(POP_SIZE):
            parent1 = self.roulette_selection(fitness_sum)
            parent2 = self.roulette_selection(fitness_sum)
            child = self.crossover(parent1, parent2)
            new_population[_] = child
        return new_population

    def mutation(self):
        pass
    
    def TSP_run(self, generations):
        self.generate_population()
        # self.population = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)
        # self.best_individual = self.population[0]
        self.best_individual = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)[0]

        for generation in range(generations):
            self.population = self.generational_succession()
            # self.population = sorted(self.generational_succession(), key=lambda individual: individual.fitness)
            self.mutation()
            # if self.best_individual.fitness > sorted_population[0].fitness:
            #     self.best_individual = sorted_population[0]
            generation_best_individual = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)[0]
            if self.best_individual.fitness < generation_best_individual.fitness:
                self.best_individual = generation_best_individual

            if generation % 10 == 0:
                for individual in self.population:
                    print(f"{individual.solution}\n")
                print(f"\n\n\n{self.best_individual.solution}, {self.best_individual.evaluation}\n\n\n")

        return self.best_individual
