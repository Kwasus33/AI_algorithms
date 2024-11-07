import numpy as np
from solution_utils import evaluate_solution, generate_solution, validate_solution
POP_SIZE = 1000
MUTATION_PROB = 0.05
SHIFT = 1000


class Route:
    def __init__(self, solution, evaluation):
        self.solution = solution
        self.evaluation = evaluation
        self.fitness = None

    def set_fitness(self, fitness):
        self.fitness = fitness**2
        # fitness is (1/(evaluation - min_eval + EPS)) - lowest eval means best fitness value and ^2 quarantees bigger bias
        # + SHIFT guarantees that fitness for individual with lowest route is not 1/0, it's (1/EPS)**2 so it's highest fitness value
    
    def mutate(self):
        gene1 = np.random.randint(1,len(self.solution)-1)
        gene2 = np.random.randint(1, len(self.solution)-1)
        self.solution[gene1], self.solution[gene2] = self.solution[gene2], self.solution[gene1]

class TSP:
    def __init__(self, data):
        self.data = data
        self.population = [None]*POP_SIZE
        self.best_solution = None

    def calculate_fitnesses(self):
        min_evaluation = min(individual.evaluation for individual in self.population)
        for individual in self.population:
            individual.set_fitness(1/(individual.evaluation - min_evaluation + SHIFT))

    def generate_population(self):
        for _ in range(POP_SIZE):
            solution = generate_solution(self.data)
            evaluation = evaluate_solution(self.data, solution)
            self.population[_] = Route(solution, evaluation)
        self.calculate_fitnesses()

    def generational_succession(self):
        new_population = [None]*POP_SIZE
        for _ in range(POP_SIZE):
            # parent1 = self.roulette_selection_1()
            # parent2 = self.roulette_selection_1()
            parent1, parent2 = self.roulette_selection_2()
            child = self.crossover(parent1, parent2) 
            new_population[_] = self.mutation(child)
        self.population = new_population
        self.calculate_fitnesses()

    def roulette_selection_1(self):
        total_prob = 0
        cross_prob = np.random.random()
        fitness_sum = sum(individual.fitness for individual in self.population)
        for individual in self.population:
            total_prob+=(individual.fitness)/fitness_sum
            if cross_prob <= total_prob:
                return individual
            
    def roulette_selection_2(self):
        fitness_sum = sum(individual.fitness for individual in self.population)
        probs = [individual.fitness/fitness_sum for individual in self.population]
        parent1, parent2 = np.random.choice(self.population, size=2, p=probs)
        return (parent1, parent2)

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

    def mutation(self, child):
        mutation_prob = np.random.random()
        if mutation_prob <= MUTATION_PROB:
            child.mutate()
            validate_solution(self.data, child.solution)
        return child

    def TSP_run(self, generations):
        self.generate_population()
        self.best_individual = sorted(self.population, key=lambda individual: individual.evaluation)[0]
        print(f'First population best individual: {self.best_individual.evaluation}')

        for generation in range(generations):
            self.generational_succession()
            
            generation_best_individual = sorted(self.population, key=lambda individual: individual.evaluation)[0]
            if self.best_individual.evaluation > generation_best_individual.evaluation:
                self.best_individual = generation_best_individual

        return self.best_individual
