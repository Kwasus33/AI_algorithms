from solution_utils import evaluate_solution as fitness
from solution_utils import generate_solution as solution
POP_SIZE = 100


class Route:
    def __init__(self, data):
        self.solution = solution(data)
        self.fitness = fitness(data, self.solution)

class TSP:
    def __init__(self, data):
        self.data = data
        self.population = [Route(self.data) for _ in range(POP_SIZE)]
        self.best_solution = None

    def crossover(self):
        pass

    def mutation(self):
        pass
    
    def TSP_run(self, generations):
        self.population = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)
        self.best_solution = self.population[0]

        for _ in range(generations):
            self.population = self.population[:len(self.population)/2]
            self.crossover()
            self.mutation()
