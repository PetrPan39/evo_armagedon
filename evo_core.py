import numpy as np

class EvoCore:
    def __init__(self, pop_size=128, dim=24, mutation_rate=0.1, elitism=0.1, fitness_func=None):
        self.pop_size = pop_size
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.fitness_func = fitness_func or self.default_fitness
        self.population = np.random.randint(0, 24, (pop_size, dim))
        self.fitness = np.zeros(pop_size)

    def default_fitness(self, vec):
        return np.sum(vec)

    def evaluate(self):
        self.fitness = np.array([self.fitness_func(indiv) for indiv in self.population])

    def select(self):
        num_elite = int(self.pop_size * self.elitism)
        elite_idx = np.argsort(self.fitness)[-num_elite:]
        elite = self.population[elite_idx]
        rest_size = self.pop_size - num_elite
        rest = []
        for _ in range(rest_size):
            a, b = np.random.randint(0, self.pop_size, 2)
            winner = self.population[a] if self.fitness[a] > self.fitness[b] else self.population[b]
            rest.append(winner)
        return np.vstack((elite, rest))

    def mutate(self, pop):
        for i in range(pop.shape[0]):
            for j in range(self.dim):
                if np.random.rand() < self.mutation_rate:
                    pop[i, j] = np.random.randint(0, 24)
        return pop

    def step(self):
        self.evaluate()
        new_pop = self.select()
        new_pop = self.mutate(new_pop)
        self.population = new_pop

    def best(self):
        idx = np.argmax(self.fitness)
        return self.population[idx], self.fitness[idx]