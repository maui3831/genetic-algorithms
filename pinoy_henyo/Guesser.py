"""
Guesser Module - Genetic Algorithm Implementation
Chromosome Encoding: String Encoding (ASCII Representation)
Parent Selection: Tournament Selection
Crossover: Single Point Crossover (Probability: 0.8)
Mutation: Gaussian Mutation (Probability: 0.1-0.4)
Stopping Condition: Maximum generation (100 generations)
"""

import random
import string
import math

class guesser:
    def __init__(self, target_word, population_size=50, max_generations=100, game_master=None):
        self.target_word = target_word
        self.word_length = len(target_word)
        self.population_size = population_size
        self.max_generations = max_generations
        self.game_master = game_master

        # Genetic Algorithm Parameters
        self.crossover_rate = 0.8
        self.mutation_counts = [0, 1, 2, 3]
        self.mutation_rates = [4/10, 3/10, 2/10, 1/10]

        # Character set for chromosome encoding
        self.alphabet = string.ascii_lowercase + string.ascii_uppercase

        # Population and fitness tracking
        self.population = []
        self.fitness_scores = []
        self.current_generation = 0
        self.best_individual = None
        self.best_fitness = float('inf')

    def create_individual(self):
        return ''.join(random.choices(self.alphabet, k=self.word_length))

    def initialize_population(self):
        self.population = [self.create_individual() for _ in range(self.population_size)]
        self.evaluate_population()

    def evaluate_population(self):
        self.fitness_scores = []
        for individual in self.population:
            fitness = self.game_master.compute_cost(individual)
            self.fitness_scores.append(fitness)

            # Track best individual
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual

    def tournament_selection(self, tournament_size=3):
        """
        Tournament Selection: Select parent through tournament
        Randomly select tournament_size individuals and return the best one
        """
        # Random selection of tournament participants
        tournament_indices = random.choices(range(len(self.population)), k=tournament_size)

        # Find the best individual in the tournament (lowest cost)
        best_index = tournament_indices[0]
        for idx in tournament_indices[1:]:
            if self.fitness_scores[idx] < self.fitness_scores[best_index]:
                best_index = idx

        return self.population[best_index]

    def single_point_crossover(self, parent1, parent2):
        """
        Single Point Crossover: Create offspring by crossing over at a random point
        """
        if len(parent1) != len(parent2):
            return parent1, parent2  # Return parents if lengths don't match

        if len(parent1) <= 1:
            return parent1, parent2  # Can't crossover with length 1

        crossover_point = random.randint(1, len(parent1) - 1)

        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

        return offspring1, offspring2

    def gaussian_mutation(self, individual):
        """
        Gaussian Mutation: Apply small random changes to characters
        Mutation count is probabilistically determined
        """
        mutation_count = random.choices(self.mutation_counts, weights=self.mutation_rates, k=1)[0]

        if mutation_count == 0 or len(individual) < 1:
            return individual

        mutated = list(individual)

        for _ in range(mutation_count):
            # Select random position to mutate
            position = random.randint(0, len(individual) - 1)

            # Apply Gaussian mutation (small random change)
            current_ascii = ord(mutated[position])

            # Gaussian noise with mean=0, std=2
            noise = int(random.gauss(0, 2))
            new_ascii = current_ascii + noise

            # Keep within printable ASCII range (32-126)
            new_ascii = max(32, min(126, new_ascii))

            mutated[position] = chr(new_ascii)

        return ''.join(mutated)

    def create_next_generation(self):
        new_population = []

        # Keep the best individual
        best_idx = self.fitness_scores.index(min(self.fitness_scores))
        new_population.append(self.population[best_idx])

        # Generate the rest of the population
        while len(new_population) < self.population_size:
            # Parent selection using tournament selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Crossover
            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self.single_point_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1, parent2

            # Mutation
            offspring1 = self.gaussian_mutation(offspring1)
            offspring2 = self.gaussian_mutation(offspring2)

            # Add offspring to new population
            new_population.extend([offspring1, offspring2])

        # Trim population to exact size
        self.population = new_population[:self.population_size]

    def evolve(self):
        """
        Main evolution loop - Run the genetic algorithm
        """
        print(f"Target word length: {self.word_length}")
        print("-" * 50)

        self.game_master.start_timing()

        self.initialize_population()

        # Evolution loop
        for generation in range(1, self.max_generations + 1):
            self.current_generation = generation

            # Record and display current generation
            self.game_master.record_generation(generation, self.best_individual, self.best_fitness)
            self.game_master.display_generation(generation, self.best_individual, self.best_fitness)

            # Check if solution found
            if self.best_fitness == 0:
                print(f"\nSolution found at generation {generation}!")
                print("-" * 50)
                break

            # Create next generation
            self.create_next_generation()
            self.evaluate_population()

        self.game_master.stop_timing()

        self.game_master.display_final_results()

        return self.best_individual

    def get_statistics(self):
        if not self.fitness_scores:
            return {}

        return {
            'generation': self.current_generation,
            'best_fitness': min(self.fitness_scores),
            'average_fitness': sum(self.fitness_scores) / len(self.fitness_scores),
            'worst_fitness': max(self.fitness_scores),
            'population_size': len(self.population)
        }