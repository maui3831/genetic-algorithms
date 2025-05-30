import numpy as np
import random
from typing import List, Tuple


class Chromosome:
    def __init__(self, route: List[int], distance_matrix: np.ndarray):
        self.route = route
        self.distance_matrix = distance_matrix
        self.fitness = self._calculate_fitness()

    def _calculate_fitness(self) -> float:
        """Calculate the total distance of the route"""
        total_distance = 0
        for i in range(len(self.route)):
            from_city = self.route[i]
            to_city = self.route[(i + 1) % len(self.route)]
            total_distance += self.distance_matrix[from_city][to_city]
        return 1 / total_distance


class GeneticAlgorithm:
    def __init__(
        self,
        distance_matrix: np.ndarray,
        population_size: int = 100,
        tournament_size: int = 5,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
        max_generations: int = 5000,
    ):
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_generations = max_generations
        self.num_cities = len(distance_matrix)

    def initialize_population(self) -> List[Chromosome]:
        """Create initial population with random routes"""
        population = []
        for _ in range(self.population_size):
            route = list(range(self.num_cities))
            random.shuffle(route)
            population.append(Chromosome(route, self.distance_matrix))
        return population

    def tournament_selection(self, population: List[Chromosome]) -> Chromosome:
        """Select parent using tournament selection"""
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def order_crossover(
        self, parent1: Chromosome, parent2: Chromosome
    ) -> Tuple[List[int], List[int]]:
        """Perform Order Crossover (OX)"""
        size = len(parent1.route)
        # Select random crossover points
        point1, point2 = sorted(random.sample(range(size), 2))

        def create_child(p1: List[int], p2: List[int]) -> List[int]:
            # Create a copy of p2's route
            child = [-1] * size
            # Copy the segment from p1
            child[point1:point2] = p1[point1:point2]
            # Fill remaining positions with elements from p2
            remaining = [x for x in p2 if x not in child[point1:point2]]
            j = 0
            for i in range(size):
                if child[i] == -1:
                    child[i] = remaining[j]
                    j += 1
            return child

        child1 = create_child(parent1.route, parent2.route)
        child2 = create_child(parent2.route, parent1.route)
        return child1, child2

    def swap_mutation(self, route: List[int]) -> List[int]:
        """Perform swap mutation"""
        if random.random() < self.mutation_prob:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def evolve(self) -> Tuple[List[Chromosome], List[float]]:
        """Main evolution loop"""
        population = self.initialize_population()
        best_fitness_history = []
        intermediate_solutions = []

        # Variables for tracking stagnation
        best_fitness_so_far = float("-inf")
        generations_without_improvement = 0
        stagnation_limit = 50

        for generation in range(self.max_generations):
            new_population = []

            # Keep the best individual
            best_individual = max(population, key=lambda x: x.fitness)
            new_population.append(best_individual)

            # Create new population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)

                # Crossover
                if random.random() < self.crossover_prob:
                    child1_route, child2_route = self.order_crossover(parent1, parent2)
                else:
                    child1_route, child2_route = (
                        parent1.route.copy(),
                        parent2.route.copy(),
                    )

                # Mutation
                child1_route = self.swap_mutation(child1_route)
                child2_route = self.swap_mutation(child2_route)

                # Create new chromosomes
                new_population.append(Chromosome(child1_route, self.distance_matrix))
                if len(new_population) < self.population_size:
                    new_population.append(
                        Chromosome(child2_route, self.distance_matrix)
                    )

            population = new_population
            best_fitness = max(population, key=lambda x: x.fitness).fitness
            best_fitness_history.append(1 / best_fitness)

            # Save solution every generation
            intermediate_solutions.append(max(population, key=lambda x: x.fitness))

            # Check for stagnation
            if best_fitness > best_fitness_so_far:
                best_fitness_so_far = best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Terminate if no improvement for stagnation_limit generations
            if generations_without_improvement >= stagnation_limit:
                print(
                    f"Terminated after {generation + 1} generations due to no improvement for {stagnation_limit} generations"
                )
                break

        return intermediate_solutions, best_fitness_history
