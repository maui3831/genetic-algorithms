import unittest
import numpy as np
from genetic_algorithm import Chromosome, GeneticAlgorithm

class TestTSPGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        # Create a simple 4-city distance matrix for testing
        self.distance_matrix = np.array([
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ])
        self.ga = GeneticAlgorithm(
            distance_matrix=self.distance_matrix,
            population_size=10,
            tournament_size=3,
            crossover_prob=0.8,
            mutation_prob=0.2,
            max_generations=100
        )

    def test_chromosome_creation(self):
        """Test if chromosome is created correctly with valid route"""
        route = [0, 1, 2, 3]
        chromosome = Chromosome(route, self.distance_matrix)
        self.assertEqual(chromosome.route, route)
        self.assertIsInstance(chromosome.fitness, float)
        self.assertGreater(chromosome.fitness, 0)

    def test_fitness_calculation(self):
        """Test if fitness is calculated correctly"""
        # Test with a known route
        route = [0, 1, 2, 3]
        chromosome = Chromosome(route, self.distance_matrix)
        expected_distance = 10 + 35 + 30 + 20  # Sum of distances in the route
        self.assertAlmostEqual(1/chromosome.fitness, expected_distance)

    def test_population_initialization(self):
        """Test if initial population is created correctly"""
        population = self.ga.initialize_population()
        self.assertEqual(len(population), self.ga.population_size)
        
        # Check if all routes are valid permutations
        for chromosome in population:
            self.assertEqual(len(chromosome.route), len(self.distance_matrix))
            self.assertEqual(sorted(chromosome.route), list(range(len(self.distance_matrix))))

    def test_tournament_selection(self):
        """Test if tournament selection works correctly"""
        population = self.ga.initialize_population()
        selected = self.ga.tournament_selection(population)
        self.assertIsInstance(selected, Chromosome)
        self.assertIn(selected, population)

    def test_order_crossover(self):
        """Test if order crossover produces valid offspring"""
        parent1 = Chromosome([0, 1, 2, 3], self.distance_matrix)
        parent2 = Chromosome([3, 2, 1, 0], self.distance_matrix)
        child1, child2 = self.ga.order_crossover(parent1, parent2)
        
        # Check if children are valid permutations
        self.assertEqual(sorted(child1), list(range(len(self.distance_matrix))))
        self.assertEqual(sorted(child2), list(range(len(self.distance_matrix))))

    def test_swap_mutation(self):
        """Test if swap mutation works correctly"""
        route = [0, 1, 2, 3]
        mutated = self.ga.swap_mutation(route.copy())
        self.assertEqual(sorted(mutated), sorted(route))
        self.assertEqual(len(mutated), len(route))

    def test_evolution(self):
        """Test if evolution process improves solution"""
        solutions, fitness_history = self.ga.evolve()
        
        # Check if we got solutions
        self.assertGreater(len(solutions), 0)
        self.assertGreater(len(fitness_history), 0)
        
        # Check if fitness improved over time
        self.assertGreaterEqual(fitness_history[-1], fitness_history[0])
        
        # Check if best solution is valid
        best_solution = solutions[-1]
        self.assertEqual(sorted(best_solution.route), list(range(len(self.distance_matrix))))

    def test_convergence(self):
        """Test if algorithm converges to a solution"""
        solutions, fitness_history = self.ga.evolve()
        
        # Check if fitness history shows convergence
        # (fitness should not decrease significantly in later generations)
        last_quarter = fitness_history[-len(fitness_history)//4:]
        self.assertGreaterEqual(min(last_quarter), max(fitness_history) * 0.9)

if __name__ == '__main__':
    unittest.main()
