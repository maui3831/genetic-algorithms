"""
guesser3.py
This module contains the Guesser3 class, which implements a guessing game
using a genetic algorithm approach. It evolves a population of guesses towards a target word by selecting, crossing over, and mutating individuals based on their fitness scores.

NOTE : modified version of guesser2.py with early stopping functionality
"""

import random
import string

class Guesser:
    def __init__(self, word_length, population_size=100):
        self.word_length = word_length
        self.population_size = population_size
        self.max_generations = 100
        
        self.crossover_probability = 0.7
        self.mutation_probability = 0.4
        self.tournament_size = 5
        self.elite_size = 8
        
        self.population = self._initialize_population()
        self.fitness_scores = []
        self.generations_without_improvement = 0
        self.best_fitness_ever = float('inf')
    
    def _initialize_population(self):
        population = []
        # Better initialization with common English letter frequencies
        common_letters = 'etaoinsrhdlucmfywgpbvkxqjz'
        
        for _ in range(self.population_size):
            individual = ''.join(random.choices(common_letters[:15], k=self.word_length))
            population.append(individual)
        
        # Add some completely random individuals
        for _ in range(self.population_size // 4):
            individual = ''.join(random.choices(string.ascii_lowercase, k=self.word_length))
            population.append(individual)
        
        return population[:self.population_size]
    
    def _tournament_selection(self, fitness_scores):
        tournament_indices = random.sample(range(len(self.population)), 
                                         min(self.tournament_size, len(self.population)))
        best_index = min(tournament_indices, key=lambda i: fitness_scores[i])
        return self.population[best_index]
    
    def _single_point_crossover(self, parent1, parent2):
        if len(parent1) != len(parent2) or len(parent1) <= 1:
            return parent1, parent2
        
        crossover_point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return offspring1, offspring2
    
    def _mutate(self, individual):
        individual_list = list(individual)
        
        # Character replacement mutation (more effective)
        if random.random() < 0.7:
            pos = random.randint(0, len(individual_list) - 1)
            # Bias towards common letters
            common_letters = 'etaoinsrhdlucmfywgpbvkxqjz'
            individual_list[pos] = random.choice(common_letters)
        
        # Swap mutation
        if len(individual_list) >= 2 and random.random() < 0.3:
            pos1, pos2 = random.sample(range(len(individual_list)), 2)
            individual_list[pos1], individual_list[pos2] = individual_list[pos2], individual_list[pos1]
        
        return ''.join(individual_list)
    
    def get_best_individual(self):
        if not self.fitness_scores:
            return self.population[0]
        
        best_index = min(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i])
        return self.population[best_index]
    
    def has_found_solution(self, game_master):
        """
        Check if the current best individual is a perfect solution.
        
        Args:
            game_master: The game master instance to calculate cost
            
        Returns:
            bool: True if perfect solution found (cost = 0), False otherwise
        """
        best_individual = self.get_best_individual()
        return game_master.calculate_cost(best_individual) == 0
    
    def get_current_best_cost(self, game_master):
        """
        Get the cost of the current best individual.
        
        Args:
            game_master: The game master instance to calculate cost
            
        Returns:
            int: The cost of the best individual
        """
        best_individual = self.get_best_individual()
        return game_master.calculate_cost(best_individual)
    
    def evolve_generation(self, game_master):
        # Calculate fitness for current population
        self.fitness_scores = [game_master.calculate_cost(ind) for ind in self.population]
        
        current_best = min(self.fitness_scores)
        
        # Adaptive parameters based on progress
        if current_best < self.best_fitness_ever:
            self.best_fitness_ever = current_best
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1
        
        # Increase mutation if stuck in plateau
        adaptive_mutation = self.mutation_probability
        if self.generations_without_improvement > 20:
            adaptive_mutation = min(0.8, self.mutation_probability * 1.5)
        
        new_population = []
        
        # Elite selection - keep best individuals
        elite_indices = sorted(range(len(self.fitness_scores)), 
                             key=lambda i: self.fitness_scores[i])[:self.elite_size]
        for i in elite_indices:
            new_population.append(self.population[i])
        
        # Add diversity if population is converging
        if self.generations_without_improvement > 30:
            for _ in range(self.population_size // 10):
                diverse_individual = ''.join(random.choices(string.ascii_lowercase, k=self.word_length))
                new_population.append(diverse_individual)
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection(self.fitness_scores)
            parent2 = self._tournament_selection(self.fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_probability:
                offspring1, offspring2 = self._single_point_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1, parent2
            
            # Adaptive mutation
            if random.random() < adaptive_mutation:
                offspring1 = self._mutate(offspring1)
            if random.random() < adaptive_mutation:
                offspring2 = self._mutate(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        self.population = new_population[:self.population_size]
    
    def evolve_with_early_stopping(self, game_master, max_generations=None):
        """
        Evolve the population with early stopping when perfect solution is found.
        
        Args:
            game_master: The game master instance
            max_generations: Maximum generations to run (defaults to self.max_generations)
            
        Returns:
            dict: Results including whether solution was found and generation count
        """
        if max_generations is None:
            max_generations = self.max_generations
        
        generations_data = []
        
        for generation in range(max_generations):
            # Calculate fitness for current population
            self.fitness_scores = [game_master.calculate_cost(ind) for ind in self.population]
            
            # Get current best
            best_individual = self.get_best_individual()
            best_cost = min(self.fitness_scores)
            
            # Store generation data
            generations_data.append({
                'generation': generation + 1,
                'best_individual': best_individual,
                'best_cost': best_cost,
                'average_cost': sum(self.fitness_scores) / len(self.fitness_scores)
            })
            
            # Check for perfect solution
            if best_cost == 0:
                return {
                    'solution_found': True,
                    'generations_used': generation + 1,
                    'best_individual': best_individual,
                    'final_cost': 0,
                    'generations_data': generations_data
                }
            
            # Continue evolution if not at last generation
            if generation < max_generations - 1:
                self.evolve_generation(game_master)
        
        # No perfect solution found
        final_best = self.get_best_individual()
        final_cost = game_master.calculate_cost(final_best)
        
        return {
            'solution_found': False,
            'generations_used': max_generations,
            'best_individual': final_best,
            'final_cost': final_cost,
            'generations_data': generations_data
        }
    
    def get_population_stats(self, game_master):
        fitness_scores = [game_master.calculate_cost(ind) for ind in self.population]
        
        return {
            'best_fitness': min(fitness_scores),
            'worst_fitness': max(fitness_scores),
            'average_fitness': sum(fitness_scores) / len(fitness_scores),
            'best_individual': self.population[fitness_scores.index(min(fitness_scores))],
            'population_size': len(self.population),
            'generations_without_improvement': self.generations_without_improvement
        }
    
    def reset_population(self):
        """Reset the population and fitness tracking for a new run."""
        self.population = self._initialize_population()
        self.fitness_scores = []
        self.generations_without_improvement = 0
        self.best_fitness_ever = float('inf')