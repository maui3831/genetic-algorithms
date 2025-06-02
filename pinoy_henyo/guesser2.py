import random
import string

class Guesser:
    """
    Guesser class that implements Genetic Algorithm to guess the secret word.
    Uses permutation encoding with tournament selection, single-point crossover, 
    and swap mutation techniques.
    """
    
    def __init__(self, word_length, population_size=300):
        """
        Initialize the Guesser with GA parameters.
        
        Args:
            word_length (int): Length of the word to guess
            population_size (int): Size of the population
        """
        self.word_length = word_length
        self.population_size = population_size
        self.max_generations = 500
        
        self.crossover_probability = 0.8
        self.mutation_probability = 0.2
        self.tournament_size = 5          
        
        self.population = self._initialize_population()
        self.fitness_scores = []
    
    def _initialize_population(self):
        """
        Initialize a random population of potential solutions.
        Each individual is a string of random lowercase letters.
        
        Returns:
            list: List of random strings (population)
        """
        population = []
        for _ in range(self.population_size):
            # Generate random string of specified length
            individual = ''.join(random.choices(string.ascii_lowercase, k=self.word_length))
            population.append(individual)
        return population
    
    def _tournament_selection(self, fitness_scores):
        """
        Tournament selection - select parent based on tournament competition.
        
        Args:
            fitness_scores (list): List of fitness scores for population
            
        Returns:
            str: Selected parent individual
        """
        # Randomly select tournament_size individuals
        tournament_indices = random.sample(range(len(self.population)), self.tournament_size)
        
        # Find the best individual in tournament (lowest cost)
        best_index = min(tournament_indices, key=lambda i: fitness_scores[i])
        return self.population[best_index]
    
    def _single_point_crossover(self, parent1, parent2):
        """
        Single-point crossover - creates two offspring by crossing over at a single point.
        
        Args:
            parent1 (str): First parent
            parent2 (str): Second parent
            
        Returns:
            tuple: Two offspring strings
        """
        if len(parent1) != len(parent2):
            return parent1, parent2
        
        length = len(parent1)
        
        if length <= 1:
            return parent1, parent2
        
        crossover_point = random.randint(1, length - 1)
        
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return offspring1, offspring2
    
    def _swap_mutation(self, individual):
        """
        Swap mutation - randomly swap two characters in the string.
        
        Args:
            individual (str): Individual to mutate
            
        Returns:
            str: Mutated individual
        """
        if len(individual) < 2:
            return individual
        
        individual_list = list(individual)
        
        # Select two random positions
        pos1, pos2 = random.sample(range(len(individual_list)), 2)
        
        # Swap characters
        individual_list[pos1], individual_list[pos2] = individual_list[pos2], individual_list[pos1]
        
        return ''.join(individual_list)
    
    def get_best_individual(self):
        """
        Get the best individual from current population.
        
        Returns:
            str: Best individual (lowest cost)
        """
        if not self.fitness_scores:
            return self.population[0]
        
        best_index = min(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i])
        return self.population[best_index]
    
    def evolve_generation(self, game_master):
        """
        Evolve the population to the next generation.
        
        Args:
            game_master: GameMaster instance to calculate fitness
        """
        # Calculate fitness for current population
        self.fitness_scores = []
        for individual in self.population:
            cost = game_master.calculate_cost(individual)
            self.fitness_scores.append(cost)
        
        new_population = []
        
        # Keep best individual
        best_index = min(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i])
        new_population.append(self.population[best_index])
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            # Parent selection using tournament selection
            parent1 = self._tournament_selection(self.fitness_scores)
            parent2 = self._tournament_selection(self.fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_probability:
                offspring1, offspring2 = self._single_point_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1, parent2
            
            # Mutation
            if random.random() < self.mutation_probability:
                offspring1 = self._swap_mutation(offspring1)
            if random.random() < self.mutation_probability:
                offspring2 = self._swap_mutation(offspring2)
            
            # Add offspring to new population
            new_population.extend([offspring1, offspring2])
        
        # Ensure population size is maintained
        self.population = new_population[:self.population_size]
    
    def get_population_stats(self, game_master):
        """
        Get statistics about current population.
        
        Args:
            game_master: GameMaster instance to calculate fitness
            
        Returns:
            dict: Statistics including best, worst, and average fitness
        """
        fitness_scores = [game_master.calculate_cost(ind) for ind in self.population]
        
        return {
            'best_fitness': min(fitness_scores),
            'worst_fitness': max(fitness_scores),
            'average_fitness': sum(fitness_scores) / len(fitness_scores),
            'best_individual': self.population[fitness_scores.index(min(fitness_scores))]
        }