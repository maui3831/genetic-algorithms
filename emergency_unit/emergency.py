import numpy as np
import random
from typing import List, Tuple, Dict

class EmergencyUnitGA:
    """
    Genetic Algorithm for optimizing Emergency Unit location
    """
    
    def __init__(self, city_size: int = 10, population_size: int = 50, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        """
        Initialize the GA parameters
        
        Args:
            city_size: Size of the city grid (city_size x city_size)
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.city_size = city_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Initialize emergency frequency map (10x10 grid)
        # Higher values indicate more frequent emergencies
        self.emergency_frequency = self._generate_emergency_frequency()
        
        # Track evolution history
        self.generation_history = []
        self.cost_history = []
        self.best_solutions = []
        
    def _generate_emergency_frequency(self) -> np.ndarray:
        """
        Generate a frequency map of emergencies across the city
        This simulates historical emergency data
        """
        # Create a base frequency map
        freq_map = np.random.exponential(scale=2.0, size=(self.city_size, self.city_size))
        
        # Add some hot spots (areas with higher emergency frequency)
        hotspots = [(2, 3), (7, 2), (5, 8), (1, 7), (8, 6)]
        for x, y in hotspots:
            if 0 <= x < self.city_size and 0 <= y < self.city_size:
                freq_map[x, y] += np.random.uniform(5, 15)
        
        # Normalize to reasonable values
        freq_map = freq_map / np.max(freq_map) * 10
        return freq_map
    
    def _calculate_response_time(self, distance: float) -> float:
        """
        Calculate response time based on distance
        Formula: 1.7 + 3.4 * distance (in minutes)
        """
        return 1.7 + 3.4 * distance
    
    def _calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calculate Euclidean distance between two points
        """
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def calculate_cost(self, x_fs: float, y_fs: float) -> float:
        """
        Calculate the cost function for a given emergency unit location
        
        Cost = Sum of (emergency_frequency * response_time) for all city sections
        
        Args:
            x_fs: X coordinate of emergency unit
            y_fs: Y coordinate of emergency unit
            
        Returns:
            Total cost value
        """
        total_cost = 0.0
        
        for i in range(self.city_size):
            for j in range(self.city_size):
                # Calculate distance from emergency unit to this city section
                # Each section is represented by its center point
                section_x = i + 0.5
                section_y = j + 0.5
                
                distance = self._calculate_distance(x_fs, y_fs, section_x, section_y)
                response_time = self._calculate_response_time(distance)
                
                # Weight by emergency frequency
                cost_contribution = self.emergency_frequency[i, j] * response_time
                total_cost += cost_contribution
        
        return total_cost
    
    def _generate_individual(self) -> Tuple[float, float]:
        """
        Generate a random individual (emergency unit coordinates)
        """
        x = random.uniform(0, self.city_size)
        y = random.uniform(0, self.city_size)
        return (x, y)
    
    def _initialize_population(self) -> List[Tuple[float, float]]:
        """
        Initialize the population with random individuals
        """
        return [self._generate_individual() for _ in range(self.population_size)]
    
    def _tournament_selection(self, population: List[Tuple[float, float]], 
                            fitness_scores: List[float], tournament_size: int = 3) -> Tuple[float, float]:
        """
        Tournament selection for choosing parents
        """
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        # Select the individual with best (lowest) fitness
        best_index = tournament_indices[np.argmin(tournament_fitness)]
        return population[best_index]
    
    def _crossover(self, parent1: Tuple[float, float], 
                   parent2: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Arithmetic crossover between two parents
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Arithmetic crossover
        alpha = random.random()
        
        child1_x = alpha * parent1[0] + (1 - alpha) * parent2[0]
        child1_y = alpha * parent1[1] + (1 - alpha) * parent2[1]
        
        child2_x = (1 - alpha) * parent1[0] + alpha * parent2[0]
        child2_y = (1 - alpha) * parent1[1] + alpha * parent2[1]
        
        # Ensure coordinates are within city bounds
        child1 = (np.clip(child1_x, 0, self.city_size), np.clip(child1_y, 0, self.city_size))
        child2 = (np.clip(child2_x, 0, self.city_size), np.clip(child2_y, 0, self.city_size))
        
        return child1, child2
    
    def _mutate(self, individual: Tuple[float, float]) -> Tuple[float, float]:
        """
        Gaussian mutation
        """
        if random.random() > self.mutation_rate:
            return individual
        
        # Apply Gaussian mutation
        mutation_strength = 0.5
        x = individual[0] + random.gauss(0, mutation_strength)
        y = individual[1] + random.gauss(0, mutation_strength)
        
        # Ensure coordinates are within city bounds
        x = np.clip(x, 0, self.city_size)
        y = np.clip(y, 0, self.city_size)
        
        return (x, y)
    
    def evolve(self, generations: int = 100) -> Dict:
        """
        Run the genetic algorithm evolution
        
        Args:
            generations: Number of generations to evolve
            
        Returns:
            Dictionary containing evolution results
        """
        # Initialize population
        population = self._initialize_population()
        
        # Reset history
        self.generation_history = []
        self.cost_history = []
        self.best_solutions = []
        
        best_overall_individual = None
        best_overall_cost = float('inf')
        
        for generation in range(generations):
            # Calculate fitness for all individuals
            fitness_scores = [self.calculate_cost(x, y) for x, y in population]
            
            # Find best individual in this generation
            best_gen_index = np.argmin(fitness_scores)
            best_gen_individual = population[best_gen_index]
            best_gen_cost = fitness_scores[best_gen_index]
            
            # Update overall best
            if best_gen_cost < best_overall_cost:
                best_overall_cost = best_gen_cost
                best_overall_individual = best_gen_individual
            
            # Store history
            self.generation_history.append(generation + 1)
            self.cost_history.append(best_gen_cost)
            self.best_solutions.append(best_gen_individual)
            
            # Create new population
            new_population = []
            
            # Elitism: Keep best individual
            new_population.append(best_gen_individual)
            
            # Generate rest of population through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                child1, child2 = self._crossover(parent1, parent2)
                
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:self.population_size]
        
        # Calculate final statistics
        final_distance_avg = self._calculate_average_response_distance(
            best_overall_individual[0], best_overall_individual[1]
        )
        
        return {
            'best_coordinates': best_overall_individual,
            'best_cost': best_overall_cost,
            'generation_history': self.generation_history,
            'cost_history': self.cost_history,
            'best_solutions': self.best_solutions,
            'average_response_distance': final_distance_avg,
            'emergency_frequency_map': self.emergency_frequency
        }
    
    def _calculate_average_response_distance(self, x_fs: float, y_fs: float) -> float:
        """
        Calculate average response distance from emergency unit to all city sections
        """
        total_distance = 0.0
        total_weight = 0.0
        
        for i in range(self.city_size):
            for j in range(self.city_size):
                section_x = i + 0.5
                section_y = j + 0.5
                distance = self._calculate_distance(x_fs, y_fs, section_x, section_y)
                weight = self.emergency_frequency[i, j]
                
                total_distance += distance * weight
                total_weight += weight
        
        return total_distance / total_weight if total_weight > 0 else 0
    
    def get_generation_table(self) -> List[Dict]:
        """
        Get formatted table data for display with the requested format
        """
        table_data = []
        for i, (gen, cost, coords) in enumerate(zip(
            self.generation_history, self.cost_history, self.best_solutions
        )):
            avg_distance = self._calculate_average_response_distance(coords[0], coords[1])
            avg_response_time = self._calculate_response_time(avg_distance)
            
            # Format coordinates as requested
            coord_text = f"Coordinate ({coords[0]:.3f}, {coords[1]:.3f})"
            
            # Check if this is the last generation (optimized solution)
            if i == len(self.generation_history) - 1:
                coord_text = f"Optimized Coordinate ({coords[0]:.3f}, {coords[1]:.3f})"
            else:
                coord_text = f"Coordinate ({coords[0]:.3f}, {coords[1]:.3f})"
            
            table_data.append({
                'Generation': gen,
                'Proposed Coordinates': coord_text,
                'Cost Value': round(cost, 2),
                'Response Time (min)': round(avg_response_time, 2)
            })
        
        return table_data