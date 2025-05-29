import numpy as np
import random
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
import pygame
import sys
from genetic_algorithm import GeneticAlgorithm
import xml.etree.ElementTree as ET
import os


class TSPVisualizer:
    def __init__(self, width=1000, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("TSP Solver Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)  
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        
        # Load map background
        try:
            self.map_image = pygame.image.load('map.png')
            self.map_image = pygame.transform.scale(self.map_image, (width, height))
        except:
            print("Warning: map.png not found. Using white background.")    
            self.map_image = None
        
        self.cities = [
            (285, 258),
            (168, 342),
            (220, 520),
            (395, 633),
            (518, 490),
            (680, 570),
            (460, 355),
            (650, 395),
            (650, 395),
            (585, 165),
            (810, 292),       
        ]
                
        # Initialize algorithm
        self.distance_matrix = self.calculate_distance_matrix()
        self.ga = GeneticAlgorithm(
            distance_matrix=self.distance_matrix,
            population_size=100,
            tournament_size=5,
            crossover_prob=0.8,
            mutation_prob=0.2,
            max_generations=5000
        )
        
        # Solution state
        self.intermediate_solutions = None
        self.current_solution_index = 0
        self.fitness_history = None
        self.generation = 0
        
        # Run the genetic algorithm first
        print("Running genetic algorithm...")
        self.intermediate_solutions, self.fitness_history = self.ga.evolve()
        print("Algorithm complete. Press SPACE to step through solutions.")
        self.best_solution = self.intermediate_solutions[0]
        
    def calculate_distance_matrix(self):
        """Calculate distance matrix between cities"""
        if not self.cities:
            return None
            
        n = len(self.cities)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.cities[i]
                    x2, y2 = self.cities[j]
                    matrix[i][j] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return matrix
    
    def draw_cities(self):
        """Draw cities as circles"""
        for i, (x, y) in enumerate(self.cities):

            number_text = self.font.render(str(i + 1), True, self.WHITE)
            text_rect = number_text.get_rect(center=(int(x), int(y)))
            self.screen.blit(number_text, text_rect)
    
    def draw_route(self, route):
        """Draw the current route"""
        if route is None:
            return
            
        seq_font = pygame.font.Font(None, 24)
            
        for i in range(len(route)):
            city1 = self.cities[route[i]]
            city2 = self.cities[route[(i + 1) % len(route)]]
            
            # Draw the line
            pygame.draw.line(self.screen, self.WHITE, 
                           (int(city1[0]), int(city1[1])),
                           (int(city2[0]), int(city2[1])), 2)
            
            # Calculate midpoint for sequence number
            mid_x = (city1[0] + city2[0]) / 2
            mid_y = (city1[1] + city2[1]) / 2
            
            # Draw sequence number with background
            seq_text = seq_font.render(str(i + 1), True, self.WHITE)
            text_rect = seq_text.get_rect(center=(int(mid_x), int(mid_y)))
            pygame.draw.rect(self.screen, self.BLACK, text_rect.inflate(8, 8))
            self.screen.blit(seq_text, text_rect)
    
    def draw_info(self):
        """Draw information about the current state"""
        if self.best_solution:
            distance = 1/self.best_solution.fitness
            fitness = self.best_solution.fitness
            
            # Display distance
            text = self.font.render(f"Distance: {distance:.2f}", True, self.WHITE)
            self.screen.blit(text, (10, 10))
            
            # Display fitness
            text = self.font.render(f"Fitness: {fitness:.6f}", True, self.WHITE)
            self.screen.blit(text, (10, 40))
            
            # Display generation
            text = self.font.render(f"Generation: {self.generation}", True, self.WHITE)
            self.screen.blit(text, (10, 70))
            
            # Display the sequence of cities
            sequence = " -> ".join(str(city + 1) for city in self.best_solution.route)
            # Split sequence into multiple lines if too long
            max_chars_per_line = 100
            for i in range(0, len(sequence), max_chars_per_line):
                line = sequence[i:i + max_chars_per_line]
                text = self.font.render(f"Route: {line}", True, self.WHITE)
                self.screen.blit(text, (10, 100 + (i // max_chars_per_line) * 25))
    
    def reset(self):
        """Reset the simulation"""
        self.current_solution_index = 0
        self.best_solution = self.intermediate_solutions[0]
        self.generation = 0
        
    def run(self):
        """Main game loop"""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_SPACE:
                        # Move to next solution
                        if self.current_solution_index < len(self.intermediate_solutions) - 1:
                            self.current_solution_index += 1
                            self.best_solution = self.intermediate_solutions[self.current_solution_index]
                            self.generation = self.current_solution_index  # Increment by 1
            
            # Draw everything
            self.screen.fill(self.WHITE)
            if self.map_image:
                self.screen.blit(self.map_image, (0, 0))
            
            self.draw_cities()
            if self.best_solution:
                self.draw_route(self.best_solution.route)
            self.draw_info()
            
            pygame.display.flip()
            self.clock.tick(60)


if __name__ == "__main__":
    visualizer = TSPVisualizer()
    visualizer.run()
