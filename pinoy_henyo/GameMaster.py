"""
GameMaster Module
Handles cost computation, display of results, and timing
"""

import time
from tabulate import tabulate
import matplotlib.pyplot as plt

class game_master:
    def __init__(self, target_word):
        self.target_word = target_word
        self.history = []
        self.start_time = None
        self.end_time = None

        self.run_genetic_algorithm()

    def compute_cost(self, guess_string):
        """
        Compute the cost value using sum of squared differences of ASCII values
        """
        if len(guess_string) != len(self.target_word):
            return float('inf')

        cost = sum((ord(guess_char) - ord(target_char)) ** 2
                  for guess_char, target_char in zip(guess_string, self.target_word))
        return cost

    def start_timing(self):
        self.start_time = time.time()

    def stop_timing(self):
        self.end_time = time.time()

    def get_elapsed_time(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    def record_generation(self, generation, best_guess, cost_value):
        self.history.append({
            'generation': generation,
            'best_guess': best_guess,
            'cost_value': cost_value
        })

    def display_generation(self, generation, best_guess, cost_value):
        if generation == 1:
            print(f"{'Generation':<11} {'Best Guess':<15} {'Cost Value':<12}")
            print("-" * 11 + " " + "-" * 15 + " " + "-" * 12)

        print(f"{generation:>10} {best_guess:<15} {cost_value:>12.0f}")

    def display_final_summary(self, best_solution):
        if best_solution and self.compute_cost(best_solution) == 0:
            print(f"SUCCESS! Found the word: '{best_solution}'")
        else:
            print(f"Evolution completed. Best guess: '{best_solution}'")
            if best_solution:
                print(f"Final cost: {self.compute_cost(best_solution)}")

    def display_final_results(self):
        elapsed_time = self.get_elapsed_time()
        print(f"Time taken: {elapsed_time:.4f} seconds")
        print(f"Total generations: {len(self.history)}")

    def display_summary_table(self, interval=10):
        """Display a summary table showing every nth generation"""
        if not self.history:
            return

        print(f"\nSUMMARY (Every {interval} generations):")
        print("-" * 50)

        summary_data = []
        for i, record in enumerate(self.history):
            if record['generation'] % interval == 0 or record['generation'] == 1 or i == len(self.history) - 1:
                summary_data.append([
                    record['generation'],
                    record['best_guess'],
                    f"{record['cost_value']:.2f}"
                ])

        print(tabulate(
            summary_data,
            headers=['Generation', 'Best Guess', 'Cost Value'],
            tablefmt='simple'
        ))

    def plot_convergence(self):
        generations = [record['generation'] for record in self.history]
        costs = [record['cost_value'] for record in self.history]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, costs, 'b-', label='Cost Value', alpha=0.5)
        plt.plot(generations, costs, 'b.', markersize=5)

        plt.title('Cost vs. Generation')
        plt.xlabel('Generation (N)')
        plt.ylabel('Cost Value')
        plt.grid(True, alpha=0.3)

        plt.yscale('log')
        plt.legend()
        plt.tight_layout()

        plt.xlim(0, 120)
        plt.ylim(0, 600)

        plt.show()

    def run_genetic_algorithm(self):
        from Guesser import guesser

        genetic_guesser = guesser(
            target_word=self.target_word,
            population_size=50,
            max_generations=100,
            game_master=self
        )

        best_solution = genetic_guesser.evolve()

        self.display_final_summary(best_solution)
        self.plot_convergence()

    def get_best_solution(self):
        if not self.history:
            return None, float('inf')

        best_record = min(self.history, key=lambda x: x['cost_value'])
        return best_record['best_guess'], best_record['cost_value']