import matplotlib.pyplot as plt
from game_master import GameMaster
from guesser2 import Guesser

def main():
    secret_word = input("Enter word: ").lower().strip()
    
    game_master = GameMaster(secret_word)
    guesser = Guesser(len(secret_word))
    
    generations = []
    best_guesses = []
    cost_values = []
    
    print(f"\nRunning Genetic Algorithm for {guesser.max_generations} generations...")
    print("=" * 60)
    print(f"{'Generation':<12} {'Best Guess':<20} {'Cost Value':<12}")
    print("=" * 60)
    
    # Run GA for specified generations
    for generation in range(guesser.max_generations):
        best_guess = guesser.get_best_individual()
        
        cost = game_master.calculate_cost(best_guess)
        
        generations.append(generation + 1)
        best_guesses.append(best_guess)
        cost_values.append(cost)
        
        print(f"{generation + 1:<12} {best_guess:<20} {cost:<12}")
        
        if cost == 0:
            print(f"\nPerfect solution found at generation {generation + 1}!")
            break
        
        # Evolve to next generation (except for last generation)
        if generation < guesser.max_generations - 1:
            guesser.evolve_generation(game_master)
    
    print("=" * 60)
    
    final_best_guess = best_guesses[-1]
    final_cost = cost_values[-1]
    
    print(f"\nFinal Results:")
    print(f"Secret Word: {secret_word}")
    print(f"Best Guess: {final_best_guess}")
    print(f"Final Cost: {final_cost}")
    
    if final_cost == 0:
        print("SUCCESS: Perfect match found!")
    else:
        print(f"Close match with cost difference of {final_cost}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, cost_values, 'b-', linewidth=2, marker='o', markersize=4)
    plt.title('Cost Value vs Generation', fontsize=14, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Cost Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, max(generations))
    plt.ylim(0, max(cost_values) * 1.1 if cost_values else 1)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()