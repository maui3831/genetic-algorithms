from GameMaster import game_master
from Guesser import guesser

def main():
    target_word = "halo"

    print("Genetic Algorithm Word Guessing Game")
    print("-" * 40)
    print(f"Target Word: {target_word}")

    game_master(target_word)

if __name__ == "__main__":
    main()