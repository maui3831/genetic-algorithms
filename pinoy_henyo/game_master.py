class GameMaster:
    """
    Game Master class that stores the secret word and computes the cost function.
    The cost function calculates the sum of squared differences between 
    ASCII values of characters in the guess and the target word.
    """
    
    def __init__(self, secret_word):
        """
        Initialize the Game Master with a secret word.
        
        Args:
            secret_word (str): The target word to be guessed
        """
        self.secret_word = secret_word.lower()
        self.word_length = len(self.secret_word)
    
    def calculate_cost(self, guess):
        """
        Calculate the cost function based on the formula:
        cost = Σ(guess[n] - answer[n])² for n=1 to letters
        
        Args:
            guess (str): The guessed word
            
        Returns:
            int: The cost value (lower is better, 0 is perfect match)
        """
        if len(guess) != len(self.secret_word):
            return float('inf')
        
        cost = 0
        for i in range(len(self.secret_word)):
            guess_char_value = ord(guess[i])
            answer_char_value = ord(self.secret_word[i])
            cost += (guess_char_value - answer_char_value) ** 2
        
        return cost
    
    def get_secret_word(self):
        """
        Get the secret word (for debugging purposes).
        
        Returns:
            str: The secret word
        """
        return self.secret_word
    
    def get_word_length(self):
        """
        Get the length of the secret word.
        
        Returns:
            int: Length of the secret word
        """
        return self.word_length
    
    def is_perfect_match(self, guess):
        """
        Check if the guess is a perfect match.
        
        Args:
            guess (str): The guessed word
            
        Returns:
            bool: True if perfect match, False otherwise
        """
        return self.calculate_cost(guess) == 0