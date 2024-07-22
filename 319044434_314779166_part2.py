import numpy as np


class Recommender:
    # Your recommender system class
   
    def __init__(self, L, S, p):
        """_summary_
        
        Args:
        L (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
                         will give a like to a clip from genre i.
        S (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
                        won't leave the system after being recommended a clip from genre i and not liking it.
        p (np.ndarray): The prior over user types. The entry i represents the probability that a user is of type i."""

        self.L = self.calculate_staying_probability(L, S)
        self.S = S
        self.p = p
        self.num_genres, self.num_user_types = L.shape
        self.successes = np.ones((self.num_genres, self.num_user_types))
        self.failures = np.ones((self.num_genres, self.num_user_types))
    
    # Take the simulation code from simulation.py and run 1 turn (not 15).
    # Define success and failure as 1 and 0.
         
    def calculate_staying_probability(self, L, S):
        stay_prob = 10*L + S * (1 - L)
        # Normalize each row to sum to 1
        L_adjusted = stay_prob / stay_prob.sum(axis=1, keepdims=True)
        return L_adjusted
    
    def recommend(self):
        """_summary_
        
        Returns:
        integer: The index of the clip that the recommender recommends to the user."""

        # Sample user type based on prior
        self.user_type = np.random.choice(range(len(self.p)), p=self.p)
        
        # Sample from the Beta distribution for the like probabilities
        like_probabilities = np.random.beta(self.successes[self.user_type] + 1, self.failures[self.user_type] + 1)
        
        # Recommend the item with the highest sampled like probability
        self.recommended_item = np.argmax(like_probabilities)
        
        return self.recommended_item
    
    
    def update(self, signal):

        """_summary_
        
        Args:
        signal (integer): A binary variable that represents whether the user liked the recommended clip or not. 
                          It is 1 if the user liked the clip, and 0 otherwise."""
        
        recommended_item = self.recommended_item
        if signal:
            self.successes[self.user_type, recommended_item] += 1
        else:
            self.failures[self.user_type, recommended_item] += 1


    
# an example of a recommender that always recommends the item with the highest probability of being liked
class GreedyRecommender:
    def __init__(self, L, S, p):
        self.L = L
        self.S = S
        self.p = p
        
    def recommend(self):
        return np.argmax(np.dot(self.L, self.p))
    
    def update(self, signal):
        pass


