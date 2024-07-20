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

        raise NotImplementedError("Implement this method")
    
    def recommend(self):
        """_summary_
        
        Returns:
        integer: The index of the clip that the recommender recommends to the user."""

        raise NotImplementedError("Implement this method")
    
    def update(self, signal):

        """_summary_
        
        Args:
        signal (integer): A binary variable that represents whether the user liked the recommended clip or not. 
                          It is 1 if the user liked the clip, and 0 otherwise."""
        
        raise NotImplementedError("Implement this method")
    
    
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


