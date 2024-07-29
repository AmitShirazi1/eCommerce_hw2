import numpy as np
from time import time
from matrix_calc import simul_matrix

class Recommender:

    def __init__(self, L, S, p):
        """Initialize the Recommender system.
        
        Args:
        L (np.ndarray): A matrix where entry (i,j) represents the probability that a user of type j
                        will like a clip from genre i.
        S (np.ndarray): A matrix where entry (i,j) represents the probability that a user of type j
                        won't leave the system after being recommended a clip from genre i and not liking it.
        p (np.ndarray): The prior over user types. Entry i is the probability that a user is of type i.
        """
        self.S = S
        self.L = L
        self.p = p 
        #simul_matrix = self.simulation(L, S)  # Compute the simulation matrix once
        self.recommendation = None
        self.simul_matrix = simul_matrix

    def simulation(self, L, S):
        """Simulate user interactions based on like probabilities (L) and staying probabilities (S)."""
        num_genres, num_user_types = L.shape
        simul_matrix = np.zeros((num_genres, num_user_types))

        max_time = 25  # Maximum time for simulation in seconds
        start_time = time()
        num_iterations = np.zeros((num_genres, num_user_types))  # Each cell has its own iteration count

        while time() - start_time < max_time:
            # Generate random interactions for all genre-user pairs
            likes = np.random.rand(num_genres, num_user_types, 15) < L[:, :, np.newaxis]
            stays = np.random.rand(num_genres, num_user_types, 15) < (L[:, :, np.newaxis] + (1 - L[:, :, np.newaxis]) * S[:, :, np.newaxis])
            
            for g in range(num_genres):
                for u in range(num_user_types):
                    interaction_likes = likes[g, u]
                    interaction_stays = stays[g, u]
                    cumulative_likes = 0
                    interactions = 0  # Track the number of interactions for this genre-user pair

                    for i in range(15):
                        if interaction_stays[i] or interaction_likes[i]:
                            cumulative_likes += interaction_likes[i]
                            interactions += 1  # Increment interactions only if there was an interaction
                        else:
                            break

                    if interactions > 0:  # Check to avoid division by zero
                        num_iterations[g, u] += 1  # Increment the number of total iterations for this cell
                        simul_matrix[g, u] += (cumulative_likes - simul_matrix[g, u]) / num_iterations[g, u]

        return simul_matrix




    def simulation(self, L, S):
        """Simulate user interactions based on like probabilities (L) and staying probabilities (S)."""
        num_genres, num_user_types = self.L.shape
        simul_matrix = np.zeros((num_genres, num_user_types))

        start_time =time()  # Corrected to use time.time() for start time capture
        num_iterations = np.zeros((num_genres, num_user_types))  # Each cell has its own iteration count

        while time() - start_time < 30:
            for g in range(num_genres):
                for u in range(num_user_types):
                    cumulative_likes = 0
                    interactions = 0  # Track the number of interactions for this genre-user pair
                
                    for _ in range(15):  # Assuming each user gets 15 interaction attempts
                        like = np.random.rand() < L[g, u]
                        stay = like or np.random.rand()  < (L[g, u]+(1 - L[g, u]) * S[g, u])
                        
                        if not stay:
                            break

                        cumulative_likes += like
                        interactions += 1  # Increment interactions only if there was an interaction

                    if interactions > 0:  # Check to avoid division by zero
                        # Update the simulation matrix with the new average for this interaction session
                        num_iterations[g, u] += 1  # Increment the number of total iterations for this cell
                        simul_matrix[g, u] += (cumulative_likes - simul_matrix[g, u]) / num_iterations[g, u]
        return simul_matrix



    def recommend(self):
        """Recommend the best genre for the user.

        Returns:
        integer: The index of the clip that the recommender recommends to the user.
        """
        recommendation = np.argmax(self.simul_matrix.dot(self.p))
        self.recommendation = recommendation
        return recommendation

    # def update(self, signal):
    #     """Update the prior probabilities based on the user's response.

    #     Args:
    #     signal (integer): A binary variable that represents whether the user liked the recommended clip or not. 
    #                       It is 1 if the user liked the clip, and 0 otherwise.
    #     """
    #     if signal:
    #         # User liked the recommendation
    #         likelihood = self.simul_matrix[self.recommendation]
    #     else:
    #         # User did not like the recommendation
    #         likelihood = self.S[self.recommendation]
        
    #     # Update the prior with the likelihood
    #     self.p *= likelihood
        
    #     if np.sum(self.p) > 0:  # Ensure there is no division by zero
    #         # Normalize the probabilities to ensure they sum to 1
    #         self.p = self.p / np.sum(self.p)
    #     # else:
    #     #     raise ValueError("Sum of updated probabilities is zero. Cannot normalize.")

    #     # # Debugging: Print the updated probabilities
    #     # print("Updated probabilities:", self.p)


    def update(self, signal):
        """Update the prior probabilities based on the user's response.

        Args:
        signal (integer): A binary variable that represents whether the user liked the recommended clip or not. 
                        It is 1 if the user liked the clip, and 0 otherwise.
        """
        if signal:
            # User liked the recommendation
            likelihood = self.simul_matrix[self.recommendation]
        else:
            # User did not like the recommendation
            likelihood = self.S[self.recommendation]
        
        # Update the prior with the likelihood
        self.p *= likelihood
        
        # Normalize the probabilities to ensure they sum to 1
        self.p /= np.sum(self.p)



   