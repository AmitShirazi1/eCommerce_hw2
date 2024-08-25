import numpy as np
from time import time
from copy import deepcopy


# This class is for simulating the 
class Simulation:
    _instance = None # 

    def __new__(cls, L=None, S=None, max_time=60, num_interactions=15):
        if (cls._instance is None) or ((not np.array_equal(cls._instance.L, L)) or (not np.array_equal(cls._instance.S, S))):
            cls._instance = super(Simulation, cls).__new__(cls)
            cls._instance._initialize(L, S, max_time, num_interactions)
        return cls._instance

    def _initialize(self, L, S, max_time, num_interactions): 
        self.L = L
        self.S = S
        self.max_time = max_time
        self.num_interactions = num_interactions
        self.simul_matrix = self.calculate_simulation_matrix()


    #simulation calculation of the matrix of usere types and genres
    def calculate_simulation_matrix(self):
        num_genres, num_user_types = self.L.shape
        simul_matrix = np.zeros((num_genres, num_user_types))

        start_time = time()
        num_iterations = np.zeros((num_genres, num_user_types))

        while time() - start_time < self.max_time:
            for g in range(num_genres):
                for u in range(num_user_types):
                    cumulative_likes = 0
                    interactions = 0

                    for _ in range(self.num_interactions):
                        like = np.random.rand() < self.L[g, u]
                        stay = like or np.random.rand() < (1-self.L[g,u])*(self.S[g, u])

                        if not stay:
                            break

                        cumulative_likes += like
                        interactions += 1

                    if interactions > 0:
                        num_iterations[g, u] += 1
                        simul_matrix[g, u] += (cumulative_likes - simul_matrix[g, u]) / num_iterations[g, u]

        return simul_matrix

    @staticmethod
    def get_simul_matrix():
        if Simulation._instance is None:
            raise ValueError("Simulation class has not been instantiated.")
        return Simulation._instance.simul_matrix


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
        self.p = deepcopy(p)
        self.simulation_instance = Simulation(L, S)  # Instantiate Simulation class if it hasn't been instantiated yet
        self.simul_matrix = self.simulation_instance.get_simul_matrix()
        self.recommendation = None

    def recommend(self):
        """Recommend the best genre for the user.

        Returns:
        integer: The index of the clip that the recommender recommends to the user.
        """
        recommendation = np.argmax(self.simul_matrix.dot(self.p))
        self.recommendation = recommendation
        return recommendation

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


