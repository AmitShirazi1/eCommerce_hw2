import numpy as np
from time import time

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
        self.S=S
        self.L = L
        self.Stay_adjusted = self.calculate_staying_probability(L, S)
        self.p = p
        self.num_genres, self.num_user_types = L.shape
        self.successes = np.ones((self.num_genres, self.num_user_types))
        self.failures = np.ones((self.num_genres, self.num_user_types))
        self.posterior_of_genres_recommendations, self.successes_per_genre, self.num_simulations_per_genre, self.successes, self.num_simulations = self.simulate_staying()
      #  print(self.posterior_of_genres_recommendations)





    # def simulate_staying(self):
    #     p, Stay_adjusted = self.p, self.Stay_adjusted
    #     num_genres = Stay_adjusted.shape[0]
    #     successes_per_genre = np.ones(num_genres)
    #     num_simulations_per_genre = np.ones(num_genres)
    #     successes = 1
    #     num_simulations = 1
    #     posterior_of_genres_recommendations = np.zeros(num_genres)
    #     time_limit = 10
    #     start_time = time()
    #     while time() - start_time < time_limit:
    #         for i in range(num_genres):
    #             genre_score = 0
    #             for j in range(len(p)):
    #                 genre_score += Stay_adjusted[i, j] * p[j]
    #             posterior_of_genres_recommendations[i] = ((successes_per_genre[i] / num_simulations_per_genre[i])**(1/3)) *2* genre_score / ((successes / num_simulations)**(1/3))
    #           # print("posterior_of_genres_recommendations ",posterior_of_genres_recommendations)
    #         normalized_posterior_of_genres_recommendations = posterior_of_genres_recommendations/posterior_of_genres_recommendations.sum()  # Normalize to create a probability distribution
    #         print(posterior_of_genres_recommendations)
    #         recommendation = np.random.choice(range(len(normalized_posterior_of_genres_recommendations)), p=normalized_posterior_of_genres_recommendations)
    #         #recommendation = np.argmax(posterior_of_genres_recommendations)
    #         user = np.random.choice(range(len(p)), p=p)
    #       #  print("index_item reccomend ",recommendation,"user index", user)
    #         like = np.random.rand() < self.L[recommendation, user]
    #        # print(like)
    #         # user of type j will stay in the system even though they don't like the item with probability S[i, j]
    #         stay = 1 if like else (0.5  if np.random.rand() < self.S[recommendation, user] else 0)


    #         #print("stay", stay)
            
    #         successes += stay
    #        # print("successes ",successes)
    #         num_simulations += 1
    #         successes_per_genre[recommendation] += stay
    #       #  print("successes_per_genre ", successes_per_genre)
    #         num_simulations_per_genre[recommendation] += 1
        
    #     return posterior_of_genres_recommendations, successes_per_genre, num_simulations_per_genre, successes, num_simulations






    def simulate_staying(self):
        p, Stay_adjusted = self.p, self.Stay_adjusted
        num_genres = Stay_adjusted.shape[0]
        successes_per_genre = np.ones(num_genres)
        num_simulations_per_genre = np.ones(num_genres)
        total_successes = 1
        total_simulations = 1
        posterior_of_genres_recommendations = np.zeros(num_genres)
        time_limit = 10
        start_time = time()

        while time() - start_time < time_limit:
            for i in range(num_genres):
                genre_score = np.dot(Stay_adjusted[i], p)  # Simplified genre score calculation
                posterior_of_genres_recommendations[i] = genre_score*((successes_per_genre[i] / num_simulations_per_genre[i]) ** (1/2))  / ((total_successes / total_simulations) ** (1/2))
                posterior_of_genres_recommendations[i]=posterior_of_genres_recommendations[i]**10
            posterior_of_genres_recommendations /= posterior_of_genres_recommendations.sum()  # Normalize to create a probability distribution
          #  print("Posterior distribution:", posterior_of_genres_recommendations)

            recommendation = np.random.choice(range(len(posterior_of_genres_recommendations)), p=posterior_of_genres_recommendations)
            user = np.random.choice(range(len(p)), p=p)

            like = np.random.rand() < self.L[recommendation, user]
            stay = 1 if like else ((1/15) if np.random.rand() < self.S[recommendation, user] else 0)

            total_successes += stay
            total_simulations += 1
            successes_per_genre[recommendation] += stay
            num_simulations_per_genre[recommendation] += 1
        
        return posterior_of_genres_recommendations, successes_per_genre, num_simulations_per_genre, total_successes, total_simulations





    def calculate_staying_probability(self, L, S):
        stay_prob = 1*L + 5*S * (1 - L)#probability of staying in the system after not liking the clip and if liked the clip
        # Normalize each row to sum to 1
       # if max(L) >0.9: 
        S_adjusted = stay_prob / stay_prob.sum(axis=1, keepdims=True)
        return S_adjusted
    

    def recommend(self):
        """_summary_
        
        Returns:
        integer: The index of the clip that the recommender recommends to the user."""

        posterior = self.posterior_of_genres_recommendations
        
        #  Option 1: Randomly sample a genre to recommend
        #recommendation = np.random.choice(range(len(posterior)), p=posterior)
        
        # Option 2: Recommend the item with the highest sampled like probability
        recommendation = np.argmax(posterior)

        self.recommended_item = recommendation
       # print("recommendation", recommendation)
        return recommendation
    
    
    def update(self, signal):
        """_summary_
        
        Args:
        signal (integer): A binary variable that represents whether the user liked the recommended clip or not. 
                          It is 1 if the user liked the clip, and 0 otherwise."""
        
        self.successes_per_genre[self.recommended_item] += signal
        self.num_simulations_per_genre[self.recommended_item] += 1
        self.successes += signal
        self.num_simulations += 1
        posterior_of_genres_recommendations = np.zeros(self.num_genres)

        p, Stay_adjusted = self.p, self.Stay_adjusted
        num_genres = Stay_adjusted.shape[0]

        for i in range(num_genres):
            genre_score = 0
            for j in range(len(p)):
                genre_score += Stay_adjusted[i, j] * p[j]

            posterior_of_genres_recommendations[i] = genre_score* (self.successes_per_genre[i] / self.num_simulations_per_genre[i])**(1/2)  / (self.successes / self.num_simulations)**(1/2)
            posterior_of_genres_recommendations[i]=posterior_of_genres_recommendations[i]**10
        self.posterior_of_genres_recommendations = posterior_of_genres_recommendations / posterior_of_genres_recommendations.sum()  # Normalize to create a probability distribution

    
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


