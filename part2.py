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
        self.S = S
        self.L = L
        self.p = p
        self.S_adjusted = self.calculate_staying_probability(L, S, p)
        self.num_genres, self.num_user_types = L.shape
        self.posterior_of_genres_recommendations, self.successes_per_genre, self.num_simulations_per_genre, self.successes, self.num_simulations = self.simulate_and_update_posterior()
        self.i = 0


    def simulate_and_update_posterior(self):
        p, S_adjusted = self.p, self.S_adjusted
        num_genres = S_adjusted.shape[0]
        successes_per_genre = np.ones(num_genres)
        num_simulations_per_genre = np.ones(num_genres)
        total_successes = 1
        total_simulations = 1
        posterior_of_genres_recommendations = np.zeros(num_genres)
        time_limit = 15
        start_time = time()

        while time() - start_time < time_limit:
            for i in range(num_genres):
                genre_score = np.dot(S_adjusted[i], p)  # Simplified genre score calculation
                posterior_of_genres_recommendations[i] = genre_score * ((successes_per_genre[i] / num_simulations_per_genre[i]) ** (1/2))  / ((total_successes / total_simulations) ** (1/2))
                posterior_of_genres_recommendations[i] = posterior_of_genres_recommendations[i]**10
            posterior_of_genres_recommendations /= posterior_of_genres_recommendations.sum()  # Normalize to create a probability distribution

            successes_per_genre, num_simulations_per_genre, total_successes, total_simulations = self.recommend_for_simulation(posterior_of_genres_recommendations, successes_per_genre, num_simulations_per_genre, total_successes, total_simulations)
        print("Posterior distribution after init:", posterior_of_genres_recommendations)
        return posterior_of_genres_recommendations, successes_per_genre, num_simulations_per_genre, total_successes, total_simulations


    def recommend_for_simulation(self, posterior_of_genres_recommendations, successes_per_genre, num_simulations_per_genre, total_successes, total_simulations):
        p = self.p
        user = np.random.choice(range(len(p)), p=p)
        for i in range(1, 16):
            recommendation = np.random.choice(range(len(posterior_of_genres_recommendations)), p=posterior_of_genres_recommendations)

            like = np.random.rand() < self.L[recommendation, user]
            #staying_reward = 1/i if i < 15 else 0
            stay = 1 if like else (self.S[recommendation, user]*(1-(i/15)) if np.random.rand() < self.S[recommendation, user] else 0)
            
            #stay=np.random.rand() < stay_prob
            total_successes += stay
            total_simulations += 1
            successes_per_genre[recommendation] += stay
            num_simulations_per_genre[recommendation] += 1
            
        return successes_per_genre, num_simulations_per_genre, total_successes, total_simulations


    def calculate_staying_probability(self, L, S, p):
        stay_prob = 5*L + S * (1 - L)#probability of staying in the system after not liking the clip and if liked the clip
        # Normalize each row to sum to 1
       # if max(L) >0.9: 
        #S_adjusted = stay_prob / stay_prob.sum(axis=1, keepdims=True)
        stay_prob_sum = stay_prob.sum(axis=1, keepdims=True)
         # Calculate the user-weighted adjustment
         # Reshape p to ensure correct broadcasting
        p_reshaped = p[:, np.newaxis]
         
        weighted_stay_prob = stay_prob* p_reshaped

        # Normalize the adjusted probabilities to ensure they sum to 1 for each user
        S_adjusted = weighted_stay_prob / stay_prob_sum

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
        self.i += 1
        print("\nTurn number", self.i)
        print("Recommended item:", self.recommended_item)
        print("Signal:", signal)
        if signal:
            self.successes_per_genre[self.recommended_item] += 1
            
        self.num_simulations_per_genre[self.recommended_item] += 1
        self.successes += signal
        self.num_simulations += 1
        posterior_of_genres_recommendations = self.posterior_of_genres_recommendations

        p, S_adjusted = self.p, self.S_adjusted
        num_genres = S_adjusted.shape[0]
        
        for i in range(num_genres):
            genre_score = 0
            for j in range(len(p)):
                genre_score += S_adjusted[i, j] * p[j]

            posterior_of_genres_recommendations[i] = genre_score* (self.successes_per_genre[i] / self.num_simulations_per_genre[i])**(1/2)  / (self.successes / self.num_simulations)**(1/2)
            posterior_of_genres_recommendations[i] = posterior_of_genres_recommendations[i]**10
            if i == self.recommended_item:
                if signal:# more weight to the genres that are liked
                    posterior_of_genres_recommendations[i] = posterior_of_genres_recommendations[i]*10
                else:#less weight to the genres that are not liked
                    posterior_of_genres_recommendations[i] = posterior_of_genres_recommendations[i]
        self.posterior_of_genres_recommendations = posterior_of_genres_recommendations / posterior_of_genres_recommendations.sum()  # Normalize to create a probability distribution
        print("Posterior distribution:", self.posterior_of_genres_recommendations)
       











# #THOMPSON SAMPLING
# class Recommender:
#     # Your recommender system class
   
#     def __init__(self, L, S, p):
#         """_summary_
        
#         Args:
#         L (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
#                          will give a like to a clip from genre i.
#         S (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
#                         won't leave the system after being recommended a clip from genre i and not liking it.
#         p (np.ndarray): The prior over user types. The entry i represents the probability that a user is of type i."""
#         self.S=S
#         self.L = L
#         self.Stay_adjusted = self.calculate_staying_probability(L, S)
#         self.p = p
#         self.num_genres, self.num_user_types = L.shape
#         self.successes = np.ones((self.num_genres, self.num_user_types))
#         self.failures = np.ones((self.num_genres, self.num_user_types))
#         self.posterior_of_genres_recommendations, self.successes_per_genre, self.num_simulations_per_genre, self.successes, self.num_simulations = self.simulate_staying()
#       #  print(self.posterior_of_genres_recommendations)





    # def simulate_staying(self):
    #     p, Stay_adjusted = self.p, self.Stay_adjusted
    #     num_genres = Stay_adjusted.shape[0]
    #     successes_per_genre = np.ones(num_genres)
    #     num_simulations_per_genre = np.ones(num_genres)
    #     total_successes = 1
    #     total_simulations = 1
    #     posterior_of_genres_recommendations = np.zeros(num_genres)
    #     time_limit = 10
    #     start_time = time()

    #     while time() - start_time < time_limit:
    #         for i in range(num_genres):
    #             genre_score = np.dot(Stay_adjusted[i], p)  # Simplified genre score calculation
    #             posterior_of_genres_recommendations[i] = genre_score*((successes_per_genre[i] / num_simulations_per_genre[i]) ** (1/2))  / ((total_successes / total_simulations) ** (1/2))
    #             posterior_of_genres_recommendations[i]=posterior_of_genres_recommendations[i]**10
    #         posterior_of_genres_recommendations /= posterior_of_genres_recommendations.sum()  # Normalize to create a probability distribution
    #       #  print("Posterior distribution:", posterior_of_genres_recommendations)

    #         recommendation = np.random.choice(range(len(posterior_of_genres_recommendations)), p=posterior_of_genres_recommendations)
    #         user = np.random.choice(range(len(p)), p=p)

    #         like = np.random.rand() < self.L[recommendation, user]
    #         stay = 1 if like else (0.5 if np.random.rand() < self.S[recommendation, user] else 0)

    #         total_successes += stay
    #         total_simulations += 1
    #         successes_per_genre[recommendation] += stay
    #         num_simulations_per_genre[recommendation] += 1
        
    #     return posterior_of_genres_recommendations, successes_per_genre, num_simulations_per_genre, total_successes, total_simulations





    # def calculate_staying_probability(self, L, S):
    #     stay_prob = 1*L + 5*S * (1 - L)#probability of staying in the system after not liking the clip and if liked the clip
    #     # Normalize each row to sum to 1
    #    # if max(L) >0.9: 
    #     S_adjusted = stay_prob / stay_prob.sum(axis=1, keepdims=True)
    #     return S_adjusted
    

    # def recommend(self):
    #     """_summary_
        
    #     Returns:
    #     integer: The index of the clip that the recommender recommends to the user."""

    #     posterior = self.posterior_of_genres_recommendations
        
    #     #  Option 1: Randomly sample a genre to recommend
    #     #recommendation = np.random.choice(range(len(posterior)), p=posterior)
        
    #     # Option 2: Recommend the item with the highest sampled like probability
    #     recommendation = np.argmax(posterior)

    #     self.recommended_item = recommendation
    #    # print("recommendation", recommendation)
    #     return recommendation
    
    
    # def update(self, signal):
    #     """_summary_
        
    #     Args:
    #     signal (integer): A binary variable that represents whether the user liked the recommended clip or not. 
    #                       It is 1 if the user liked the clip, and 0 otherwise."""
        
    #     self.successes_per_genre[self.recommended_item] += signal
    #     self.num_simulations_per_genre[self.recommended_item] += 1
    #     self.successes += signal
    #     self.num_simulations += 1
    #     posterior_of_genres_recommendations = np.zeros(self.num_genres)

    #     p, Stay_adjusted = self.p, self.Stay_adjusted
    #     num_genres = Stay_adjusted.shape[0]

    #     for i in range(num_genres):
    #         genre_score = 0
    #         for j in range(len(p)):
    #             genre_score += Stay_adjusted[i, j] * p[j]

    #         posterior_of_genres_recommendations[i] = genre_score* (self.successes_per_genre[i] / self.num_simulations_per_genre[i])**(1/2)  / (self.successes / self.num_simulations)**(1/2)
    #         posterior_of_genres_recommendations[i]=posterior_of_genres_recommendations[i]**10
    #     self.posterior_of_genres_recommendations = posterior_of_genres_recommendations / posterior_of_genres_recommendations.sum()  # Normalize to create a probability distribution

    
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


