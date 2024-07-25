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
        self.L = L
        self.S = self.calculate_staying_probability(L, S)
        self.p = p
        self.num_genres, self.num_user_types = L.shape
        self.successes = np.ones((self.num_genres, self.num_user_types))
        self.failures = np.ones((self.num_genres, self.num_user_types))
        self.posterior_of_genres_recommendations, self.successes_per_genre, self.num_simulations_per_genre, self.successes, self.num_simulations = self.simulate_staying()


    def simulate_staying(self):
        p, L = self.p, self.L
        num_genres = L.shape[0]
        successes_per_genre = np.ones(num_genres)
        num_simulations_per_genre = np.ones(num_genres)
        successes = 1
        num_simulations = 1
        posterior_of_genres_recommendations = np.zeros(num_genres)

        for _ in range(10):
            for i in range(num_genres):
                genre_score = 0
                for j in range(len(p)):
                    genre_score += L[i, j] * p[j]
                posterior_of_genres_recommendations[i] = (successes_per_genre[i] / num_simulations_per_genre[i]) * genre_score / (successes / num_simulations)

            posterior_of_genres_recommendations /= posterior_of_genres_recommendations.sum()  # Normalize to create a probability distribution

            recommendation = np.random.choice(range(len(posterior_of_genres_recommendations)), p=posterior_of_genres_recommendations)
            user = np.random.choice(range(len(p)), p=p)
            like = np.random.rand() < L[recommendation, user]

            # user of type j will stay in the system even though they don't like the item with probability S[i, j]
            stay = 1 if like else 0.5 * (np.random.rand() < self.S[recommendation, user])
            
            successes += stay
            num_simulations += 1
            successes_per_genre[recommendation] += stay
            num_simulations_per_genre[recommendation] += 1

        return posterior_of_genres_recommendations, successes_per_genre, num_simulations_per_genre, successes, num_simulations


    def calculate_staying_probability(self, L, S):
        stay_prob = 1*L + S * (1 - L)#probability of staying in the system after not liking the clip and if liked the clip
        # Normalize each row to sum to 1
        L_adjusted = stay_prob / stay_prob.sum(axis=1, keepdims=True)
        return L_adjusted
    

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

        p, L = self.p, self.L
        num_genres = L.shape[0]

        for i in range(num_genres):
            genre_score = 0
            for j in range(len(p)):
                genre_score += L[i, j] * p[j]
            posterior_of_genres_recommendations[i] = (self.successes_per_genre[i] / self.num_simulations_per_genre[i]) * genre_score / (self.successes / self.num_simulations)

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


