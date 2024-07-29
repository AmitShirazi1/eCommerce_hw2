

import numpy as np
from time import time
#1
L1 = np.array([[0.8, 0.7, 0.6], [0.79, 0.69, 0.59], [0.78, 0.68, 0.58]])
S1 = np.array([[0.56, 0.46, 0.36], [0.55, 0.45, 0.35], [0.54, 0.44, 0.34]])
p1 = np.array([0.35, 0.45, 0.2])

# Instance 2
L2 = np.array([[0.9, 0.75], [0.64, 0.5]])
S2 = np.array([[0.2, 0.4], [0.7, 0.8]])
p2 = np.array([0.3, 0.7])


# Instances 3a, 3b, 3c (same matrices L3, S3, different priors p3a, p3b, p3c)
L3 = np.array([[0.99, 0.2, 0.2], 
                [0.2, 0.99, 0.2], 
                [0.2, 0.2, 0.99], 
                [0.93, 0.93, 0.4],
                [0.4, 0.93, 0.93],
                [0.93, 0.4, 0.93],
                [0.85, 0.85, 0.85]])
S3 = np.zeros((7, 3))
p3a = np.array([0.9, 0.05, 0.05])
p3b = np.array([1/3, 1/3, 1/3])
p3c = np.array(object=[0.45, 0.25, 0.3])

# Instance 4
L4 = np.array([[0.94, 0.21, 0.02, 0.05, 0.86, 0.61, 0.59, 0.26],
               [0.91, 0.46, 0.87, 0.19, 0.64, 0.40, 0.83, 0.67],
               [0.25, 0.52, 0.32, 0.13, 0.15, 0.82, 0.46, 0.41],
               [0.10, 0.85, 0.70, 0.95, 0.06, 0.49, 0.68, 0.98]])
S4 = np.array([[0.51, 0.26, 0.98, 0.12, 0.99, 0.15, 0.74, 0.21],
               [0.92, 0.37, 0.17, 0.45, 0.81, 0.56, 0.28, 0.55],
               [0.61, 0.40, 0.21, 0.87, 0.25, 0.03, 0.85, 0.21],
               [0.62, 0.47, 0.06, 0.28, 0.90, 0.75, 0.48, 0.79]])
p4 = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])

# Instance 5
L5 = np.array([[0.88, 0.12, 0.08, 0.29, 0.01, 0.34, 0.83, 0.61, 0.05, 0.07],
              [0.04, 0.01, 0.42, 0.24, 0.79, 0.24, 0.98, 0.88, 0.83, 0.38],
              [0.34, 0.76, 0.08, 0.07, 0.52, 0.43, 0.43, 0.82, 0.62, 0.88],
              [0.52, 0.58, 0.54, 0.59, 0.83, 0.79, 0.71, 0.72, 0.39, 0.28],
              [0.47, 0.49, 0.21, 0.51, 0.15, 0.22, 0.43, 0.56, 0.83, 0.04],
              [0.94, 0.73, 0.53, 0.54, 0.70, 0.79, 0.26, 0.21, 0.80, 0.56],
              [0.15, 0.72, 0.87, 0.83, 0.45, 0.90, 0.49, 0.45, 0.58, 0.95],
              [0.60, 0.23, 0.48, 0.74, 0.37, 0.90, 0.56, 0.82, 0.90, 0.86],
              [0.10, 0.57, 0.80, 0.47, 0.18, 0.91, 0.68, 0.52, 0.04, 0.42],
              [0.61, 0.11, 0.95, 0.39, 0.23, 0.13, 0.50, 0.10, 1.00, 0.26]])
S5 = np.array([[0.67, 0.83, 0.24, 0.07, 0.54, 0.15, 0.79, 0.44, 0.93, 0.49],
              [0.96, 0.23, 0.89, 0.54, 0.36, 0.43, 0.74, 0.32, 0.23, 0.88],
              [0.03, 0.88, 0.33, 0.79, 0.21, 0.10, 0.01, 0.62, 0.39, 0.86],
              [0.88, 0.84, 0.84, 0.65, 0.33, 0.44, 0.98, 0.85, 0.42, 0.42],
              [0.28, 0.45, 0.99, 0.25, 0.85, 0.16, 1.00, 0.87, 0.88, 0.82],
              [0.55, 0.81, 0.76, 0.25, 0.78, 0.80, 0.36, 0.37, 0.55, 0.75],
              [0.65, 0.94, 0.03, 0.32, 0.51, 0.89, 0.61, 0.89, 0.55, 0.96],
              [0.35, 0.03, 0.78, 0.96, 0.20, 0.44, 0.08, 0.82, 0.51, 0.28],
              [0.16, 0.57, 0.93, 0.81, 0.94, 0.48, 0.93, 0.35, 0.73, 0.37],
              [0.12, 0.42, 0.81, 0.25, 0.44, 0.99, 0.08, 0.51, 0.16, 0.38]])
p5 = np.array([0.11, 0.12, 0.07, 0.1, 0.05, 0.13, 0.1, 0.11, 0.11, 0.1])


L,S=L5, S5        
"""Simulate user interactions based on like probabilities (L) and staying probabilities (S)."""
num_genres, num_user_types = L.shape
# Initialize the simulation matrix if it's not already initialized
simul_matrix = np.zeros((num_genres, num_user_types))

start_time =time()  # Corrected to use time.time() for start time capture
num_iterations = np.zeros((num_genres, num_user_types))  # Each cell has its own iteration count

while time() - start_time < 25:
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