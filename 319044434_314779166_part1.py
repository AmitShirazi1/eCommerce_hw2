#TASK 1
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

# Load the data
user_clip_train = pd.read_csv('user_clip.csv').dropna()

# Calculate the average rating
r_avg = user_clip_train['weight'].mean()

# Create mappings for user_id and clip_id to indices
user_ids = pd.Categorical(user_clip_train['user_id'])
user_map = {user: i for i, user in enumerate(user_ids.categories)}
clip_ids = pd.Categorical(user_clip_train['clip_id'])
clip_map = {clip: i for i, clip in enumerate(clip_ids.categories)}

# Extract user and clip indices
user_indices = user_clip_train['user_id'].map(user_map).values
clip_indices = user_clip_train['clip_id'].map(clip_map).values

# Create the sparse design matrix
num_ratings = len(user_clip_train)
num_users = len(user_ids.categories)
num_clips = len(clip_ids.categories)

# Build the sparse matrix using user and clip indices
row_indices = np.arange(num_ratings)
col_indices_users = user_indices
col_indices_clips = clip_indices + num_users

data_users = np.ones(num_ratings)
data_clips = np.ones(num_ratings)

row_indices_combined = np.concatenate([row_indices, row_indices])
col_indices_combined = np.concatenate([col_indices_users, col_indices_clips])
data_combined = np.concatenate([data_users, data_clips])

# Create the sparse matrix
A = csr_matrix((data_combined, (row_indices_combined, col_indices_combined)), shape=(num_ratings, num_users + num_clips))

# Create the target vector by subtracting the average weight from the actual weights
y = np.array(user_clip_train['weight'] - r_avg)

# Solve the linear system to find the biases for users and clips using the least-squares method
b = lsqr(A, y)[0]

# Store the biases in a dictionary
bias_dict = pd.Series(b, index=[f"user_{user_id}" for user_id in user_ids.categories] + [f"clip_{clip_id}" for clip_id in clip_ids.categories])

# Function to predict rank for a single row
def predict_viewtime(row):
    user_bias = bias_dict.get(f"user_{row['user_id']}", 0)
    clip_bias = bias_dict.get(f"clip_{row['clip_id']}", 0)
    return r_avg + user_bias + clip_bias

# Load the test data
test_df = pd.read_csv('test.csv').dropna()

# Predict weights for the test dataset
test_df['weight_pred'] = test_df.apply(predict_viewtime, axis=1)

# Ensure no negative predictions
test_df['weight_pred'] = test_df['weight_pred'].clip(lower=0)

# Save the predictions to a CSV file
test_df[['user_id', 'clip_id', 'weight_pred']].to_csv('319044434_314779166_task1.csv', index=False)

# Function to calculate the objective function (error + regularization)
def f1(df, user_bias, clip_bias):
    error = ((df['prediction'] - df['weight']) ** 2).sum()
    regularization = 0.1 * ((user_bias ** 2).sum() + (clip_bias ** 2).sum())
    return error + regularization

# Calculate predictions for training data
user_clip_train['prediction'] = user_clip_train.apply(predict_viewtime, axis=1)

# Extract user and clip biases from the bias series
user_bias_values = bias_dict.iloc[:num_users].values
clip_bias_values = bias_dict.iloc[num_users:].values

# Calculate the F1 score for training data
f1_score = f1(user_clip_train, user_bias_values, clip_bias_values)
print(f'F1 SCORE: {f1_score}')





#TASK 2
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Load the training data and remove any rows with missing values
user_clip = pd.read_csv('user_clip.csv').dropna()

# Create a pivot table with 'user_id' as the index, 'clip_id' as the columns, and 'weight' as the values
# Fill any missing values with 0
user_clip_matrix = user_clip.pivot(index='user_id', columns='clip_id', values='weight').fillna(0)

# Extract the indices and columns from the pivot table
user_ids = user_clip_matrix.index
clip_ids = user_clip_matrix.columns

# Convert the pivot table to a sparse matrix format
user_clip_matrix = csr_matrix(user_clip_matrix.values)

# Perform Singular Value Decomposition (SVD) to decompose the sparse matrix
U, Σ, V_T = svds(user_clip_matrix, k=20)

# Reconstruct the matrix using the decomposed matrices
predicted_user_clip_matrix = U @ np.diag(Σ) @ V_T

# Convert the reconstructed matrix back to a DataFrame for easier manipulation
predicted_user_clip_matrix = pd.DataFrame(predicted_user_clip_matrix, columns=clip_ids, index=user_ids)

# Define a function to predict the weight for a given user and clip
def predict(user_id_test, clip_id_test):
    if user_id_test in predicted_user_clip_matrix.index and clip_id_test in predicted_user_clip_matrix.columns:
        return predicted_user_clip_matrix.loc[user_id_test, clip_id_test]
    else:
        return 0

# Load the test data and filter out any rows with missing 'user_id' or 'clip_id'
test_df = pd.read_csv('test.csv').filter(['user_id', 'clip_id']).dropna()

# Apply the predict function to each row in the test data to get the predicted weights
test_df['weight'] = test_df.apply(lambda row: predict(row['user_id'], row['clip_id']), axis=1)

test_df.to_csv('319044434_314779166_task2.csv', index=False)

# Convert the predicted matrix to a long format DataFrame for comparison with the original weights
predicted_user_clip = predicted_user_clip_matrix.reset_index().melt(id_vars='user_id', var_name='clip_id', value_name='weight')
predicted_user_clip = predicted_user_clip.rename(columns={'weight': 'prediction'})

# Define a function to calculate the sum of squared errors (SSE) between the actual and predicted weights
def f2(weights, predictions):
    merged_df = weights.merge(predictions, on=['user_id', 'clip_id'])
    sse = ((merged_df['weight'] - merged_df['prediction']) ** 2).sum()
    return sse

# Print the SSE to evaluate the performance of the model
f2_score=f2(user_clip, predicted_user_clip)
print(f'F2 SCORE: {f2_score}')


