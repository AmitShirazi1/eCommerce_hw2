# Task 1

import pandas as pd

user_clip = pd.read_csv('user_clip.csv').dropna()
r_avg = user_clip['weight'].mean()
initial_users_bias = user_clip.groupby('user_id')['weight'].mean() - r_avg
initial_clips_bias = user_clip.groupby('clip_id')['weight'].mean() - r_avg

# Initialize parameters
users = user_clip['user_id'].unique()
clips = user_clip['clip_id'].unique()
num_users = len(users)
num_clips = len(clips)

user_map = {user: i for i, user in enumerate(users)}
clip_map = {clip: i for i, clip in enumerate(clips)}

user_indices = user_clip['user_id'].map(user_map).values
clip_indices = user_clip['clip_id'].map(clip_map).values
train_weights = user_clip['weight'].values

import numpy as np
import pandas as pd
import tqdm

user_bias = np.zeros(num_users)
clip_bias = np.zeros(num_clips)

learning_rate = 0.01
num_iterations = 10

# Initialize matrices
R = np.zeros((num_users, num_clips))
for u, i, w in zip(user_indices, clip_indices, train_weights):
    R[u, i] = w

# Regularization parameter
lambda_reg = 0.1

# Identity matrices
I_u = np.eye(num_users)
I_i = np.eye(num_clips)
print('starting')

# Solve for user biases (b_u)
b_i = np.zeros(num_clips)
for i in range(num_clips):
    b_i[i] = np.mean(R[:, i])

b_u = np.linalg.inv(I_u + lambda_reg * I_u).dot(R - b_i)

# Solve for item biases (b_i)
b_i = np.linalg.inv(I_i + lambda_reg * I_i).dot(R.T - b_u).T
print(user_bias)
print(clip_bias)

test_df = pd.read_csv('test.csv').filter(['user_id', 'clip_id']).dropna()
users_bias = users_bias.reset_index().rename(columns={'index': 'user_id', 'weight': 'user_bias'})
clips_bias = clips_bias.reset_index().rename(columns={'index': 'clip_id', 'weight': 'clip_bias'})
test_df = test_df.merge(users_bias, on=['user_id'], how='left')
test_df = test_df.merge(clips_bias, on=['clip_id'], how='left')
test_df['prediction'] = r_avg + test_df['user_bias'] + test_df['clip_bias']
test_df.filter(['user_id', 'clip_id', 'prediction']).rename(columns={'prediction': 'weight'}).to_csv('319044434_314779166_task1.csv', index=False)

user_clip = user_clip.merge(users_bias, on=['user_id'], how='left')
user_clip = user_clip.merge(clips_bias, on=['clip_id'], how='left')
user_clip['prediction'] = r_avg + user_clip['user_bias'] + user_clip['clip_bias']
user_clip['prediction'] = user_clip['prediction'].clip(lower=0)



print('f1 value:', f1(user_clip))

# TODO: Question for the metargel:
# 1. What do we do with predictions below 0 (both in train and test)? Nothing.
# 2. What do we do with missing values (currently - we drop them)?
# 3. Is f1 and f2 on the test or train?
# 4. If a movie\user is missing from the test, maybe fill in the average?

# Task 2
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

user_clip = pd.read_csv('user_clip.csv').dropna()
user_clip_matrix = user_clip.pivot(index='user_id', columns='clip_id', values='weight').fillna(0)
user_ids = user_clip_matrix.index
clip_ids = user_clip_matrix.columns

user_clip_matrix = csr_matrix(user_clip_matrix.values)
U, Σ, V_T = svds(user_clip_matrix, k=20)
predicted_user_clip_matrix = U @ np.diag(Σ) @ V_T
predicted_user_clip_matrix = pd.DataFrame(predicted_user_clip_matrix, columns=clip_ids, index=user_ids)

def predict(user_id_test, clip_id_test):
    if user_id_test in predicted_user_clip_matrix.index and clip_id_test in predicted_user_clip_matrix.columns:
        return predicted_user_clip_matrix.loc[user_id_test, clip_id_test]
    else:
        return 0

test_df = pd.read_csv('test.csv').filter(['user_id', 'clip_id']).dropna()
test_df['weight'] = test_df.apply(lambda row: predict(row['user_id'], row['clip_id']) ,axis=1)
test_df.to_csv('319044434_314779166_task2.csv', index=False)

predicted_user_clip = predicted_user_clip_matrix.reset_index().melt(id_vars='user_id', var_name='clip_id', value_name='weight')
predicted_user_clip = predicted_user_clip.rename(columns={'weight': 'prediction'})

def f2(weights, predictions):
    merged_df = weights.merge(predictions, on=['user_id', 'clip_id'])
    sse = ((merged_df['weight'] - merged_df['prediction']) ** 2).sum()
    return sse

print('f2 value:', f2(user_clip, predicted_user_clip))


