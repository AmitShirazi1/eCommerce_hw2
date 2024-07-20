# Task 1

import pandas as pd

user_clip = pd.read_csv('user_clip.csv').dropna()
r_avg = user_clip['weight'].mean()
users_bias = user_clip.groupby('user_id')['weight'].mean() - r_avg
clips_bias = user_clip.groupby('clip_id')['weight'].mean() - r_avg

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

def f1(df):
    error = ((df['prediction'] - df['weight']) ** 2).sum()
    regularization = 0.1 * ((df['user_bias'] ** 2).sum() + (df['clip_bias'] ** 2).sum())
    return error + regularization

print('f1 value:', f1(user_clip))

# TODO: Question for the metargel:
# 1. What do we do with predictions below 0 (both in train and test)?
# 2. What do we do with missing values (currently - we drop them)?
# 3. Is f1 and f2 on the test or train?

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


