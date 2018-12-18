import pandas as pd
import numpy as np

df = pd.read_csv('../data/initial/true_car_listings.csv')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.8

train = df[msk]
test = df[~msk]

del test['split']
del train['split']

test.to_csv('../data/initial_random/test.csv', index=False)
train.to_csv('../data/initial_random/train.csv', index=False)
