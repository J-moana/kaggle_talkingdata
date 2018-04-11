import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from sklearn.model_selection import train_test_split


df = pd.read_csv('train.csv')
# df = df.drop(columns=['attributed_time'], axis=1)

# df_sub = df[df['is_attributed'] == 1]
# print(df_sub.shape)
s = df[df['is_attributed'] == 1].shape

# k = np.random.randint(0,df.shape[0],size=(df_sub.shape[0]))
# kk = df.iloc[np.random.randint(0,df.shape[0],size=(df_sub.shape[0]))]

df = pd.concat([df[df['is_attributed'] == 1], df.iloc[np.random.randint(0,df.shape[0],size=(s[0]))]])
df = df.iloc[np.random.permutation(len(df))]
# print(df.head(30))
# print(df.tail(30))
# print(df.shape)
# print(df_sub.shape)

df.to_csv('train_sampling.csv')
