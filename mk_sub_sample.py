import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')
print(df.shape)
df, test = train_test_split(df,test_size=0.1)
print(df.shape,test.shape)
s = df[df['is_attributed'] == 1].shape

df = pd.concat([df[df['is_attributed'] == 1], df.iloc[np.random.randint(0,df.shape[0],size=(s[0]*9))]])
df = df.iloc[np.random.permutation(len(df))]


df.to_csv('train_sampling3.csv')
test.to_csv('train_test3.csv')
