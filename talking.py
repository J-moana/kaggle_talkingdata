
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


df = pd.read_csv('train_sample.csv')
df = df.drop(columns=['click_time','attributed_time'], axis=1)
y = df['is_attributed']
df = df.iloc[:,:-1]

x_train, x_val, y_train, y_val = train_test_split(df,y,test_size=0.2)
# x_train = x_train.transpose()
# x_val = x_val.transpose()
# y_train = y_train.transpose()
# y_val = y_val.transpose()
print(x_train.shape,x_val.shape,y_train.shape,y_val.shape)
print(y_train.sum(),y_val.sum())

model = Sequential()
model.add(Dense(512,activation='relu',input_shape=(5,)))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=64,epochs=5,validation_data=(x_val,y_val))

score, acc = model.evaluate(x_val,y_val,batch_size=32)
print('Test score:',score)
print('Test accuracy:',acc)

#
#
df_sub = pd.DataFrame()
#
x_test = pd.read_csv('test.csv')
df_sub['click_id'] = x_test['click_id']
x_test = x_test.drop(columns=['click_id','click_time'], axis=1)
df_sub['is_attributed'] = model.predict(x_test,batch_size=64,verbose=0)

print(df_sub['is_attributed'].sum())
# print(preds)

df_sub.to_csv('result1.csv')
