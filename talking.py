
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from keras import initializers
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import TensorBoard

df = pd.read_csv('train_sampling.csv')
df = df.drop(columns=['attributed_time'], axis=1)
df['click_time'] = pd.to_datetime(df['click_time'])
df['weekday'] = df['click_time'].dt.dayofweek
df['hour'] = df['click_time'].dt.hour
y = df['is_attributed']
df = df.drop(columns=['click_time','is_attributed'], axis=1)
print(df.head())


x_train, x_val, y_train, y_val = train_test_split(df,y,test_size=0.1)
# x_train = x_train.transpose()
# x_val = x_val.transpose()
# y_train = y_train.transpose()
# y_val = y_val.transpose()
print(x_train.shape,x_val.shape,y_train.shape,y_val.shape)
print(y.size - y.sum())
class_weight = {0:1., 1:1.}
del df, y

# def my_init(shape, name=None):
#     return initializers.normal(shape, scale=0.01, name=name)


model = Sequential()
model.add(Dense(32,activation='relu',kernel_initializer='random_uniform',
                bias_initializer='zeros',input_shape=(x_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(64,kernel_initializer='random_uniform',
                bias_initializer='zeros',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(16,kernel_initializer='random_uniform',
                bias_initializer='zeros',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1,activation='sigmoid'))


#
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

batch_size = 128
eopoch = 10
# default = (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size = batch_size,epochs=eopoch,validation_data=(x_val,y_val),callbacks=[tensorboard])

score, acc = model.evaluate(x_val,y_val,batch_size=1024)
print('Test score:',score)
print('Test accuracy:',acc)

#
#
df_sub = pd.DataFrame()
#
result_out = False
if result_out == True:
    x_test = pd.read_csv('test.csv')
    df_sub['click_id'] = x_test['click_id']
    x_test = x_test.drop(columns=['click_id','click_time'], axis=1)
    df_sub['is_attributed'] = model.predict(x_test,batch_size=batch_size,verbose=0)

    print(df_sub['is_attributed'].sum())
    # print(preds)

    df_sub.to_csv('result1.csv')
