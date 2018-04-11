import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

def clear_data(df):
    df = df.drop(columns=['attributed_time'], axis=1)
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['weekday'] = df['click_time'].dt.dayofweek
    df['hour'] = df['click_time'].dt.hour
    y = df['is_attributed']
    df = df.drop(columns=['click_time','is_attributed'], axis=1)
    return df, y


def clear_test(df):
    df = df.drop(columns=['attributed_time'], axis=1)
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['weekday'] = df['click_time'].dt.dayofweek
    df['hour'] = df['click_time'].dt.hour
    df = df.drop(columns=['click_time','is_attributed'], axis=1)
    return df


batch_size = 256

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("weights_256_10.h5")
print("Loaded model from disk")




# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
x_test = pd.read_csv('train_test3.csv')
x_test, y_test = clear_data(x_test)
print("read test data")


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score, acc = model.evaluate(x_test,y_test,batch_size=batch_size,verbose=0)
yy = model.predict(x_test,batch_size=batch_size,verbose=0)
y_test['predict'] = yy

print(y_test.head())
print(y_test['predict'])
print('Test score:',score)
print('Test accuracy:',acc)

del x_test, y_test, yy

result_out = False
if result_out == True:
    df_sub = pd.DataFrame()
    xx_test = pd.read_csv('test.csv')
    xx_test = clear_test(xx_test)
    df_sub['click_id'] = x_test['click_id']
    # x_test = x_test.drop(columns=['click_id','click_time'], axis=1)
    df_sub['is_attributed'] = model.predict(xx_test,batch_size=batch_size,verbose=0)

    print(df_sub['is_attributed'].sum())
    # print(preds)

    df_sub.to_csv('result2.csv')
