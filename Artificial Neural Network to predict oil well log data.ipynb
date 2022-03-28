# Artificial Neural Network built to predict oil well log parameters ('GR', 'SONIC', 'NPHI', 'DENS') of five out of six well log data using the sixth well log data as the input
# Click "Eddie_Portfolio" to see the predicted values. (Excel spreadsheets)
from numpy import sqrt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import Input
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import pandas as pd
from tensorflow.keras import backend
from tensorflow.keras.layers import concatenate
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error
from math import sqrt
import xlsxwriter

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv (r'PLYT-WELL6-CSV.csv')
data = pd.DataFrame(df, columns=['GR','SONIC'])

indexValue = data[data['GR']==-999.25].index
data.drop(indexValue, inplace=True)

indexValue = data[data['SONIC']==-999.25].index
data.drop(indexValue, inplace=True)

indexValue = data[data['NPHI']==-999.25].index
data.drop(indexValue, inplace=True)

indexValue = data[data['DENS']==-999.25].index
data.drop(indexValue, inplace=True)

col_x = pd.DataFrame(data, columns=['GR','SONIC'])
col_y = pd.DataFrame(data, columns=['NPHI'])

X = col_x.values
Y = col_y.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
n_features = X_train.shape[1]

x_in = Input(shape=(n_features,))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(500, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(400, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(300, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(200, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(50, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(1))


opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=opt, loss='mse')
model.fit(X_train, Y_train, epochs=1000, batch_size=15, verbose=1)

y_pred = model.predict(X_test)

print("MSE: ",mean_squared_error(Y_test, y_pred))
print("RMSE: ",sqrt(mean_squared_error(Y_test, y_pred)))
print("R2 score: ",r2_score(Y_test, y_pred))
print("MAE: ",mean_absolute_error(Y_test, y_pred))
