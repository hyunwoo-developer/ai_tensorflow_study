import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_csv('./대학교 합격 예측하기/gpascore.csv')
data = data.dropna()

x_data = []
for i, rows in data.iterrows():
    x_data.append([rows['gre'], rows['gpa'], rows['rank']])

y_data = data['admit'].values

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(x_data), np.array(y_data), epochs=1000)

#예측

predict = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(predict)