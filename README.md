Making the dataset file to a variable to access the arrayed datas.

# importing pandas to read the dataset in csv.
import pandas as pd

# imported file directing to a variable.
dataset = pd.read_csv('cancer.csv')
Need to define spesific needed datas to variables as prefered to access.

# defining x axis datas as x variable.
x = dataset.drop(columns=['diagnosis(1=m, 0=b)'])
# defining y axis datas as y variable.
y = dataset['diagnosis(1=m, 0=b)']
Take train_test_split library to access and split data as we want and access the all the data.

# import train_test_split from sklearn.
from sklearn.model_selection import train_test_split

# defining necessory variables to use for the model.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
Take tensorflow library and keras to access models in sequential mode.

# import tensorflow as 'tf'.
import tensorflow as tf

# defining a variable for a model and to take data in a sequentially. 
model = tf.keras.models.Sequential()
Addding to tf keras and assigning a dense to set prorities and giving inputs to the model. To analize data use Sigmoid to simplify the all predictions.

# adding to model and set inputs.
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape, activation='sigmoid'))

# again assigning to use the sigmoid with a dense.
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))

# lowing dense and saying sigmoid to analyze the finals.
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
Compile the model and use metrics and optimizer to fine the output and make a priority phase.

# compiling the model using an optimizer and metrics.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
Set trains to model and fit the x_train and y_train and use epochs to make how many turns the predictions have to be done. And evaluvating the final putput to get a fine readable output.

# set data with the model 
model.fit(x_train, y_train, epochs = 1000)
# evaluating the output datasets to simply and final answer.
model.evaluate(x_test, y_test)
