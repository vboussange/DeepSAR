"""
Power model
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
traindata = pd.read_csv('../../data/processed_data/forest_t1_SAR_traindata.csv')
testrawdata = pd.read_csv('../../data/processed_data/forest_t1_SAR_testrawdata.csv')

# Extract the 'a' column and feature columns
a_data = traindata['a'].values
x_data = traindata.drop(columns=['a', 'sr']).values

# Standardize the data
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

# Perform PCA to reduce the number of features
n_components = 5  # Choose the number of components you want to keep
pca = PCA(n_components=n_components)
x_data = pca.fit_transform(x_data)

def custom_model(input_dim, nn1_units, nn2_units):
    # Input layer for the feature vector x
    x_input = Input(shape=(input_dim,), name='x_input')
    
    # Input layer for the scalar feature A
    a_input = Input(shape=(1,), name='a_input')
    
    # Define the first neural network (NN1) for the feature vector x
    nn1 = Dense(nn1_units, activation='relu', use_bias = False)(x_input)
    
    # Define the second neural network (NN2) for the feature vector x
    nn2 = Dense(nn2_units, activation='relu', use_bias = False)(x_input)
        
    # Multiply the output of NN1 with the exponentiated scalar feature A
    output = nn1 * tf.pow(a_input, nn2)
    
    # Create the custom model
    model = Model(inputs=[x_input, a_input], outputs=output)
    
    return model

# Instantiate the custom model with your desired parameters
input_dim = x_data.shape[1]  # Input dimension is the number of feature columns
nn1_units = 1  # Define the number of units in NN1
nn2_units = 1  # Define the number of units in NN2


model = custom_model(input_dim, nn1_units, nn2_units)

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.Poisson(), metrics=['mean_absolute_error'])

# Train the model
y_data = traindata['sr'].values# Your target values
model.fit([x_data, a_data], y_data, epochs=100, batch_size=64)

# Make predictions
x_test_data = # Test feature data
a_test_data = # Test scalar feature 'a' data
predictions = model.predict([x_test_data, a_test_data])

import matplotlib.pyplot as plt

# Assuming you have already trained the model and have predictions and observed values
observed_values = # Your observed (actual) values
predicted_values = predictions  # Replace with your model's predictions

# Create a scatter plot
plt.scatter(observed_values, predicted_values)
plt.xlabel('Observed Values')
plt.ylabel('Predicted Values')
plt.title('Observed vs. Predicted Values')
plt.show()
