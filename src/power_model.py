import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import numpy as np

# TODO: this throws
class ConstantLayer(keras.layers.Layer):
    def __init__(self, val = np.random.rand()):
        super().__init__()
        self.cst = tf.Variable(
            initial_value=val,
            trainable=True,
        )

    def call(self, inputs):
        return self.cst

class PowerLayer(keras.layers.Layer):
    def __init__(self):
        super(PowerLayer, self).__init__()

    def call(self, c, a, z):
        return tf.multiply(c, tf.pow(a, z))

class FFNN3(keras.layers.Layer):
    def __init__(self, n_neurons = 10, activation="tanh"):
        super(FFNN3, self).__init__()
        self.n_neurons = n_neurons
        self.layer1 = Dense(units=n_neurons, activation=activation)
        self.layer2 = Dense(units=n_neurons, activation=activation)
        self.layer3 = Dense(units=1, activation=activation)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class NNPowerModel(tf.keras.Model):
    def __init__(self, n_neurons=10, activation="relu"):
        super(NNPowerModel, self).__init__()
        self.nn1 = FFNN3(n_neurons, activation)
        self.nn2 = FFNN3(n_neurons, activation)
        self.power_layer = PowerLayer()

    def call(self,inputs):
        a, pred = inputs
        nnpred1 = self.nn1(pred)
        nnpred2 = self.nn2(pred)
        
        x = self.power_layer((a, nnpred1, nnpred2))
        return x


class NNPowerModelSimple(tf.keras.Model):
    def __init__(self, n_neurons=10, activation="relu"):
        super().__init__()
        self.nn1 = FFNN3(n_neurons, activation)
        self.nn2 = FFNN3(n_neurons, activation)

    def call(self,inputs):
        a, pred = inputs
        nnpred1 = self.nn1(pred)
        nnpred2 = self.nn2(pred)
        
        x = tf.multiply(nnpred1, tf.pow(a, nnpred2))
        # x = nnpred1
        return x

if __name__ == '__main__':
    ################################
    ### testing 
    ################################
    model = NNPowerModelSimple()
    a = tf.ones(shape=(10,1))
    pred = tf.ones(shape=(10,2))
    model((a,pred))
    model.predict((a,pred)).shape

    ################################
    ### Further testing 
    ################################
    import pandas as pd
    import numpy as np

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA

    # Load the dataset
    traindata = pd.read_csv('../data/processed_data/forest_t1_SAR_traindata.csv')
    testrawdata = pd.read_csv('../data/processed_data/forest_t1_SAR_testrawdata.csv')

    # Extract the 'a' column and feature columns
    a_data = traindata['a'].values.reshape(-1,1)
    x_data = traindata.drop(columns=['a', 'sr']).values

    # Standardize the data
    scaler = MinMaxScaler()
    x_data = scaler.fit_transform(x_data)

    # Perform PCA to reduce the number of features
    n_components = 5  # Choose the number of components you want to keep
    pca = PCA(n_components=n_components)
    x_data = pca.fit_transform(x_data)

    y_data = traindata['sr'].values# Your target values

    model = NNPowerModelSimple(n_neurons = 10, activation="relu")

    # testing model
    model((a_data,x_data))

    # Compile the model
    # loss = tf.keras.losses.Poisson()
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(0.001), loss='mse')

    # Train the model
    model.fit((a_data,x_data), y_data, epochs=100, batch_size=128)

    model.predict((a_data,x_data))

    # y_data
    # # Make predictions
    # x_test_data = # Test feature data
    # a_test_data = # Test scalar feature 'a' data
    # predictions = model.predict([x_test_data, a_test_data])

    # import matplotlib.pyplot as plt

    # # Assuming you have already trained the model and have predictions and observed values
    # observed_values = # Your observed (actual) values
    # predicted_values = predictions  # Replace with your model's predictions

    # # Create a scatter plot
    # plt.scatter(observed_values, predicted_values)
    # plt.xlabel('Observed Values')
    # plt.ylabel('Predicted Values')
    # plt.title('Observed vs. Predicted Values')
    # plt.show()
