import pytest
import tensorflow as tf
import numpy as np
from src.power_model import ConstantLayer, PowerLayer, FFNN3, NNPowerModel, NNPowerModelSimple

def test_constant_layer():
    value = 5.0
    constant_layer = ConstantLayer(val=value)
    input_tensor = tf.constant(np.array([[1.0, 2.0, 3.0]]), dtype=tf.float32)
    output = constant_layer(input_tensor)
    assert np.allclose(output.numpy(), value)

# Define test cases for the PowerLayer class
def test_power_layer():
    power_layer = PowerLayer()
    inputs = (tf.constant(3.0), tf.constant(2.0), tf.constant(2.0))
    output = power_layer(inputs)
    assert tf.experimental.numpy.isclose(output, 18.0, rtol=1e-5)
    
# Define test cases for the FFNN3 class
def test_ffnn3():
    ffnn3 = FFNN3()
    inputs = tf.constant([[1.0, 2.0, 3.0]])
    output = ffnn3(inputs)
    assert output.shape == (1, 1)

# Define test cases for the NNPowerModel class
def test_nn_power_model():
    nn_power_model = NNPowerModel()
    a = tf.constant(2.0)
    pred = tf.constant([[1.0, 2.0, 3.0]])
    inputs = (a, pred)
    output = nn_power_model(inputs)
    assert output.shape == (1, 1)

# Define test cases for the NNPowerModelSimple class
def test_nn_power_model_simple():
    nn_power_model_simple = NNPowerModelSimple()
    a = tf.constant(2.0)
    pred = tf.constant([[1.0, 2.0, 3.0]])
    inputs = (a, pred)
    output = nn_power_model_simple(inputs)
    assert output.shape == (1, 1)
    
    # testing with multi entries tensors
    nn_power_model_simple = NNPowerModelSimple()
    a = tf.ones(shape=(10,1))
    pred = tf.ones(shape=(10,2))
    output = nn_power_model_simple((a,pred))
    assert output.shape == (10, 1)
