import os
import pathlib
os.chdir(pathlib.Path(__file__).parent.resolve()) # changing working dir to file's directory
import pytest
import numpy as np
from mixed_model_Weibull import *

class TestComposableWeibull:
    weib_lay = ComposableWeibull(d=1., c=1.)
    assert weib_lay((tf.zeros(1), tf.ones(1))) == 0.
    assert weib_lay((tf.ones(1), tf.ones(1))) == 1. - np.exp(-1.)

class TestFFNN3:
    ffnn3 = FFNN3()
    pred = ffnn3(tf.ones(shape=(1,4)))
    assert isinstance(pred, tf.Tensor)
    assert all(tf.shape(pred) == [1,1])

class TestWeibullNN:
    ffnn3 = WeibullNN()
    assert isinstance(pred, tf.Tensor)
    assert all(tf.shape(pred) == [1,1])

n_neurons = 10
laydense = Dense(units=n_neurons, activation="tanh")
laydense(tf.ones(shape=(1,3)))

weibullnn = WeibullNN()
a = tf.ones(shape=(10,1))
pred = tf.ones(shape=(10,2))
weibullnn(a,pred)
