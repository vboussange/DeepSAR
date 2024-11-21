import pytest
import tensorflow as tf
from src.sklearn_models import MultiInputRegressor, build_power_model
from src.power_model import ConstantLayer
import numpy as np
from tensorflow import keras
from keras.initializers import Constant

def test_power_model():
    npoints = 1000
    aa = np.linspace(1, 1000, num=npoints).reshape(-1,1)
    pred = np.random.rand(npoints,2)
    X_sklearn = np.column_stack((aa, pred))

    c = 1.
    z = 0.1
    y = c * aa ** z
    
    c_fun = ConstantLayer()
    z_fun = ConstantLayer()
    model = MultiInputRegressor(model = build_power_model, 
                                c_fun=c_fun, 
                                z_fun=z_fun,
                                loss = keras.losses.MeanSquaredLogarithmicError(), 
                                epochs = 100,
                                batch_size = 100,
                                optimizer = keras.optimizers.legacy.Adam(learning_rate=0.1), 
                                # run_eagerly = True
                                )
    model.fit(X_sklearn, y)
    assert np.allclose(model.model_.get_weights(), [c, z], rtol = 0.1)
    
    # testing with non-zero predictors
    y = np.sum(pred, axis=1).reshape(-1,1) * aa ** z
    c_fun = keras.layers.Dense(units=1, kernel_initializer='ones', bias_initializer=Constant(value=0.0))
    c_fun.trainable = False
    z_fun = ConstantLayer(z)
    z_fun.trainable = False

    model = MultiInputRegressor(model = build_power_model, 
                                c_fun=c_fun, 
                                z_fun=z_fun,
                                loss = keras.losses.MeanSquaredLogarithmicError(), 
                                epochs = 100,
                                batch_size = 100,
                                optimizer = keras.optimizers.legacy.Adam(learning_rate=0.01))
    model.fit(X_sklearn, y)
    assert model.history_["loss"][-1] < 1e-10
    
    # making sure that this is not spurious results
    pred = np.random.rand(npoints,2)
    X_sklearn2 = np.column_stack((aa, pred))
    model = MultiInputRegressor(model = build_power_model, 
                                c_fun=c_fun, 
                                z_fun=z_fun,
                                loss = keras.losses.MeanSquaredLogarithmicError(), 
                                epochs = 100,
                                batch_size = 100,
                                optimizer = keras.optimizers.legacy.Adam(learning_rate=0.01))
    model.fit(X_sklearn2, y)
    assert model.history_["loss"][-1] > 1e-1
    
    # testing neural net
    c_fun = keras.layers.Dense(units=1, kernel_initializer='ones')
    z_fun = keras.layers.Dense(units=1, kernel_initializer='ones')
    model = MultiInputRegressor(model = build_power_model, 
                                c_fun=c_fun, 
                                z_fun=z_fun,
                                loss = keras.losses.MeanSquaredLogarithmicError(), 
                                epochs = 200,
                                batch_size = 32,
                                optimizer = keras.optimizers.legacy.Adam(learning_rate=0.01))
    model.fit(X_sklearn2, y)
    # assert model.history_["loss"][-1] < 1e-1