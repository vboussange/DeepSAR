from sklearn.preprocessing import FunctionTransformer
from scikeras.wrappers import KerasRegressor
from src.power_model import PowerLayer, ConstantLayer
from typing import Dict, Iterable, Any
from scikeras.wrappers import KerasRegressor
import keras
import tensorflow as tf


## This is not working, see https://github.com/adriangb/scikeras/issues/311
# class SklPowerModel(KerasRegressor):

#     @property
#     def feature_encoder(self):
#         return FunctionTransformer(
#             func=lambda X: (X[:, 0], X[:, 1:]),
#         )
    
#     def __init__(
#         self,
#         activation="relu",
#         n_neurons=10,
#         optimizer="adam",
#         optimizer__learning_rate=0.001,
#         epochs=200,
#         verbose=0,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.activation = activation
#         self.n_neurons = n_neurons
#         self.optimizer = optimizer
#         self.epochs = epochs
#         self.verbose = verbose

#     def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
#         model = NNPowerModel(n_neurons = self.n_neurons, activation = self.activation)
#         model.compile(loss="mse", optimizer=compile_kwargs["optimizer"])
#         return model
    

# def build_power_model(nn1, nn2, n_neurons=10, activation="relu"):

#     a = keras.layers.Input(shape=(1, ))
#     pred = keras.layers.Input(shape=(1, ))

#     nnpred1 = FFNN3(n_neurons, activation)(a)
#     nnpred2 = FFNN3(n_neurons, activation)(pred)

#     out = tf.multiply(nnpred1, tf.pow(a, nnpred2))

#     model = keras.Model(inputs=[a, pred], outputs=out)

#     return model


## This is an other approach using self containd classes
## which may not be as flexible as build a SciKeras model from a function

# class SklPowerModel(KerasRegressor):

#     @property
#     def feature_encoder(self):
#         return FunctionTransformer(
#             func=lambda X: (X[:, 0], X[:, 1:]),
#         )
    
#     def __init__(
#         self,
#         c_fun,
#         z_fun,
#         optimizer="adam",
#         loss = 
#         optimizer__learning_rate=0.001,
#         epochs=200,
#         verbose=0,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.c_fun = c_fun
#         self.z_fun = z_fun
#         self.optimizer = optimizer
#         self.loss = loss
#         self.epochs = epochs
#         self.verbose = verbose

#     def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
#         model = keras.Sequential()
#         a = keras.layers.Input(shape=(self.n_features_in_))
#         pred = keras.layers.Input(shape=(self.n_features_in_-1))
#         c = self.c_fun(n_neurons, activation)(pred)
#         z = self.z_fun(n_neurons, activation)(pred)

#         out = tf.multiply(c, tf.pow(a, z))

#         model = keras.Model(inputs=[a, pred], outputs=out)
#         model.compile(loss="mse", optimizer=compile_kwargs["optimizer"])
#         return model

# most convenient option:
def build_power_model(c_fun, z_fun, meta: Dict[str, Any], compile_kwargs: Dict[str, Any]):
    model = keras.Sequential()
    a = keras.layers.Input(shape=(1,))
    pred = keras.layers.Input(shape=(meta["n_features_in_"]-1))
    c = c_fun(pred)
    z = z_fun(pred)

    out = PowerLayer()(c, a, z)

    model = keras.Model(inputs=[a, pred], outputs=out)
    return model

class MultiInputRegressor(KerasRegressor):
    @property
    def feature_encoder(self):
        return FunctionTransformer(
            func=lambda X: [X[:, 0], X[:, 1:]],
        )

if __name__ == "main":
    a = np.random.rand(10,1)
    pred = np.random.randn(10,10)
    X_sklearn = np.column_stack((a, pred))

    y = np.random.rand(10)
    
    c_fun = keras.layers.Dense(1)
    z_fun = keras.layers.Dense(1)
    model = MultiInputRegressor(model = build_power_model, 
                                c_fun=c_fun, 
                                z_fun=z_fun,
                                loss = keras.losses.Poisson(), 
                                optimizer = keras.optimizers.Adam())
    model.fit(X_sklearn, y)
    