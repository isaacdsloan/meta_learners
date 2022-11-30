import tensorflow as tf
import numpy as np
from keras import layers

class S_learner(tf.keras.Model):

  def __init__(self, hidden_dim):
    super().__init__()
    self.dense1 = layers.Dense(hidden_dim, activation=tf.nn.elu)
    self.dense2 = layers.Dense(hidden_dim, activation=tf.nn.elu)
    self.dense3 = layers.Dense(hidden_dim, activation=tf.nn.elu)
    self.dense4 = layers.Dense(hidden_dim, activation=tf.nn.elu)
    self.dense5 = layers.Dense(1)
    

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    x = self.dense3(x)
    x = self.dense4(x)
    return self.dense5(x)


class ResidualBlock(layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer1 = layers.Dense(hidden_dim, activation='elu')
        



