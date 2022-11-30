import tensorflow as tf
import numpy as np
from keras import layers

def make_slearner(reg_l2, hidden_dim, num_layers):
  s_learner = tf.keras.models.Sequential(
    [layers.Dense(units=hidden_dim, activation='elu', kernel_initializer='RandomNormal') for i in np.arange(num_layers)])
  s_learner.add(layers.Dense(units=100, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(reg_l2)))
  s_learner.add(layers.Dense(units=100, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(reg_l2)))
  s_learner.add(layers.Dense(units=1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(reg_l2)))
  return s_learner

  



