import tensorflow as tf
import numpy as np
from keras import layers

def make_tlearner(input_dim, reg_l2, hidden_dim):
    '''
    The first argument is the column dimension of our data.
    It needs to be specified because the functional API creates a static computational graph
    The second argument is the strength of regularization we'll apply to the output layers
    '''
    x = layers.Input(shape=(input_dim,), name='input')

    #in TF2/Keras it is idiomatic to instantiate a layer and pass its inputs on the same line unless the layer will be reused
    # HYPOTHESIS
    y0_hidden = layers.Dense(units=hidden_dim, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(reg_l2),name='y0_hidden_1')(x)
    y1_hidden = layers.Dense(units=hidden_dim, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(reg_l2),name='y1_hidden_1')(x)

    # second layer
    y0_hidden = layers.Dense(units=hidden_dim, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(reg_l2),name='y0_hidden_2')(y0_hidden)
    y1_hidden = layers.Dense(units=hidden_dim, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(reg_l2),name='y1_hidden_2')(y1_hidden)

    # third
    y0_predictions = layers.Dense(units=1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
    y1_predictions = layers.Dense(units=1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)

    #a convenience "layer" that concatenates arrays as columns in a matrix
    concat_pred = layers.Concatenate(1)([y0_predictions, y1_predictions])
    #the declarations above have specified the computational graph of our network, now we instantiate it
    model = tf.keras.Model(inputs=x, outputs=concat_pred)

    return model

def regression_loss(concat_true, concat_pred):
    #computes a standard MSE loss for TARNet
    y_true = concat_true[:, 0] #get individual vectors
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    #Each head outputs a prediction for both potential outcomes
    #We use t_true as a switch to only calculate the factual loss
    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))
    #note Shi uses tf.reduce_sum for her losses instead of tf.reduce_mean.""
    #They should be equivalent but it's possible that having larger gradients accelerates convergence.
    #You can always try changing it!
    return loss0 + loss1




        



