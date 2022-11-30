import tensorflow as tf
import numpy as np
from keras import layers

class T_learner(tf.keras.Model):

  def __init__(self, hidden_dim):
    super().__init__()
    # mu_0 model
    self.mu0_layer1 = layers.Dense(hidden_dim, activation=tf.nn.elu)
    self.mu0_layer2 = layers.Dense(hidden_dim, activation=tf.nn.elu)
    self.mu0_final_layer = layers.Dense(1)
    
    # mu_1 model
    self.mu1_layer1 = layers.Dense(hidden_dim, activation=tf.nn.elu)
    self.mu1_layer2 = layers.Dense(hidden_dim, activation=tf.nn.elu)
    self.mu1_final_layer = layers.Dense(1)
    

  def call(self, inputs):
    # mu_0 model
    y0_pred = self.mu0_layer1(inputs)
    y0_pred = self.mu0_layer1(y0_pred)
    y0_pred = self.mu0_final_layer(y0_pred)

    # mu_1 model
    y1_pred = self.mu1_layer1(inputs)
    y1_pred = self.mu1_layer1(y1_pred)
    y1_pred = self.mu1_final_layer(y1_pred)

    return layers.Concatenate(1)([y0_pred, y1_pred])

def t_learner_loss(pred, true):
    # every loss function in TF2 takes 2 arguments, a vector of true values and a vector predictions
    y_true = true[:, 0] #get individual vectors
    t_true = true[:, 1]

    y0_pred = pred[:, 0]
    y1_pred = pred[:, 1]

    #Each head outputs a prediction for both potential outcomes
    #We use t_true as a switch to only calculate the factual loss
    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))
    #note Shi uses tf.reduce_sum for her losses instead of tf.reduce_mean.""
    #They should be equivalent but it's possible that having larger gradients accelerates convergence.
    #You can always try changing it!
    return loss0 + loss1




        



