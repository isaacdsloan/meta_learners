import pandas as pd
import numpy as np
import tensorflow as tf

def plot_cates(y0_pred,y1_pred,data):
  #dont forget to rescale the outcome before estimation!
  # what is the inverse transform for
  # use transform on y to train DL model easier, must inverse to get true values back
  y0_pred = data['y_scaler'].inverse_transform(y0_pred)
  y1_pred = data['y_scaler'].inverse_transform(y1_pred)
  cate_pred=(y1_pred-y0_pred).squeeze()
  cate_true=(data['mu_1']-data['mu_0']).squeeze() #Hill's noiseless true values
  ate_pred=tf.reduce_mean(cate_pred)

  print(pd.Series(cate_pred).plot.kde(color='blue'))
  print(pd.Series(cate_true).plot.kde(color='green'))

  print(pd.Series(cate_true-cate_pred).plot.kde(color='red'))
  pehe=tf.reduce_mean( tf.square( ( cate_true - cate_pred) ) )
  sqrt_pehe=tf.sqrt(pehe).numpy()
  print("\nSQRT PEHE:",sqrt_pehe)
  print("Estimated ATE (True is %.3f): %.3f \n\n" % (tf.reduce_mean(data['mu_1']-data['mu_0']), ate_pred.numpy()))
  
  print("\nError CATE Estimates: RED")
  print("Individualized CATE Estimates: BLUE")
  print("Individualized CATE True: GREEN")
  return sqrt_pehe,np.abs(ate_pred.numpy()-4)

def plot_cates_uber(cate_pred, data):
  cate_true=(data['mu_1']-data['mu_0']).squeeze() #Hill's noiseless true values
  ate_pred=tf.reduce_mean(cate_pred)

  print(pd.Series(cate_pred).plot.kde(color='blue'))
  print(pd.Series(cate_true).plot.kde(color='green'))

  print(pd.Series(cate_true-cate_pred).plot.kde(color='red'))
  pehe=tf.reduce_mean( tf.square( ( cate_true - cate_pred) ) )
  sqrt_pehe=tf.sqrt(pehe).numpy()
  print("\nSQRT PEHE:",sqrt_pehe)
  print("Estimated ATE (True is %.3f): %.3f \n\n" % (tf.reduce_mean(data['mu_1']-data['mu_0']), ate_pred.numpy()))
  
  print("\nError CATE Estimates: RED")
  print("Individualized CATE Estimates: BLUE")
  print("Individualized CATE True: GREEN")
  return sqrt_pehe,np.abs(ate_pred.numpy()-4)