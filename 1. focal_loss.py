# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:16:41 2023

@author: alexd
"""

from keras import backend as K
import tensorflow as tf

# Compatible with tensorflow backend

def focal_loss_fixed(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed


# Register the custom loss function with Keras
tf.keras.utils.get_custom_objects().update({'focal_loss_fixed': focal_loss_fixed})

