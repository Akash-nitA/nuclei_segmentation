import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from model import unet_model
from skimage.io import imshow
import matplotlib.pyplot as plt

with open("redata.pkl", "rb") as f:
    data = pickle.load(f)

x_train=data['train']['X']
y_train=data['train']['Y']


callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)
model=unet_model()
model.fit(x_train,y_train,epochs=25,batch_size=16,validation_split=0.1,callbacks=callback)
model.save('model.h5')