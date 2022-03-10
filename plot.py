from random import seed
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf

seed=42

np.random.seed=seed
with open('redata.pkl','rb') as f:
    data=pickle.load(f)

model=tf.keras.models.load_model('model.h5')

train_Path="./plot/training_data_/"
test_Path="./plot/test_data_/"



x_train=data['train']['X']
y_train=data['train']['Y']

x_test=data['test']['X']
ix=np.random.randint(0,x_train.shape[0])
iy=np.random.randint(0,x_test.shape[0])
np.random.randint(0,x_train)
fig,ax=plt.subplots(1,3,figsize=(15,15))
plt.title("training_data")
ax[0].imshow(x_train[ix])
ax[1].imshow(y_train[ix])
ax[2].imshow(model.predict(x_train[ix:ix+1])[0])
plt.show()
fig.savefig(train_Path+"training_data_"+str(ix)+".png")

fig,ax=plt.subplots(1,2,figsize=(15,15))
plt.title("test_data")
ax[0].imshow(x_test[iy])
ax[1].imshow(model.predict(x_test[iy:iy+1])[0])
plt.show()
fig.savefig(test_Path+"test_data_"+str(iy)+".png")