import tensorflow as tf
import keras

from keras.layers import Dropout,BatchNormalization,Conv1D
from keras.layers import MaxPooling1D,Flatten,Dense

from keras.models import Sequential
from keras.layers import LeakyReLU
from keras.optimizers import Adam

from keras import metrics

def run_model(train_vecs_features,test_vecs_features,train_l):

	model = Sequential()
	model.add(Conv1D(32,kernel_size=7,input_shape=(50,26)))
	model.add(Conv1D(20,kernel_size=7))
	model.add(Conv1D(12,kernel_size=5))
	model.add(MaxPooling1D(2))
	model.add(Conv1D(6,kernel_size=3))
	model.add(Flatten())
	model.add(Dense(8))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dense(4))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dense(1,activation='sigmoid'))

	model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["accuracy"])

	model.fit(train_vecs_features,train_l,batch_size=128,epochs=50,verbose=2,shuffle=True)
	
	pred = model.predict(test_vecs_features,batch_size=32)

	return pred
