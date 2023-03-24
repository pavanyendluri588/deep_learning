from keras.models import Sequential
from keras.layers import Activation,Dense
import keras as kk
import numpy as np 


#creating  a neural network 
nn= Sequential()
#step2 add input layer  by using the dense function (input_dim=6 or bu using the input_SHAPE=(6,)or by using the keras.Input(shape(16,))
#STEP 3 add the activation function by using the .add(Activation) or by using the parameter inside the activation function  activation  
nn.add(Dense(2,activation="sigmoid",input_dim=2))

#after first layer we do not need to create the input 
#because after the is rest all are the hidden layers except the last layer because it is a oupput layer 
nn.add(Dense(12))
nn.add(Activation("relu"))
nn.add(Dense(6,activation="relu"))
#adding the last layer output layer 
nn.add(Dense(2))


#getting the summaery of the model 
nn.summary()

model=Sequential(
    [
    Dense(5,activation="relu"),
    Dense(7,activation="relu"),
    Dense(4,activation="relu"),
    Dense(2)
    ]
)

#model.weights

#model.summary()
model.build((None,22, 32, 32, 3))
x=[[1,2,3,4,5],
   [11,22,33,44,55],
   [21,22,23,24,25],
   [31,32,33,34,35],
   [41,42,43,44,45]]
print(model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy']) )
model.weights
model.summary()
