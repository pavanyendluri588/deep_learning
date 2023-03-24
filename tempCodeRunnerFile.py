from keras.layers  import Dense,Activation
from keras.models import Sequential
from keras.activations import sigmoid 
from keras.activations import relu


model = Sequential()
model.add(Dense(5,input_dim=5))
model.add(Activation("sigmoid"))
model.add(Dense(4))
model.add(Activation("sigmoid"))
model.add(Dense(3))
model.add(Activation("sigmoid"))
model.add(Dense(2))
model.add(Activation("sigmoid"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.summary()
