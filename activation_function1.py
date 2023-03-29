from keras.layers  import Dense,Activation
from keras.models import Sequential
from keras.activations import sigmoid 
from keras.activations import relu
from keras.utils.vis_utils import plot_model
import keras 



print("\n===========\n",keras.backend.backend(),"\n==========")

model = Sequential()
model.add(Dense(5,input_dim=2))
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


#plot_model(model, to_file='model.png')

import tensorflow as tf
# define the input tensor
input_tensor = tf.constant([-2.0,-1.0,0.0,1.0,2.0])

# define the ReLU activation function
sigmoid1 = tf.keras.activations.sigmoid

# apply the ReLU function to the input tensor
output = sigmoid1(input_tensor)
print("sigmoid function:\ninput tensor",input_tensor,"\nsigmoid activation function result:",output)


input_tensor2 = tf.constant([-3.5,-2.5,-2.0,-1.7,-1.1,0,1.1,1.2,1.6,1.8])
output=tf.keras.activations.relu(input_tensor2)
print("relu function\ninput tensor:",input_tensor2,"relu activation function result:",output)




#creating the simple neural network 
#with the input of the 3 
#and the 2 hidden layers of the each of 5 eurons 
#and the output laayer is 1 neuron

model1 = Sequential(
    [
    tf.keras.layers.Dense(5,activation = "relu",input_dim=3),
    tf.keras.layers.Dense(5,activation="relu"),
    tf.keras.layers.Dense(1)
    ]
    
)
model1.summary()

#loading the data from keras 
(x_train,x_test),(y_train,y_test) = keras.datasets.mnist.load_data()
print(x_train.shape,x_test.shape,y_test.shape,y_train.shape)


#t0 view the images now we are using 
import matplotlib.pyplot as  plt
print(x_train[1]) 
plt.matshow(x_train[1])
plt.show()

x_train_flatten_changed= x_train.reshape(len(x_train),28*28)

y_train_flatten_changed= y_train.reshape(len(y_train),28*28)


model12 = Sequential(
    [
    tf.keras.layers.Dense(100,activation = "relu",input_dim=784),
    tf.keras.layers.Dense(100,activation = "relu"),
    tf.keras.layers.Dense(100,activation = "relu"),
    tf.keras.layers.Dense(100,activation = "relu"),
tf.keras.layers.Dense(100,activation = "relu"),
tf.keras.layers.Dense(100,activation = "relu"),
tf.keras.layers.Dense(100,activation = "relu"),
    tf.keras.layers.Dense(10,activation="sigmoid")
    ]
    
)
model12.summary()
model12.compile(
    optimizer="sgd",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
import numpy as np
model12.fit(x_train_flatten_changed,x_test,epochs=40)
print("weights:\n",model12.get_weights())
y_predict=model12.predict(y_train_flatten_changed)
y_predict = [np.argmax(i) for i in y_predict]
conf_matrix=tf.math.confusion_matrix(labels=y_test,predictions=y_predict)
print(tf.math.confusion_matrix(labels=y_test,predictions=y_predict))



from sklearn.metrics import r2_score

coefficient_of_dermination = r2_score(y_test, y_predict)
print("corellation:",coefficient_of_dermination)
import seaborn  as sn 
sn.heatmap(conf_matrix)

#saving the model to  file  using the pickle file.
import pickle
with open("pickle_ml_model_file","wb") as f:
    pickle.dump(model12,f)


#loading the model
with open("pickle_ml_model_file","rb") as f:
    model123=pickle.load(f)     