import keras_tuner
import tensorflow as tf 
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam as adam
import keras


#building the mode  build_model functio is used to create and return the model
def build_model(hp):
    model = Sequential()
    for i in range(hp.Int("number_of_layers",2,20)):
        model.add(Dense(hp.Int("unit_"+str(i),
                               min_value=32,
                               max_value=512),activation='relu'))
    model.add(Dense(units=1,activation='linear'))
    model.compile(optimizer=adam(),
                loss="mean_absolute_error",
                metrics=['mean_absolute_error'])
    return model


import keras_tuner

build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_mean_absolute_error",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="project1",
)

tuner.search_space_summary()

#loading the data from keras 
(x_train, y_train), (x_test, y_test)  = keras.datasets.mnist.load_data()
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


#t0 view the images now we are using 
import matplotlib.pyplot as  plt
print(x_train[1]) 
plt.matshow(x_train[1])
plt.show()

x_train_flatten_changed= x_train.reshape(len(x_train),28*28)

x_test_flatten_changed= x_test.reshape(len(x_test),28*28)


#importing the data
import pandas as  pd 
path="D:\\datasets\\Churn_Modelling.csv"
data1 = pd.read_csv(path)
data=data1.copy()
print(data)
print(data.columns)
print(data.info)



x=data.iloc[:,3:13]
y=data.iloc[:,13]
print(x.columns)
Gender_update= pd.get_dummies(x['Gender'])
Geography_update= pd.get_dummies(x['Geography'])
x.drop(['Gender'],axis=1,inplace=True)
x.drop(["Geography"],axis=1,inplace=True)

print(data1['Gender'])
x=pd.concat([x,Gender_update,Geography_update],axis=1)
print(x.shape,y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=5)


tuner.search(x_train, y_train, epochs=100, batch_size=100,validation_data=(x_test, y_test)) 