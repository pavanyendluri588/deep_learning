import tensorflow as tf 
import pandas as pd 

from tensorflow import keras 

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation

#importing the data
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
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=5)
print(x,y)

#part 2 model creation 
model= Sequential()
#,kernel_initializer="he_uniform"
model.add(Dense(units=40 ,kernel_initializer="he_normal",activation="relu",input_dim=13))
#model.add(Dropout(0.5))
model.add(Dense(units=30 ,kernel_initializer="he_normal",activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(units=40 ,kernel_initializer="he_normal",activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(units=30 ,kernel_initializer="he_normal",activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(units=20 ,kernel_initializer="he_normal",activation="relu"))
#,kernel_initializer="glorot_uniform"
model.add(Dense(units=1))

model.summary()

model.compile(
    optimizer="adamax",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.fit(x_train,y_train ,epochs=100)
y_predict=model.predict(x_test)
from sklearn.metrics import r2_score
error = r2_score(y_test, y_predict)