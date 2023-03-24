import pickle
with open("pickle_ml_model_file","rb") as f:
    model=pickle.load(f)
import tensorflow  as tf
(x_train,x_test),(y_train,y_test)= tf.keras.datasets.mnist.load_data()
x_train_flattened = x_train.reshape(len(x_train),28*28)
y_train_flattened = y_train.reshape(len(y_train),28*28)
y_predict = model.predict(y_train_flattened)
import numpy
y_predict =[numpy.argmax(i) for i in y_predict]
conf_matrix=tf.math.confusion_matrix(labels=y_test,predictions=y_predict)
print("=======================\n",conf_matrix)

from sklearn.metrics import r2_score
print("r2 score for model:",r2_score(y_test,y_predict))
