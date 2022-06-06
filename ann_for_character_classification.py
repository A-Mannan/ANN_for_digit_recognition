
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

#-------------------- Training ANN -------------------
#(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()

# X_train=X_train/255
# X_test=X_test/255

# model=tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(128,activation='sigmoid'))
# model.add(tf.keras.layers.Dense(100,activation='softmax'))

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# model.fit(X_train,y_train,epochs=20)

# model.save('handwritten.model')




#------------------- retrieving the saved model ----------------
model=tf.keras.models.load_model('handwritten.model')

#------------------- Testing the model -------------------------
model.evaluate(X_test,y_test)

img=cv2.imread("test_4.jpeg",0)
img=cv2.resize(img,(28,28),interpolation = cv2.INTER_CUBIC)
img=(np.invert(img))/255
plt.imshow(img)
img=img.reshape(1,28,28)

print(np.argmax(model.predict(img)))
