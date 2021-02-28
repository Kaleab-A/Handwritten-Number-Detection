import tensorflow as tf
import matplotlib.pyplot as plt

dataset = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = dataset.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

for i in range(len(x_train)):
    for row in range(28):
        for col in range(28):
            if x_train[i][row][col] > 0.2:
                x_train[i][row][col] = 1
            else:
                x_train[i][row][col] = 0
                
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5) # Traing for 4 epochs

model.save("model4.model") # Saving the Model
