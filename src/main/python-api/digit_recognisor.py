import tensorflow as tf

# Load the MNIST dataset, seperating training and testing 
# mnist_dataset = tf.keras.datasets.mnist
# (training_images, training_labels), (testing_images, testing_labels) = mnist_dataset.load_data()

# training_images = tf.keras.utils.normalize(training_images, axis=1)
# testing_images = tf.keras.utils.normalize(testing_images, axis=1)


# Initialise the MLP model & add layers
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the model for 3 epochs, then save it
# model.fit(training_images, training_labels, epochs=3)
# model.save('digit-recogniser')

model = tf.keras.models.load_model('digit-recogniser')



