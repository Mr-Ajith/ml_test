import tensorflow as tf
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense,Flatten
import numpy as np
from PIL import Image

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print("dataset loaded")

""""
  ----------- demo data print------------
  image_index = 1

# Print the array of the chosen image
for row in train_images[image_index]:
    for pixel in row:
        print(f"{pixel:3}", end=" ")
    print()
"""""


# Normalize the image data to values between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reshape the images to flatten them (28x28 images to 1D arrays of length 784)

train_images = train_images.reshape(-1, 28*28)
test_images = test_images.reshape(-1, 28*28)
'''
image = Image.open("E:\Downloads\path_to_save_image.png")
image = image.resize((28, 28))
image = image.convert('L')
image_array = np.array(image)
image_array=image_array/255

image_vector = image_array.reshape(1, 784) 

# Combine the MNIST dataset and your external data
train_images = np.concatenate([train_images, image_vector])
train_labels = np.concatenate([train_labels,np.array([4]).astype(int)])
'''
'''
model =  tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu',input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])'''


train_images = np.reshape(train_images, (train_images.shape[0], 28, 28, 1))
test_images = np.reshape(test_images, (test_images.shape[0], 28, 28, 1))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(28, 28, 1))) 

# Convolutional layers
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# Flatten layer to convert the 2D matrix data to a vector
model.add(tf.keras.layers.Flatten())

# Fully connected layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))  # Dropout layer for regularization
model.add(tf.keras.layers.Dense(10, activation='softmax')) 

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(train_images, train_labels, epochs=7, batch_size=30)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss*100} %")
print(f"Test Accuracy: {test_accuracy*100} %")

test_loss, test_accuracy = model.evaluate(train_images, train_labels)
print(f"Test Loss: {test_loss*100} %")
print(f"Test Accuracy: {test_accuracy*100} %")

model.save("my_saved_model")  # Save



""""" 
----------- testing------------  
import numpy as np

# Choose an image from the test set
image_idx = 12
sample_image = np.expand_dims(test_images[image_idx], axis=0)

# Make predictions
predictions = loaded_model.predict(sample_image)
predicted_label = np.argmax(predictions)

print(f"True Label: {test_labels[image_idx]}, Predicted Label: {predicted_label}") 



image = Image.open("E:\Downloads\path_to_save_image.png")
image = image.resize((28, 28))
image = image.convert('L')
image_array = np.array(image)
image_array=image_array/255

image_vector = image_array.reshape(1, 784) 
prediction = model.predict(image_vector)

predicted_digit = np.argmax(prediction)
print(predicted_digit)"""""