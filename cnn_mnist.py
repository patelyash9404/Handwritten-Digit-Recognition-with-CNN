import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras import backend as k
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Define input shape
img_rows, img_cols = 28, 28
if k.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Normalize data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Model Architecture
inp = Input(shape=input_shape)
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inp)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(10, activation='softmax')(x)

# Compile Model
model = Model(inp, out)
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(x_train, y_train, epochs=15, batch_size=500, validation_data=(x_test, y_test))

# Evaluate Model
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Loss: {score[0]:.4f}')

print(f'Accuracy: {score[1]:.4f}')

# Example Prediction
random_index = np.random.randint(0, x_test.shape[0])
test_sample = x_test[random_index].reshape(1, 28, 28, 1)
prediction = model.predict(test_sample)
predicted_label = np.argmax(prediction)

# Plot the test image
plt.imshow(x_test[random_index].reshape(28, 28), cmap='gray')
plt.title(f'Predicted Label: {predicted_label}')
plt.axis('off')
plt.show()
