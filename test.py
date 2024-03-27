import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import ModelCheckpoint

# Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize pixel values to the range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define class names (optional)
# class_names = [...]  # List of class names omitted for brevity

# Print dataset shapes and number of classes
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)
print("Number of classes:", len(np.unique(y_train)))

activations = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential", "leaky_relu", "relu6", "hard_silu", "gelu", "hard_sigmoid", "linear", "mish", "log_softmax"]
no_layers = 5
max_pow = 6
lay = range(2, no_layers)

k_neurons = [2**i for i in range(2,max_pow)]

# Define the checkpoint filepath
checkpoint_filepath = './best_model.keras'

best_accuracy = 0.0
best_activation = None

for a in activations:
    for k in k_neurons:
        model = Sequential()
        for _ in lay:
            for i in range(2, no_layers-1):
                model.add(Flatten(input_shape=(32, 32, 3)))
                model.add(Dense(k, activation=a))
            model.add(Dense(len(np.unique(y_train)), activation=a))

            # Compile the model
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Define the ModelCheckpoint callback
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )

            # Train the model on CIFAR-100 training data
            history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.5, callbacks=[checkpoint], verbose=1)

            # Evaluate the model on CIFAR-100 test data
            _, accuracy = model.evaluate(x_test, y_test)
            print(f"Neural Network Accuracy on CIFAR-100 test data: {accuracy:.2f}")

            # Save the best performing activation function
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_activation = a

# Load the best performing model
best_model = tf.keras.models.load_model(checkpoint_filepath)

print(f"Best performing activation function: {best_activation}")
print(f"Accuracy of the best model: {best_accuracy:.2f}")

# Plot training history of the best model
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
