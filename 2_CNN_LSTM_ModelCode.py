#CNN-LSTM model
from tensorflow import keras
import numpy as np

# Load and preprocess the data
train_ds = keras.utils.image_dataset_from_directory(
    directory='C:\\Users\\User\\Desktop\\Preprocess\\All_Data\\Training',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(13, 6))

validation_ds = keras.utils.image_dataset_from_directory(
    directory='C:\\Users\\User\\Desktop\\Preprocess\\All_Data\\Testing',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(13, 6))

# Build the CNN-LSTM model
model = keras.Sequential([
    keras.layers.Reshape(target_shape=(13, 6, 3), input_shape=(13, 6, 3)),
    keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.TimeDistributed(keras.layers.Flatten()),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
# Train the model
#
model.fit(train_ds, epochs=150, validation_data=validation_ds)

# Save the trained model
model.save('C:\\Users\\User\\Desktop\\Preprocess\\All_Data\\new-cnn_lstm_model.h5')

# Save the predicted labels on the test dataset
test_images = []
test_labels = []

for images, labels in validation_ds:
    test_images.extend(images.numpy())
    test_labels.extend(labels.numpy())

test_images = np.array(test_images)
test_labels = np.array(test_labels)

predicted_labels = model.predict(test_images)
predicted_classes = np.argmax(predicted_labels, axis=1)

with open('C:\\Users\\User\\Desktop\\Preprocess\\All_Data\\testing1_predictions.txt', 'w') as file:
    for filename, predicted_class in zip(validation_ds.file_paths, predicted_classes):
        file.write(f'{filename}: {predicted_class}\n')