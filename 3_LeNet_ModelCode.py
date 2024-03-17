#LeNet Model Inspiration

from tensorflow import keras
from keras import layers

# Load and preprocess the data
train_ds = keras.preprocessing.image_dataset_from_directory(
    directory='C:\\Users\\User\\Desktop\\Preprocess\\All_Data\\Training',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(13, 6))

validation_ds = keras.preprocessing.image_dataset_from_directory(
    directory='C:\\Users\\User\\Desktop\\Preprocess\\All_Data\\Testing',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(13, 6))

# Modify the LeNet model
def modified_lenet_model(input_shape=(13, 6, 3)):
    inputs = keras.Input(shape=input_shape)
    
    # Convolutional layers
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Attention mechanism
    attention = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(32, activation='relu')(attention)
    attention = layers.Dense(64, activation='sigmoid')(attention)
    attention = layers.Reshape((1, 1, 64))(attention)
    attention = layers.UpSampling2D((x.shape[1], x.shape[2]))(attention)
    
    # Expand dimensions for broadcasting
    attention = layers.Lambda(lambda x: keras.backend.expand_dims(x, axis=-1))(attention)
    
    # Apply attention to feature map
    attention_applied = layers.multiply([x, attention])
    
    # Flatten and fully connected layers
    x = layers.Flatten()(attention_applied)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create the model
model = modified_lenet_model(input_shape=(13, 6, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
model.fit(train_ds, epochs=100, validation_data=validation_ds)

# Save the trained model
model.save('C:\\Users\\User\\Desktop\\Preprocess\\All_Data\\Lenet_Modified_Model.h5')
