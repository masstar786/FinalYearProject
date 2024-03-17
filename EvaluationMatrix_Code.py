from tensorflow import keras
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load the model
model = load_model('C:\\Users\\User\\Desktop\\Preprocess\\All_Data\\CNN_LSTM\\new-cnn_lstm_model.h5')

# Load the validation dataset
validation_ds = keras.utils.image_dataset_from_directory(
    directory='C:\\Users\\User\\Desktop\\Preprocess\\All_Data\\Training',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(13, 6))

# Convert the validation dataset to numpy arrays
X_val = []
y_val = []
for images, labels in validation_ds:
    X_val.append(images.numpy())
    y_val.append(labels.numpy())

X_val = np.concatenate(X_val)
y_val = np.concatenate(y_val)

# Make predictions on the validation dataset
y_pred = model.predict(X_val)
y_pred = np.argmax(y_pred, axis=1)

# Compute evaluation metrics
accuracy = np.mean(y_pred == np.argmax(y_val, axis=1))
classification_report = classification_report(np.argmax(y_val, axis=1), y_pred)
confusion_mat = confusion_matrix(np.argmax(y_val, axis=1), y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report)
print("Confusion Matrix:")
print(confusion_mat)