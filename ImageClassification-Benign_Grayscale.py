import pandas as pd
import numpy as np
import cv2, os

# Load the CSV file
data = pd.read_csv('C:/Users/User/Desktop/Preprocess/test1_data.csv')
data = data.drop(columns=['Timestamp'])
benign_data = data.loc[data['Label'] == 'Benign']
print(data.shape)
print(benign_data.shape)
print(len(data.columns))

# Extract the fields and label from the data
fields = benign_data.iloc[:, :78].values
label = benign_data.iloc[:, 78].values

i = 0
# Reshape the fields into images
images = []
for field in fields:
    # Normalize the field values between 0 and 255
    field_normalized = (field - np.min(field)) / (np.max(field) - np.min(field))
    field_normalized = (field_normalized * 255).astype(np.uint8)
    
    # Convert the field to a grayscale image
    image = cv2.cvtColor(field_normalized, cv2.COLOR_GRAY2BGR)
    
    image_resized = cv2.resize(image, (13, 6))
    
    # Save the image as PNG
    filename = f'image_{i}.png'  # Set the filename
    folder = 'C:/Users/User/Desktop/Preprocess/images'  # Set the folder name
    os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
    filepath = os.path.join(folder, filename)  # Create the file path
    cv2.imwrite(filepath, image_resized)  # Save the image as PNG
    
    # Append the resized image to the list
    images.append(image_resized)

    i+=1

# Convert the list of images into a numpy array
images = np.array(images)

# Perform any additional preprocessing on the images (e.g., resizing)

# Print the shape of the images array
print('Shape of images:', images.shape)
