# Import the necessary packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import Model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16


# Define a function that loads an image, resizes it, converts it to a numpy array and normalizes the pixel values
# to [0, 1]
def load_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)  # Loads the image in the specified target size
    img_array = image.img_to_array(img)  # Converts the image to an array
    img_array = img_array / 255.0  # normalizes the pixel value to 0-1 by dividing by 255 (maximum color channel value)
    return img_array


# Define the path to the folder containing the images
# ADJUST TO PERSONAL FOLDER
image_folder = '/Users/maxvanhout/Documents/School/Maastricht/Machine Learning/Project/Dataset2/Images'
image_files = sorted(os.listdir(image_folder))

# Load and preprocess all images in the image folder using the above defined 'load_image' function
images = [load_image(os.path.join(image_folder, img_file)) for img_file in image_files]
# Convert the images to numpy array and print its shape
images_array = np.array(images)
# The dimensions represent: (number of images, pixel height=224, pixel width=224, color values=3)
print("The image data has dimensions:", images_array.shape)

# Import the fMRI data and store it as a pandas dataframe
# ADJUST TO PERSONAL FOLDER
brain_data_np = np.load('/Users/maxvanhout/Documents/School/Maastricht/Machine Learning/Project/Dataset2/fMRI-data/training.npy')
brain_data_df = pd.DataFrame(brain_data_np)
# The dimensions represent: (number of images, number of voxels)
print("The fMRI data has dimensions:", brain_data_df.shape)

# Check if the images have been correctly loaded and preprocessed by plotting the first 5 images from the image array
first_5_images_array = images_array[:5]
# Display the first 5 images
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, ax in enumerate(axes):
    ax.imshow(first_5_images_array[i])
    ax.axis('off')
    ax.set_title(image_files[i])
plt.show()

# Split the image and fMRI data into 80/20% training and testing data
X_train, X_test, y_train, y_test = train_test_split(images_array, brain_data_df, test_size=0.2, random_state=20)

# Load the VGG16 model with the pre-trained weights based on the imagenet dataset
# We exclude the fully connected layers, by setting 'include_top=False', to get rid of the classification part
# We set the input shape
VGG_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
# Freeze the layers in the VGG16 CNN model by setting all the model weights to be non-trainable
VGG_model.trainable = False
# Check the summary of the pre-trained model
VGG_model.summary()

# Define layer to extract features from
# Adjust this layer to 'Block5_conv3', 'Block5_conv2', 'Block5_conv1', 'Block4_pool', 'Block4_conv3', 'Block3_pool',...
# ATTENTION: deeper layers require significantly more computational power to extract the features from
layer_name = 'block5_pool'
# Create a model that outputs features from the specified layer
feature_extractor = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer(layer_name).output)
# Extract features from our images by passing our images through the VGG model to generate feature vector predictions
feature_train = feature_extractor.predict(X_train)
feature_test = feature_extractor.predict(X_test)
# Reshape the resulting four-dimensional data to a 2d array
feature_train = feature_train.reshape(feature_train.shape[0], -1)
feature_test = feature_test.reshape(feature_test.shape[0], -1)

# Transform the features using quantile transformation
scaler = QuantileTransformer(output_distribution='uniform')
feature_train = scaler.fit_transform(feature_train)
feature_test = scaler.transform(feature_test)
# Each row represents an image and each column represents a feature
print("The flattened feature data has dimensions:", feature_train.shape)

# Apply PCA to reduce the number of features
pca = PCA(n_components=200)  # You can adjust this
pca_train = pca.fit_transform(feature_train)
pca_test = pca.transform(feature_test)

# Initialize and fit the linear regression model
regression_model = LinearRegression()
regression_model.fit(pca_train, y_train)

# Predict voxel values using the regression model based on the PCA of the test features
y_pred = regression_model.predict(pca_test)

# Calculate mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print(f'R-squared for {layer_name}: {r_squared}')
print(f'Mean Squared Error for {layer_name}: {mse}')
