import cv2
import numpy as np
from skimage import transform
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained CNN model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def preprocess_image(image):
    # Resize the image to match the input size of the CNN model
    resized_image = transform.resize(image, (224, 224), anti_aliasing=True)

    # Expand dimensions to create a batch of size 1
    preprocessed_image = np.expand_dims(resized_image, axis=0)

    # Preprocess the input image based on the requirements of the CNN model
    preprocessed_image = tf.keras.applications.mobilenet_v2.preprocess_input(preprocessed_image)

    return preprocessed_image

def process_image(input_image):
    # Load the image
    img = cv2.imread(input_image)

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    preprocessed_image = preprocess_image(img_rgb)

    # Run the preprocessed image through the CNN model
    features = model.predict(preprocessed_image)

    # Resize the features to match the shape of the original image
    resized_features = transform.resize(features[0], img_rgb.shape[:2], anti_aliasing=True)

    # Threshold the features to create a binary mask
    mask = resized_features > 0.5

    # Apply the mask to the original image
    result = img_rgb * mask.astype(np.uint8)

    return result

# Call the function witimport cv2
# import numpy as np
# from skimage import transform
# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# # Load the pre-trained CNN model
# model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#
# def preprocess_image(image):
#     # Resize the image to match the input size of the CNN model
#     resized_image = transform.resize(image, (224, 224), anti_aliasing=True)
#
#     # Expand dimensions to create a batch of size 1
#     preprocessed_image = np.expand_dims(resized_image, axis=0)
#
#     # Preprocess the input image based on the requirements of the CNN model
#     preprocessed_image = tf.keras.applications.mobilenet_v2.preprocess_input(preprocessed_image)
#
#     return preprocessed_image
#
# def process_image(input_image):
#     # Load the image
#     img = cv2.imread(input_image)
#
#     # Convert the image from BGR to RGB
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # Preprocess the image
#     preprocessed_image = preprocess_image(img_rgb)
#
#     # Run the preprocessed image through the CNN model
#     features = model.predict(preprocessed_image)
#
#     # Resize the features to match the shape of the original image
#     resized_features = transform.resize(features[0], img_rgb.shape[:2], anti_aliasing=True)
#
#     # Threshold the features to create a binary mask
#     mask = resized_features > 0.5
#
#     # Apply the mask to the original image
#     result = img_rgb * mask.astype(np.uint8)
#
#     return result
#
# # Call the function with an input image
# input_image = 'long_togue.png'
# output_image = process_image(input_image)
#
# # Display the output image
# plt.imshow(output_image.astype(np.uint8))
# plt.axis('off')
# plt.show()h an input image
input_image = 'long_togue.png'
output_image = process_image(input_image)

# Display the output image
plt.imshow(output_image.astype(np.uint8))
plt.axis('off')
plt.show()