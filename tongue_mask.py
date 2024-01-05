import cv2
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.measure import label, regionprops
from skimage.morphology import dilation, erosion
from skimage import io
import matplotlib.pyplot as plt


def process_image(input_image):
    # Load the image
    img = cv2.imread(input_image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Create a mask for the largest contour
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [max_contour], 0, 255, -1)

    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Save the result
    cv2.imwrite('result.png', result)

    # Load the result image
    sample = imread('result.png')
    sample_g = rgb2gray(sample)

    # Binarize image
    sample_b = sample_g > 0.6

    # Clean the binary image using dilation and erosion
    square = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    def multi_dil(im, num, element=square):
        for _ in range(num):
            im = dilation(im, element)
        return im

    def multi_ero(im, num, element=square):
        for _ in range(num):
            im = erosion(im, element)
        return im

    sample_c = multi_ero(multi_dil(sample_b, 5), 5)

    # Label the connected components
    sample_l = label(sample_c)

    # Extract region properties
    sample_rp = regionprops(sample_l)

    # Sort the blobs by area and select the largest ones
    list1 = []
    for x in sample_rp:
        list1.append(x.area)
    list2 = sorted(list(enumerate(list1)), key=lambda x: x[1], reverse=True)[:7]

    # Save the biggest blob image
    biggest_blob = sample_rp[list2[0][0]].image
    io.imsave('biggest_blob.png', biggest_blob)

    # Store the coordinates of the edge of the biggest blob
    coords = sample_rp[list2[0][0]].coords

    # Create a blank mask with the same shape as the original image
    mask_tongue = np.zeros_like(img[:, :, 0])

    # Set the pixel values at the blob coordinates to 255 (white)
    mask_tongue[coords[:, 0], coords[:, 1]] = 255

    # Convert the mask to a grayscale image
    gray_tongue = cv2.cvtColor(mask_tongue, cv2.COLOR_GRAY2BGR)

    # Apply bitwise AND operation to the original image and the mask
    img_tongue_masked = cv2.bitwise_and(img, gray_tongue)

    # Draw the contours of the mask on the original image
    contours, hierarchy = cv2.findContours(mask_tongue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_tongue_masked, contours, -1, (0, 255, 0), 3)

    # Convert the image from BGR to RGB format for displaying in Matplotlib
    img_tongue_rgb = cv2.cvtColor(img_tongue_masked, cv2.COLOR_BGR2RGB)

    return img_tongue_rgb


# ========== TEST CODE ====================
# Call the function with an input image
input_image = 'wide_tongue.png'
output_image = process_image(input_image)

# Display the output image
plt.imshow(output_image)
plt.show()