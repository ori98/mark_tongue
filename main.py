from flask import Flask, request, send_file
import werkzeug
import cv2
import pathlib
import os
# from machine_learning import predict_iris
# Custom library imports
from tongue_mask import process_image

app = Flask(__name__)

# provide the default filename for image
global filename

PATH_FOR_RESULT_IMAGE = os.path.join(os.getcwd(), 'test_images')

@app.route('/', methods=['GET', 'POST'])
def homepage():
    response = "The app is connected to the server!"
    return response


@app.route("/upload", methods=['GET', 'POST'])
def upload_image():
    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print('\n Received image file: ' + filename)
    imagefile.save(filename)
    response = "Image Uploaded to Server:"
    # add_shape(filename) this function was used to send the image for processing
    # instead we now process the tongue image:
    processed_image = process_image(input_image=filename)
    cv2.imwrite(os.path.join(PATH_FOR_RESULT_IMAGE, "resultImage.jpg"), processed_image)

    return response

def add_shape(image_name):
    image = cv2.imread(image_name)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

    # building a classifier
    clf = cv2.CascadeClassifier(str(cascade_path))

    # converting image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detecting the faces
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # looping over the faces detected
    for (x, y, width, height) in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 255), 2)

    cv2.imwrite(os.path.join(PATH_FOR_RESULT_IMAGE, "resultImage.jpg"), image)

@app.route("/getImage", methods=['GET', 'POST'])
def get_image():
    return send_file(os.path.join(PATH_FOR_RESULT_IMAGE, "resultImage.jpg"), mimetype='image/jpeg')

# Commenting out the Iris prediction code
# @app.route("/getPredictions", methods=['GET', 'POST'])
# def get_iris_prediction():
#     result = predict_iris()
#     return result


@app.route("/test", methods=['GET', 'POST'])
def test_response():
    return "Connection Active.\nPlease SCAN FACE and then press UPLOAD for result"


if __name__ == "__main__":
    # Bind to all interfacesf
    app.run(host='10.0.0.85', port=5001, debug=True)