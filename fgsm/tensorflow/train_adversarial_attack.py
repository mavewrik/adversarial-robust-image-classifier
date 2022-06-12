# ----------------------------------
#   USAGE
# ----------------------------------
# python train_adversarial_attack.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from utils.simplecnn import SimpleCNN
from utils.fgsm import generate_image_adversary
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import numpy as np
from copy import deepcopy
from PIL import Image, ImageOps, ImageDraw, ImageFont
from utils.cifar_10 import load_data

# Specify font to draw on images
font = ImageFont.truetype("utils/arial.ttf", 9)

# Load the CIFAR-10 dataset. If server is down (error 503), follow the big comment below.
# If facing certificate error, follow this: https://stackoverflow.com/questions/69687794/unable-to-manually-load-cifar10-dataset

# Download/Get the Python3-compatible CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz or anywhere else.
# Make sure tar.gz file is fully unzipped and in the same location as this .py file.
# Use "tar -zxvf cifar-10-python.tar.gz" command to completely unzip the CIFAR-10 dataset to get a directory
# named "cifar-10-batches-py" in the same location as this current .py file. 
print("[INFO] Loading CIFAR-10 dataset...")
# (trainX, trainY), (testX, testY) = cifar10.load_data()
(trainX, trainY), (testX, testY) = load_data()

# Scale the pixel values to the range [0, 1]
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# Add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

# One-hot encode the labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Initialize the optimizer and the model
print("[INFO] Compiling the model...")
opt = Adam(lr=1e-3)
model = SimpleCNN.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the simple CNN on the CIFAR-10 dataset
print("[INFO] Training the network...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=10, verbose=1)

# Make predictions on the testing set for the model trained on non-adversarial images
(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] Loss: {:.4f}, Accuracy: {:.4f}".format(loss, acc))

# Loop over a sample of the testing images
for i in np.random.choice(np.arange(0, len(testX)), size=(10,)):
    # Grab the current image and label
    image = testX[i]
    label = testY[i]
    # Generate an image adversary for the current image and make a prediction on the adversary image
    adversary = generate_image_adversary(model, image.reshape(1, 32, 32, 3), label, eps=0.01)
    pred = model.predict(adversary)
    # Scale both the original image and the adversary image to the range [0, 255]
    # and convert them to unsigned 8-bit integers
    adversary = adversary.reshape((32, 32, 3)) * 255
    adversary = np.clip(adversary, 0, 255).astype("uint8")
    image = image.reshape((32, 32, 3)) * 255
    image = image.astype("uint8")
    # # Convert the image and adversarial image from grayscale to three channel (in order to draw on the image)
    # image = np.dstack([image] * 3)
    # adversary = np.dstack([adversary] * 3)
    # Resize the images in order to visualize them later
    image = Image.fromarray(image).resize((96, 96))
    adversary = Image.fromarray(adversary).resize((96, 96))
    # image = cv2.resize(image, (96, 96))
    # adversary = cv2.resize(adversary, (96, 96))
    # Determine the predicted label for both the original image and the adversarial image
    imagePred = label.argmax()
    adversaryPred = pred[0].argmax()
    color = (0, 255, 0)
    # If the image prediction does not match with the adversarial prediction then update the color
    if imagePred != adversaryPred:
        color = (255, 0, 0)
    # Draw the predictions on the respective output images
    # cv2.putText(image, str(imagePred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
    # cv2.putText(adversary, str(adversaryPred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    image_copy = deepcopy(image)
    adversary_copy = deepcopy(adversary)
    draw_image = ImageDraw.Draw(image_copy)
    draw_image.text((10, 10), "{}".format(labelNames[imagePred]), (0,255,0), font=font)
    draw_adversary = ImageDraw.Draw(adversary_copy)
    draw_adversary.text((10, 10), "{}".format(labelNames[adversaryPred]), color, font=font)
    # Stack the two images horizontally and then show the original image and its adversary
    output = np.hstack([image_copy, adversary_copy])
    # cv2.imshow("FGSM Adversarial Images", output)
    # cv2.waitKey(0)
    Image.fromarray(output)


