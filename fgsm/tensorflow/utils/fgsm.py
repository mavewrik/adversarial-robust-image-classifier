# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from tensorflow.keras.losses import MSE
import tensorflow as tf


# -----------------------------
#   FUNCTIONS
# -----------------------------
def generate_image_adversary(model, image, label, eps=2/255.0):
    # Cast the image to a float-32 tensor
    image = tf.cast(image, tf.float32)
    # Record the gradients
    with tf.GradientTape() as tape:
        # Explicitly indicate that the image should be tracked for gradient updates
        tape.watch(image)
        # Use the model to make predictions on the input image and then compute the loss
        pred = model(image)
        loss = MSE(label, pred)
    # Calculate the gradients of loss with respect to the image, then compute the sign of the gradient
    gradient = tape.gradient(loss, image)
    signedGrad = tf.sign(gradient)
    # Construct the image adversary following FGSM strategy
    adversary = (image + (signedGrad * eps)).numpy()
    # Return the image adversary to the calling function
    return adversary

