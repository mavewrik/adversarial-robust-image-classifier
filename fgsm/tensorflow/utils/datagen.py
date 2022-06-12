# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from .fgsm import generate_image_adversary
from sklearn.utils import shuffle
import numpy as np


# -----------------------------
#   FUNCTIONS
# -----------------------------
def generate_adversarial_batch(model, total, images, labels, dims, eps=0.01):
    # Unpack the image dimensions into convenience variables
    (h, w, c) = dims
    # Construct a data generator here to loop indefinitely
    while True:
        # Initialize the perturbed images and labels
        perturbImages = []
        perturbLabels = []
        # Randomly sample indexes (without replacement) from the input data
        idxs = np.random.choice(range(0, len(images)), size=total, replace=False)
        # Loop over the indexes
        for i in idxs:
            # Grab the current image and label
            image = images[i]
            label = labels[i]
            # Generate an adversarial image
            adversary = generate_image_adversary(model, image.reshape(1, h, w, c), label, eps=eps)
            # Update the perturbed images and labels lists
            perturbImages.append(adversary.reshape(h, w, c))
            perturbLabels.append(label)
        # Yield the perturbed images and labels
        yield np.array(perturbImages), np.array(perturbLabels)


def generate_mixed_adversarial_batch(model, total, images, labels, dims, eps=0.01, split=0.5):
    # Unpack the image dimensions into convenience variables
    (h, w, c) = dims
    # Compute the total number of training images to keep along with the number of adversarial images to generate
    totalNormal = int(total * split)
    totalAdv = int(total * (1 - split))
    # Construct a data generator here to loop indefinitely
    while True:
        # Randomly sample indexes (without replacement) from the input data and then use those indexes to sample
        # the normal images and labels
        idxs = np.random.choice(range(0, len(images)), size=totalNormal, replace=False)
        mixedImages = images[idxs]
        mixedLabels = labels[idxs]
        # Again, randomly sample indexes from the input data, this time to construct the adversarial images
        idxs = np.random.choice(range(0, len(images)), size=totalAdv, replace=False)
        # Loop over the indexes
        for i in idxs:
            # Grab the current image and label, then use that data to generate the adversarial example
            image = images[i]
            label = labels[i]
            adversary = generate_image_adversary(model, image.reshape(1, h, w, c), label, eps=eps)
            # Reorder and expand dimensions of adversary to match dimensions of mixedImages
            adversary = adversary.reshape(h, w, c, 1)
            adversary = np.expand_dims(adversary, axis=0)
            # Update the mixed images and labels lists
            mixedImages = np.vstack([mixedImages, adversary])
            mixedLabels = np.vstack([mixedLabels, label])
        # Shuffle the images and labels together
        (mixedImages, mixedLabels) = shuffle(mixedImages, mixedLabels)
        # Yield the mixed images and labels to the calling function. Drop last column of mixedImages to match dimensions.
        yield mixedImages[:,:,:,:,0], mixedLabels


