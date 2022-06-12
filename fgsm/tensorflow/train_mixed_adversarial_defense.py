#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary packages
from utils.simplecnn import SimpleCNN
from utils.datagen import generate_mixed_adversarial_batch
from utils.datagen import generate_adversarial_batch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from utils.cifar_10 import load_data
import numpy as np
from copy import deepcopy
from PIL import Image, ImageOps, ImageDraw, ImageFont
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Specify font to draw on images
font = ImageFont.truetype("utils/arial.ttf", 9)


# In[3]:


# Load the CIFAR-10 dataset. If server is down (error 503), follow the big comment below.
# If facing certificate error, follow this: https://stackoverflow.com/questions/69687794/unable-to-manually-load-cifar10-dataset

# Download/Get the Python3-compatible CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz or anywhere else.
# Make sure tar.gz file is fully unzipped and in the same location as this .py file.
# Use "tar -zxvf cifar-10-python.tar.gz" command to completely unzip the CIFAR-10 dataset to get a directory
# named "cifar-10-batches-py" in the same location as this current .py file. 
print("[INFO] Loading CIFAR-10 dataset...")
# (trainX, trainY), (testX, testY) = cifar10.load_data()
(trainX, trainY), (testX, testY) = load_data()


# In[4]:


# Scale the pixel values to the range [0, 1]
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0


# In[5]:


# Add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)


# In[6]:


# One-hot encode the labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)


# In[7]:


# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# In[8]:


# Initialize the optimizer and model
print("[INFO] Compiling the model...")
opt = Adam(lr=1e-3)
model = SimpleCNN.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# In[9]:


# Train the simple CNN on CIFAR-10
print("[INFO] Training the network...")
H1 = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=20, verbose=1)


# In[10]:


# Make predictions on the testing set for the model trained on non-adversarial images
(loss, accuracy) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] Normal testing images: ")
print("[INFO] Loss: {:.4f}, Accuracy: {:.4f}\n".format(loss, accuracy))


# In[11]:


# Predictions on non-adversarial images for classification report
print("[INFO] Evaluating network...")
predictions_normal = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions_normal.argmax(axis=1), target_names=labelNames))


# In[12]:


# Generate a set of adversarial images from the test set (in order to evaluate the model performance
# *before* and *after* the mixed adversarial training)
print("[INFO] Generating adversarial examples with FGSM...\n")
(advX, advY) = next(generate_adversarial_batch(model, len(testX), testX, testY, (32, 32, 3), eps=0.01))


# In[13]:


# Re-evaluate the model on the adversarial images
(loss, accuracy) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] Adversarial testing images:")
print("[INFO] Loss: {:.4f}, Accuracy: {:.4f}\n".format(loss, accuracy))


# In[14]:


# Predictions on adversarial images for classification report
print("[INFO] Evaluating network...")
predictions_adv = model.predict(advX, batch_size=32)
print(classification_report(advY.argmax(axis=1), predictions_adv.argmax(axis=1), target_names=labelNames))


# In[15]:


# plot the training loss and accuracy over time before applying defense
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H1.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H1.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H1.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 20), H1.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy *before* applying FGSM defense")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# ## FGSM Defense

# In[16]:


# Lower the learning rate and re-compile the model (in order to fine-tune the model on the mixed batches
# of normal images and the dynamically generated adversarial images)
print("[INFO] Re-compiling the model...")
opt = Adam(lr=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# In[17]:


# Initialize the data generator to create data batches containing a mix of both
# *normal* and *adversarial* images
print("[INFO] Creating the mixed data generator...")
dataGen = generate_mixed_adversarial_batch(model, 64, trainX, trainY, (32, 32, 3), eps=0.01, split=0.5)


# In[18]:


# Fine-tune the CNN on the adversarial images
print("[INFO] Fine-tuning the network in dynamic mixed data...")
# H2 = model.fit(dataGen, steps_per_epoch=len(trainX)//64, epochs=10, verbose=1)
# H2 = model.fit(dataGen, steps_per_epoch=len(trainX)//64, validation_data=(testX, testY), epochs=10, verbose=1)
H2 = model.fit(dataGen, steps_per_epoch=len(trainX)//64, validation_data=(advX, advY), epochs=10, verbose=1)


# In[ ]:


# Save model after applying FGSM defense
model.save('robust_model')


# In[19]:


# Now that the model is fine-tuned, evaluate it on the test set (i.e, non-adversarial images) again to
# see if the overall performance of the model has degraded
(loss, accuracy) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] Normal testing images *after* fine-tuning:")
print("[INFO] Loss: {:.4f}, Accuracy: {:.4f}\n".format(loss, accuracy))


# In[20]:


# Predictions on non-adversarial images after defense for classification report
print("[INFO] Evaluating network...")
predictions_normal_after = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions_normal_after.argmax(axis=1),                             target_names=labelNames))


# In[21]:


# Do a final evaluation of the model on the adversarial images
(loss, accuracy) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] Adversarial images *after* fine-tuning:")
print("[INFO] Loss: {:.4f}, Accuracy: {:.4f}\n".format(loss, accuracy))


# In[22]:


# Predictions on adversarial images after defense for classification report
print("[INFO] Evaluating network...")
predictions_adv_after = model.predict(advX, batch_size=32)
print(classification_report(advY.argmax(axis=1), predictions_adv_after.argmax(axis=1), target_names=labelNames))


# In[23]:


# plot the training loss and accuracy over time before applying defense
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 10), H2.history["loss"], label="train_loss")
plt.plot(np.arange(0, 10), H2.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 10), H2.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 10), H2.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy *after* applying FGSM defense")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# In[ ]:




