"""
Gebaseerd op https://keras.io/examples/vision/siamese_network/
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from keras import applications
from keras import layers
from keras import losses
from keras import optimizers
from keras import metrics
from keras import Model
from keras.applications import resnet
from keras.models import save_model

# Doel grootte van afbeeldingen
target_shape = (200, 200)

cache_dir = Path(Path.home()) / ".keras"
# Basis afbeeldingen/anker afbeelding, invoeren registraties van merken
anchor_images_path = cache_dir / "Anchor"
# Vergelijkbare afbeeldingen, invoeren afbeeldingen die lijken op de merken (zoals op auto etc.)
positive_images_path = cache_dir / "Positive"


def preprocess_image(filename):
    # Afbeeldingen inlezen, decoden, en resizen
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    # Drieling afbeeldingen retourneren
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


# We need to make sure both the anchor and positive images are loaded in
# sorted order so we can match them together.
anchor_images = sorted(
    [str(anchor_images_path / f) for f in os.listdir(anchor_images_path)]
)

print(anchor_images)

positive_images = sorted(
    [str(positive_images_path / f) for f in os.listdir(positive_images_path)]
)

# Pak afbeeldinglocatie van input afbeelding
print(os.path.dirname(__file__))
input_folder_path = os.path.join(os.path.dirname(__file__), 'input')
for file in os.listdir(input_folder_path):
    input_image = os.path.join(input_folder_path, file)

input_data = [input_image]
image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
input_dataset = tf.data.Dataset.from_tensor_slices(input_data)
#positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
# dataset aanmaken door lijst mee te geven van het bestandspad!
# input_data = tf.data.Dataset.from_tensor_slices(input_image)

anchor_iterator = iter(anchor_dataset)
input_iterator = iter(input_dataset)
input_positive = input_iterator.get_next()

# Testen van iteratie aanpak
# Na trainen van embeddingmodel, pak de embedding van elk plaatje en vergelijk met cosimilar!
for i in range(0, image_count):
    # Pak embedding
    anchor = anchor_iterator.get_next()
    print(anchor)
    print("Embedding van afbeelding " + str(i+1) + " gepakt.")
    print(input_positive)
    print("Embedding van input ...")
    # Geef cosmilarity weer, zet het in dictionary met de filename als key, dan sla het op in database, om json vanuit database te dumpen!

    #print(anchor_iterator.get_next()) # Anchor pakken
    #print(positive_iterator.get_next()) # Positive pakken


