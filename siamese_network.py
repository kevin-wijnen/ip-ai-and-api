"""
Gebaseerd op https://keras.io/examples/vision/siamese_network/

Apache 2.0-licentie vanwege source code dat Apache 2.0-licensed is.

CHANGES:

- Changed folder location
- Added waiting for image
- Added support for SQL-database for API status report support
- Added export of results in JSON for API results report support
"""
import sqlite3
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import time
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

input_folder_path = os.path.join(os.path.dirname(__file__), 'input')

image_available = False


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


con = sqlite3.connect('API/AI_Status')
cur = con.cursor()
cur.execute("""UPDATE info SET status = "Wachtend op afbeelding" WHERE id = 1""")
con.commit()
# Wachten tot dat er een afbeelding is
while not image_available:

    print("Checking for image...")
    print(os.listdir(input_folder_path))
    if len(os.listdir(input_folder_path)) != 0:
        image_available = True
        print("Image found!")
        cur.execute("""UPDATE info SET status = "Afbeelding gevonden" WHERE id = 1""")
        con.commit()
    else:
        print("Image not available, retrying in 3 seconds...")
        time.sleep(3)
con.close()
# We need to make sure both the anchor and positive images are loaded in
# sorted order so we can match them together.
anchor_images = sorted(
    [str(anchor_images_path / f) for f in os.listdir(anchor_images_path)]
)

original_anchor_images = sorted(
    [str(anchor_images_path / f) for f in os.listdir(anchor_images_path)]
)

positive_images = sorted(
    [str(positive_images_path / f) for f in os.listdir(positive_images_path)]
)

# Pak afbeeldinglocatie van input afbeelding
print(os.path.dirname(__file__))
# input_folder_path = os.path.join(os.path.dirname(__file__), 'input')
for file in os.listdir(input_folder_path):
    input_image = os.path.join(input_folder_path, file)

image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

input_data = [input_image]
input_data = tf.data.Dataset.from_tensor_slices(input_data)
anchor_process_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)

# To generate the list of negative images, let's randomize the list of
# available images and concatenate them together.
rng = np.random.RandomState(seed=42)
rng.shuffle(anchor_images)
rng.shuffle(positive_images)

negative_images = anchor_images + positive_images
np.random.RandomState(seed=32).shuffle(negative_images)

negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
negative_dataset = negative_dataset.shuffle(buffer_size=4096)

dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

input_data = input_data.map(preprocess_image)
anchor_process_dataset = anchor_process_dataset.map(preprocess_image)

# Let's now split our dataset in train and validation.
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)


# Pre-trained embedding model voor het compilen van de embedding model
base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="Embedding")
# embedding.save("embedding_model")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable


class DistanceLayer(layers.Layer):
    # Afstanden tussen anchor, positive en negative afbeeldingen berekenen voor embeddings

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(resnet.preprocess_input(anchor_input)),
    embedding(resnet.preprocess_input(positive_input)),
    embedding(resnet.preprocess_input(negative_input)),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

#siamese_network.save("siamese_network")


class SiameseModel(Model):
    """Siamees datamodel definitie:

    Formule van triplet loss:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


con = sqlite3.connect('API/AI_Status')
cur = con.cursor()
cur.execute("""UPDATE info SET status = "Trainen van model" WHERE id = 1""")
con.commit()
con.close()
siamese_model = SiameseModel(siamese_network)
# siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.compile(optimizer=optimizers.adam_v2.Adam(0.0001))
siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset)

# Input als positive afbeeldign pakken
input_positive = next(iter(input_data))
# Reshapen want mist anders de "None" dimensie
input_positive = input_positive[None, :, :, :]
cosine_similarity = metrics.CosineSimilarity()
results = {}
for i in range(0, image_count):
    print(original_anchor_images[i])
    # Pak embedding
    anchor = next(iter(anchor_process_dataset))
    print(anchor)
    anchor = anchor[None, :, :, :]
    # print(anchor)
    anchor_embedding = embedding(resnet.preprocess_input(anchor))
    print("Embedding van afbeelding " + str(i + 1) + " gepakt.")
    # print(anchor_embedding)
    # print(input_positive)
    input_embedding = embedding(resnet.preprocess_input(input_positive))
    print("Embedding van input ...")
    # print(input_embedding)

    input_similarity = cosine_similarity(anchor_embedding, input_embedding)
    print("Input similarity (to anchor) ", input_similarity.numpy())

    split_file_name = original_anchor_images[i].split("/")
    file_name = split_file_name[-1]
    results[file_name] = str(input_similarity.numpy())

# Voor debuggen, VOORBEELD LOOP
for i in range(0, 5):
    sample = next(iter(train_dataset))
    # print(sample)

    anchor, positive, negative = sample
    # print(anchor)
    # print(negative)
    anchor_embedding, positive_embedding, negative_embedding = (
        embedding(resnet.preprocess_input(anchor)),
        embedding(resnet.preprocess_input(positive)),
        embedding(resnet.preprocess_input(negative)),
    )
    # print(anchor_embedding)

    positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
    print("Positive similarity:", positive_similarity.numpy())

    negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
    print("Negative similarity", negative_similarity.numpy())

con = sqlite3.connect('API/AI_Status')
cur = con.cursor()
cur.execute("""UPDATE info SET status = "Resultaten zijn bekend" WHERE id = 1""")
con.commit()
con.close()

print(results)
with open("API/results.json", "w+") as results_file:
    json.dump(results, results_file)
os.remove(input_image)
