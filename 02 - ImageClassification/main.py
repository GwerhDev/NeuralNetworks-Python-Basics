import tensorflow as tf
import tensorflow_datasets as tfds

data, metadata = tfds.load('fashion_mnist', as_supervised = True, with_info = True)

data_training, data_test = data['tarin'], data['test']

class_name = metadata.features['label'].names

def normalizer(images, tags):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, tags

data_training = data_training.map(normalizer)
data_test = data_test.map(normalizer)

data_training = data_training.cache()
data_test = data_test.cache()

for image, tag in data_training.take(1):
  break
image = image.numpy().reshape((28,28))

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(image, cmap = plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()
plt.figure(figsize=(10,10))
for i, (image, tag) in enumerate(data_training.take(25)):
  image = image.numpy().reshape((28,28))
  plt.subplot(5, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image, cmap = plt.cm.binary)
  plt.xlabel(class_name[tag])
plt.show()

# Se crea el modelo

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape = (28, 28, 1)),
  tf.keras.layers.Dense(50, activation = tf.nn.relu),
  tf.keras.layers.Dense(50, activation = tf.nn.relu),
  tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.compile(
  optimizer = "adam",
  loss = tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics = ['accuracy']
)

num_ex_training = metadata.splits["train"].num_examples
num_ex_test = metadata.splits["test"].num_examples

print(num_ex_training)
print(num_ex_test)

LOTE_SIZE = 32

data_training = data_training.repeat().shuffle(num_ex_training).batch(LOTE_SIZE)
data_test = data_test.batch(LOTE_SIZE)

import math

# Entrenamos el modelo

historial = model.fit(data_training, epochs=5, steps_per_epoch = math.ceil(num_ex_training/LOTE_SIZE))

plt.xlabel("# Epoch")
plt.ylabel("Loss")
plt.plot(historial.history("loss"))

import numpy as np

for images_test, tags_test in data_test.take(1):
  images_test = images_test.numpy()
  tags_test = tags_test.numpy()
  predictions = model.predict(images_test)

def graph_image(i, arr_predictions, real_tags, images):
  arr_prediction, real_tag, image = arr_predictions[i], real_tags[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow([..., 0], cmap = plt.cm.binary)

  tag_prediction = np.argmax(arr_prediction)
  if tag_prediction == real_tag:
    color = 'green'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2,0f}% ({})".format(
    class_name[tag_prediction],
    100*np.max(arr_prediction),
    class_name[real_tag],
    color = color
  ))

def graph_arr_value(i, arr_predictions, real_tag):
  arr_predictions, real_tag = arr_predictions[i], real_tag[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  graph = plt.bar(range(10), arr_predictions, color="#777777")
  plt.ylim([0, 1])
  tag_prediction = np.argmax(arr_predictions)

  graph[tag_prediction].set_color('red')
  graph[real_tag].set_color('green')

rows = 5
columns = 5
num_images = rows * columns
plt.figure(figsize=(2 * 2 * columns, 2 * rows))

for i in range(num_images):
  plt.subplot(rows, 2 * columns, 2 * i + 1)
  graph_image(i, predictions, tags_test, images_test)
  plt.subplot(rows, 2 * columns, 2 * i + 2)
  graph_arr_value(i, predictions, tags_test)

# Cualquier index

image = images_test[10]
image = np.array([image])

prediction = model.predict(image)

print ("Prediction: " + class_name[np.argmax(prediction[0])])