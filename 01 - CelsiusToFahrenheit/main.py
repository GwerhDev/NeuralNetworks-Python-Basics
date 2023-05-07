import tensorflow as tf
import numpy as np
import matplotlib as plt

# La conversión de Celsius a Fahrenheit corresponde a la siguiente función lineal:
# Fahrenheit = Celsius * 1.8 + 32
# En el siguiente código entrenaremos un modelo de red neuronal simple para llegar
# a esta fórmula (o acercarnos)

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)

fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

layer = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([layer])

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print("Training has began...")

historial = model.fit(celsius, fahrenheit, epochs = 1000, verbose = False)

print("Model has been trained!")

plt.xlabel("# Epoch")
plt.ylabel("Loss")
plt.plot(historial.history["loss"])

print("Let's predict!")

result = model.predict([100,0])

print("Result is " + str(result) + " fahrenheit!")

print("Inner variables of model")

print(layer.get_weights())