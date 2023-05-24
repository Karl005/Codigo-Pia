import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Definir el entorno y las variables específicas del problema
num_states = 24  # Número de estados posibles (representando diferentes momentos del día)
num_actions = 3  # Número de acciones posibles (por ejemplo, apagar, mantener o encender un dispositivo)
energy_consumption = np.array([5, 10, 15])  # Consumo de energía asociado a cada acción
rewards = np.array([
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1]
])

# Hiperparámetros
learning_rate = 0.8
discount_factor = 0.95
hidden_units = 64

# Definir la red neuronal
model = tf.keras.Sequential([
    layers.Dense(hidden_units, activation='relu', input_shape=(num_states,)),
    layers.Dense(hidden_units, activation='relu'),
    layers.Dense(num_actions)
])

# Cargar los pesos del modelo entrenado
model.load_weights('model_weights.h5')

# Función para seleccionar una acción basada en la política Q-Value
def choose_action(state):
    return np.argmax(model.predict(state.reshape(1, -1))[0])

# Utilizar el modelo para tomar acciones
current_state = np.random.randint(0, num_states)
while True:
    action = choose_action(np.eye(num_states)[current_state])
    next_state = (current_state + 1) % num_states
    reward = rewards[current_state, action]
    energy = energy_consumption[action]
    print(f"State: {current_state}, Action: {action}, Reward: {reward}, Energy: {energy}")
    current_state = next_state
    if current_state == 0:
        break
