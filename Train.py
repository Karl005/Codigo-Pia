import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Definir el entorno y las variables específicas del problema
num_states = 24  # Número de estados posibles (representando diferentes momentos del día)
num_actions = 3  # Número de acciones posibles (por ejemplo, apagar, mantener o encender un dispositivo)
energy_consumption = np.array([5, 10, 15])  # Consumo de energía asociado a cada acción

# Definir las recompensas
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
num_episodes = 10
hidden_units = 64

# Definir la red neuronal
model = tf.keras.Sequential([
    layers.Dense(hidden_units, activation='relu', input_shape=(num_states,)),
    layers.Dense(hidden_units, activation='relu'),
    layers.Dense(num_actions)
])



# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

# Función para seleccionar una acción basada en la política Q-Value
def choose_action(state):
    return np.argmax(model.predict(state.reshape(1, -1))[0])

# Entrenamiento del agente
for episode in range(num_episodes):
    # Reiniciar el estado del agente al inicio de cada episodio
    current_state = np.random.randint(0, num_states)
    
    while True:
        # Seleccionar una acción basada en la política actual (exploración vs. explotación)
        action = choose_action(np.eye(num_states)[current_state])
        
        # Realizar la acción y obtener el siguiente estado
        next_state = (current_state + 1) % num_states
        
        # Obtener la recompensa para la transición actual
        reward = rewards[current_state, action]
        
        # Calcular el consumo de energía para la acción realizada
        energy = energy_consumption[action]
        
        # Actualizar el modelo mediante el aprendizaje supervisado
        target = reward + discount_factor * np.max(model.predict(np.eye(num_states)[next_state].reshape(1, -1)))
        target_vec = model.predict(np.eye(num_states)[current_state].reshape(1, -1))
        target_vec[0, action] = target

        
        # Entrenar el modelo con la nueva muestra
        model.fit(np.eye(num_states)[current_state].reshape(1, -1), target_vec.reshape(1, -1), epochs=1, verbose=0)
        
        # Actualizar el estado actual
        current_state = next_state
        
        # Verificar si se alcanzó el estado objetivo
        if current_state == 0:
            break
            
model.save_weights('model_weights.h5')