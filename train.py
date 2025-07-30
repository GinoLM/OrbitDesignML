# train_final.py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pykep as pk

from ml_environment import TrajectoryEnvironment
from ml_agent import TD3, ReplayBuffer

def evaluate_agent(agent, env, eval_episodes=5):
    avg_reward = 0.
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        for _ in range(int(env.MAX_SECONDS / env.DT_SECONDS)):
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            episode_reward += reward
            if done: break
        avg_reward += episode_reward
    avg_reward /= eval_episodes
    return avg_reward

def plot_learning_curve(timesteps, rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, rewards)
    plt.title('Curva de Aprendizaje del Agente')
    plt.xlabel('Pasos de Simulación (Timesteps)'); plt.ylabel('Recompensa Media de Evaluación')
    plt.grid(True); plt.show()

def analyze_and_plot_final_trajectory(agent, env):
    print("\n" + "="*50 + "\n--- ANÁLISIS DE LA TRAYECTORIA FINAL APRENDIDA ---\n" + "="*50)
    state = env.reset(); trajectory, thrust_profile = [state[0:2] * env.L_char / 1000], []
    done, initial_mass, total_delta_v = False, env.mass0, 0.0
    for i in range(int(env.MAX_SECONDS / env.DT_SECONDS)):
        action = agent.select_action(state); trajectory.append(state[0:2] * env.L_char / 1000)
        thrust_profile.append(np.linalg.norm(action)); thrust_vector = np.array([action[0], action[1], 0.0]) * env.Tmax
        current_mass = state[-1] * env.mass0
        acceleration = np.linalg.norm(thrust_vector) / current_mass if current_mass > 0 else 0
        total_delta_v += acceleration * env.DT_SECONDS; state, _, done = env.step(action)
        if done: break
    final_mass = state[-1] * env.mass0; elapsed_time_days = (i + 1) * env.DT_SECONDS / pk.DAY2SEC
    print(f"Tiempo de Vuelo Total: {elapsed_time_days:.2f} días"); print(f"Masa Final: {final_mass:.2f} kg")
    print(f"Consumo de Combustible: {initial_mass - final_mass:.2f} kg"); print(f"Delta-V Total Acumulado: {total_delta_v / 1000:.3f} km/s"); print("="*50)
    
    plt.figure(figsize=(10, 5)); plt.plot(np.linspace(0, elapsed_time_days, len(thrust_profile)), thrust_profile)
    plt.title('Perfil de Empuje (Throttle)'); plt.xlabel('Tiempo de Vuelo (días)'); plt.ylabel('Magnitud del Empuje Normalizado (0 a 1)')
    plt.grid(True); plt.ylim(0, 1.1)
    
    trajectory = np.array(trajectory); plt.figure(figsize=(10, 10)); plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trayectoria del Agente')
    earth = plt.Circle((0, 0), pk.EARTH_RADIUS / 1000, color='blue', label='Tierra')
    target_orbit = plt.Circle((0, 0), env.target_radius / 1000, color='red', fill=False, linestyle='--', label='Órbita Objetivo')
    plt.gca().add_artist(earth); plt.gca().add_artist(target_orbit)
    plt.title('Trayectoria Óptima Aprendida por el Agente'); plt.xlabel('X (km)'); plt.ylabel('Y (km)')
    plt.legend(); plt.axis('equal'); plt.grid(True); plt.show()

# --- SCRIPT PRINCIPAL ---
env_name = "LARES3_to_Moon_v9"
MAX_TIMESTEPS, START_TIMESTEPS = 500_000, 25_000
EVAL_FREQ, SAVE_FREQ = 50_000, 50_000
file_name = f"TD3_{env_name}"

lunar_orbit_altitude = 384400.0 - pk.EARTH_RADIUS/1e3 + 2000.0
env = TrajectoryEnvironment(target_altitude_km=lunar_orbit_altitude)
state_dim, action_dim, max_action = env.get_state().shape[0], 2, 1.0
policy = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(state_dim, action_dim)

if not os.path.exists("./models"): os.makedirs("./models")
try:
    policy.load(f"./models/{file_name}"); print(f"Modelo cargado: ./models/{file_name}")
except:
    print("No se encontró un modelo guardado, empezando desde cero.")

state, done = env.reset(), False
episode_reward, episode_timesteps, episode_num = 0, 0, 0
eval_timesteps, eval_rewards = [], []

print(f"--- Iniciando entrenamiento por {MAX_TIMESTEPS} timesteps ---")
for t in range(MAX_TIMESTEPS):
    episode_timesteps += 1
    if t < START_TIMESTEPS:
        action = np.random.uniform(-1, 1, size=action_dim)
    else:
        noise = np.random.normal(0, max_action * 0.1, size=action_dim)
        action = (policy.select_action(state) + noise).clip(-max_action, max_action)
    
    next_state, reward, done = env.step(action)
    replay_buffer.add(state, action, next_state, reward, float(done))
    state = next_state
    episode_reward += reward

    if t >= START_TIMESTEPS: policy.train(replay_buffer)

    if done:
        print(f"T: {t+1}, Ep: {episode_num+1}, Steps: {episode_timesteps}, Reward: {episode_reward:.2f}")
        state, done = env.reset(), False
        episode_reward, episode_timesteps = 0, 0
        episode_num += 1

    if (t + 1) % EVAL_FREQ == 0 and t >= START_TIMESTEPS:
        avg_reward = evaluate_agent(policy, env)
        eval_rewards.append(avg_reward)
        eval_timesteps.append(t + 1)
        print("-" * 40); print(f"Evaluación tras {t+1} timesteps. Recompensa media: {avg_reward:.3f}"); print("-" * 40)
        policy.save(f"./models/{file_name}")
        print("**** Modelo Guardado ****")

plot_learning_curve(eval_timesteps, eval_rewards)
analyze_and_plot_final_trajectory(policy, env)