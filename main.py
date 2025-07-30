
# main.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import pykep as pk
from astropy import units as u
from poliastro.bodies import Earth, Moon
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from scipy.integrate import solve_ivp

# --- CONFIGURACIÓN ---
# Establecer el dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Evitar problemas de duplicación de librerías en algunos entornos
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- CLASES DEL AGENTE DE APRENDIZAJE POR REFUERZO (TD3) ---

class ReplayBuffer:
    """Buffer para almacenar y muestrear experiencias (transiciones)."""
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        """Añade una nueva transición al buffer."""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Muestrea un lote de transiciones del buffer."""
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
        )

class Actor(nn.Module):
    """Red neuronal del Actor: mapea estados a acciones."""
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    """Red neuronal del Crítico: evalúa el valor de las acciones en un estado."""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Arquitectura Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Arquitectura Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3:
    """Implementación del algoritmo Twin-Delayed Deep Deterministic Policy Gradient (TD3)."""
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state):
        """Selecciona una acción basada en el estado actual (para evaluación)."""
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        """Entrena al agente durante un paso."""
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        """Guarda los modelos del agente."""
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.actor.state_dict(), filename + "_actor")

    def load(self, filename):
        """Carga los modelos del agente."""
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)

# --- CLASE DEL ENTORNO DE SIMULACIÓN DE TRAYECTORIA ---

def equations_of_motion(t, state, T_vec, isp, g0, mu):
    """Ecuaciones de movimiento para la propagación de la órbita."""
    r_vec, v_vec, m = state[0:3], state[3:6], state[6]
    r_norm = np.linalg.norm(r_vec)
    
    # Aceleración gravitacional
    a_g = -mu * r_vec / (r_norm**3) if r_norm > 0 else np.zeros(3)
    
    # Aceleración por empuje
    a_t = T_vec / m if m > 0 else np.zeros(3)
    
    # Derivadas
    dr = v_vec
    dv = a_g + a_t
    dm = -np.linalg.norm(T_vec) / (isp * g0)
    
    return np.concatenate([dr, dv, [dm]])

class TrajectoryEnvironment:
    """Entorno de simulación para la optimización de la trayectoria."""
    def __init__(self, target_orbit):
        # --- PARÁMETROS DE LA MISIÓN ---
        self.mass0 = 1000.0 * u.kg
        self.Tmax = 1.5 * u.N
        self.Isp = 2500.0 * u.s
        self.g0 = 9.80665 * u.m / u.s**2

        # --- PARÁMETROS DE SIMULACIÓN ---
        self.DT_SECONDS = 120.0  # Duración de cada paso de simulación
        self.MAX_SECONDS = 600 * pk.DAY2SEC # Duración máxima de la misión

        # --- ÓRBITA INICIAL (LARES 3) ---
        r_p = pk.EARTH_RADIUS * u.m + 250 * u.km
        r_a = pk.EARTH_RADIUS * u.m + 5700 * u.km
        self.initial_orbit = Orbit.from_classical(
            attractor=Earth,
            a=(r_p + r_a) / 2,
            ecc=(r_a - r_p) / (r_a + r_p),
            inc=0 * u.deg,
            raan=0 * u.deg,
            argp=0 * u.deg,
            nu=0 * u.deg
        )
        
        # --- ÓRBITA OBJETIVO ---
        self.target_orbit = target_orbit
        self.target_radius = self.target_orbit.a.to(u.m).value
        self.target_speed = np.sqrt(self.target_orbit.attractor.k.to(u.m**3 / u.s**2).value / self.target_radius)

        # --- ESTADO INICIAL ---
        self.current_state_vector = self.get_initial_state_vector()
        self.current_time = 0.0
        
        # --- VALORES DE NORMALIZACIÓN ---
        self.L_char = pk.EARTH_RADIUS
        self.V_char = np.sqrt(pk.MU_EARTH / pk.EARTH_RADIUS)
        
        print("Entorno de trayectoria inicializado.")

    def get_initial_state_vector(self):
        r0 = self.initial_orbit.r.to(u.m).value
        v0 = self.initial_orbit.v.to(u.m / u.s).value
        m0 = self.mass0.to(u.kg).value
        return np.concatenate([r0, v0, [m0]])

    def get_state(self):
        """Obtiene el estado normalizado del entorno."""
        r, v, m = self.current_state_vector[0:3], self.current_state_vector[3:6], self.current_state_vector[6]
        return np.concatenate([r / self.L_char, v / self.V_char, [m / self.mass0.to(u.kg).value]])

    def reset(self):
        """Reinicia el entorno a su estado inicial."""
        self.current_state_vector = self.get_initial_state_vector()
        self.current_time = 0.0
        return self.get_state()

    def step(self, action):
        """Avanza la simulación un paso de tiempo."""
        action = np.clip(action, -1.0, 1.0)
        
        # Lógica de empuje "bang-bang"
        throttle_magnitude = np.linalg.norm(action)
        final_thrust_magnitude = self.Tmax.to(u.N).value if throttle_magnitude > 0.5 else 0.0

        # Dirección del empuje
        direction_3d = np.array([action[0], action[1], 0.0])
        norm = np.linalg.norm(direction_3d)
        unit_direction = direction_3d / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
        thrust_vector = unit_direction * final_thrust_magnitude

        # Propagación de la órbita
        r_prev = self.current_state_vector[0:3]
        apogee_prev = Orbit.from_vectors(Earth, r_prev * u.m, self.current_state_vector[3:6] * u.m / u.s).a.to(u.m).value * (1 + Orbit.from_vectors(Earth, r_prev * u.m, self.current_state_vector[3:6] * u.m / u.s).ecc)

        sol = solve_ivp(
            fun=equations_of_motion,
            t_span=[0, self.DT_SECONDS],
            y0=self.current_state_vector,
            args=(thrust_vector, self.Isp.to(u.s).value, self.g0.to(u.m / u.s**2).value, pk.MU_EARTH),
            rtol=1e-7,
            atol=1e-7
        )
        
        self.current_state_vector = sol.y[:, -1]
        self.current_time += self.DT_SECONDS
        
        return self.calculate_reward_and_done(apogee_prev)
        
    def calculate_reward_and_done(self, apogee_prev):
        """Calcula la recompensa y determina si el episodio ha terminado."""
        reward, done = 0, False
        r_vec, v_vec, mass = self.current_state_vector[0:3], self.current_state_vector[3:6], self.current_state_vector[6]
        current_radius = np.linalg.norm(r_vec)
        current_speed = np.linalg.norm(v_vec)
        
        # --- RECOMPENSAS Y PENALIZACIONES ---
        radius_error = abs(current_radius - self.target_radius)
        speed_error = abs(current_speed - self.target_speed)
        
        # Condición de éxito
        if radius_error < 1000e3 and speed_error < 100:
            reward = 2000
            done = True
            print("¡OBJETIVO ALCANZADO!")
            return self.get_state(), reward, done

        # Condiciones de fracaso
        if self.current_time > self.MAX_SECONDS:
            reward = -100
            done = True
        elif mass < 100:
            reward = -100
            done = True
        elif current_radius < self.L_char:
            reward = -1000
            done = True
        
        if done:
            return self.get_state(), reward, done

        # Recompensa de guía (Reward Shaping)
        current_orbit = Orbit.from_vectors(Earth, r_vec * u.m, v_vec * u.m / u.s)
        apogee_curr = current_orbit.a.to(u.m).value * (1 + current_orbit.ecc)
        perigee_curr = current_orbit.a.to(u.m).value * (1 - current_orbit.ecc)
        
        # 1. Recompensa por aumentar el apogeo (con mayor incentivo)
        reward += (apogee_curr - apogee_prev) / self.target_radius * 2
        
        # 2. Penalización por baja altitud
        if perigee_curr < (self.L_char + 2000e3):
            reward -= 0.1
        
        # 3. Penalización por tiempo
        reward -= 0.01

        return self.get_state(), reward, done

# --- FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN ---

def evaluate_agent(agent, env, eval_episodes=5):
    """Evalúa el rendimiento del agente durante un número de episodios."""
    avg_reward = 0.
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            episode_reward += reward
        avg_reward += episode_reward
    avg_reward /= eval_episodes
    return avg_reward

def plot_learning_curve(timesteps, rewards):
    """Grafica la curva de aprendizaje del agente."""
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, rewards)
    plt.title('Curva de Aprendizaje del Agente')
    plt.xlabel('Pasos de Simulación (Timesteps)')
    plt.ylabel('Recompensa Media de Evaluación')
    plt.grid(True)
    plt.show()

def analyze_and_plot_final_trajectory(agent, env):
    """Analiza y grafica la trayectoria final aprendida por el agente."""
    print("\n" + "="*50 + "\n--- ANÁLISIS DE LA TRAYECTORIA FINAL APRENDIDA ---\n" + "="*50)
    state = env.reset()
    trajectory = [state[0:2] * env.L_char / 1000]
    thrust_profile = []
    done = False
    initial_mass = env.mass0.to(u.kg).value
    total_delta_v = 0.0
    
    i = 0
    while not done and i < int(env.MAX_SECONDS / env.DT_SECONDS):
        action = agent.select_action(state)
        trajectory.append(state[0:2] * env.L_char / 1000)
        thrust_profile.append(np.linalg.norm(action))
        
        thrust_vector = np.array([action[0], action[1], 0.0]) * env.Tmax.to(u.N).value
        current_mass = state[-1] * initial_mass
        acceleration = np.linalg.norm(thrust_vector) / current_mass if current_mass > 0 else 0
        total_delta_v += acceleration * env.DT_SECONDS
        
        state, _, done = env.step(action)
        i += 1

    final_mass = state[-1] * initial_mass
    elapsed_time_days = i * env.DT_SECONDS / pk.DAY2SEC
    
    print(f"Tiempo de Vuelo Total: {elapsed_time_days:.2f} días")
    print(f"Masa Final: {final_mass:.2f} kg")
    print(f"Consumo de Combustible: {initial_mass - final_mass:.2f} kg")
    print(f"Delta-V Total Acumulado: {total_delta_v / 1000:.3f} km/s")
    print("="*50)
    
    # Gráfica del perfil de empuje
    plt.figure(figsize=(10, 5))
    plt.plot(np.linspace(0, elapsed_time_days, len(thrust_profile)), thrust_profile)
    plt.title('Perfil de Empuje (Throttle)')
    plt.xlabel('Tiempo de Vuelo (días)')
    plt.ylabel('Magnitud del Empuje Normalizado (0 a 1)')
    plt.grid(True)
    plt.ylim(0, 1.1)
    
    # Gráfica de la trayectoria
    trajectory = np.array(trajectory)
    plt.figure(figsize=(10, 10))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trayectoria del Agente')
    earth = plt.Circle((0, 0), pk.EARTH_RADIUS / 1000, color='blue', label='Tierra')
    target_orbit_plot = plt.Circle((0, 0), env.target_radius / 1000, color='red', fill=False, linestyle='--', label='Órbita Objetivo')
    plt.gca().add_artist(earth)
    plt.gca().add_artist(target_orbit_plot)
    plt.title('Trayectoria Óptima Aprendida por el Agente')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    # --- PARÁMETROS DE ENTRENAMIENTO ---
    env_name = "Earth_to_Moon_Transfer"
    MAX_TIMESTEPS = 350_000
    START_TIMESTEPS = 25_000
    EVAL_FREQ = 50_000
    SAVE_FREQ = 50_000
    file_name = f"TD3_{env_name}"

    # --- DEFINICIÓN DEL ENTORNO ---
    lunar_altitude = 384400 * u.km
    lunar_orbit = Orbit.circular(Earth, alt=lunar_altitude)
    env = TrajectoryEnvironment(target_orbit=lunar_orbit)
    
    state_dim = env.get_state().shape[0]
    action_dim = 2  # Control en 2D (plano de la órbita)
    max_action = 1.0
    
    # --- INICIALIZACIÓN DEL AGENTE ---
    policy = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # --- CARGAR MODELO (SI EXISTE) ---
    if not os.path.exists("./models"):
        os.makedirs("./models")
    try:
        policy.load(f"./models/{file_name}")
        print(f"Modelo cargado: ./models/{file_name}")
    except:
        print("No se encontró un modelo guardado, empezando desde cero.")

    # --- BUCLE DE ENTRENAMIENTO ---
    state, done = env.reset(), False
    episode_reward, episode_timesteps, episode_num = 0, 0, 0
    eval_timesteps, eval_rewards = [], []

    print(f"--- Iniciando entrenamiento por {MAX_TIMESTEPS} timesteps ---")
    for t in range(MAX_TIMESTEPS):
        episode_timesteps += 1
        
        # Acción aleatoria al principio para exploración
        if t < START_TIMESTEPS:
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            noise = np.random.normal(0, max_action * 0.1, size=action_dim)
            action = (policy.select_action(state) + noise).clip(-max_action, max_action)
        
        next_state, reward, done = env.step(action)
        replay_buffer.add(state, action, next_state, reward, float(done))
        state = next_state
        episode_reward += reward

        if t >= START_TIMESTEPS:
            policy.train(replay_buffer)

        if done:
            print(f"T: {t+1}, Ep: {episode_num+1}, Steps: {episode_timesteps}, Reward: {episode_reward:.2f}")
            state, done = env.reset(), False
            episode_reward, episode_timesteps = 0, 0
            episode_num += 1

        # Evaluación y guardado del modelo
        if (t + 1) % EVAL_FREQ == 0 and t >= START_TIMESTEPS:
            avg_reward = evaluate_agent(policy, env)
            eval_rewards.append(avg_reward)
            eval_timesteps.append(t + 1)
            print("-" * 40)
            print(f"Evaluación tras {t+1} timesteps. Recompensa media: {avg_reward:.3f}")
            print("-" * 40)
            policy.save(f"./models/{file_name}")
            print("**** Modelo Guardado ****")

    # --- FINALIZACIÓN ---
    plot_learning_curve(eval_timesteps, eval_rewards)
    analyze_and_plot_final_trajectory(policy, env)

    print("--- Entrenamiento Finalizado ---")
