# ml_environment_v9.py

import pykep as pk
import numpy as np
from math import sqrt
from scipy.integrate import solve_ivp

def eom(t, state, T_vec, isp, g0, mu):
    r_vec, v_vec, m = state[0:3], state[3:6], state[6]
    dr, r_norm = v_vec, np.linalg.norm(r_vec)
    a_g = -mu * r_vec / (r_norm**3) if r_norm > 0 else 0
    a_t = T_vec / m if m > 0 else 0
    dv = a_g + a_t
    dm = -np.linalg.norm(T_vec) / (isp * g0)
    return np.concatenate([dr, dv, [dm]])

class TrajectoryEnvironment:
    def __init__(self, target_altitude_km):
        # --- PARÁMETROS DE LA MISIÓN (Puedes manipularlos aquí) ---
        self.mass0 = 1000.0
        self.Tmax = 0.6  # Empuje en Newtons
        self.Isp = 2500.0 # Impulso específico en segundos
        
        # --- PARÁMETROS DE SIMULACIÓN ---
        self.DT_SECONDS = 120  # Duración de cada paso
        self.MAX_SECONDS = 600 * pk.DAY2SEC # Duración máxima de la misión

        # Órbita de salida elíptica (LARES 3)
        r_p = pk.EARTH_RADIUS + 250e3; r_a = pk.EARTH_RADIUS + 5700e3
        a_inicial, e_inicial = (r_p + r_a) / 2.0, (r_a - r_p) / (r_a + r_p)
        r0_tuple, v0_tuple = pk.par2ic([a_inicial, e_inicial, 0, 0, 0, 0], pk.MU_EARTH)
        self.r0, self.v0 = np.array(r0_tuple), np.array(v0_tuple)
        
        self.target_radius = pk.EARTH_RADIUS + target_altitude_km * 1000
        self.target_speed = sqrt(pk.MU_EARTH / self.target_radius)
        self.current_state_vector = np.concatenate([self.r0, self.v0, [self.mass0]])
        self.current_time = 0.0
        self.L_char, self.V_char = pk.EARTH_RADIUS, sqrt(pk.MU_EARTH / pk.EARTH_RADIUS)
        print("Entorno de trayectoria v9 (conciencia de peligro) inicializado.")
        
    def get_state(self):
        r, v, m = self.current_state_vector[0:3], self.current_state_vector[3:6], self.current_state_vector[6]
        return np.concatenate([r / self.L_char, v / self.V_char, [m / self.mass0]])

    def reset(self):
        self.current_state_vector = np.concatenate([self.r0, self.v0, [self.mass0]])
        self.current_time = 0.0
        return self.get_state()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        # --- CORRECCIÓN DE LA LÓGICA "BANG-BANG" ---
        throttle_magnitude = np.linalg.norm(action)

        if throttle_magnitude > 0.5: # Umbral para encender
            final_thrust_magnitude = self.Tmax
        else:
            final_thrust_magnitude = 0.0 # Apagado

        # Calculamos la dirección unitaria en 3D
        direction_3d = np.array([action[0], action[1], 0.0])
        norm = np.linalg.norm(direction_3d)
        unit_direction = direction_3d / norm if norm > 0 else np.array([1.0, 0.0, 0.0])

        # El vector de empuje final es 3D
        thrust_vector = unit_direction * final_thrust_magnitude
        # --- FIN DE LA CORRECCIÓN ---

        # El resto del método no cambia
        r_prev, v_prev = self.current_state_vector[0:3], self.current_state_vector[3:6]
        try:
            elements_prev = pk.ic2par(list(r_prev), list(v_prev), pk.MU_EARTH)
            apogee_prev = elements_prev[0] * (1 + elements_prev[1])
        except:
            apogee_prev = np.linalg.norm(r_prev)
            
        sol = solve_ivp(fun=eom, t_span=[0, self.DT_SECONDS], y0=self.current_state_vector,
                        args=(thrust_vector, self.Isp, pk.G0, pk.MU_EARTH), rtol=1e-7, atol=1e-7)
        
        self.current_state_vector = sol.y[:, -1]
        self.current_time += self.DT_SECONDS
        
        return self.calculate_reward_and_done(apogee_prev)
        
    def calculate_reward_and_done(self, apogee_prev):
        reward, done = 0, False
        r_vec, v_vec, mass = self.current_state_vector[0:3], self.current_state_vector[3:6], self.current_state_vector[6]
        current_radius, current_speed = np.linalg.norm(r_vec), np.linalg.norm(v_vec)
        
        # --- RECOMPENSAS Y CASTIGOS (Puedes manipularlos aquí) ---
        radius_error = abs(current_radius - self.target_radius)
        speed_error = abs(current_speed - self.target_speed)
        
        # Condición de Éxito
        if radius_error < 1000e3 and speed_error < 100:
            reward = 2000; done = True; print("🎉 ¡OBJETIVO ALCANZADO!")
            return self.get_state(), reward, done

        # Condiciones de Fracaso
        if self.current_time > self.MAX_SECONDS: reward = -100; done = True
        elif mass < 100: reward = -100; done = True
        elif current_radius < self.L_char: reward = -1000; done = True
        if done: return self.get_state(), reward, done

        # Recompensa de Guía (Reward Shaping)
        try:
            elements_curr = pk.ic2par(list(r_vec), list(v_vec), pk.MU_EARTH)
            apogee_curr = elements_curr[0] * (1 + elements_curr[1])
            perigee_curr = elements_curr[0] * (1 - elements_curr[1])
        except:
            apogee_curr, perigee_curr = current_radius, current_radius
        
        # 1. Recompensa por aumentar apogeo
        reward += (apogee_curr - apogee_prev) / self.target_radius
        
        # 2. Penalización por proximidad
        if perigee_curr < (self.L_char + 2000e3):
            reward -= 0.1
        
        # 3. Penalización por tiempo
        reward -= 0.01

        return self.get_state(), reward, done