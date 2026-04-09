import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from IPython.display import HTML
import pickle

def simular_tubo(FIS, altura_deseada=10.0, coef_fuerza = 0.05, coef_suavidad = 0.05, pasos_tiempo=100, tiempo_final = 10,
                 init = (0,0), noisy = True, noise = None, penalizar_fuerza = True, penalizar_suavidad = True):
    
    """
    Simula la física de la bola y el controlador.
    Parámetros de entrada del PSO: [K_pos, K_desplazamiento, Fuerza_Base]
    """

    dt = tiempo_final/pasos_tiempo
    
    y = init[0]       # Posición inicial
    v = init[1]       # Velocidad inicial
    y_prev = y - v*dt # Posición en el instante anterior
        
    m = 1.0      # Masa de la bola (kg)
    g = 9.81     # Gravedad (m/s^2)
    
    historial_y = []
    historial_fuerza = []
    for este_paso in range(pasos_tiempo):
        # Posición (vista como error respecto al objetivo)
        error_posicion = y - altura_deseada
        # Velocidad media desde el último instante
        velocidad = (y - y_prev)/dt
        
        fuerza = max(0, FIS.eval(error_posicion, velocidad))
        historial_fuerza.append(fuerza)
        # --- PERTURBACIÓN Y FÍSICA ---
        if noisy:
            if noise is None:
                interferencia = np.random.normal(0, 1.5)
            elif len(noise) == pasos_tiempo:
                interferencia = noise[este_paso]
            else: 
                interferencia = 0
        else: 
            interferencia = 0
        
        # Segunda Ley de Newton: Fuerza Neta = Masa * Aceleración
        fuerza_neta = fuerza - (m * g) + interferencia
        aceleracion = fuerza_neta / m
        
        # Actualización de cinemática (Método de Euler)
        v += aceleracion * dt
        y_prev = y
        y += v * dt
        
        # Restricción del suelo del tubo
        if y < 0:
            y = 0.0
            v = -0.3 * v
            
        historial_y.append(y)
        
    # FUNCIÓN DE PÉRDIDA: MAE
    perdida = np.mean(np.abs(np.array(historial_y) - altura_deseada))
    if penalizar_fuerza:
        perdida += coef_fuerza * np.mean(np.abs(np.array(historial_fuerza)))
    if penalizar_suavidad:
        perdida += coef_suavidad * np.mean(np.abs(np.diff(historial_fuerza)))

    return perdida, historial_y

class FS:
    def __init__(self, centro, sigma, tipo):
        self.tipo = tipo
        self.c = centro
        self.sigma = sigma
        if self.sigma< 1e-4: 
            self.sigma = 1e-4
    
    def eval(self, x):
        if self.tipo == "gaussiano":
            return np.exp(-0.5 * ((x - self.c) / self.sigma)**2)
        if self.tipo == "triangular":
            return np.maximum(0, 1 - np.abs(x - self.c) / self.sigma)

class SugenoFIS:
    def __init__(self, parametros, tipo = "gaussiano"):
        """
        Inicializa el FIS con un vector de 21 parámetros.
        """
        self.FSs_errores = []
        self.FSs_velocidades = []

        parametros_error, parametros_vel, consecuentes = parametros
        centros_error, sigmas_error = parametros_error
        centros_vel, sigmas_vel = parametros_vel

        # Parámetros de las Funciones de Pertenencia del ERROR (Centros y Sigmas)1
        if (len(centros_error) != len(sigmas_error)):
            raise ValueError(f"Los centros ({len(centros_error)}) y las sigmas ({len(sigmas_error)}) de los errores no tienen la misma longitud")
        if (len(centros_vel) != len(sigmas_vel)):
            raise ValueError(f"Los centros ({len(centros_vel)}), las sigmas ({len(sigmas_vel)}) de las velocidades no tienen la misma longitud")
        if consecuentes.shape != (len(centros_error), len(centros_vel)):
            raise ValueError(f"La dimensión de la matriz de los consecuentes {consecuentes.shape} no coincide con la esperada {(len(centros_error), len(centros_vel))}")
        
        for i in range(len(centros_error)):
            self.FSs_errores.append(FS(centros_error[i], sigmas_error[i], tipo))
        
        for i in range(len(centros_vel)):
            self.FSs_velocidades.append(FS(centros_vel[i], sigmas_vel[i], tipo))
        
        self.consecuentes = consecuentes

    def eval(self, error, vel):
        """
        Evalúa las entradas y devuelve la fuerza de salida calculada.
        """
        eval_FSs_errores = []
        eval_FSs_velocidades = []
        for FS_error in self.FSs_errores:
            eval_FSs_errores.append(FS_error.eval(error))
        for FS_vel in self.FSs_velocidades:
            eval_FSs_velocidades.append(FS_vel.eval(vel))

        evals_reglas = np.outer(eval_FSs_errores, eval_FSs_velocidades)
        
        # 3. Defuzzificación (Suma ponderada de Sugeno)
        suma_pesos = np.sum(evals_reglas)
        if suma_pesos == 0:
            return 0.0
            
        salida = np.sum(evals_reglas * self.consecuentes) / suma_pesos
        return salida
    
def fis_desde_particula(particula, n_e, n_v, tipo = "gaussiano"):
    """ Vector -> objeto SugenoFIS."""
    idx = 0
    
    c_error = particula[idx : idx+n_e]; idx += n_e
    s_error = particula[idx : idx+n_e]; idx += n_e
    
    c_vel = particula[idx : idx+n_v]; idx += n_v
    s_vel = particula[idx : idx+n_v]; idx += n_v
    
    consecuentes = particula[idx : idx+(n_e*n_v)].reshape((n_e, n_v))
    
    # Esta es exactamente la tupla de tuplas que espera el __init__ de tu clase
    parametros = ((c_error, s_error), (c_vel, s_vel), consecuentes)
    return SugenoFIS(parametros, tipo)

def particula_desde_fis(fis):
    
    idx = 0
    n_e = len(fis.FSs_errores)
    n_v = len(fis.FSs_velocidades)

    c_error = [FS.c for FS in fis.FSs_errores]
    s_error = [FS.sigma for FS in fis.FSs_errores]

    c_vel = [FS.c for FS in fis.FSs_velocidades]
    s_vel = [FS.sigma for FS in fis.FSs_velocidades]

    consecuentes = fis.consecuentes.flatten()
    
    particula = np.concatenate([c_error, s_error, c_vel, s_vel, consecuentes])
    return particula

def generar_poblacion(fis_buenos, tamano, lim_min, lim_max, 
                      porcentaje_expertos=0.15, 
                      porcentaje_interpolacion=0.25, 
                      ruido_sigma=0.03):
    """
    Crea una población robusta y continua para el PSO combinando:
    1. Expertos (FIS procedentes del AG).
    2. Interpolación: Cruce lineal entre expertos para rellenar el espacio entre ellos.
    3. Exploración local: Clones con ruido de los expertos.
    4. Diversidad: Individuos aleatorios.
    """
    dim = len(lim_min)
    poblacion_total = []

    if fis_buenos is None or len(fis_buenos) == 0:
        return np.random.uniform(lim_min, lim_max, (tamano, dim))
    
    particulas_expertas = [particula_desde_fis(f) for f in fis_buenos]
    num_expertos = len(particulas_expertas)

    poblacion_total.extend(particulas_expertas)

    # Creamos individuos que están entre de dos expertos aleatorios
    num_interp = int(tamano * porcentaje_interpolacion)
    for _ in range(num_interp):
        idx1, idx2 = np.random.choice(num_expertos, 2, replace=False)
        padre1 = particulas_expertas[idx1]
        padre2 = particulas_expertas[idx2]
        
        alfa = np.random.rand(dim)
        hijo = alfa * padre1 + (1 - alfa) * padre2
        poblacion_total.append(np.clip(hijo, lim_min, lim_max))

    # Generamos variaciones alrededor de los expertos
    num_clones = int(tamano * porcentaje_expertos)
    rango = lim_max - lim_min
    for _ in range(num_clones):
        idx = np.random.randint(num_expertos)
        ruido = np.random.normal(0, rango * ruido_sigma)
        clon = particulas_expertas[idx] + ruido
        poblacion_total.append(np.clip(clon, lim_min, lim_max))

    num_actual = len(poblacion_total)
    num_restante = max(0, tamano - num_actual)
    
    if num_restante > 0:
        randoms = np.random.uniform(lim_min, lim_max, (num_restante, dim))
        poblacion_total.extend(randoms)

    # Convertir a array de numpy y asegurar el tamaño exacto
    return np.array(poblacion_total)[:tamano]

def graficar_fis(fis, historial, init, pasos = 100, tiempo_final = 5, resolucion = 50):  
    tiempo = np.linspace(0, tiempo_final, pasos)

    perdida_final = 0
    trayectorias = []
    for i in range(len(init)):   
        perdida, trayectoria = simular_tubo(fis, altura_deseada=10.0, pasos_tiempo=pasos, tiempo_final=tiempo_final, init = init[i])
        trayectorias.append(trayectoria)
        perdida_final += perdida/len(init)
    print(f"\nOptimización finalizada. Pérdida final: {perdida_final:.3f} m")
    
    # 3. Visualización de los resultados
    fig = plt.figure(figsize=(16, 10))
    
    # --- Gráfica 1: Trayectoria de la bola ---
    ax1 = plt.subplot(2, 2, 1)
    for trayectoria in trayectorias:
        ax1.plot(tiempo, trayectoria, lw=2)
    ax1.axhline(y=10.0, color='red', linestyle='--', label="Objetivo (10 m)")
    ax1.set_title("Trayectorias controladas por SugenoFIS")
    ax1.set_xlabel("Tiempo")
    ax1.set_ylabel("Altura")
    ax1.grid(True, alpha=0.3)
    
# --- Gráfica 2: Convergencia del PSO ---
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(historial, color='#2ca02c', lw=2)
    ax2.set_title("Curva de Aprendizaje")
    ax2.set_xlabel("Iteración")
    ax2.set_ylabel("Pérdida")
    ax2.grid(True, alpha=0.3)
    
    # --- Gráfica 3: Funciones de pertenencia del ERROR ---
    ax3 = plt.subplot(2, 2, 3)
    x_error = np.linspace(-10, 15, 200)
    colores = ['red', 'green', 'blue', 'orange', 'purple']
    for i, fs in enumerate(fis.FSs_errores):
        ax3.plot(x_error, fs.eval(x_error), color=colores[i%len(colores)], label=f"Error {i+1}")
    ax3.set_title("Conjuntos Difusos: Error de Posición")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
# --- Gráfica 4: Funciones de pertenencia de la VELOCIDAD ---
    ax4 = plt.subplot(2, 2, 4)
    x_vel = np.linspace(-20, 20, 200)
    for i, fs in enumerate(fis.FSs_velocidades):
        ax4.plot(x_vel, fs.eval(x_vel), color=colores[i%len(colores)], label=f"Velocidad {i+1}")
    ax4.set_title("Conjuntos Difusos: Velocidad")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    rango_error = np.linspace(-10, 5, resolucion)
    rango_velocidad = np.linspace(-20, 20, resolucion)

    X_err, Y_vel = np.meshgrid(rango_error, rango_velocidad)
    Z_fuerza = np.zeros_like(X_err)

    for i in range(X_err.shape[0]):
        for j in range(X_err.shape[1]):
            Z_fuerza[i, j] = fis.eval(X_err[i, j], Y_vel[i, j])

    fig_3d = plt.figure(figsize=(12, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    surf = ax_3d.plot_surface(X_err, Y_vel, Z_fuerza, 
                           cmap=cm.coolwarm, 
                           linewidth=0.1, 
                           antialiased=True,
                           edgecolor='none', 
                           alpha=0.9)

    ax_3d.set_title("Salida del FIS Optimizado", pad=20, fontsize=14)
    ax_3d.set_xlabel("\nError de Posición (m)", fontsize=11)
    ax_3d.set_ylabel("\nVelocidad (m/s)", fontsize=11)
    ax_3d.set_zlabel("\nFuerza de Control Producida", fontsize=11)

    ax_3d.view_init(elev=20., azim=45) # Elevación y Azimut
    plt.show()

def graficar_fis_limpio(fis, init, pasos = 100, tiempo_final = 5, resolucion = 50):  
    tiempo = np.linspace(0, tiempo_final, pasos)

    perdida_final = 0
    trayectorias = []
    for i in range(len(init)):   
        perdida, trayectoria = simular_tubo(fis, altura_deseada=10.0, pasos_tiempo=pasos, tiempo_final=tiempo_final, init=init[i])
        trayectorias.append(trayectoria)
        perdida_final += perdida / len(init)
    
    # --- 1. PREPARAR DATOS PARA LA GRÁFICA 3D ---
    rango_error = np.linspace(-10, 10, resolucion)
    rango_velocidad = np.linspace(-20, 20, resolucion)
    X_err, Y_vel = np.meshgrid(rango_error, rango_velocidad)
    Z_fuerza = np.zeros_like(X_err)

    for i in range(X_err.shape[0]):
        for j in range(X_err.shape[1]):
            Z_fuerza[i, j] = fis.eval(X_err[i, j], Y_vel[i, j])
    
    # --- 2. VISUALIZACIÓN EN UN SOLO PLOT DE 4 PANELES ---
    fig = plt.figure(figsize=(16, 10))
    colores = ['red', 'green', 'blue', 'orange', 'purple']
    
    # --- Gráfica 1: Trayectoria de la bola (2D) ---
    ax1 = fig.add_subplot(2, 2, 1)
    for trayectoria in trayectorias:
        ax1.plot(tiempo, trayectoria, lw=2)
    ax1.axhline(y=10.0, color='red', linestyle='--', label="Objetivo (10 m)")
    ax1.set_title(f"Trayectorias. Pérdida : {perdida_final:.3f}", fontsize=12)
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Altura (m)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Gráfica 2: Superficie de Control (3D) ---
    # Usamos projection='3d' para este subplot específico
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax2.plot_surface(X_err, Y_vel, Z_fuerza, 
                           cmap=cm.coolwarm, 
                           linewidth=0.1, 
                           antialiased=True,
                           edgecolor='none', 
                           alpha=0.9)

    ax2.set_title("Salida del FIS", pad=15, fontsize=12)
    ax2.set_xlabel("\nError Posición (m)")
    ax2.set_ylabel("\nVelocidad (m/s)")
    ax2.set_zlabel("\nFuerza (N)")
    ax2.view_init(elev=20., azim=45)
    
    # --- Gráfica 3: Funciones de pertenencia del ERROR (2D) ---
    ax3 = fig.add_subplot(2, 2, 3)
    x_error = np.linspace(-20, 20, 200)
    for i, fs in enumerate(fis.FSs_errores):
        ax3.plot(x_error, fs.eval(x_error), color=colores[i%len(colores)], label=f"Error {i+1}")
    ax3.set_title("Conjuntos Difusos: Error de Posición", fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- Gráfica 4: Funciones de pertenencia de la VELOCIDAD (2D) ---
    ax4 = fig.add_subplot(2, 2, 4)
    x_vel = np.linspace(-20, 20, 200)
    for i, fs in enumerate(fis.FSs_velocidades):
        ax4.plot(x_vel, fs.eval(x_vel), color=colores[i%len(colores)], label=f"Velocidad {i+1}")
    ax4.set_title("Conjuntos Difusos: Velocidad", fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def animar_control(FIS, init=(0,0), altura_deseada = 10.0, tiempo_final = 5, pasos = 100):
    """
    Crea una animación de doble panel:
    Izquierda: Simulación física de la bola redonda en el tubo.
    Derecha: Dos barras verticales indicando Velocidad y Fuerza aplicada en tiempo real.
    """
    
    # 1. GENERAR DATOS COMPLETOS DE LA TRAYECTORIA (Física y Control)
    # Necesitamos una versión de simular_tubo que devuelva y, v, y fuerza.
    # Como no la tengo definida, asumo que 'simular_tubo' ya lo hace o la simulo rápido aquí:
    
    y_hist, vel_hist, fuerza_hist = [], [], []
    dt = tiempo_final/pasos
    m, g = 1.0, 9.81
    y, v = init[0], init[1]
    y_prev = y - v*dt
    
    # Simulación limpia (noisy=False) para la animación
    for _ in range(pasos):
        err_pos = y - altura_deseada
        fuerza = max(0, FIS.eval(err_pos, v)) # Asumimos FIS.eval(err, vel)
        
        # Guardar estado actual
        y_hist.append(y)
        vel_hist.append(v)
        fuerza_hist.append(fuerza)
        
        # Física (Euler)
        fuerza_neta = fuerza - (m * g)
        aceleracion = fuerza_neta / m
        v += aceleracion * dt
        y_prev = y
        y += v * dt
        if y < 0: y, v = 0.0, -0.3*v # Suelo

    
    fig, (ax_tubo, ax_vel, ax_fuerza) = plt.subplots(1, 3, figsize=(4, 7), 
                                                     gridspec_kw={'width_ratios': [1.5, 1, 1]})
    
    # --- A. Panel Tubo (Izquierda) ---
    ax_tubo.set_xlim(-1.3, 1.3)
    ax_tubo.set_ylim(0, 22)
    ax_tubo.set_aspect('equal', adjustable='box') 
    ax_tubo.set_xticks([])
    ax_tubo.set_title("Simulación Física", fontsize=12, pad=10)
    
    # Dibujar paredes del tubo (ensanchadas a +/-1.1 para dar aire)
    ax_tubo.plot([-1.1, -1.1], [0, 22], color='black', lw=4)
    ax_tubo.plot([ 1.1,  1.1], [0, 22], color='black', lw=4)
    
    # Línea de objetivo
    ax_tubo.axhline(y=altura_deseada, color='red', linestyle='--', alpha=0.7)
    
    # La bola (Radio 0.9 es grande, ensanchamos límites X para que quepa bien)
    bola = patches.Circle((0, y_hist[0]), 0.9, color='dodgerblue', zorder=3, ec='black', lw=1)
    ax_tubo.add_patch(bola)

    # --- B. Panel Barra Velocidad (Centro) ---
    ax_vel.set_title("Velocidad\n(m/s)", fontsize=11)
    # Límites para velocidad: Asumimos rango +/- 15 m/s
    v_limit = 15 
    ax_vel.set_xlim(0, 1)
    ax_vel.set_ylim(-v_limit, v_limit*1.2)
    ax_vel.set_xticks([])
    ax_vel.grid(True, axis='y', alpha=0.3)
    ax_vel.axhline(y=0, color='black', lw=1) # Línea de v=0
    
    # Barra de velocidad (inicialmente en v=0, centrada en x=0.5)
    bar_vel = ax_vel.bar(0.5, 0, width=0.6, color='seagreen', alpha=0.8, align='center')[0]

    # --- C. Panel Barra Fuerza (Derecha) ---
    ax_fuerza.set_title("Fuerza Control\n(N)", fontsize=11)
    # Límites para fuerza: De 0 a la fuerza máxima esperada (ej. 40N)
    f_limit = 60 
    ax_fuerza.set_xlim(0, 1)
    ax_fuerza.set_ylim(0, f_limit)
    ax_fuerza.set_xticks([])
    ax_fuerza.grid(True, axis='y', alpha=0.3)
    ax_fuerza.axhline(y=9.81, color='red', linestyle=':', alpha=0.5) # Línea de Gravedad
    
    # Barra de fuerza (inicialmente en F=0)
    bar_fuerza = ax_fuerza.bar(0.5, 0, width=0.6, color='tomato', alpha=0.8, align='center')[0]

    # 3. FUNCIÓN DE ACTUALIZACIÓN PARA LA ANIMACIÓN
    def update(frame):
        # A. Actualizar bola
        bola.set_center((0, y_hist[frame]))
        
        # B. Actualizar Barra Velocidad
        v_actual = vel_hist[frame]
        bar_vel.set_height(v_actual)
        
        # C. Actualizar Barra Fuerza
        f_actual = fuerza_hist[frame]
        bar_fuerza.set_height(f_actual)
        
        # Retornamos los artistas que han cambiado para el blitting
        return bola, bar_vel, bar_fuerza

    ani = FuncAnimation(fig, update, frames=pasos, interval=(tiempo_final/pasos)*1000, blit=True)
    
    plt.tight_layout()
    plt.close()
    
    
    return ani

def optimizar_pso(n_e=3, n_v=3, num_particulas = 500, num_iteraciones = 100, init = [(0,0)], repes = 5,\
                  tipo = "gaussiano", paciencia = 30, tolerancia = 1e-4, noisy = True, common_noise = True,
                  pasos_tiempo = 100, t_max = 5, fis_iniciales = []):
    if common_noise == False:
        noise = None

    if init == "random":
        lista_inits = [(np.random.uniform(0, 20), np.random.uniform(-5, 5)) for _ in range(repes)]
    else:
        lista_inits = init
        repes = len(init)
        
    dim = (n_e * 2) + (n_v * 2) + (n_e * n_v)
    
    limites_min = []
    limites_max = []
    
    limites_min.extend(np.linspace(-15, -2, n_e))
    limites_max.extend(np.linspace(2, 15, n_e))
    limites_min.extend([1] * n_e)
    limites_max.extend([10.0] * n_e)
    
    limites_min.extend(np.linspace(-20, -2, n_v))
    limites_max.extend(np.linspace(2, 20, n_v))
    limites_min.extend([1.0] * n_v)
    limites_max.extend([10.0] * n_v)
    
    limites_min.extend([0.0] * (n_e * n_v))
    limites_max.extend([60.0] * (n_e * n_v))
    
    lim_min = np.array(limites_min)
    lim_max = np.array(limites_max)
    
    # Inicialización del PSO
    if fis_iniciales is not None and len(fis_iniciales) > 0:
        velocidades_dadas = np.zeros((len(fis_iniciales), dim)) 
        num_random = num_particulas - len(fis_iniciales)
    else:
        velocidades_dadas = np.empty((0, dim))
        num_random = num_particulas

    velocidades_random = np.random.uniform(-2, 2, (num_random, dim))

    posiciones = generar_poblacion(fis_iniciales, num_particulas, lim_min, lim_max)
    velocidades = np.vstack((velocidades_dadas, velocidades_random))

    pbest_pos = np.copy(posiciones)
    
    # Evaluamos el enjambre inicial instanciando los FIS
    pbest_perdidas = np.zeros(num_particulas)
    for i in range(num_particulas):
        fis_temp = fis_desde_particula(posiciones[i], n_e, n_v, tipo)
        perdida = 0
        for salida in lista_inits:
            perdida += simular_tubo(fis_temp, init = salida, noisy = noisy, tiempo_final=t_max, 
                                    pasos_tiempo=pasos_tiempo)[0]/repes
        pbest_perdidas[i] = perdida
    
    gbest_idx = np.argmin(pbest_perdidas)
    gbest_pos = np.copy(pbest_pos[gbest_idx])
    gbest_perdida = pbest_perdidas[gbest_idx]
    
    # Inicializamos los parámetros de actualización
    w, c1, c2 = 0.99, 2.5, 0.5
    w_final, c1_final, c2_final = 0.4, 0.5, 2.5
    w_step, c1_step, c2_step = (w_final - w)/num_iteraciones, (c1_final-c1)/num_iteraciones, (c2_final- c2)/num_iteraciones

    historial_convergencia = []

    generaciones_sin_mejora = 0
    mejor_historico = float('inf')
    
    print(f"Iniciando PSO para SugenoFIS ({n_e}x{n_v} reglas | {dim} dim | {num_particulas} particulas | {num_iteraciones} iteraciones)")
    for iteracion in range(num_iteraciones):
        if common_noise == True:
                noise = np.random.normal(0, 1.5, size = pasos_tiempo)
        for j in range(num_particulas):
            # Ecuaciones estándar de PSO
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocidades[j] = (w * velocidades[j] + 
                              c1 * r1 * (pbest_pos[j] - posiciones[j]) + 
                              c2 * r2 * (gbest_pos - posiciones[j]))
            
            posiciones[j] += velocidades[j]
            posiciones[j] = np.clip(posiciones[j], lim_min, lim_max)
            
            # Evaluación
            fis_actual = fis_desde_particula(posiciones[j], n_e, n_v, tipo)

            perdida = 0
            for i in range(repes):
                perdida += simular_tubo(fis_actual, init=lista_inits[i], noisy = noisy, noise = noise, tiempo_final=t_max, 
                                    pasos_tiempo=pasos_tiempo)[0]/repes
            
            fis_pbest = fis_desde_particula(pbest_pos[j], n_e, n_v, tipo)
            nueva_perdida_pbest = 0
            if noisy:
                for init in lista_inits:
                    l_pbest, _ = simular_tubo(fis_pbest, init=init, noisy=noisy)
                    nueva_perdida_pbest += l_pbest / len(lista_inits)

                pbest_perdidas[j] = nueva_perdida_pbest

            if perdida < pbest_perdidas[j]:
                pbest_perdidas[j] = perdida
                pbest_pos[j] = np.copy(posiciones[j])

        gbest_idx = np.argmin(pbest_perdidas)
        gbest_pos = np.copy(pbest_pos[gbest_idx])
        gbest_perdida = pbest_perdidas[gbest_idx]
                
        historial_convergencia.append(gbest_perdida)
        w  = w  + w_step
        c1 = c1 + c1_step
        c2 = c2 + c2_step

        if (mejor_historico - gbest_perdida) > tolerancia:
            mejor_historico = gbest_perdida
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1
            
        if (iteracion+1) % 5 == 0:
            print(f"Iteración {iteracion+1}/{num_iteraciones} | Error: {gbest_perdida:.3f} m | Estancado: {generaciones_sin_mejora}/{paciencia}")
            
        if generaciones_sin_mejora >= paciencia:
            print(f"\n Early Stopping activado en la iteración {iteracion+1}.")
            print(f"El error no ha mejorado más de {tolerancia} en {paciencia} iteraciones consecutivas.")

            mejor_fis = fis_desde_particula(gbest_pos, n_e, n_v, tipo = tipo)
            return mejor_fis, historial_convergencia
            
    # Devuelve el mejor FIS
    mejor_fis = fis_desde_particula(gbest_pos, n_e, n_v, tipo = tipo)
    return mejor_fis, historial_convergencia

def optimizar_ga(n_e=3, n_v=3, tamano_poblacion = 1000, num_generaciones = 100, tasa_mutacion=0.1, fuerza_mutacion_base=0.1, elitismo=2, 
                 init=[(0,0)], repes=5, noisy=True, common_noise=True, pasos_tiempo=100, t_max=5, fis_iniciales = [], return_poblacion = False):

    if isinstance(init, list):
        repes = len(init)
        
    dim = (n_e * 2) + (n_v * 2) + (n_e * n_v)
    
    # Límites dinámicos (Permitiendo mayor solapamiento cerca del 0)
    limites_min, limites_max = [], []
    limites_min.extend(np.linspace(-15, -2, n_e)); limites_max.extend(np.linspace(2, 15, n_e))
    limites_min.extend([1] * n_e); limites_max.extend([10.0] * n_e)
    limites_min.extend(np.linspace(-20, -2, n_v)); limites_max.extend(np.linspace(2, 20, n_v))
    limites_min.extend([1] * n_v); limites_max.extend([10.0] * n_v)
    limites_min.extend([0.0] * (n_e * n_v)); limites_max.extend([60.0] * (n_e * n_v))
    
    lim_min = np.array(limites_min)
    lim_max = np.array(limites_max)
    

    poblacion = generar_poblacion(fis_iniciales, tamano_poblacion, lim_min, lim_max)
    historial_convergencia = []
    
    print(f"Iniciando AG ({dim} dim | Población: {tamano_poblacion} | Generaciones: {num_generaciones})")
    
    # Evaluamos la población inicial (fuera del bucle principal para tener un punto de partida)
    perdidas = np.zeros(tamano_poblacion)
    for ind in range(tamano_poblacion):
        fis_temp = fis_desde_particula(poblacion[ind], n_e, n_v)
        for rep in range(repes):
            l, _ = simular_tubo(fis_temp, init=init[rep], noisy=False, pasos_tiempo=pasos_tiempo, tiempo_final=t_max)
            perdidas[ind] += l / repes

    # Ordenamos la población inicial
    indices_ordenados = np.argsort(perdidas)
    poblacion = poblacion[indices_ordenados]
    perdidas = perdidas[indices_ordenados]

    for generacion in range(num_generaciones):
        
        # 1. GENERAR RUIDO COMÚN PARA TODA LA GENERACIÓN
        if common_noise:
            noise_generacion = np.random.normal(0, 1.5, size=pasos_tiempo)
        else:
            noise_generacion = None

        # 2. RE-EVALUAR A LOS ÉLITES (Prueba de robustez)
        for i in range(elitismo):
            fis_elite = fis_desde_particula(poblacion[i], n_e, n_v)
            perdida_revaluada = 0
            for rep in range(repes):
                l, _ = simular_tubo(fis_elite, init=init[rep], noisy=noisy, noise=noise_generacion, 
                                    pasos_tiempo=pasos_tiempo, tiempo_final=t_max)
                perdida_revaluada += l / repes
            perdidas[i] = perdida_revaluada

        # Re-ordenamos por si algún élite era un "farsante"
        indices_ordenados = np.argsort(perdidas)
        poblacion = poblacion[indices_ordenados]
        perdidas = perdidas[indices_ordenados]
        
        mejor_error_gen = perdidas[0]
        historial_convergencia.append(mejor_error_gen)
        
        if (generacion + 1) % 5 == 0:
            print(f"Generación {generacion+1}/{num_generaciones} | Mejor Pérdida: {mejor_error_gen:.3f} ")
            
        nueva_poblacion = np.zeros_like(poblacion)
        nueva_poblacion[0:elitismo] = poblacion[0:elitismo]
        
        # 3. CREACIÓN DE LA NUEVA GENERACIÓN
        for i in range(elitismo, tamano_poblacion):
            # Selección por Torneo
            t1 = np.random.choice(tamano_poblacion, 3, replace=False)
            t2 = np.random.choice(tamano_poblacion, 3, replace=False)
            idx_p1, idx_p2 = min(t1), min(t2) # Menor índice = mejor individuo
            padre1, padre2 = poblacion[idx_p1], poblacion[idx_p2]
            
            # CRUCE HEURÍSTICO: El padre con menor error aporta más genes (ej. 60-80%)
            peso_padre1 = perdidas[idx_p2] / (perdidas[idx_p1] + perdidas[idx_p2] + 1e-6)
            # Acotamos para que haya mezcla real y no sea solo clonar
            peso_padre1 = np.clip(peso_padre1, 0.4, 0.9) 
            hijo = peso_padre1 * padre1 + (1 - peso_padre1) * padre2
            
            # MUTACIÓN ADAPTATIVA BASADA EN RANKING
            # i es el índice actual. i/tamano_poblacion va de ~0.0 a 1.0.
            # Los peores individuos (i alto) sufren mutaciones hasta 3 veces más fuertes.
            multiplicador_ranking = 1.0 + (2.0 * (i / tamano_poblacion))
            fuerza_actual = fuerza_mutacion_base * multiplicador_ranking
            
            for gen_idx in range(dim):
                if np.random.rand() < tasa_mutacion:
                    rango = lim_max[gen_idx] - lim_min[gen_idx]
                    ruido = np.random.normal(0, rango * fuerza_actual)
                    hijo[gen_idx] += ruido
                    
            nueva_poblacion[i] = np.clip(hijo, lim_min, lim_max)
            
        # Evaluamos solo a los hijos (los élites ya están evaluados)
        for i in range(elitismo, tamano_poblacion):
            fis_temp = fis_desde_particula(nueva_poblacion[i], n_e, n_v)
            perdida_hijo = 0
            for rep in range(repes):
                l, _ = simular_tubo(fis_temp, init=init[rep], noisy=noisy, noise=noise_generacion, 
                                    pasos_tiempo=pasos_tiempo, tiempo_final=t_max)
                perdida_hijo += l / repes
            perdidas[i] = perdida_hijo
            
        poblacion = nueva_poblacion
        # Decaimiento global suave de la mutación base
        fuerza_mutacion_base *= 0.985

    mejor_fis = fis_desde_particula(poblacion[0], n_e, n_v)

    if return_poblacion:
        return mejor_fis, historial_convergencia, poblacion
    return mejor_fis, historial_convergencia