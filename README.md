# Optimizacion-de-Sistemas-de-Inferencia-Difusa-con-algoritmos-bio-inspirados
El código utilizado para el trabajo de la asignatura de Modelización con incertidumbre, información difusa y soft computing. Se implementa un algoritmo de optimización por enjambre de partículas (PSO) y un algoritmo genético clásico (GA) para un problema de control muy sencillo.

En **Trabajo IDIF.ipynb* están ejecutadas (en versiones reducidas por lo largo del tiempo de ejecución) todos los algoritmos utilizados en la presentación.

**Descripción de las funciones del código fuente (FISControlOpt.py)**

- **simular_tubo**: Es el motor físico del proyecto. Simula la gravedad, el tubo neumático y el comportamiento de la bola bajo la influencia del controlador difuso en cada instante de tiempo. Al final de la simulación, calcula una "pérdida" (error) evaluando qué tan lejos se quedó la bola de la altura objetivo, penalizando también el uso excesivo o brusco de la fuerza del ventilador.

- **Clases FS y SugenoFIS (y sus métodos eval)**: Definen la estructura matemática del controlador lógico difuso. FS crea los conjuntos difusos (funciones de pertenencia, como campanas de Gauss) para categorizar el error de posición y la velocidad. Por su parte, SugenoFIS agrupa estos conjuntos, evalúa las reglas lógicas (inferencia de Sugeno de orden 0) y decide la fuerza exacta que debe aplicar el ventilador (defuzzificación).

- **fis_desde_particula y particula_desde_fis**: Son funciones traductoras bidireccionales. Convierten toda la estructura compleja del sistema difuso (centros, anchuras de campanas y reglas) en un vector numérico simple (un array unidimensional) que los algoritmos de optimización pueden procesar, y viceversa.

- **generar_poblacion**: Construye el punto de partida para los algoritmos de optimización. Crea un grupo diverso de posibles controladores iniciales combinando soluciones totalmente aleatorias, individuos "expertos" previos, interpolaciones entre ellos y pequeñas variaciones con ruido para garantizar una buena exploración inicial.

- **graficar_fis y graficar_fis_limpio**: Son las herramientas de análisis visual estático. Generan un panel de métricas que muestra la trayectoria física de la bola, la curva de aprendizaje (convergencia) del algoritmo, la adaptación de los conjuntos difusos y una representación en 3D de la superficie de control (cómo responde el sistema ante cualquier combinación de error y velocidad).

- **animar_control**: Genera la representación visual dinámica de los resultados. Crea una animación paso a paso donde se observa a la bola levitando físicamente dentro del tubo, acompañada de gráficos de barras en tiempo real que indican la velocidad del objeto y la fuerza aplicada por el motor.

- **optimizar_pso**: Implementa la Optimización por Enjambre de Partículas (PSO). Lanza una "bandada" de posibles controladores matemáticos y ajusta sus parámetros iterativamente, haciendo que cada individuo aprenda de sus propios aciertos pasados y del mejor resultado global de todo el enjambre, buscando afinar el control del sistema de levitación.

- **optimizar_ga**: Implementa el Algoritmo Genético (GA). Simula la evolución biológica sometiendo a una población de controladores a un proceso de selección por torneos, cruces heurísticos (mezclando los "genes" o parámetros de los mejores controladores) y mutaciones adaptativas para explorar de manera robusta el espacio de posibles soluciones matemáticas.
