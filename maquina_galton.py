#Importamos las librerías necesarias
import random # Librería para generar números aleatorios
import matplotlib.pyplot as plt # Librería para graficar

# Definimos la función llamada máquina_Galton
# simula la máquina de Galton
def maquina_galton(cantidad_canicas, cantidad_niveles):
    # Creamos una lista para almacenar la cantidad de canicas que caen en cada nivel
    contenedores = [0] * (cantidad_niveles + 1) 
    # Realizamos la simulación para cada canica
    for i in range(cantidad_canicas):
        posicion_actual = cantidad_niveles // 2  # Comenzamos en el nivel central
        # Simulamos el recorrido de la canica por los diferentes niveles
        for j in range(cantidad_niveles):
            # Si el número aleatorio es menor a 0.5, cae a la izquierda
            if random.random() < 0.5:  
                posicion_actual = max(posicion_actual - 1, 0)  
            # Si el número aleatorio es mayor o igual a 0.5, cae a la derecha
            else: 
                posicion_actual = min(posicion_actual + 1, cantidad_niveles) 
        # Añadimos la canica al contenedor correspondiente 
        contenedores[posicion_actual] += 1
    # Calculamos el total de canicas  
    total_canicas = sum(contenedores)
    print(f"Total de canicas: {total_canicas}")
    # Devolvemos la lista con la cantidad de canicas en cada contenedor
    return contenedores

# Definimos la función para graficar los resultados de la simulación
def graficar_histograma(contenedores):
    # Creamos un histograma con la cantidad de canicas en cada contenedor
    plt.bar(range(len(contenedores)), contenedores)
    # Añadimos etiquetas al eje x, y y al título del gráfico
    plt.xlabel('Contenedor')
    plt.ylabel('Cantidad de canicas')
    plt.title('Resultados de la simulación de la máquina de Galton')
    # Mostramos el gráfico
    plt.show()

# Realizamos la simulación con 3000 canicas y 12 niveles
contenedores_resultantes = maquina_galton(3000, 12)
# Graficamos los resultados
graficar_histograma(contenedores_resultantes)