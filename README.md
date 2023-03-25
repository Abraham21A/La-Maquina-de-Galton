# La-Maquina-de-Galton
### La-Maquina-de-Galton en Python 

[![Galton-box.jpg](https://i.postimg.cc/2ScdwDTm/Galton-box.jpg)](https://postimg.cc/WdZJNxQf)

### Ejemplo del programa con Python 

[![Python](https://img.shields.io/badge/Python-4479A1?style=for-the-badge&logo=python&logoColor=white&labelColor=101010)]()

Máquina de Galton
Este código es una simulación de la máquina de Galton

La simulación se realiza mediante la función maquina_galton(), que toma dos argumentos: 
cantidad_canicas, que es la cantidad de canicas que se lanzarán en la simulación, y 
cantidad_niveles, que es la cantidad de niveles que tendrá la máquina de Galton.

La función simula el recorrido de cada canica por la máquina y devuelve una lista con la 
cantidad de canicas en cada contenedor. Esta lista se utiliza en la función 
graficar_histograma() para graficar los resultados de la simulación.

### Este código utiliza dos librerías de Python:

- random: para generar números aleatorios
- matplotlib: para graficar los resultados

### Uso:
Para ejecutar la simulación, simplemente llame a la función maquina_galton() con los 
parámetros deseados. Por ejemplo, para simular la máquina de Galton con 3000 canicas y 12 niveles, 
puede hacer lo siguiente:

contenedores_resultantes = maquina_galton(3000, 12)

La función maquina_galton() imprimirá el total de canicas y devolverá una lista con la 
cantidad de canicas en cada contenedor. Puede utilizar esta lista para graficar los resultados 
llamando a la función graficar_histograma(), como se muestra en el siguiente ejemplo:

graficar_histograma(contenedores_resultantes)


[![image-378.jpg](https://i.postimg.cc/fLghz6TK/image-378.jpg)](https://postimg.cc/xJHZgpmJ)
