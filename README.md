# Automatic-counting-of-epidermal-cells-with-Deep-Learning
¡Hola! Soy Javier Ramírez G., estudiante de ciencias computacionales en UABC, y este es mi proyecto de investigación para desarrollar un sistema de conteo automático de células epidérmicas en hojas de chile mediante el uso de ResNet50 y Detectron2.

## Descripción del Proyecto

### Características Clave
- **ResNet50 como Espina Dorsal:** Utilizo la arquitectura preentrenada ResNet50 para extraer características clave de las imágenes de micrografía de hojas de chile.
- **Detectron2 para Detección de Objetos:** Implemento Detectron2, simplificando el proceso de detección de células y permitiendo un conteo eficiente.

### Objetivos
- Desarrollar un modelo robusto para el conteo automático de células epidérmicas.
- Optimizar la arquitectura de ResNet50 para adaptarse a las características específicas de las imágenes de hojas de chile.
- Facilitar la detección eficiente de células mediante Detectron2.

### Contribuciones Esperadas
- Avance en la automatización de la cuantificación de células en estudios biológicos.
- Código abierto y documentación para fomentar la colaboración y replicabilidad.
- Mejoras continuas basadas en la retroalimentación y avances en investigación.

## Instalación

### Crear entorno virtual

#### Instalar venv
```
sudo apt install python3-venv
```

Crear un entorno virtual dentro del directorio raíz del proyecto.

GNU/Linux:
```
python3 -m venv venv
source ./venv/bin/activate
```
#### Desactivar el entorno virtual

Para desactivar el entorno virtual, ejecuta el siguiente comando:

GNU/Linux:
```
$ deactivate
```


### Dependencias

GNU/Linux:
```
python3 -m pip install -r requirements.txt
```




## Estructura del Repositorio

```plaintext
|-- 📁 data
|   |-- [Datos de Entrenamiento y Pruebas]
|-- 📁 notebooks
|   |-- [Cuadernos Jupyter para Análisis y Experimentación]
|-- 📁 src
|   |-- 📁 models
|   |   |-- [Implementación de Modelos, incluyendo ResNet50]
|   |-- 📁 utils
|   |   |-- [Herramientas y utilidades diversas]
|-- 📄 requirements.txt
|-- 📄 LICENSE
|-- 📄 README.md
