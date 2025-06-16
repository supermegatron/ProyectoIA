# Proyecto IA
INSTRUCCIONES DE EJECUCIÓN 

Para empezar se deben instalar todas las dependencias necesarias para ello he prearado un 
txt llamado requeriments.txt, puedes crear un entorno virtual en caso de ser necesario.

Ejecutar: pip install -r requirements.txt

Una vez instaladas las dependencias debes ejecutar los scripts en el siguiente orden y
mediante los siguients comandos:

1. "python src/video_generator.py"
2. "python src/object_detector.py"
3. "python src/scene_comparator.py"
4. "python src/visualizer.py"
5. "python src/heatmap_generator.py"
6. "python src/temporal_analyzer.py"

De todas formas lo he preparado para que cuando ejecutes uno te avise de cual es el siguiente.

DOCUMENTACIÓN COMPLETA DEL PROYECTO
================================================================================

INFORMACIÓN GENERAL
===================

Nombre del Proyecto: ProyectoAI

Fecha de Creación: 06/2024

Autor: Alejandro Plasencia Querol

DESCRIPCIÓN DEL PROYECTO
========================

Este proyecto implementa un sistema completo de análisis de objetos utilizando
técnicas avanzadas de inteligencia artificial y visión por computadora. El sistema
es capaz de detectar, rastrear, comparar y analizar objetos en secuencias de video,
proporcionando informacion detallada sobre el comportamiento y movimiento de los
objetos a lo largo del tiempo.

El proyecto está diseñado para detectar los movimientos producidos dentro de una nevera
para asi poder detectar los alimentos que ya no estan, los que han cambiado de lugar y 
los nuevos añadidos. Esto lo muestra de manera visual y ofrece difrerentes datos para que 
el usuario tenga toda la información posible. Como una foto comparativa del antes y el despues, 
un mapa de calor y un analisis temporar(idea nueva implementada)

CARACTERÍSTICAS PRINCIPALES
===========================

1. DETECCIÓN DE OBJETOS CON YOLOv8
   - Utiliza el modelo YOLOv8s para detección en tiempo real
   - Detecta múltiples categorías de objetos (frutas, verduras, botellas, etc.)
   - Proporciona coordenadas precisas y niveles de confianza

2. COMPARACIÓN INTELIGENTE DE ESCENAS
   - Detecta objetos nuevos, desaparecidos, desplazados y estáticos
   - Calcula distancias de movimiento con precisión

3. ANÁLISIS TEMPORAL AVANZADO
   - Rastrea trayectorias de objetos a lo largo del tiempo
   - Calcula scores de estabilidad para cada objeto
   - Detecta patrones de movimiento (estático, errático, constante)

4. VISUALIZACIONES AVANZADAS
   - Gráficos de comparación con códigos de colores
   - Mapas de calor de densidad de objetos
   - Mapas de movimiento con flechas direccionales
   - Análisis de trayectorias con visualización temporal


COMPONENTES DETALLADOS
======================


1. video_generator.py

PROPÓSITO:
Genera un video suave de transición entre dos imágenes de estado de la nevera
utilizando técnicas de interpolación de frames basadas en deep learning.


PROCESO:
1. Carga y preprocesa las imágenes de entrada (nevera_1.png y nevera_2.png)
2. Redimensiona automáticamente si las imágenes tienen tamaños diferentes
3. Aplica padding para compatibilidad con la arquitectura
4. Genera los frames intemedios
5. Genera transición de apertura y cierre 


ENTRADA:
- nevera_1.png: Imagen inicial del estado de la nevera
- nevera_2.png: Imagen final del estado de la nevera
- Formato soportado: PNG, JPG (convertido automáticamente a RGB)

SALIDA:
- nevera_interpolada.mp4: Video completo con transiciones suaves


2. object_detector.py

   
PROPÓSITO:
Detecta objetos en frames de video utilizando el modelo YOLOv8.

ALGORITMO:
- Carga el modelo YOLOv8s preentrenado
- Procesa cada frame del directorio frames/
- Aplica detección de objetos con umbral de confianza 0.5
- Extrae coordenadas 
- Guarda resultados en formato JSON 

ENTRADA:
- Frames de video en formato JPG 
- Modelo YOLOv8s.pt

SALIDA:
- detected_objects.json: Lista de detecciones por frame
  Estructura:
  {
    "frame_index": 0,
    "frame_file": "frame_0000.jpg",
    "detections": [
      {
        "label": "apple",
        "confidence": 0.85,
        "box": [x1, y1, x2, y2]
      }
    ]
  }


3. scene_comparator.py


PROPÓSITO:
Compara dos escenas para detectar los cambios de cada una.

ENTRADA:
- detected_objects.json: Lista de objetos detectados en cada frame

SALIDA:
- comparison.json: Resultados de la comparación
  Categorías: nuevos, desaparecidos, desplazados, estaticos


4. visualicer.py


PROPÓSITO:
Genera visualizaciones gráficas de los resultados de comparación.

SALIDA:
- comparison_plot.png: Gráfico de comparación en alta resolución


5. heatmap_generator.py


PROPÓSITO:
Crea mapas de calor para visualizar densidad y movimientos de objetos.

TIPOS DE MAPAS:

- MAPAS DE DENSIDAD
- MAPA DE MOVIMIENTOS


SALIDA:
- heatmap_densidad_inicial.png
- heatmap_densidad_final.png
- heatmap_movimientos.png


6. temporal_analyzer.py


PROPÓSITO:
Analiza el comportamiento temporal de objetos a lo largo de toda la secuencia.

ALGORITMOS IMPLEMENTADOS:

SALIDA:
- analisis_temporal_trayectorias.png: Visualización de trayectorias
- reporte_analisis_temporal.json: Datos detallados del análisis


DEPENDENCIAS
=========================

LIBRERÍAS PRINCIPALES:
- ultralytics
- opencv-python
- numpy
- matplotlib
- scipy
- json

CASOS DE USO
============

1. MONITOREO DE REFRIGERADORES:
   - Detectar productos añadidos o removidos
   - Rastrear reorganización de alimentos
   - Análisis de patrones de consumo

2. CONTROL DE INVENTARIO:
   - Seguimiento de productos en almacenes
   - Detección de movimientos no autorizados
   - Optimización de disposición espacial

3. SEGURIDAD Y VIGILANCIA:
   - Monitoreo de objetos en áreas restringidas
   - Detección de cambios sospechosos

4. INVESTIGACIÓN CIENTÍFICA:
   - Estudio de comportamiento de objetos
   - Análisis de patrones temporales


SOLUCIÓN DE PROBLEMAS
====================

DEBUGGING:
- Activar logs detallados modificando nivel de logging
- Verificar archivos intermedios en directorio resultados/
- Usar modo verbose en scripts individuales

================================================================================
                            FIN DE LA DOCUMENTACIÓN
================================================================================
