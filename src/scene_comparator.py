import json
import os
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

# --- Rutas Absolutas ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados')

DETECTIONS_FILE = os.path.join(RESULTS_DIR, 'detected_objects.json')
COMPARISON_FILE = os.path.join(RESULTS_DIR, 'comparison.json')

# Parámetros ajustados
IOU_THRESHOLD = 0.1  # Umbral mínimo para considerar emparejamiento
MOVEMENT_THRESHOLD = 20  # Píxeles mínimos para considerar "desplazado"
MAX_DISTANCE_THRESHOLD = 150  # Distancia máxima para considerar el mismo objeto

def calculate_iou(boxA, boxB):
   
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if xB <= xA or yB <= yA:
        return 0.0

    inter_area = (xB - xA) * (yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

def get_centroid(box):
    """Calcula el centroide de una caja delimitadora."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def calculate_distance(centroid1, centroid2):
    """Calcula la distancia euclidiana entre dos centroides."""
    return math.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)

def calculate_box_area(box):
    """Calcula el área de una caja delimitadora."""
    return (box[2] - box[0]) * (box[3] - box[1])

def calculate_similarity_score(obj1, obj2):
    """
    Calcula un score de similitud entre dos objetos basado en:
    - IOU
    - Distancia entre centroides
    - Similitud de área
    - Misma etiqueta
    """
    if obj1['label'] != obj2['label']:
        return 0.0  # Objetos de diferente tipo no pueden emparejarse
    
    iou = calculate_iou(obj1['box'], obj2['box'])
    
    centroid1 = get_centroid(obj1['box'])
    centroid2 = get_centroid(obj2['box'])
    distance = calculate_distance(centroid1, centroid2)
    
    # Normalizar distancia (menor distancia = mayor score)
    max_distance = 300  # Distancia máxima esperada en la imagen
    distance_score = max(0, 1 - (distance / max_distance))
    
    # Similitud de área
    area1 = calculate_box_area(obj1['box'])
    area2 = calculate_box_area(obj2['box'])
    area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
    
    # Score combinado (pesos ajustables)
    combined_score = (
        0.4 * iou +           # IOU es importante pero no decisivo
        0.4 * distance_score + # Distancia es muy importante
        0.2 * area_ratio      # Área similar es un buen indicador
    )
    
    # Penalizar si la distancia es demasiado grande
    if distance > MAX_DISTANCE_THRESHOLD:
        combined_score *= 0.1
    
    return combined_score

def get_scene_data(detections, scene_index):
    """Extrae la información de una escena específica de la lista de detecciones."""
    if scene_index < len(detections):
        return detections[scene_index]['detections']
    return []

def compare_scenes_hungarian(initial_objects, final_objects):
    """
    Compara objetos usando el algoritmo húngaro para emparejamiento óptimo.
    """
    if not initial_objects or not final_objects:
        return {
            'nuevos': final_objects if final_objects else [],
            'desaparecidos': initial_objects if initial_objects else [],
            'desplazados': [],
            'estaticos': []
        }
    
    # Crear matriz de costos (1 - similarity_score para convertir a problema de minimización)
    n_initial = len(initial_objects)
    n_final = len(final_objects)
    
    # Crear matriz cuadrada agregando objetos "dummy" si es necesario
    max_size = max(n_initial, n_final)
    cost_matrix = np.ones((max_size, max_size))  # Inicializar con costo alto
    
    # Llenar la matriz de costos reales
    for i in range(n_initial):
        for j in range(n_final):
            similarity = calculate_similarity_score(initial_objects[i], final_objects[j])
            cost_matrix[i, j] = 1 - similarity  # Convertir a costo
    
    # Aplicar algoritmo húngaro
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Procesar resultados
    nuevos = []
    desaparecidos = []
    desplazados = []
    estaticos = []
    
    matched_final = set()
    
    for i, j in zip(row_indices, col_indices):
        if i < n_initial and j < n_final:
            similarity = 1 - cost_matrix[i, j]
            
            # Solo considerar emparejamientos con suficiente similitud
            if similarity > 0.3:  # Umbral de similitud mínima
                obj_initial = initial_objects[i]
                obj_final = final_objects[j]
                
                matched_final.add(j)
                
                # Calcular movimiento
                centroid_initial = get_centroid(obj_initial['box'])
                centroid_final = get_centroid(obj_final['box'])
                movement_distance = calculate_distance(centroid_initial, centroid_final)
                
                if movement_distance > MOVEMENT_THRESHOLD:
                    desplazados.append({
                        'label': obj_initial['label'],
                        'initial_box': obj_initial['box'],
                        'final_box': obj_final['box'],
                        'initial_centroid': centroid_initial,
                        'final_centroid': centroid_final,
                        'movement_distance': round(movement_distance, 2),
                        'similarity_score': round(similarity, 3)
                    })
                else:
                    estaticos.append({
                        'label': obj_initial['label'],
                        'initial_box': obj_initial['box'],
                        'final_box': obj_final['box'],
                        'movement_distance': round(movement_distance, 2),
                        'similarity_score': round(similarity, 3)
                    })
            else:
                # Similitud muy baja, considerar como desaparecido
                desaparecidos.append(initial_objects[i])
        elif i < n_initial:
            # Objeto inicial sin emparejamiento
            desaparecidos.append(initial_objects[i])
    
    # Objetos finales no emparejados son nuevos
    for j in range(n_final):
        if j not in matched_final:
            nuevos.append(final_objects[j])
    
    return {
        'nuevos': nuevos,
        'desaparecidos': desaparecidos,
        'desplazados': desplazados,
        'estaticos': estaticos
    }



def main():
    """
    Función principal para cargar detecciones y comparar escenas.
    """
    if not os.path.exists(DETECTIONS_FILE):
        print(f"Error: El archivo de detecciones '{DETECTIONS_FILE}' no fue encontrado.")
        return

    with open(DETECTIONS_FILE, 'r') as f:
        all_detections = json.load(f)

    if not all_detections or len(all_detections) < 2:
        print("No hay suficientes detecciones para comparar (se necesitan al menos 2).")
        return

    # El primer frame es el inicial, el último es el final
    initial_detections = get_scene_data(all_detections, 0)
    final_detections = get_scene_data(all_detections, 81)  # Usar frame 81 en lugar del último
    
    print(f"Escena inicial (Frame 0): {len(initial_detections)} objetos detectados")
    print(f"Escena final (Frame 81): {len(final_detections)} objetos detectados")
    print("Comparando objetos usando algoritmo húngaro para emparejamiento óptimo...")
    
    comparison_results = compare_scenes_hungarian(initial_detections, final_detections)

    print("\nResultados de la comparación:")
    print(f"  - Objetos nuevos: {len(comparison_results['nuevos'])}")
    for obj in comparison_results['nuevos']:
        print(f"    - {obj['label']} (confianza: {obj['confidence']:.2f})")
    
    print(f"  - Objetos desaparecidos: {len(comparison_results['desaparecidos'])}")
    for obj in comparison_results['desaparecidos']:
        print(f"    - {obj['label']} (confianza: {obj['confidence']:.2f})")
    
    print(f"  - Objetos desplazados: {len(comparison_results['desplazados'])}")
    for obj in comparison_results['desplazados']:
        print(f"    - {obj['label']} (movimiento: {obj['movement_distance']} píxeles, similitud: {obj['similarity_score']})")
    
    print(f"  - Objetos estáticos: {len(comparison_results['estaticos'])}")
    for obj in comparison_results['estaticos']:
        print(f"    - {obj['label']} (movimiento mínimo: {obj['movement_distance']} píxeles, similitud: {obj['similarity_score']})")

    print(f"\nGuardando resultados de la comparación en '{COMPARISON_FILE}'...")
    with open(COMPARISON_FILE, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    print("Resultados guardados con éxito.")
    print("\nTarea 3 completada: Comparación entre escenas.")
    print("El siguiente paso es la visualización de resultados con Matplotlib. Ejecuta el script 'visualizer.py'.")

if __name__ == '__main__':
    main() 