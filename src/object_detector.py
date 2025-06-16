import torch
import cv2
import json
import os
from ultralytics import YOLO

# Obtener la ruta del directorio actual del script (src)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Ir un nivel arriba para llegar a la raíz del proyecto
BASE_DIR = os.path.dirname(SRC_DIR)

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8s.pt')
VIDEO_PATH = os.path.join(BASE_DIR, 'resultados', 'nevera_interpolada.mp4')
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'detected_objects.json')

# Crea la carpeta de resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)

# Cargar el modelo YOLOv8
model = YOLO(MODEL_PATH)

def detect_objects_in_video():
    """
    Analiza un vídeo frame por frame, detecta objetos con YOLO y guarda los resultados.
    """
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error al abrir el vídeo: {VIDEO_PATH}")
        return

    all_detections = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar detección
        results = model(frame, stream=True)

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                
                detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "label": model.names[int(cls)],
                    "confidence": float(conf)
                })

        all_detections.append({
            "frame_id": frame_count,
            "detections": detections
        })

        frame_count += 1
        print(f"Procesando frame {frame_count}...")

    cap.release()
    return all_detections

def main():
    """
    Función principal para ejecutar la detección y guardar los resultados.
    """
    print("Iniciando la detección de objetos en el vídeo con YOLOv8...")
    detections = detect_objects_in_video()

    if detections:
        print(f"Detección completada. Guardando resultados en '{RESULTS_FILE}'...")
        with open(RESULTS_FILE, 'w') as f:
            json.dump(detections, f, indent=4)
        print("Resultados guardados con éxito.")
        print("\nTarea 2 completada: Detección de objetos.")
        print("El siguiente paso es la comparación de escenas. Ejecuta el script 'scene_comparator.py'.")
    else:
        print("No se detectaron objetos o hubo un error en el proceso.")

if __name__ == '__main__':
    main() 