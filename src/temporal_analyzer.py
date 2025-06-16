import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from collections import defaultdict
import math

# --- Rutas Absolutas ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DETECTIONS_FILE = os.path.join(BASE_DIR, 'resultados', 'detected_objects.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados')

class TemporalAnalyzer:
    def __init__(self, detections_data):
        self.detections = detections_data
        self.object_trajectories = defaultdict(list)
        self.stability_scores = {}
        self.movement_patterns = {}
        
    def calculate_centroid(self, box):
        """Calcula el centroide de una caja delimitadora."""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def calculate_distance(self, point1, point2):
        """Calcula la distancia euclidiana entre dos puntos."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def track_object_trajectories(self):
        """
        Rastrea las trayectorias de objetos a lo largo del tiempo usando un algoritmo simple de seguimiento.
        """
        print("🔍 Analizando trayectorias de objetos...")
        
        # Agrupar objetos por tipo y posición aproximada
        for frame_idx, frame in enumerate(self.detections):
            frame_objects = defaultdict(list)
            
            # Agrupar por tipo de objeto
            for obj in frame['detections']:
                frame_objects[obj['label']].append({
                    'centroid': self.calculate_centroid(obj['box']),
                    'box': obj['box'],
                    'confidence': obj['confidence'],
                    'frame': frame_idx
                })
            
            # Para cada tipo de objeto, intentar hacer seguimiento
            for obj_type, objects in frame_objects.items():
                for obj in objects:
                    # Buscar la trayectoria más cercana o crear una nueva
                    best_trajectory = None
                    min_distance = float('inf')
                    
                    for traj_id in self.object_trajectories:
                        if (traj_id.startswith(obj_type) and 
                            len(self.object_trajectories[traj_id]) > 0):
                            
                            last_point = self.object_trajectories[traj_id][-1]
                            # Solo considerar si el frame anterior está cerca temporalmente
                            if frame_idx - last_point['frame'] <= 5:  # Máximo 5 frames de diferencia
                                distance = self.calculate_distance(
                                    obj['centroid'], 
                                    last_point['centroid']
                                )
                                if distance < min_distance and distance < 100:  # Umbral de distancia
                                    min_distance = distance
                                    best_trajectory = traj_id
                    
                    # Si encontramos una trayectoria cercana, agregar a ella
                    if best_trajectory:
                        self.object_trajectories[best_trajectory].append(obj)
                    else:
                        # Crear nueva trayectoria
                        new_traj_id = f"{obj_type}_{len([k for k in self.object_trajectories.keys() if k.startswith(obj_type)])}"
                        self.object_trajectories[new_traj_id] = [obj]
    
    def analyze_stability(self):
        """
        Analiza la estabilidad de cada objeto (qué tan quieto se mantiene).
        """
        print("📊 Calculando scores de estabilidad...")
        
        for traj_id, trajectory in self.object_trajectories.items():
            if len(trajectory) < 3:  # Necesitamos al menos 3 puntos
                continue
                
            # Calcular varianza de posición
            centroids = [point['centroid'] for point in trajectory]
            x_coords = [c[0] for c in centroids]
            y_coords = [c[1] for c in centroids]
            
            x_variance = np.var(x_coords)
            y_variance = np.var(y_coords)
            total_variance = x_variance + y_variance
            
            # Calcular distancia total recorrida
            total_distance = 0
            for i in range(1, len(centroids)):
                total_distance += self.calculate_distance(centroids[i-1], centroids[i])
            
            # Score de estabilidad (menor varianza y distancia = más estable)
            stability_score = 1 / (1 + total_variance/1000 + total_distance/100)
            
            self.stability_scores[traj_id] = {
                'score': stability_score,
                'variance': total_variance,
                'total_distance': total_distance,
                'frames_tracked': len(trajectory),
                'avg_confidence': np.mean([p['confidence'] for p in trajectory])
            }
    
    def detect_movement_patterns(self):
        """
        Detecta patrones de movimiento interesantes.
        """
        print("Detectando patrones de movimiento...")
        
        for traj_id, trajectory in self.object_trajectories.items():
            if len(trajectory) < 5:  # Necesitamos suficientes puntos
                continue
                
            centroids = [point['centroid'] for point in trajectory]
            frames = [point['frame'] for point in trajectory]
            
            # Detectar diferentes tipos de patrones
            pattern_info = {
                'type': 'unknown',
                'description': '',
                'intensity': 0
            }
            
            # Calcular velocidades
            velocities = []
            for i in range(1, len(centroids)):
                frame_diff = frames[i] - frames[i-1]
                if frame_diff > 0:
                    distance = self.calculate_distance(centroids[i-1], centroids[i])
                    velocity = distance / frame_diff
                    velocities.append(velocity)
            
            if len(velocities) > 0:
                avg_velocity = np.mean(velocities)
                velocity_variance = np.var(velocities)
                
                # Clasificar patrones
                if avg_velocity < 1:
                    pattern_info['type'] = 'estatico'
                    pattern_info['description'] = 'Objeto prácticamente inmóvil'
                    pattern_info['intensity'] = 1 - avg_velocity
                elif avg_velocity < 5 and velocity_variance < 2:
                    pattern_info['type'] = 'movimiento_constante'
                    pattern_info['description'] = 'Movimiento lento y constante'
                    pattern_info['intensity'] = avg_velocity
                elif velocity_variance > 10:
                    pattern_info['type'] = 'movimiento_erratico'
                    pattern_info['description'] = 'Movimiento irregular o errático'
                    pattern_info['intensity'] = velocity_variance
                else:
                    pattern_info['type'] = 'movimiento_normal'
                    pattern_info['description'] = 'Movimiento moderado'
                    pattern_info['intensity'] = avg_velocity
                
                # Detectar direcciones predominantes
                directions = []
                for i in range(1, len(centroids)):
                    dx = centroids[i][0] - centroids[i-1][0]
                    dy = centroids[i][1] - centroids[i-1][1]
                    if abs(dx) > 5 or abs(dy) > 5:  # Solo movimientos significativos
                        angle = math.atan2(dy, dx)
                        directions.append(angle)
                
                if len(directions) > 0:
                    # Convertir ángulos a direcciones cardinales
                    avg_angle = np.mean(directions)
                    if -math.pi/4 <= avg_angle <= math.pi/4:
                        pattern_info['direction'] = 'derecha'
                    elif math.pi/4 < avg_angle <= 3*math.pi/4:
                        pattern_info['direction'] = 'abajo'
                    elif avg_angle > 3*math.pi/4 or avg_angle <= -3*math.pi/4:
                        pattern_info['direction'] = 'izquierda'
                    else:
                        pattern_info['direction'] = 'arriba'
                else:
                    pattern_info['direction'] = 'sin_direccion'
            
            self.movement_patterns[traj_id] = pattern_info
    
    def create_trajectory_visualization(self):
        """
        Crea una visualización avanzada de las trayectorias.
        """
        print("Creando visualización de trayectorias...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Colores para diferentes tipos de objetos
        colors = {
            'banana': 'gold',
            'apple': 'red', 
            'carrot': 'orange',
            'bottle': 'blue',
            'bowl': 'green',
            'broccoli': 'darkgreen',
            'cup': 'purple'
        }
        
        # Gráfico 1: Trayectorias completas
        ax1.set_title('Trayectorias de Objetos a lo Largo del Tiempo', fontsize=14, weight='bold')
        
        for traj_id, trajectory in self.object_trajectories.items():
            if len(trajectory) < 3:
                continue
                
            obj_type = traj_id.split('_')[0]
            color = colors.get(obj_type, 'gray')
            
            # Extraer coordenadas
            x_coords = [point['centroid'][0] for point in trajectory]
            y_coords = [point['centroid'][1] for point in trajectory]
            frames = [point['frame'] for point in trajectory]
            
            # Dibujar trayectoria
            ax1.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7, label=f'{traj_id}')
            
            # Marcar inicio y fin
            ax1.scatter(x_coords[0], y_coords[0], color=color, s=100, marker='o', edgecolor='black', linewidth=2)
            ax1.scatter(x_coords[-1], y_coords[-1], color=color, s=100, marker='s', edgecolor='black', linewidth=2)
            
            # Agregar flechas de dirección
            for i in range(0, len(x_coords)-1, max(1, len(x_coords)//5)):
                if i+1 < len(x_coords):
                    ax1.annotate('', xy=(x_coords[i+1], y_coords[i+1]), xytext=(x_coords[i], y_coords[i]),
                               arrowprops=dict(arrowstyle='->', color=color, alpha=0.6))
        
        ax1.set_xlim(0, 640)
        ax1.set_ylim(480, 0)
        ax1.set_xlabel('Coordenada X (píxeles)')
        ax1.set_ylabel('Coordenada Y (píxeles)')
        ax1.grid(True, alpha=0.3)
        
        # Leyenda personalizada
        legend_elements = [
            mpatches.Patch(color='black', label='○ Inicio de trayectoria'),
            mpatches.Patch(color='black', label='□ Fin de trayectoria')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Gráfico 2: Análisis de estabilidad
        ax2.set_title('Análisis de Estabilidad de Objetos', fontsize=14, weight='bold')
        
        if self.stability_scores:
            objects = list(self.stability_scores.keys())
            scores = [self.stability_scores[obj]['score'] for obj in objects]
            distances = [self.stability_scores[obj]['total_distance'] for obj in objects]
            
            # Crear scatter plot
            scatter = ax2.scatter(distances, scores, s=100, alpha=0.7, 
                                c=[colors.get(obj.split('_')[0], 'gray') for obj in objects])
            
            # Agregar etiquetas
            for i, obj in enumerate(objects):
                ax2.annotate(obj, (distances[i], scores[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            ax2.set_xlabel('Distancia Total Recorrida (píxeles)')
            ax2.set_ylabel('Score de Estabilidad (0-1)')
            ax2.grid(True, alpha=0.3)
            
            # Líneas de referencia
            ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Muy Estable')
            ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderadamente Estable')
            ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Inestable')
            ax2.legend()
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'analisis_temporal_trayectorias.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualización de trayectorias guardada en '{output_path}'")
    
    def create_pattern_report(self):
        """
        Crea un reporte detallado de patrones encontrados.
        """
        print("Generando reporte de patrones...")
        
        report = {
            'resumen_general': {
                'total_trayectorias': len(self.object_trajectories),
                'objetos_rastreados': len([t for t in self.object_trajectories.values() if len(t) >= 3]),
                'frames_analizados': len(self.detections)
            },
            'patrones_movimiento': self.movement_patterns,
            'scores_estabilidad': self.stability_scores,
            'insights': []
        }
        
        # Generar insights automáticos
        if self.stability_scores:
            # Objeto más estable
            most_stable = max(self.stability_scores.items(), key=lambda x: x[1]['score'])
            report['insights'].append(f"🏆 Objeto más estable: {most_stable[0]} (score: {most_stable[1]['score']:.3f})")
            
            # Objeto más móvil
            most_mobile = max(self.stability_scores.items(), key=lambda x: x[1]['total_distance'])
            report['insights'].append(f"🏃 Objeto más móvil: {most_mobile[0]} (distancia: {most_mobile[1]['total_distance']:.1f} píxeles)")
            
            # Patrones interesantes
            erratic_objects = [obj for obj, pattern in self.movement_patterns.items() 
                             if pattern.get('type') == 'movimiento_erratico']
            if erratic_objects:
                report['insights'].append(f"⚡ Objetos con movimiento errático: {', '.join(erratic_objects)}")
            
            static_objects = [obj for obj, pattern in self.movement_patterns.items() 
                            if pattern.get('type') == 'estatico']
            if static_objects:
                report['insights'].append(f"🔒 Objetos prácticamente inmóviles: {', '.join(static_objects)}")
        
        # Guardar reporte
        report_path = os.path.join(OUTPUT_DIR, 'reporte_analisis_temporal.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Reporte guardado en '{report_path}'")
        return report
    
    def print_summary(self, report):
        """
        Imprime un resumen legible del análisis.
        """
        print("\n" + "="*60)
        print(" ANÁLISIS TEMPORAL AVANZADO - RESUMEN")
        print("="*60)
        
        print(f"\n ESTADÍSTICAS GENERALES:")
        print(f"   • Trayectorias detectadas: {report['resumen_general']['total_trayectorias']}")
        print(f"   • Objetos rastreados exitosamente: {report['resumen_general']['objetos_rastreados']}")
        print(f"   • Frames analizados: {report['resumen_general']['frames_analizados']}")
        
        print(f"\n INSIGHTS PRINCIPALES:")
        for insight in report['insights']:
            print(f"   {insight}")
        
        print(f"\n PATRONES DE MOVIMIENTO DETECTADOS:")
        pattern_counts = defaultdict(int)
        for pattern in self.movement_patterns.values():
            pattern_counts[pattern['type']] += 1
        
        for pattern_type, count in pattern_counts.items():
            print(f"   • {pattern_type.replace('_', ' ').title()}: {count} objetos")
        
        print(f"\n TOP 3 OBJETOS MÁS ESTABLES:")
        if self.stability_scores:
            sorted_stability = sorted(self.stability_scores.items(), 
                                    key=lambda x: x[1]['score'], reverse=True)[:3]
            for i, (obj, data) in enumerate(sorted_stability, 1):
                print(f"   {i}. {obj}: {data['score']:.3f} (distancia: {data['total_distance']:.1f}px)")

def main():
    """
    Función principal del analizador temporal.
    """
    print("🚀 Iniciando Análisis Temporal Avanzado...")
    
    if not os.path.exists(DETECTIONS_FILE):
        print(f"❌ Error: No se encontró el archivo de detecciones '{DETECTIONS_FILE}'.")
        print("   Ejecuta primero 'object_detector.py' para generar las detecciones.")
        return
    
    # Cargar datos
    with open(DETECTIONS_FILE, 'r') as f:
        detections_data = json.load(f)
    
    if len(detections_data) < 10:
        print("Advertencia: Pocos frames para análisis temporal significativo.")
    
    # Crear analizador
    analyzer = TemporalAnalyzer(detections_data)
    
    # Ejecutar análisis
    analyzer.track_object_trajectories()
    analyzer.analyze_stability()
    analyzer.detect_movement_patterns()
    
    # Generar visualizaciones y reportes
    analyzer.create_trajectory_visualization()
    report = analyzer.create_pattern_report()
    analyzer.print_summary(report)
    
    print(f"\nAnálisis temporal completado!")
    print(f"Archivos generados en '{OUTPUT_DIR}':")
    print(f"   • analisis_temporal_trayectorias.png")
    print(f"   • reporte_analisis_temporal.json")

if __name__ == "__main__":
    main() 