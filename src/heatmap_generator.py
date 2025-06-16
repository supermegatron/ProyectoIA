import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# --- Rutas Absolutas ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

DETECTIONS_FILE = os.path.join(BASE_DIR, 'resultados', 'detected_objects.json')
COMPARISON_FILE = os.path.join(BASE_DIR, 'resultados', 'comparison.json')
IMG1_PATH = os.path.join(BASE_DIR, 'imagenes', 'nevera_1.png')
IMG2_PATH = os.path.join(BASE_DIR, 'imagenes', 'nevera_2.png')
OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados')

def create_density_heatmap(detections, title, output_filename):
   
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Crear una matriz de densidad
    density_map = np.zeros((480, 640))  # Altura x Ancho de la imagen
    
    # Colores para diferentes tipos de objetos
    colors = {
        'banana': 'yellow',
        'apple': 'red', 
        'carrot': 'orange',
        'bottle': 'blue',
        'bowl': 'green',
        'broccoli': 'darkgreen',
        'cup': 'purple',
        'refrigerator': 'lightgray'
    }
    
    # Dibujar las cajas de detección
    for det in detections:
        box = det['box']
        label = det['label']
        confidence = det['confidence']
        
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Agregar densidad a la matriz
        if y1 >= 0 and y2 < 480 and x1 >= 0 and x2 < 640:
            density_map[y1:y2, x1:x2] += confidence
        
        # Dibujar rectángulo
        color = colors.get(label, 'gray')
        rect = Rectangle((x1, y1), width, height, 
                        linewidth=2, edgecolor=color, 
                        facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        
        # Agregar etiqueta
        ax.text(x1, y1-5, f'{label}\n{confidence:.2f}', 
               fontsize=8, color=color, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Mostrar el mapa de densidad como fondo
    im = ax.imshow(density_map, cmap='hot', alpha=0.6, extent=[0, 640, 480, 0])
    
    # Configurar el plot
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_title(f'{title}\nTotal de objetos: {len(detections)}', fontsize=14, weight='bold')
    ax.set_xlabel('Coordenada X (píxeles)', fontsize=12)
    ax.set_ylabel('Coordenada Y (píxeles)', fontsize=12)
    
    # Crear leyenda
    legend_elements = []
    for label, color in colors.items():
        count = sum(1 for det in detections if det['label'] == label)
        if count > 0:
            legend_elements.append(mpatches.Patch(color=color, label=f'{label} ({count})'))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Agregar barra de color para densidad
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Densidad de Confianza', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mapa de densidad guardado en '{output_filename}'")

def create_movement_heatmap(comparison_data, output_filename):
    """
    Crea un mapa de calor que muestra los movimientos de objetos.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Colores para diferentes tipos de cambios
    colors = {
        'estaticos': 'lightblue',
        'desplazados': 'red',
        'nuevos': 'green',
        'desaparecidos': 'gray'
    }
    
    # Dibujar objetos estáticos
    for obj in comparison_data.get('estaticos', []):
        box = obj['initial_box']
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        rect = Rectangle((x1, y1), width, height, 
                        linewidth=2, edgecolor=colors['estaticos'], 
                        facecolor=colors['estaticos'], alpha=0.5)
        ax.add_patch(rect)
        
        ax.text(x1, y1-5, f'{obj["label"]}\nEstático', 
               fontsize=8, color='blue', weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Dibujar objetos desplazados con flechas
    for obj in comparison_data.get('desplazados', []):
        initial_box = obj['initial_box']
        final_box = obj['final_box']
        
        # Posición inicial (rojo claro)
        x1, y1, x2, y2 = initial_box
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height, 
                        linewidth=2, edgecolor='red', 
                        facecolor='red', alpha=0.3)
        ax.add_patch(rect)
        
        # Posición final (rojo oscuro)
        x1f, y1f, x2f, y2f = final_box
        widthf = x2f - x1f
        heightf = y2f - y1f
        rect = Rectangle((x1f, y1f), widthf, heightf, 
                        linewidth=3, edgecolor='darkred', 
                        facecolor='red', alpha=0.7)
        ax.add_patch(rect)
        
        # Flecha de movimiento
        center_initial = ((x1 + x2) / 2, (y1 + y2) / 2)
        center_final = ((x1f + x2f) / 2, (y1f + y2f) / 2)
        
        ax.annotate('', xy=center_final, xytext=center_initial,
                   arrowprops=dict(arrowstyle='->', lw=3, color='red'))
        
        # Etiqueta
        ax.text(x1, y1-5, f'{obj["label"]}\n{obj["movement_distance"]:.1f}px', 
               fontsize=8, color='red', weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Dibujar objetos nuevos
    for obj in comparison_data.get('nuevos', []):
        box = obj['box']
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        rect = Rectangle((x1, y1), width, height, 
                        linewidth=3, edgecolor=colors['nuevos'], 
                        facecolor=colors['nuevos'], alpha=0.6)
        ax.add_patch(rect)
        
        ax.text(x1, y1-5, f'{obj["label"]}\nNuevo', 
               fontsize=8, color='green', weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Dibujar objetos desaparecidos
    for obj in comparison_data.get('desaparecidos', []):
        box = obj['box']
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        rect = Rectangle((x1, y1), width, height, 
                        linewidth=2, edgecolor=colors['desaparecidos'], 
                        facecolor=colors['desaparecidos'], alpha=0.4,
                        linestyle='--')
        ax.add_patch(rect)
        
        ax.text(x1, y1-5, f'{obj["label"]}\nDesaparecido', 
               fontsize=8, color='gray', weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Configurar el plot
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_title('Mapa de Movimientos de Objetos\n(Frame 0 → Frame 81)', fontsize=14, weight='bold')
    ax.set_xlabel('Coordenada X (píxeles)', fontsize=12)
    ax.set_ylabel('Coordenada Y (píxeles)', fontsize=12)
    
    # Crear leyenda
    legend_elements = [
        mpatches.Patch(color=colors['estaticos'], label=f'Estáticos ({len(comparison_data.get("estaticos", []))})'),
        mpatches.Patch(color=colors['desplazados'], label=f'Desplazados ({len(comparison_data.get("desplazados", []))})'),
        mpatches.Patch(color=colors['nuevos'], label=f'Nuevos ({len(comparison_data.get("nuevos", []))})'),
        mpatches.Patch(color=colors['desaparecidos'], label=f'Desaparecidos ({len(comparison_data.get("desaparecidos", []))})')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mapa de movimientos guardado en '{output_filename}'")

def main():
    
    if not os.path.exists(DETECTIONS_FILE):
        print(f"Error: No se encontró el archivo de detecciones '{DETECTIONS_FILE}'.")
        return

    with open(DETECTIONS_FILE, 'r') as f:
        all_detections = json.load(f)

    if not all_detections or len(all_detections) < 2:
        print("No hay suficientes detecciones para generar los mapas de calor.")
        return

    # Usar Frame 0 y Frame 81 (igual que en el comparador)
    initial_detections = all_detections[0]['detections']
    final_detections = all_detections[81]['detections']

    # Crear mapas de densidad
    create_density_heatmap(initial_detections, 
                          "Mapa de Densidad - Frame 0 (Inicial)", 
                          os.path.join(OUTPUT_DIR, 'heatmap_densidad_inicial.png'))
    
    create_density_heatmap(final_detections, 
                          "Mapa de Densidad - Frame 81 (Final)", 
                          os.path.join(OUTPUT_DIR, 'heatmap_densidad_final.png'))

    # Crear mapa de movimientos si existe el archivo de comparación
    if os.path.exists(COMPARISON_FILE):
        with open(COMPARISON_FILE, 'r') as f:
            comparison_data = json.load(f)
        
        create_movement_heatmap(comparison_data, 
                               os.path.join(OUTPUT_DIR, 'heatmap_movimientos.png'))
    else:
        print("No se encontró el archivo de comparación. Ejecuta primero scene_comparator.py")

    print(f"\nMapas de calor generados en '{OUTPUT_DIR}'.")
    print("\nTarea 5 completada: Generación de mapas de calor mejorados. Ejecuta el script 'temporal_analyzer.py'.")

if __name__ == "__main__":
    main() 