import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# --- Rutas Absolutas ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

COMPARISON_FILE = os.path.join(BASE_DIR, 'resultados', 'comparison.json')
IMG1_PATH = os.path.join(BASE_DIR, 'imagenes', 'nevera_1.png')
IMG2_PATH = os.path.join(BASE_DIR, 'imagenes', 'nevera_2.png')
# Crea la carpeta de resultados si no existe
os.makedirs(os.path.join(BASE_DIR, 'resultados'), exist_ok=True)
OUTPUT_PLOT = os.path.join(BASE_DIR, 'resultados', 'comparison_plot.png')

def get_centroid(box):
    """Calcula el centroide de una caja delimitadora."""
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

def visualize_comparison(comparison_data, img1, img2):
    """
    Crea una visualización con Matplotlib que muestra los cambios entre dos imágenes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Análisis Comparativo de Escenas en la Nevera', fontsize=20)

    # --- Escena Inicial (Imagen 1) ---
    ax1.imshow(img1)
    ax1.set_title('Escena Inicial', fontsize=16)
    ax1.axis('off')

    # Objetos desaparecidos (en rojo)
    for obj in comparison_data['desaparecidos']:
        box = obj['box']
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                linewidth=3, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        centroid = get_centroid(box)
        ax1.text(centroid[0], centroid[1] - 15, obj['label'], color='white', fontsize=10, 
                ha='center', bbox=dict(facecolor='red', alpha=0.8))

    # Objetos desplazados (posición inicial en amarillo)
    for obj in comparison_data['desplazados']:
        box = obj['initial_box']
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                linewidth=3, edgecolor='yellow', facecolor='none')
        ax1.add_patch(rect)
        centroid = get_centroid(box)
        ax1.text(centroid[0], centroid[1] - 15, obj['label'], color='black', fontsize=10, 
                ha='center', bbox=dict(facecolor='yellow', alpha=0.8))

    # Objetos estáticos (en azul claro)
    if 'estaticos' in comparison_data:
        for obj in comparison_data['estaticos']:
            box = obj['initial_box']
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                    linewidth=2, edgecolor='lightblue', facecolor='none')
            ax1.add_patch(rect)
            centroid = get_centroid(box)
            ax1.text(centroid[0], centroid[1] - 15, obj['label'], color='black', fontsize=9, 
                    ha='center', bbox=dict(facecolor='lightblue', alpha=0.7))
        
    # --- Escena Final (Imagen 2) ---
    ax2.imshow(img2)
    ax2.set_title('Escena Final', fontsize=16)
    ax2.axis('off')
    
    # Objetos nuevos (en verde)
    for obj in comparison_data['nuevos']:
        box = obj['box']
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                linewidth=3, edgecolor='green', facecolor='none')
        ax2.add_patch(rect)
        centroid = get_centroid(box)
        ax2.text(centroid[0], centroid[1] - 15, obj['label'], color='white', fontsize=10, 
                ha='center', bbox=dict(facecolor='green', alpha=0.8))

    # Objetos desplazados (posición final en amarillo)
    for obj in comparison_data['desplazados']:
        box = obj['final_box']
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                linewidth=3, edgecolor='yellow', facecolor='none')
        ax2.add_patch(rect)
        
        # Flecha que muestra el movimiento
        initial_centroid = obj['initial_centroid']
        final_centroid = obj['final_centroid']
        
        # Convertir coordenadas para la flecha entre imágenes
        coord1 = ax1.transData.transform(initial_centroid)
        coord2 = ax2.transData.transform(final_centroid)
        fig_coord1 = fig.transFigure.inverted().transform(coord1)
        fig_coord2 = fig.transFigure.inverted().transform(coord2)
        
        arrow_fig = patches.FancyArrowPatch(
            fig_coord1, fig_coord2,
            connectionstyle="arc3,rad=.1",
            color='orange',
            linewidth=3,
            mutation_scale=25,
            transform=fig.transFigure
        )
        fig.add_artist(arrow_fig)
        
        # Etiqueta en la posición final
        centroid = get_centroid(box)
        ax2.text(centroid[0], centroid[1] - 15, f"{obj['label']}", color='black', fontsize=10, 
                ha='center', bbox=dict(facecolor='yellow', alpha=0.8))

    # Objetos estáticos (en azul claro en ambas imágenes)
    if 'estaticos' in comparison_data:
        for obj in comparison_data['estaticos']:
            box = obj['final_box']
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                    linewidth=2, edgecolor='lightblue', facecolor='none')
            ax2.add_patch(rect)
            centroid = get_centroid(box)
            ax2.text(centroid[0], centroid[1] - 15, obj['label'], color='black', fontsize=9, 
                    ha='center', bbox=dict(facecolor='lightblue', alpha=0.7))

    # --- Leyenda mejorada ---
    handles = [
        patches.Patch(color='green', label='Nuevo'),
        patches.Patch(color='red', label='Desaparecido'),
        patches.Patch(color='yellow', label='Desplazado'),
        patches.Patch(color='lightblue', label='Estático')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=14, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"Gráfico de comparación guardado en '{OUTPUT_PLOT}'")

    # Mostrar estadísticas
    print(f"\nEstadísticas de la comparación:")
    print(f"  - Objetos nuevos: {len(comparison_data['nuevos'])}")
    print(f"  - Objetos desaparecidos: {len(comparison_data['desaparecidos'])}")
    print(f"  - Objetos desplazados: {len(comparison_data['desplazados'])}")
    if 'estaticos' in comparison_data:
        print(f"  - Objetos estáticos: {len(comparison_data['estaticos'])}")

def main():
    if not os.path.exists(COMPARISON_FILE):
        print(f"Error: El archivo de comparación '{COMPARISON_FILE}' no existe.")
        print("Por favor, ejecuta 'scene_comparator.py' primero.")
        return

    with open(COMPARISON_FILE, 'r') as f:
        comparison_data = json.load(f)

    try:
        img1 = Image.open(IMG1_PATH)
        img2 = Image.open(IMG2_PATH)
    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar el archivo de imagen: {e.filename}")
        return

    # Asegurarse de que las imágenes tengan el mismo tamaño para la visualización
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    visualize_comparison(comparison_data, img1, img2)
    
    print("\nTarea 4 completada: Visualización de resultados.")
    print("El último paso es la generación de mapas de calor. Ejecuta el script 'heatmap_generator.py'.")

if __name__ == '__main__':
    main() 