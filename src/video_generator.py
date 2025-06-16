import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(6, 64, 3, 2, 1), nn.PReLU(64),
            nn.Conv2d(64, 128, 3, 2, 1), nn.PReLU(128),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.PReLU(128),
            nn.Conv2d(128, 128, 3, 1, 1), nn.PReLU(128),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.PReLU(64),
            nn.ConvTranspose2d(64, 5, 4, 2, 1),
        )

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x) + x
        x = self.block2(x)
        return x

    def warp(self, img, flow):
        B, C, H, W = img.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if img.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(img, vgrid, align_corners=True)
        mask = torch.ones(img.size(), device=img.device)
        mask = F.grid_sample(mask, vgrid, align_corners=True)
        mask[mask < 0.999] = 0
        mask[mask > 0] = 1
        return output * mask


def get_model():

    model = IFNet()

    return model

def interpolate(model, img1, img2, num_frames=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    img1_torch = torch.from_numpy(img1.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.
    img2_torch = torch.from_numpy(img2.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.

    h, w = img1_torch.shape[2], img1_torch.shape[3]
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    
    img1_padded = F.pad(img1_torch, padding)
    img2_padded = F.pad(img2_torch, padding)
    
    frames = []
    for i in range(num_frames):
        ratio = (i + 1) / (num_frames + 1)
        with torch.no_grad():
            flow_data = model(torch.cat((img1_padded, img2_padded), 1))
        
        flow = flow_data[:, :4]
        mask = flow_data[:, 4:5]
        
        img0_warped = model.warp(img1_padded, flow[:, :2] * ratio)
        img1_warped = model.warp(img2_padded, flow[:, 2:4] * (1 - ratio))
        
        mask = torch.sigmoid(mask)
        merged_img = (1 - ratio) * img0_warped + ratio * img1_warped
        merged_img /= ((1 - ratio) * mask + ratio * (1 - mask))
        
        frame_np = (merged_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frame_np = frame_np[:h, :w]
        frames.append(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
        
    return frames

def create_video(frames, output_path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, size)
    for frame in frames:
        video.write(frame)
    video.release()

def main():
    # --- Rutas Absolutas ---
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SRC_DIR)

    model = get_model()
    print("Modelo cargado.")

    img1_path = os.path.join(BASE_DIR, 'imagenes', 'nevera_1.png')
    img2_path = os.path.join(BASE_DIR, 'imagenes', 'nevera_2.png')
    
    try:
        img1_pil = Image.open(img1_path).convert('RGB')
        img2_pil = Image.open(img2_path).convert('RGB')
    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar el archivo de imagen: {e.filename}")
        return

    if img1_pil.size != img2_pil.size:
        print("Las imágenes tienen diferentes tamaños. Redimensionando la segunda imagen.")
        img2_pil = img2_pil.resize(img1_pil.size)

    img1_np = np.array(img1_pil)
    img2_np = np.array(img2_pil)
    
    n_frames = 80

    print("Generando transición de apertura...")
    open_frames = interpolate(model, img1_np, img2_np, num_frames=n_frames)
    
    print("Generando transición de cierre...")
    close_frames = interpolate(model, img2_np, img1_np, num_frames=n_frames)

    all_frames = [cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)] + open_frames + \
                 [cv2.cvtColor(img2_np, cv2.COLOR_RGB2BGR)] + close_frames

    output_dir = os.path.join(BASE_DIR, 'resultados')
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'nevera_interpolada.mp4')
    fps = 30
    video_size = (img1_np.shape[1], img1_np.shape[0])

    print(f"Creando vídeo en '{output_video_path}'...")
    create_video(all_frames, output_video_path, fps, video_size)
    print("¡Vídeo creado con éxito!")
    print("El siguiente paso es la detección de objetos en el vídeo. Ejecuta el script 'object_detector.py'.")

if __name__ == '__main__':
    main() 