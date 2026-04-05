#!/usr/bin/env python3
import os
import sys
import re
import glob
import argparse
import warnings
import cv2
import numpy as np
import rasterio
from rasterio.warp import transform as transform_coords
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR           = os.path.dirname(CURRENT_SCRIPT_DIR)

INPUT_DIR        = os.path.join(BASE_DIR, "training_patches_ir")
DRONE_PHOTOS_DIR = os.path.join(BASE_DIR, "foto_drone")
MOSAIC_PATH      = os.path.join(BASE_DIR, "ortomosaicoir.tif")
OUTPUT_DIR       = os.path.join(BASE_DIR, "risultati_finali")
COMPARISON_DIR   = os.path.join(OUTPUT_DIR, "confronto_gps_corretto")

# ==============================================================================
# FUNZIONI GPS
# ==============================================================================

def get_gps_from_exif(path):
    """Estrae Lat/Lon decimali reali dalla foto del drone."""
    try:
        with Image.open(path) as img:
            exif = img._getexif()
            if not exif: return None
            info = {}
            for tag, value in exif.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    for t in value:
                        sub_tag = GPSTAGS.get(t, t)
                        info[sub_tag] = value[t]
            
            def to_dec(c, ref):
                d = float(c[0]) + float(c[1])/60.0 + float(c[2])/3600.0
                return -d if ref in ['S', 'W'] else d

            return to_dec(info['GPSLatitude'], info['GPSLatitudeRef']), \
                   to_dec(info['GPSLongitude'], info['GPSLongitudeRef'])
    except: return None

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    # 1. Carica Mosaico e ottieni CRS (Sistema di Riferimento)
    print(f"[*] Apertura mosaico e lettura sistema di coordinate...")
    with rasterio.open(MOSAIC_PATH) as src:
        mosaic_transform = src.transform
        mosaic_crs = src.crs
        print(f"    CRS Rilevato: {mosaic_crs}")

    # 2. Indicizza Foto Drone
    print(f"[*] Lettura GPS da foto drone...")
    drone_db = []
    for p in glob.glob(os.path.join(DRONE_PHOTOS_DIR, "*.[jJ][pP][gG]")):
        coords = get_gps_from_exif(p)
        if coords:
            drone_db.append({'path': p, 'lat': coords[0], 'lon': coords[1]})
    
    if not drone_db:
        print("ERRORE: Nessun dato GPS trovato nelle foto del drone.")
        return

    # 3. Elaborazione Patch
    patch_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
    print(f"[*] Elaborazione di {len(patch_files)} patch...")

    for i, p_path in enumerate(patch_files):
        filename = os.path.basename(p_path)
        m = re.search(r"tile_col_(\d+)_row_(\d+)", filename)
        if not m: continue
        
        # Centro della patch in pixel
        px_x, px_y = int(m.group(1)) + 320, int(m.group(2)) + 320
        
        # A. Coordinate nel sistema del mosaico (es. metri UTM)
        east, north = mosaic_transform * (px_x, px_y)
        
        # B. CONVERSIONE FONDAMENTALE: da Metri a Gradi (WGS84)
        # Trasformiamo il punto dal CRS del mosaico a EPSG:4326 (GPS standard)
        lons, lats = transform_coords(mosaic_crs, 'EPSG:4326', [east], [north])
        patch_lat, patch_lon = lats[0], lons[0]

        # C. Cerca la foto più vicina usando i gradi corretti
        # Usiamo la distanza euclidea quadrata (veloce)
        best_photo = min(drone_db, key=lambda x: (x['lat']-patch_lat)**2 + (x['lon']-patch_lon)**2)
        
        # D. Genera Immagine di Confronto
        img_patch = cv2.imread(p_path)
        img_drone = cv2.imread(best_photo['path'])
        
        if img_drone is not None:
            # Resize drone per match altezza patch
            h_p, w_p = img_patch.shape[:2]
            drone_res = cv2.resize(img_drone, (int(img_drone.shape[1] * (h_p / img_drone.shape[0])), h_p))
            
            # Label
            cv2.putText(img_patch, "PATCH IR (MOSAICO)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(drone_res, f"ORIGINALE: {os.path.basename(best_photo['path'])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            combined = np.hstack((img_patch, drone_res))
            cv2.imwrite(os.path.join(COMPARISON_DIR, f"match_{filename}"), combined)
            
            print(f"[{i+1}/{len(patch_files)}] OK: {filename} -> {os.path.basename(best_photo['path'])}", flush=True)

    print(f"\n[FINITO] Risultati in: {COMPARISON_DIR}")

if __name__ == "__main__":
    main()
