#!/usr/bin/env python3
import os
import sys
import re
import glob
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR        = os.path.join(BASE_DIR, "training_patches_ir")
DRONE_PHOTOS_DIR = os.path.join(BASE_DIR, "foto_drone")
MOSAIC_PATH      = os.path.join(BASE_DIR, "ortomosaicoir.tif")
OUTPUT_DIR       = os.path.join(BASE_DIR, "risultati_finali")
COMPARISON_DIR   = os.path.join(OUTPUT_DIR, "confronto_allineato")

# ==============================================================================
# FUNZIONI GPS E ALLINEAMENTO
# ==============================================================================

def get_gps_from_exif(path):
    try:
        with Image.open(path) as img:
            exif = img._getexif()
            if not exif: return None
            info = {TAGS.get(tag, tag): value for tag, value in exif.items() if TAGS.get(tag, tag) == "GPSInfo"}
            if "GPSInfo" not in info: return None
            gps = {GPSTAGS.get(t, t): info['GPSInfo'][t] for t in info['GPSInfo']}
            def to_dec(c, ref):
                d = float(c[0]) + float(c[1])/60.0 + float(c[2])/3600.0
                return -d if ref in ['S', 'W'] else d
            return to_dec(gps['GPSLatitude'], gps['GPSLatitudeRef']), to_dec(gps['GPSLongitude'], gps['GPSLongitudeRef'])
    except: return None

def allinea_e_disegna(patch, drone_img):
    """Allinea la patch alla foto del drone e trova il rettangolo di appartenenza."""
    gray_p = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray_d = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)

    # Inizializza rilevatore di punti (ORB è veloce e robusto)
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(gray_p, None)
    kp2, des2 = orb.detectAndCompute(gray_d, None)

    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Trova la trasformazione (Omografia)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            h, w = gray_p.shape
            # 1. Ruota la patch per farla combaciare con la prospettiva del drone
            patch_ruotata = cv2.warpPerspective(patch, M, (drone_img.shape[1], drone_img.shape[0]))
            
            # 2. Trova i bordi della patch nella foto del drone
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            # Disegna il perimetro sulla foto originale
            drone_annotata = drone_img.copy()
            cv2.polylines(drone_annotata, [np.int32(dst)], True, (0, 255, 0), 10, cv2.LINE_AA)
            
            # Ritaglio della patch allineata per il side-by-side
            x, y, w_r, h_r = cv2.boundingRect(np.int32(dst))
            patch_aligned = drone_img[max(0, y):y+h_r, max(0, x):x+w_r]
            
            return patch_aligned, drone_annotata

    return patch, drone_img

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    with rasterio.open(MOSAIC_PATH) as src:
        mosaic_transform = src.transform
        mosaic_crs = src.crs

    drone_db = []
    for p in glob.glob(os.path.join(DRONE_PHOTOS_DIR, "*.[jJ][pP][gG]")):
        coords = get_gps_from_exif(p)
        if coords: drone_db.append({'path': p, 'lat': coords[0], 'lon': coords[1]})

    patch_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))

    for i, p_path in enumerate(patch_files):
        filename = os.path.basename(p_path)
        m = re.search(r"tile_col_(\d+)_row_(\d+)", filename)
        if not m: continue
        
        px_x, px_y = int(m.group(1)) + 320, int(m.group(2)) + 320
        east, north = mosaic_transform * (px_x, px_y)
        lons, lats = transform_coords(mosaic_crs, 'EPSG:4326', [east], [north])
        
        # Trova foto drone tramite GPS
        best_photo_data = min(drone_db, key=lambda x: (x['lat']-lats[0])**2 + (x['lon']-lons[0])**2)
        
        img_patch = cv2.imread(p_path)
        img_drone = cv2.imread(best_photo_data['path'])

        # ALLINEAMENTO E DISEGNO RETTANGOLO
        patch_res, drone_res = allinea_e_disegna(img_patch, img_drone)

        # Resize per affiancamento
        h_target = 800
        p_h, p_w = patch_res.shape[:2]
        d_h, d_w = drone_res.shape[:2]
        
        p_final = cv2.resize(patch_res, (int(p_w * (h_target / p_h)), h_target))
        d_final = cv2.resize(drone_res, (int(d_w * (h_target / d_h)), h_target))

        combined = np.hstack((p_final, d_final))
        cv2.imwrite(os.path.join(COMPARISON_DIR, f"aligned_{filename}"), combined)
        print(f"[{i+1}/{len(patch_files)}] Allineata: {filename} -> {os.path.basename(best_photo_data['path'])}", flush=True)

if __name__ == "__main__":
    main()
