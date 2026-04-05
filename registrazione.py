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

# Silenzia log di sistema superflui
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR           = os.path.dirname(CURRENT_SCRIPT_DIR)

INPUT_DIR          = os.path.join(BASE_DIR, "training_patches_ir")
DRONE_PHOTOS_DIR   = os.path.join(BASE_DIR, "foto_drone")
MOSAIC_PATH        = os.path.join(BASE_DIR, "ortomosaicoir.tif")
OUTPUT_DIR         = os.path.join(BASE_DIR, "risultati_finali")
REGISTRATION_DIR   = os.path.join(OUTPUT_DIR, "registrazione_allineamento")
PAIRS_DIR          = os.path.join(OUTPUT_DIR, "pair")

# ==============================================================================
# FUNZIONI GPS E METADATI
# ==============================================================================
def get_gps_from_exif(path):
    """Estrae Lat/Lon reali dai metadati EXIF della foto drone."""
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
# FUNZIONE DI ALLINEAMENTO (COMPUTER VISION)
# ==============================================================================
def allinea_e_disegna(patch, drone_img):
    """Trova la patch e disegna un rettangolo DRITTO sulla foto del drone."""
    gray_p = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray_d = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)

    # Feature Matching con ORB
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(gray_p, None)
    kp2, des2 = orb.detectAndCompute(gray_d, None)

    if des1 is None or des2 is None:
        return patch, drone_img

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 15:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            h, w = gray_p.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            x, y, bw, bh = cv2.boundingRect(np.int32(dst))
            
            drone_annotata = drone_img.copy()
            cv2.rectangle(drone_annotata, (x, y), (x + bw, y + bh), (0, 255, 0), 12)
            
            y1, y2 = max(0, y), min(drone_img.shape[0], y+bh)
            x1, x2 = max(0, x), min(drone_img.shape[1], x+bw)
            patch_aligned = drone_img[y1:y2, x1:x2]
            
            if patch_aligned.size > 0:
                return patch_aligned, drone_annotata
            
    return patch, drone_img

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    os.makedirs(REGISTRATION_DIR, exist_ok=True)
    os.makedirs(PAIRS_DIR, exist_ok=True)

    print(f"[*] Lettura CRS Mosaico: {os.path.basename(MOSAIC_PATH)}...", flush=True)
    with rasterio.open(MOSAIC_PATH) as src:
        mosaic_transform = src.transform
        mosaic_crs = src.crs

    drone_photos = glob.glob(os.path.join(DRONE_PHOTOS_DIR, "*.[jJ][pP][gG]"))
    drone_db = []
    print("[*] Indicizzazione GPS foto drone...", flush=True)
    for p in drone_photos:
        coords = get_gps_from_exif(p)
        if coords: drone_db.append({'path': p, 'lat': coords[0], 'lon': coords[1]})
    
    if not drone_db:
        print("ERRORE: Nessuna foto con GPS trovata in foto_drone!")
        return

    patch_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
    print(f"[*] Allineamento di {len(patch_files)} patch con foto drone...", flush=True)

    HEADER_H = 70
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    FONT_COLOR = (0, 255, 0)
    FONT_THICKNESS = 2

    for i, p_path in enumerate(patch_files):
        filename = os.path.basename(p_path)
        img_patch = cv2.imread(p_path)
        if img_patch is None: continue

        m = re.search(r"tile_col_(\d+)_row_(\d+)", filename)
        if not m: continue
        
        px_x, px_y = int(m.group(1)) + 320, int(m.group(2)) + 320
        east, north = mosaic_transform * (px_x, px_y)
        
        lons, lats = transform_coords(mosaic_crs, 'EPSG:4326', [east], [north])
        p_lat, p_lon = lats[0], lons[0]

        best_photo_data = min(drone_db, key=lambda x: (x['lat']-p_lat)**2 + (x['lon']-p_lon)**2)
        drone_filename = os.path.basename(best_photo_data['path'])
        img_drone = cv2.imread(best_photo_data['path'])

        patch_res, drone_res = allinea_e_disegna(img_patch, img_drone)

        H_CONTENT = 800
        p_h, p_w = patch_res.shape[:2]
        d_h, d_w = drone_res.shape[:2]
        
        p_resized = cv2.resize(patch_res, (int(p_w * (H_CONTENT / p_h)), H_CONTENT))
        d_resized = cv2.resize(drone_res, (int(d_w * (H_CONTENT / d_h)), H_CONTENT))

        combined_content = np.hstack((p_resized, d_resized))
        c_h, c_w = combined_content.shape[:2]

        header_canvas = np.zeros((HEADER_H, c_w, 3), dtype=np.uint8)
        header_canvas[:] = (0, 0, 0)

        text_sx = f"Patch: {filename}"
        text_dx = f"Drone: {drone_filename}"
        y_text = int(HEADER_H * 0.7)
        
        cv2.putText(header_canvas, text_sx, (10, y_text), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        (tw_dx, _), _ = cv2.getTextSize(text_dx, FONT, FONT_SCALE, FONT_THICKNESS)
        x_dx = c_w - tw_dx - 10
        cv2.putText(header_canvas, text_dx, (x_dx, y_text), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        final_image = np.vstack((header_canvas, combined_content))

        # ID Univoco per questo ciclo
        pair_prefix = f"pair{i+1}"

        # Salvataggio della foto affiancata con il prefisso "pair" nel nome
        out_path = os.path.join(REGISTRATION_DIR, f"{pair_prefix}_{filename}")
        cv2.imwrite(out_path, final_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Salvataggio delle due foto separate nella cartella pair
        cv2.imwrite(os.path.join(PAIRS_DIR, f"{pair_prefix}_patch.jpg"), patch_res, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Usiamo img_drone (originale) invece di drone_res (che ha il rettangolo)
        cv2.imwrite(os.path.join(PAIRS_DIR, f"{pair_prefix}_drone.jpg"), img_drone, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"[{i+1}/{len(patch_files)}] ALLINEATA: {filename} -> {drone_filename}", flush=True)

    print(f"\n[FINE] Risultati registrazione in: {REGISTRATION_DIR}")
    print(f"[FINE] Coppie separate salvate in: {PAIRS_DIR}")

if __name__ == "__main__":
    main()
