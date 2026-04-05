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
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR           = os.path.dirname(CURRENT_SCRIPT_DIR)

WEIGHTS_PATH      = os.path.join(BASE_DIR, "weights.pt")
INPUT_DIR         = os.path.join(BASE_DIR, "training_patches_ir")
DRONE_PHOTOS_DIR  = os.path.join(BASE_DIR, "foto_drone")
MOSAIC_PATH       = os.path.join(BASE_DIR, "ortomosaicoir.tif")
OUTPUT_DIR        = os.path.join(BASE_DIR, "risultati_finali")
COMPARISON_DIR    = os.path.join(OUTPUT_DIR, "confronto_finale_proporzionato")

MIN_CONFIDENCE = 0.45

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
    """Calcola il quadrato proporzionato sulla foto del drone intera."""
    gray_p = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray_d = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(4000)
    kp1, des1 = orb.detectAndCompute(gray_p, None)
    kp2, des2 = orb.detectAndCompute(gray_d, None)

    if des1 is None or des2 is None:
        return drone_img

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    drone_annotata = drone_img.copy()

    if len(matches) > 15:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            h, w = gray_p.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            # Calcolo centro e scala media per mantenere il quadrato proporzionato
            center = np.mean(dst, axis=0)[0]
            # Calcoliamo la dimensione media dei lati trasformati
            side1 = np.linalg.norm(dst[0] - dst[3])
            side2 = np.linalg.norm(dst[0] - dst[1])
            avg_side = (side1 + side2) / 2
            
            # Disegno del quadrato DRITTO e PROPORZIONATO
            x1 = int(center[0] - avg_side/2)
            y1 = int(center[1] - avg_side/2)
            x2 = int(center[0] + avg_side/2)
            y2 = int(center[1] + avg_side/2)
            
            cv2.rectangle(drone_annotata, (x1, y1), (x2, y2), (0, 255, 0), 15)
            
    return drone_annotata

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    print("[*] Caricamento modello IA...", flush=True)
    from rfdetr import RFDETRSegLarge
    model = RFDETRSegLarge(pretrain_weights=WEIGHTS_PATH, num_classes=2)
    model.optimize_for_inference()

    with rasterio.open(MOSAIC_PATH) as src:
        mosaic_transform = src.transform
        mosaic_crs = src.crs

    drone_photos = glob.glob(os.path.join(DRONE_PHOTOS_DIR, "*.[jJ][pP][gG]"))
    drone_db = []
    print("[*] Indicizzazione GPS drone...", flush=True)
    for p in drone_photos:
        coords = get_gps_from_exif(p)
        if coords: drone_db.append({'path': p, 'lat': coords[0], 'lon': coords[1]})

    patch_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))

    # Parametri grafici
    HEADER_H = 80
    LINE_COLOR = (0, 0, 255) # Rosso
    LINE_WIDTH = 4
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for i, p_path in enumerate(patch_files):
        filename = os.path.basename(p_path)
        img_patch = cv2.imread(p_path)
        if img_patch is None: continue

        results = model.predict(Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)), threshold=MIN_CONFIDENCE)
        if results is None or len(results.xyxy) == 0: continue

        m = re.search(r"tile_col_(\d+)_row_(\d+)", filename)
        if not m: continue
        
        px_x, px_y = int(m.group(1)) + 320, int(m.group(2)) + 320
        east, north = mosaic_transform * (px_x, px_y)
        lons, lats = transform_coords(mosaic_crs, 'EPSG:4326', [east], [north])
        
        best_photo_data = min(drone_db, key=lambda x: (x['lat']-lats[0])**2 + (x['lon']-lons[0])**2)
        drone_filename = os.path.basename(best_photo_data['path'])
        img_drone = cv2.imread(best_photo_data['path'])

        # Elaborazione: Intera foto drone con box
        drone_res = allinea_e_disegna(img_patch, img_drone)

        # Composizione Side-by-Side (H=900) senza tagliare nulla
        H_CONTENT = 900
        # Patch proporzionale
        p_h, p_w = img_patch.shape[:2]
        p_resized = cv2.resize(img_patch, (int(p_w * (H_CONTENT / p_h)), H_CONTENT))
        
        # Drone proporzionale (intera foto)
        d_h, d_w = drone_res.shape[:2]
        d_resized = cv2.resize(drone_res, (int(d_w * (H_CONTENT / d_h)), H_CONTENT))

        # Unione orizzontale
        combined_content = np.hstack((p_resized, d_resized))
        c_h, c_w = combined_content.shape[:2]
        div_x = p_resized.shape[1] # Punto di divisione tra le due foto

        # Creazione Header
        header = np.zeros((HEADER_H, c_w, 3), dtype=np.uint8)
        cv2.putText(header, f"Patch: {filename}", (20, 50), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        text_drone = f"Drone: {drone_filename}"
        tw, _ = cv2.getTextSize(text_drone, FONT, 0.8, 2)[0]
        cv2.putText(header, text_drone, (c_w - tw - 20, 50), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Unione Finale
        final_img = np.vstack((header, combined_content))

        # DISEGNO LINEA ROSSA (Header + Foto)
        cv2.line(final_img, (div_x, 0), (div_x, HEADER_H + H_CONTENT), LINE_COLOR, LINE_WIDTH)

        out_path = os.path.join(COMPARISON_DIR, f"final_{filename}")
        cv2.imwrite(out_path, final_img)
        print(f"[{i+1}/{len(patch_files)}] OK: {filename}", flush=True)

if __name__ == "__main__":
    main()
