#!/usr/bin/env python3
import os
import sys
import re
import glob
import argparse
import warnings
import cv2
import numpy as np
from PIL import Image

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR           = os.path.dirname(CURRENT_SCRIPT_DIR)

WEIGHTS_PATH  = os.path.join(BASE_DIR, "weights.pt")
INPUT_DIR     = os.path.join(BASE_DIR, "training_patches_ir")
MOSAIC_PATH   = os.path.join(BASE_DIR, "ortomosaicoir.tif")
OUTPUT_DIR    = os.path.join(BASE_DIR, "risultati_finali")

# Parametri Filtro Dimensioni (modificabili se serve)
MIN_AREA_PX   = 800   # Area minima del pannello in pixel
MIN_SIDE_PX   = 15    # Lunghezza minima del lato più corto

COLOR_MAP = {0: (0, 0, 255), 1: (0, 200, 0), 2: (0, 200, 0)}
NAME_MAP  = {0: "Difettoso", 1: "Sano", 2: "Sano"}

# ==============================================================================
# FUNZIONI TECNICHE
# ==============================================================================

def calcola_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]) , min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def carica_mosaico(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: sys.exit(1)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--min_area", type=int, default=MIN_AREA_PX)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mosaico = carica_mosaico(MOSAIC_PATH)
    from rfdetr import RFDETRSegLarge
    model = RFDETRSegLarge(pretrain_weights=WEIGHTS_PATH, num_classes=2)
    model.optimize_for_inference()

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
    print(f"[*] Analisi di {len(files)} patch con filtro dimensioni...")
    
    rilevamenti_globali = []

    for path in files:
        filename = os.path.basename(path)
        match = re.search(r"tile_col_(\d+)_row_(\d+)", filename)
        if not match: continue
        off_x, off_y = int(match.group(1)), int(match.group(2))

        img_patch = cv2.imread(path)
        if img_patch is None: continue
        
        img_pil = Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
        results = model.predict(img_pil, threshold=args.threshold)

        if results is not None and len(results.xyxy) > 0:
            for k in range(len(results.xyxy)):
                mask = results.mask[k]
                if mask is None: continue
                
                mask_u8 = (mask.astype(np.uint8)) * 255
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                
                c_local = max(contours, key=cv2.contourArea)
                
                # --- FILTRO DIMENSIONI ---
                rect = cv2.minAreaRect(c_local)
                (w_rect, h_rect) = rect[1]
                area_rect = w_rect * h_rect
                lato_corto = min(w_rect, h_rect)

                # Se il rettangolo è troppo piccolo o troppo stretto, lo scartiamo
                if area_rect < args.min_area or lato_corto < MIN_SIDE_PX:
                    continue

                # Calcolo punti rettangolo
                box_local = np.int32(cv2.boxPoints(rect))

                # Trasla in coordinate mosaico
                box_global = box_local.copy()
                box_global[:, 0] += off_x
                box_global[:, 1] += off_y
                
                gx, gy, gw, gh = cv2.boundingRect(box_global)
                rilevamenti_globali.append({
                    'class_id': int(results.class_id[k]),
                    'score': float(results.confidence[k]),
                    'poly': box_global,
                    'bbox': [gx, gy, gx + gw, gy + gh]
                })

    # Merge (NMS)
    rilevamenti_globali.sort(key=lambda x: x['score'], reverse=True)
    finali = []
    mentre_elaboro = rilevamenti_globali.copy()

    while mentre_elaboro:
        migliore = mentre_elaboro.pop(0)
        finali.append(migliore)
        mentre_elaboro = [r for r in mentre_elaboro if calcola_iou(migliore['bbox'], r['bbox']) < 0.30]

    # Disegno Finale
    print(f"[*] Disegno di {len(finali)} pannelli validi sul mosaico...")
    for f in finali:
        color = COLOR_MAP.get(f['class_id'], (0, 200, 0))
        cv2.polylines(mosaico, [f['poly']], True, color, 2, cv2.LINE_AA)
        
        centro = np.mean(f['poly'], axis=0).astype(int)
        label = f"{NAME_MAP.get(f['class_id'], 'Sano')} {f['score']:.0%}"
        cv2.putText(mosaico, label, (centro[0]-20, centro[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    mosaic_out = os.path.join(OUTPUT_DIR, "Mosaico_Filtrato.jpg")
    cv2.imwrite(mosaic_out, mosaico, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    print(f"\nDONE! Pannelli scartati per dimensioni: {len(rilevamenti_globali) - len(finali) if len(rilevamenti_globali) > len(finali) else 0}")
    print(f"Risultato salvato in: {mosaic_out}")

if __name__ == "__main__":
    main()
