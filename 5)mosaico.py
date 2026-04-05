#!/usr/bin/env python3
import os
import sys
import glob
import re
import csv
import warnings
import cv2
import numpy as np
import rasterio
from rasterio.warp import transform as transform_coords
from PIL import Image

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAZIONE PERCORSI
# ==============================================================================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR           = os.path.dirname(CURRENT_SCRIPT_DIR)

WEIGHTS_PATH       = os.path.join(BASE_DIR, "weights.pt")
OUTPUT_DIR         = os.path.join(BASE_DIR, "risultati_finali")
PATCH_IR_DIR       = os.path.join(BASE_DIR, "training_patches_ir")
CSV_PATH           = os.path.join(OUTPUT_DIR, "efficienza_risultati", "report_efficienza.csv")
MAPPA_OUT_PATH     = os.path.join(OUTPUT_DIR, "mappa_efficienza_rgb.tif")

# Trova i mosaici (gestisce piccole variazioni di nome)
IR_MOSAIC  = os.path.join(BASE_DIR, "ortomosaicoir.tif")
RGB_MOSAIC = os.path.join(BASE_DIR, "ortomosaicorgb.tif")
if not os.path.exists(RGB_MOSAIC): 
    RGB_MOSAIC = os.path.join(BASE_DIR, "ortomosaiccrgb.tif") # Fallback typo

# ==============================================================================
# FUNZIONI DI UTILITA'
# ==============================================================================
def carica_report_csv(csv_path):
    """Legge il CSV dell'efficienza e crea un dizionario per rapida consultazione."""
    if not os.path.exists(csv_path):
        print(f"[!] ERRORE: CSV non trovato in {csv_path}. Esegui prima 4_efficienza.py")
        sys.exit(1)
        
    eff_data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nome_patch = row["File_Patch"]
            id_pannello = int(row["ID_Pannello"])
            eta_rel = float(row["Salute_Relativa_pct"])
            
            if nome_patch not in eff_data:
                eff_data[nome_patch] = {}
            eff_data[nome_patch][id_pannello] = eta_rel
            
    return eff_data

def mappa_pair_a_originale(patch_ir_dir):
    """
    Ricostruisce il collegamento tra il nome "pairX_patch.jpg" e il file
    originale "tile_col_X_row_Y.jpg" per estrarne le coordinate pixel.
    """
    original_patches = sorted(glob.glob(os.path.join(patch_ir_dir, "*.jpg")))
    mapping = {}
    for i, orig_path in enumerate(original_patches):
        pair_name = f"pair{i+1}_patch.jpg"
        mapping[pair_name] = orig_path
    return mapping

def determina_colore_rgb(eta_rel_pct):
    """Restituisce il colore (R, G, B) in base alla salute del pannello."""
    if eta_rel_pct >= 98.0:
        return (0, 255, 0)       # Verde (Sano)
    elif eta_rel_pct >= 94.0:
        return (255, 200, 0)     # Giallo (Leggero degrado/Soiling)
    else:
        return (255, 0, 0)       # Rosso (Hotspot severo)

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print("\n" + "="*60)
    print("  GENERAZIONE DIGITAL TWIN: PROIEZIONE PANNELLI SU RGB")
    print("="*60)

    if not os.path.exists(IR_MOSAIC) or not os.path.exists(RGB_MOSAIC):
        print("[!] ERRORE: Mosaico IR o RGB non trovato nella cartella Yolo.")
        return

    # 1. Carica Modello AI e Dati
    from rfdetr import RFDETRSegLarge
    print("[*] Caricamento modello IA...")
    model = RFDETRSegLarge(pretrain_weights=WEIGHTS_PATH, num_classes=2)
    
    eff_data = carica_report_csv(CSV_PATH)
    mappa_orig = mappa_pair_a_originale(PATCH_IR_DIR)

    # 2. Lettura Mosaici con Rasterio
    print("[*] Lettura Coordinate Spaziali Mosaico Termico (IR)...")
    src_ir = rasterio.open(IR_MOSAIC)
    ir_transform = src_ir.transform
    ir_crs = src_ir.crs

    print("[*] Caricamento in RAM Mosaico Visivo (RGB)... (Potrebbe richiedere memoria)")
    src_rgb = rasterio.open(RGB_MOSAIC)
    rgb_transform = src_rgb.transform
    rgb_crs = src_rgb.crs
    
    # FIX: Copiamo il profilo e forziamo count=3 per ignorare eventuali canali Alpha (trasparenze)
    rgb_profile = src_rgb.profile.copy()
    rgb_profile.update(count=3)

    # Legge solo le prime 3 bande (RGB) per evitare problemi col canale Alpha
    rgb_img_data = src_rgb.read([1, 2, 3]) 
    
    # Converte da (C, H, W) a (H, W, C) per poterci disegnare sopra con OpenCV
    rgb_canvas = np.transpose(rgb_img_data, (1, 2, 0)).copy()

    # 3. Processamento Patch e Proiezione
    pair_files = list(eff_data.keys())
    print(f"[*] Proiezione di {len(pair_files)} aree analizzate sul mosaico RGB...")

    for i, pair_name in enumerate(pair_files):
        orig_path = mappa_orig.get(pair_name)
        if not orig_path: continue
        
        # Estrai coordinate pixel dell'angolo in alto a sx della patch dal nome originale
        m = re.search(r"tile_col_(\d+)_row_(\d+)", os.path.basename(orig_path))
        if not m: continue
        patch_col_offset = int(m.group(1))
        patch_row_offset = int(m.group(2))

        # Carica e analizza l'immagine
        patch_path = os.path.join(OUTPUT_DIR, "pair", pair_name)
        img_patch = cv2.imread(patch_path)
        if img_patch is None: continue
        
        img_pil = Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
        results = model.predict(img_pil, threshold=0.30)
        
        if results is None or len(results.xyxy) == 0: continue

        # Disegna ogni pannello rilevato
        for k in range(len(results.xyxy)):
            if (k + 1) not in eff_data[pair_name]: continue
            
            eta_rel = eff_data[pair_name][k + 1]
            color_rgb = determina_colore_rgb(eta_rel)
            mask = results.mask[k]
            if mask is None: continue

            # Trova i contorni della maschera
            mask_u8 = (mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            c_big = max(contours, key=cv2.contourArea)
            
            # Algoritmo di Vettorizzazione: Proiezione Coordinate Ultra-veloce
            # Estrae tutti i punti X e Y dal contorno
            px_xs = c_big[:, 0, 0] + patch_col_offset
            px_ys = c_big[:, 0, 1] + patch_row_offset
            
            # Passo A: Da Pixel IR a Coordinate Geografiche (Es. UTM/WGS84) del Mosaico IR
            east_irs, north_irs = rasterio.transform.xy(ir_transform, px_ys, px_xs)
            
            # Passo B: Allineamento tra CRS (Se IR e RGB hanno sistemi di riferimento diversi)
            xs_rgb, ys_rgb = transform_coords(ir_crs, rgb_crs, east_irs, north_irs)
            
            # Passo C: Da Coordinate Geografiche a Pixel nel Mosaico RGB
            rows_rgb, cols_rgb = rasterio.transform.rowcol(rgb_transform, xs_rgb, ys_rgb)
            
            # Ricostruisce il poligono con le nuove coordinate
            mapped_contour = np.array([list(zip(cols_rgb, rows_rgb))], dtype=np.int32)
            
            # Disegna il poligono sul canvas RGB
            cv2.drawContours(rgb_canvas, mapped_contour, -1, color_rgb, 4)
            
            # Calcola il centro per l'etichetta
            x, y, w, h = cv2.boundingRect(mapped_contour[0])
            cx, cy = x + w//2, y + h//2
            
            # Scrive l'efficienza
            label = f"{eta_rel:.1f}%"
            # Dimensione font dinamica in base a quanto appare grande il pannello sul mosaico
            font_scale = max(0.8, w / 150.0) 
            thickness = max(2, int(font_scale * 2))
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Sfondo nero per il testo
            cv2.rectangle(rgb_canvas, (cx - tw//2 - 5, cy - th - 5), (cx + tw//2 + 5, cy + 5), (0, 0, 0), -1)
            cv2.putText(rgb_canvas, label, (cx - tw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_rgb, thickness, cv2.LINE_AA)

        print(f"  [✓] Proiettati i pannelli di: {pair_name}")

    # 4. Salvataggio Mappa GeoTIFF
    print(f"\n[*] Salvataggio della mappa vettoriale georeferenziata...")
    out_img_chw = np.transpose(rgb_canvas, (2, 0, 1))
    
    with rasterio.open(MAPPA_OUT_PATH, 'w', **rgb_profile) as dst:
        dst.write(out_img_chw)

    print(f"[FINE] Mappa completata con successo!")
    print(f"[+] File salvato in: {MAPPA_OUT_PATH}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
