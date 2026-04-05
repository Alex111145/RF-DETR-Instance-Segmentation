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
    original_patches = sorted(glob.glob(os.path.join(patch_ir_dir, "*.jpg")))
    mapping = {}
    for i, orig_path in enumerate(original_patches):
        pair_name = f"pair{i+1}_patch.jpg"
        mapping[pair_name] = orig_path
    return mapping

def determina_colore_rgb(eta_rel_pct):
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
    
    # FIX Salvataggio: Forziamo 3 bande per evitare l'errore del canale Alpha
    rgb_profile = src_rgb.profile.copy()
    rgb_profile.update(count=3)

    rgb_img_data = src_rgb.read([1, 2, 3]) 
    rgb_canvas = np.transpose(rgb_img_data, (1, 2, 0)).copy()

    # 3. Processamento Patch e Accumulo (Senza disegnare subito)
    pair_files = list(eff_data.keys())
    print(f"[*] Proiezione virtuale di {len(pair_files)} aree...")

    pannelli_globali = [] # Lista per raccogliere tutti i pannelli e filtrare i duplicati

    for i, pair_name in enumerate(pair_files):
        orig_path = mappa_orig.get(pair_name)
        if not orig_path: continue
        
        m = re.search(r"tile_col_(\d+)_row_(\d+)", os.path.basename(orig_path))
        if not m: continue
        patch_col_offset = int(m.group(1))
        patch_row_offset = int(m.group(2))

        patch_path = os.path.join(OUTPUT_DIR, "pair", pair_name)
        img_patch = cv2.imread(patch_path)
        if img_patch is None: continue
        
        img_pil = Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
        results = model.predict(img_pil, threshold=0.30)
        
        if results is None or len(results.xyxy) == 0: continue

        for k in range(len(results.xyxy)):
            if (k + 1) not in eff_data[pair_name]: continue
            
            eta_rel = eff_data[pair_name][k + 1]
            color_rgb = determina_colore_rgb(eta_rel)
            mask = results.mask[k]
            if mask is None: continue

            mask_u8 = (mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            c_big = max(contours, key=cv2.contourArea)
            
            px_xs = c_big[:, 0, 0] + patch_col_offset
            px_ys = c_big[:, 0, 1] + patch_row_offset
            
            east_irs, north_irs = rasterio.transform.xy(ir_transform, px_ys, px_xs)
            xs_rgb, ys_rgb = transform_coords(ir_crs, rgb_crs, east_irs, north_irs)
            rows_rgb, cols_rgb = rasterio.transform.rowcol(rgb_transform, xs_rgb, ys_rgb)
            
            mapped_contour = np.array([list(zip(cols_rgb, rows_rgb))], dtype=np.int32)
            
            # Calcolo Area e Centroide del poligono proiettato
            area_globale = cv2.contourArea(mapped_contour)
            M = cv2.moments(mapped_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(mapped_contour)
                cx, cy = x + w//2, y + h//2
            
            # Salviamo il pannello nella lista invece di disegnarlo
            pannelli_globali.append({
                'contour': mapped_contour,
                'centroid': (cx, cy),
                'area': area_globale,
                'eta': eta_rel,
                'color': color_rgb
            })

    # ==========================================================
    # 4. FILTRO ANTI-DUPLICATI (Global NMS per Poligoni)
    # ==========================================================
    print(f"[*] Pulizia sovrapposizioni: Trovati {len(pannelli_globali)} pannelli grezzi.")
    
    # Ordiniamo dal più grande al più piccolo (diamo priorità ai pannelli completi)
    pannelli_globali.sort(key=lambda p: p['area'], reverse=True)
    
    pannelli_filtrati = []
    
    for nuovo_pan in pannelli_globali:
        duplicato = False
        cx, cy = nuovo_pan['centroid']
        
        for pan_salvato in pannelli_filtrati:
            # Se il centroide del nuovo pannello cade dentro il perimetro di uno già salvato, è un duplicato!
            if cv2.pointPolygonTest(pan_salvato['contour'], (cx, cy), measureDist=False) >= 0:
                duplicato = True
                break
                
        if not duplicato:
            pannelli_filtrati.append(nuovo_pan)

    print(f"  [✓] Pulizia completata! Pannelli unici reali da disegnare: {len(pannelli_filtrati)}")

    # 5. DISEGNO FINALE
    for pan in pannelli_filtrati:
        contour = pan['contour']
        color = pan['color']
        eta_rel = pan['eta']
        cx, cy = pan['centroid']
        
        cv2.drawContours(rgb_canvas, contour, -1, color, 4)
        
        x, y, w, h = cv2.boundingRect(contour)
        label = f"{eta_rel:.1f}%"
        font_scale = max(0.8, w / 150.0) 
        thickness = max(2, int(font_scale * 2))
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        cv2.rectangle(rgb_canvas, (cx - tw//2 - 5, cy - th - 5), (cx + tw//2 + 5, cy + 5), (0, 0, 0), -1)
        cv2.putText(rgb_canvas, label, (cx - tw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    # 6. Salvataggio Mappa GeoTIFF
    print(f"\n[*] Salvataggio della mappa vettoriale georeferenziata...")
    out_img_chw = np.transpose(rgb_canvas, (2, 0, 1))
    
    with rasterio.open(MAPPA_OUT_PATH, 'w', **rgb_profile) as dst:
        dst.write(out_img_chw)

    print(f"[FINE] Mappa completata con successo senza sovrapposizioni!")
    print(f"[+] File salvato in: {MAPPA_OUT_PATH}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
