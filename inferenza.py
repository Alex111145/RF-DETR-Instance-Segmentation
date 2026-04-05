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
# CONFIGURAZIONE PERCORSI
# ==============================================================================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR           = os.path.dirname(CURRENT_SCRIPT_DIR)

WEIGHTS_PATH  = os.path.join(BASE_DIR, "weights.pt")
INPUT_DIR     = os.path.join(BASE_DIR, "training_patches_ir")
MOSAIC_PATH   = os.path.join(BASE_DIR, "ortomosaicoir.tif")
OUTPUT_DIR    = os.path.join(BASE_DIR, "risultati_finali")

# Mappa colori e nomi (ID 2 è Sano come scoperto prima)
COLOR_MAP = {0: (0, 0, 255), 1: (0, 200, 0), 2: (0, 200, 0)}
NAME_MAP  = {0: "Difettoso", 1: "Sano", 2: "Sano"}

# ==============================================================================
# FUNZIONI DI ELABORAZIONE
# ==============================================================================

def carica_mosaico(path):
    """Carica il mosaico TIFF e lo normalizza a 8-bit BGR."""
    print(f"[*] Caricamento mosaico: {os.path.basename(path)}...")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"ERRORE: Impossibile leggere il mosaico in {path}")
        sys.exit(1)

    # Conversione se 16-bit o float
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Conversione in BGR se scala di grigi
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4: # Rimuovi eventuale canale Alpha
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    return img

def disegna_rilevamenti(img_canvas, det_info, off_x=0, off_y=0, spessore=2):
    """
    Disegna i contorni reali dei pannelli.
    Se off_x e off_y sono forniti, trasla i punti per il mosaico.
    """
    contatore = 0
    for d in det_info:
        if d['mask'] is None: continue
        
        mask_u8 = (d['mask'].astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c_big = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c_big) > 100:
                cid = d['class_id']
                color = COLOR_MAP.get(cid, (255, 255, 255))
                label = f"{NAME_MAP.get(cid, 'Sano')} {d['score']:.0%}"

                # Traslazione coordinate per il mosaico globale
                # $$x_{global} = x_{local} + offset_{x}$$
                # $$y_{global} = y_{local} + offset_{y}$$
                cnt_global = c_big.copy()
                cnt_global[:, :, 0] += off_x
                cnt_global[:, :, 1] += off_y

                # Disegna il perimetro reale
                cv2.drawContours(img_canvas, [cnt_global], 0, color, spessore)
                
                # Testo (centroide)
                M = cv2.moments(cnt_global)
                if M["m00"] != 0:
                    tx, ty = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.putText(img_canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                contatore += 1
    return contatore

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    patch_out_dir = os.path.join(OUTPUT_DIR, "patches_annotate")
    os.makedirs(patch_out_dir, exist_ok=True)

    # 1. Carica Mosaico e Modello
    mosaico = carica_mosaico(MOSAIC_PATH)
    
    from rfdetr import RFDETRSegLarge
    model = RFDETRSegLarge(pretrain_weights=WEIGHTS_PATH, num_classes=2)
    model.optimize_for_inference() # Ottimizzazione per velocità

    # 2. Trova le patch
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
    print(f"[*] Inizio inferenza su {len(files)} patch...")

    tot_pannelli = 0

    for path in files:
        filename = os.path.basename(path)
        
        # Estrai offset dal nome: tile_col_X_row_Y
        match = re.search(r"tile_col_(\d+)_row_(\d+)", filename)
        if not match: continue
        
        off_x, off_y = int(match.group(1)), int(match.group(2))

        # Leggi patch
        img_patch = cv2.imread(path)
        if img_patch is None: continue
        
        # Inferenza
        img_pil = Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
        results = model.predict(img_pil, threshold=args.threshold)

        if results is not None and len(results.xyxy) > 0:
            lista_det = []
            for k in range(len(results.xyxy)):
                lista_det.append({
                    'class_id': int(results.class_id[k]),
                    'score': float(results.confidence[k]),
                    'mask': results.mask[k]
                })

            # A. Disegna sulla patch singola (coordinate locali 0,0)
            patch_annotata = img_patch.copy()
            disegna_rilevamenti(patch_annotata, lista_det, 0, 0, spessore=2)
            cv2.imwrite(os.path.join(patch_out_dir, f"det_{filename}"), patch_annotata)

            # B. Disegna sul Mosaico (coordinate globali off_x, off_y)
            n = disegna_rilevamenti(mosaico, lista_det, off_x, off_y, spessore=3)
            tot_pannelli += n
            
            if n > 0:
                print(f"   > {filename}: trovati {n} pannelli")

    # 3. Salvataggio Finale
    print("\n[*] Salvataggio mosaico finale...")
    mosaic_out = os.path.join(OUTPUT_DIR, "Mosaico_Annotato_Completo.jpg")
    cv2.imwrite(mosaic_out, mosaico, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    # Crea una preview leggera
    h, w = mosaico.shape[:2]
    preview = cv2.resize(mosaico, (int(w*0.2), int(h*0.2)))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Mosaico_Preview.jpg"), preview)

    print(f"\n" + "="*40)
    print(f" FINITO!")
    print(f" Pannelli totali rilevati: {tot_pannelli}")
    print(f" Mosaico salvato in: {mosaic_out}")
    print(f" Patches singole in: {patch_out_dir}")
    print("="*40)

if __name__ == "__main__":
    main()
