#!/usr/bin/env python3
import os
import sys
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

WEIGHTS_PATH       = os.path.join(BASE_DIR, "weights.pt")
INPUT_DIR          = os.path.join(BASE_DIR, "training_patches_ir")
OUTPUT_DIR         = os.path.join(BASE_DIR, "risultati_finali")
INFERENCE_DIR      = os.path.join(OUTPUT_DIR, "inferenza_pannelli")

# Mappa colori BGR
COLOR_MAP = {
    0: (0, 0, 255),    # Rosso -> Difettoso
    1: (0, 200, 0),    # Verde -> Sano (ID standard)
    2: (0, 200, 0),    # Verde -> Sano (ID rilevato)
}

NAME_MAP = {
    0: "Difettoso",
    1: "Sano",
    2: "Sano"
}

def disegna_rilevamento(img, det_info):
    canvas = img.copy()
    for d in det_info:
        cid   = d['class_id']
        score = d['score']
        
        class_name = NAME_MAP.get(cid, f"ID:{cid}")
        color      = COLOR_MAP.get(cid, (255, 255, 255))
        label      = f"{class_name} {score:.0%}"

        if d['mask'] is not None:
            # Convertiamo la maschera booleana in immagine 8-bit
            mask_u8 = (d['mask'].astype(np.uint8)) * 255
            
            # Troviamo il contorno esatto della maschera
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Prendiamo il contorno più grande
                c_big = max(contours, key=cv2.contourArea)
                
                # Filtro area minima per evitare rumore
                if cv2.contourArea(c_big) > 100:
                    # DISEGNO DEL PERIMETRO REALE (non rettangolare se tagliato)
                    cv2.drawContours(canvas, [c_big], 0, color, 2)
                    
                    # Calcolo posizione etichetta (centroide)
                    M = cv2.moments(c_big)
                    if M["m00"] != 0:
                        tx = int(M["m10"] / M["m00"])
                        ty = int(M["m01"] / M["m00"])
                    else:
                        # Fallback se il momento è zero (contorno molto sottile)
                        tx, ty = c_big[0][0][0], c_big[0][0][1]
                    
                    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        else:
            # Se non c'è maschera, usa il classico rettangolo XYXY
            x1, y1, x2, y2 = d['xyxy']
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            
    return canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_DIR)
    parser.add_argument("--output", default=INFERENCE_DIR)
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERRORE: Cartella {args.input} non trovata.")
        return

    os.makedirs(args.output, exist_ok=True)
    
    from rfdetr import RFDETRSegLarge
    model = RFDETRSegLarge(pretrain_weights=WEIGHTS_PATH, num_classes=2)

    files = sorted(glob.glob(os.path.join(args.input, "*.jpg")))
    print(f"[*] Elaborazione di {len(files)} patch...")

    for i, path in enumerate(files):
        img_bgr = cv2.imread(path)
        if img_bgr is None: continue
        
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        results = model.predict(img_pil, threshold=args.threshold)

        lista_det = []
        if results is not None and len(results.xyxy) > 0:
            for k in range(len(results.xyxy)):
                lista_det.append({
                    'class_id': int(results.class_id[k]),
                    'score': float(results.confidence[k]),
                    'xyxy': results.xyxy[k].astype(int),
                    'mask': results.mask[k] if results.mask is not None else None
                })

        if lista_det:
            annotata = disegna_rilevamento(img_bgr, lista_det)
            cv2.imwrite(os.path.join(args.output, f"det_{os.path.basename(path)}"), annotata)

    print(f"\n[FINE] Fatto! Ora i perimetri seguono esattamente la maschera rilevata. Salvati in: {args.output}")

if __name__ == "__main__":
    main()
