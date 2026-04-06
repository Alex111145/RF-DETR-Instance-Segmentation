#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import warnings
import subprocess
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
OUTPUT_DIR         = os.path.join(BASE_DIR, "risultati_finali")
INPUT_DIR          = os.path.join(OUTPUT_DIR, "pair") 
THERMAL_DIR        = os.path.join(OUTPUT_DIR, "analisi_termica")
SDK_DIR            = os.path.join(BASE_DIR, "sdk/linux") 

COLOR_MAP = {
    0: (0, 0, 255),    # Rosso -> Difettoso
    1: (0, 200, 0),    # Verde -> Sano
    2: (0, 200, 0),    # Verde -> Sano
}

NAME_MAP = {
    0: "Difettoso",
    1: "Sano",
    2: "Sano"
}

# ==============================================================================
# FUNZIONI DJI THERMAL (LETTURA 100% BINARIA)
# ==============================================================================
def estrai_matrice_termica_dji(drone_path, sdk_dir):
    """
    Estrae la matrice termica leggendo rigorosamente in binario, 
    bypassando qualsiasi problema di decodifica UTF-8.
    """
    # --- STRATEGIA 1: RAW PARSING (Zero dipendenze) ---
    try:
        with open(drone_path, "rb") as f:
            raw_bytes = f.read()

        eoi_pos = raw_bytes.rfind(b"\xff\xd9")
        if eoi_pos != -1:
            thermal_blob = raw_bytes[eoi_pos + 2:]
            # Sensori standard DJI (640x512)
            if len(thermal_blob) >= 640 * 512 * 2:
                arr_raw = np.frombuffer(thermal_blob[:640 * 512 * 2], dtype=np.uint16).reshape(512, 640)
                temp_c = arr_raw.astype(np.float32) / 100.0 - 273.15
                return temp_c
    except Exception:
        pass # Passa alla strategia 2 se fallisce in silenzio

    # --- STRATEGIA 2: CLI dji_irp (File .raw temporaneo) ---
    exe_path = os.path.join(sdk_dir, "dji_irp")
    if not os.path.exists(exe_path):
        return None

    out_raw = drone_path.replace(".jpg", ".raw")
    cmd = [exe_path, "-s", drone_path, "-a", "measure", "-o", out_raw]
    
    # Configurazione ambiente per librerie Linux (.so)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{sdk_dir}:{os.path.join(sdk_dir, 'lib')}:{env.get('LD_LIBRARY_PATH', '')}"

    try:
        # Esegue ignorando completamente l'output testuale per evitare crash UTF-8
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, cwd=sdk_dir)
        
        if os.path.exists(out_raw):
            with open(out_raw, "rb") as f:
                file_bytes = f.read()
            os.remove(out_raw)
            
            # Formato output standard dji_irp (int16 diviso 10 = °C)
            if len(file_bytes) >= 640 * 512 * 2:
                arr_16 = np.frombuffer(file_bytes[:640 * 512 * 2], dtype=np.int16).reshape(512, 640)
                return arr_16.astype(np.float32) / 10.0
                
    except Exception as e:
        print(f"[!] Impossibile estrarre dati termici da {os.path.basename(drone_path)}")
        if os.path.exists(out_raw):
            os.remove(out_raw)
        
    return None

def calcola_omografia_patch_drone(img_patch, img_drone):
    gray_p = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
    gray_d = cv2.cvtColor(img_drone, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(gray_p, None)
    kp2, des2 = orb.detectAndCompute(gray_d, None)

    if des1 is None or des2 is None: return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

    if len(matches) > 15:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M
    return None

# ==============================================================================
# DISEGNO RISULTATI
# ==============================================================================
def disegna_rilevamento_termico(img, det_info):
    canvas = img.copy()
    for d in det_info:
        cid   = d['class_id']
        temp  = d.get('temp', None)
        
        class_name = NAME_MAP.get(cid, f"ID:{cid}")
        color      = COLOR_MAP.get(cid, (255, 255, 255))
        
        if temp is not None:
            label = f"{class_name} | {temp:.1f} C"
        else:
            label = f"{class_name} | N/A C"

        if d['mask'] is not None:
            mask_u8 = (d['mask'].astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c_big = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c_big) > 100:
                    cv2.drawContours(canvas, [c_big], 0, color, 2)
                    
                    M = cv2.moments(c_big)
                    if M["m00"] != 0:
                        tx, ty = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    else:
                        tx, ty = c_big[0][0][0], c_big[0][0][1]
                    
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(canvas, (tx, ty - th - 5), (tx + tw, ty + 5), (0, 0, 0), -1)
                    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        else:
            x1, y1, x2, y2 = d['xyxy']
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            
    return canvas

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_DIR)
    parser.add_argument("--output", default=THERMAL_DIR)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERRORE: Cartella {args.input} non trovata.")
        return

    os.makedirs(args.output, exist_ok=True)
    
    from rfdetr import RFDETRSegLarge
    model = RFDETRSegLarge(pretrain_weights=WEIGHTS_PATH, num_classes=2)

    patch_files = sorted(glob.glob(os.path.join(args.input, "*_patch.jpg")))
    print(f"[*] Avvio Analisi Termica su {len(patch_files)} coppie...")

    for i, path_patch in enumerate(patch_files):
        img_patch = cv2.imread(path_patch)
        if img_patch is None: continue
        
        path_drone = path_patch.replace("_patch.jpg", "_drone.jpg")
        img_drone = cv2.imread(path_drone)
        
        img_pil = Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
        results = model.predict(img_pil, threshold=args.threshold)

        if results is None or len(results.xyxy) == 0:
            continue

        temp_matrix = None
        M_transform = None
        if os.path.exists(path_drone):
            temp_matrix = estrai_matrice_termica_dji(path_drone, SDK_DIR)
            M_transform = calcola_omografia_patch_drone(img_patch, img_drone)

        lista_det = []
        d_h, d_w = img_drone.shape[:2] if img_drone is not None else (0,0)

        for k in range(len(results.xyxy)):
            mask = results.mask[k] if results.mask is not None else None
            media_termica = None

            if mask is not None and temp_matrix is not None and M_transform is not None:
                mask_u8 = (mask.astype(np.uint8)) * 255
                mask_warped = cv2.warpPerspective(mask_u8, M_transform, (d_w, d_h))
                
                # Ridimensiona matrice termica se non combacia con la visuale
                if temp_matrix.shape[:2] != (d_h, d_w):
                    temp_matrix_resized = cv2.resize(temp_matrix, (d_w, d_h), interpolation=cv2.INTER_NEAREST)
                else:
                    temp_matrix_resized = temp_matrix

                pixel_termici = temp_matrix_resized[mask_warped > 127]
                
                if len(pixel_termici) > 0:
                    media_termica = np.mean(pixel_termici)

            lista_det.append({
                'class_id': int(results.class_id[k]),
                'xyxy': results.xyxy[k].astype(int),
                'mask': mask,
                'temp': media_termica
            })

        if lista_det:
            annotata = disegna_rilevamento_termico(img_patch, lista_det)
            out_name = os.path.basename(path_patch).replace("_patch", "_thermal")
            cv2.imwrite(os.path.join(args.output, out_name), annotata)
            print(f"[{i+1}/{len(patch_files)}] Temperatura analizzata: {out_name}")

    print(f"\n[FINE] Rilevamenti con letture termiche salvati in: {args.output}")

if __name__ == "__main__":
    main()
