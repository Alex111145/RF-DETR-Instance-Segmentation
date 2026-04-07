#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import warnings
import subprocess
import cv2
import json  # <--- Aggiunto per la gestione del file JSON
import numpy as np
from PIL import Image
from tqdm import tqdm 

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
    0: (0, 255, 0),    
    1: (0, 0, 255),    
    2: (0, 255, 0),    
}

NAME_MAP = {
    0: "SANO",
    1: "DIFETTOSO",
    2: "SANO"
}

def estrai_matrice_termica_dji(drone_path, sdk_dir):
    try:
        with open(drone_path, "rb") as f:
            raw_bytes = f.read()
        eoi_pos = raw_bytes.rfind(b"\xff\xd9")
        if eoi_pos != -1:
            thermal_blob = raw_bytes[eoi_pos + 2:]
            if len(thermal_blob) >= 640 * 512 * 2:
                arr_raw = np.frombuffer(thermal_blob[:640 * 512 * 2], dtype=np.uint16).reshape(512, 640)
                temp_c = arr_raw.astype(np.float32) / 100.0 - 273.15
                return temp_c
    except Exception: pass

    exe_path = os.path.join(sdk_dir, "dji_irp")
    if not os.path.exists(exe_path): return None
    out_raw = drone_path.replace(".jpg", ".raw")
    cmd = [exe_path, "-s", drone_path, "-a", "measure", "-o", out_raw]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{sdk_dir}:{os.path.join(sdk_dir, 'lib')}:{env.get('LD_LIBRARY_PATH', '')}"

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, cwd=sdk_dir)
        if os.path.exists(out_raw):
            with open(out_raw, "rb") as f:
                file_bytes = f.read()
            os.remove(out_raw)
            if len(file_bytes) >= 640 * 512 * 2:
                arr_16 = np.frombuffer(file_bytes[:640 * 512 * 2], dtype=np.int16).reshape(512, 640)
                return arr_16.astype(np.float32) / 10.0
    except Exception: pass
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

def disegna_rilevamento_termico(img, det_info):
    canvas = img.copy()
    for d in det_info:
        cid   = d['class_id']
        temp  = d.get('temp', None)
        class_name = NAME_MAP.get(cid, f"ID:{cid}")
        color      = COLOR_MAP.get(cid, (255, 255, 255))
        label = f"{class_name} | {temp:.1f} C" if temp is not None else f"{class_name} | N/A"

        if d['mask'] is not None:
            mask_u8 = (d['mask'].astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c_big = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c_big) > 100:
                    cv2.drawContours(canvas, [c_big], 0, color, 2)
                    M = cv2.moments(c_big)
                    tx, ty = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (c_big[0][0][0], c_big[0][0][1])
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(canvas, (tx, ty - th - 5), (tx + tw, ty + 5), (0, 0, 0), -1)
                    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas

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
    model = RFDETRSegLarge(pretrain_weights=WEIGHTS_PATH, num_classes=3)

    patch_files = sorted(glob.glob(os.path.join(args.input, "*_patch.jpg")))
    print(f"[*] Avvio Analisi Termica su {len(patch_files)} coppie...")

    # Dizionario per raccogliere tutti i risultati per il JSON
    risultati_globali = {}

    for path_patch in tqdm(patch_files, desc="Elaborazione Termica", unit="patch"):
        img_patch = cv2.imread(path_patch)
        if img_patch is None: continue
        
        nome_file_base = os.path.basename(path_patch)
        path_drone = path_patch.replace("_patch.jpg", "_drone.jpg")
        img_drone = cv2.imread(path_drone)
        
        img_pil = Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
        results = model.predict(img_pil, threshold=args.threshold)

        if results is None or len(results.xyxy) == 0:
            continue

        temp_matrix = estrai_matrice_termica_dji(path_drone, SDK_DIR) if os.path.exists(path_drone) else None
        M_transform = calcola_omografia_patch_drone(img_patch, img_drone) if img_drone is not None else None

        lista_det = []
        dati_per_json = [] # Dati puliti da salvare nel file
        d_h, d_w = img_drone.shape[:2] if img_drone is not None else (0,0)

        for k in range(len(results.xyxy)):
            cid = int(results.class_id[k])
            mask = results.mask[k] if hasattr(results, 'mask') and results.mask is not None else None
            
            t_media, t_max = None, None
            valore_termico_label = None

            if mask is not None and temp_matrix is not None and M_transform is not None:
                mask_u8 = (mask.astype(np.uint8)) * 255
                mask_warped = cv2.warpPerspective(mask_u8, M_transform, (d_w, d_h))
                
                temp_matrix_resized = cv2.resize(temp_matrix, (d_w, d_h), interpolation=cv2.INTER_NEAREST) if temp_matrix.shape[:2] != (d_h, d_w) else temp_matrix
                pixel_termici = temp_matrix_resized[mask_warped > 127]
                
                if len(pixel_termici) > 0:
                    t_media = float(np.mean(pixel_termici))
                    t_max = float(np.max(pixel_termici))
                    
                    # Logica etichetta: Max per difettosi, Media per sani
                    if NAME_MAP.get(cid) == "DIFETTOSO":
                        valore_termico_label = t_max
                    else:
                        valore_termico_label = t_media

            lista_det.append({
                'class_id': cid,
                'xyxy': results.xyxy[k].astype(int),
                'mask': mask,
                'temp': valore_termico_label
            })

            dati_per_json.append({
                "class_id": cid,
                "label": NAME_MAP.get(cid, "SCONOSCIUTO"),
                "temp_media": round(t_media, 2) if t_media is not None else None,
                "temp_max": round(t_max, 2) if t_max is not None else None,
                "temp_utilizzata": round(valore_termico_label, 2) if valore_termico_label is not None else None
            })

        if lista_det:
            # Salvataggio Immagine
            annotata = disegna_rilevamento_termico(img_patch, lista_det)
            out_name = nome_file_base.replace("_patch.jpg", "_thermal.jpg")
            cv2.imwrite(os.path.join(args.output, out_name), annotata)
            
            # Aggiunta dati al database globale
            risultati_globali[nome_file_base] = dati_per_json

    # Salvataggio del file JSON globale
    json_path = os.path.join(args.output, "analisi_dati.json")
    with open(json_path, "w") as f:
        json.dump(risultati_globali, f, indent=4)

    print(f"\n[FINE] Analisi completata.")
    print(f" -> Immagini salvate in: {args.output}")
    print(f" -> Dati JSON salvati in: {json_path}")

if __name__ == "__main__":
    main()
