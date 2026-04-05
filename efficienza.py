#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import warnings
import subprocess
import tempfile
import cv2
import csv
import numpy as np
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
INPUT_DIR          = os.path.join(OUTPUT_DIR, "pair") 
EFF_DIR            = os.path.join(OUTPUT_DIR, "efficienza_risultati")
SDK_DIR            = os.path.join(BASE_DIR, "sdk/linux") 

# ==============================================================================
# COSTANTI TERMODINAMICHE ED ECONOMICHE (Tratte dalla Tesi)
# ==============================================================================
# Default: Modulo Policristallino
ETA_NOM     = 0.165      # 16.5% Efficienza nominale
GAMMA       = -0.0042    # -0.42%/°C Coefficiente di temperatura
EPSILON     = 0.90       # Emissività vetro temperato
T_AMB_C     = 25.0       # Temperatura ambiente in °C
T_STC_K     = 298.15     # 25°C in Kelvin (Condizioni Standard)

G_IRR       = 1000.0     # W/m^2 Irraggiamento
AREA_MODULO = 1.67       # m^2 (Superficie fisica standard)

ESH_MEDIE   = 3.18       # Ore equivalenti di pieno sole giornaliere
GIORNI_UTIL = 300        # Giorni utili di produzione annua
COSTO_KWH   = 0.40       # €/kWh Costo opportunità energia

# ==============================================================================
# MODELLO FISICO MATEMATICO (Caso 1: Pannello Sano)
# ==============================================================================
def applica_stefan_boltzmann(t_app_c, eps, t_amb_c):
    """Calcola la temperatura reale della cella depurata dai riflessi ambientali (in Kelvin)."""
    t_app_k = t_app_c + 273.15
    t_amb_k = t_amb_c + 273.15
    
    # Legge di Stefan-Boltzmann
    valore_interno = (t_app_k**4 - (1 - eps) * t_amb_k**4) / eps
    
    if valore_interno < 0:
        return t_app_k # Fallback fisico di sicurezza
        
    t_reale_k = valore_interno**0.25
    return t_reale_k

def calcola_efficienza_reale(eta_nom, gamma, t_reale_k):
    """Calcola il degrado termico dell'efficienza in base alla temperatura reale in Kelvin."""
    eta_reale = eta_nom * (1 + gamma * (t_reale_k - T_STC_K))
    return max(0.0, eta_reale) # Impedisce efficienze negative assurde

def calcola_potenze_e_perdite(eta_reale):
    """Calcola P_teorica, P_reale, e mancato guadagno in Euro."""
    p_teo_w = G_IRR * AREA_MODULO * ETA_NOM
    p_real_w = G_IRR * AREA_MODULO * eta_reale
    
    if p_teo_w <= p_real_w:
        return p_teo_w, p_real_w, 0.0, 0.0
        
    p_persa_w = p_teo_w - p_real_w
    e_persa_kwh = (p_persa_w * ESH_MEDIE) / 1000.0
    perdita_euro = e_persa_kwh * GIORNI_UTIL * COSTO_KWH
    
    return p_teo_w, p_real_w, p_persa_w, perdita_euro

# ==============================================================================
# FUNZIONI DJI THERMAL (LETTURA BINARIA SICURA)
# ==============================================================================
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
    except Exception:
        pass

    exe_path = os.path.join(sdk_dir, "dji_irp")
    if not os.path.exists(exe_path): return None

    out_raw = drone_path.replace(".jpg", ".raw")
    cmd = [exe_path, "-s", drone_path, "-a", "measure", "-o", out_raw]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{sdk_dir}:{os.path.join(sdk_dir, 'lib')}:{env.get('LD_LIBRARY_PATH', '')}"

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, cwd=sdk_dir)
        if os.path.exists(out_raw):
            with open(out_raw, "rb") as f: file_bytes = f.read()
            os.remove(out_raw)
            if len(file_bytes) >= 640 * 512 * 2:
                arr_16 = np.frombuffer(file_bytes[:640 * 512 * 2], dtype=np.int16).reshape(512, 640)
                return arr_16.astype(np.float32) / 10.0
    except Exception:
        if os.path.exists(out_raw): os.remove(out_raw)
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
def disegna_risultati_efficienza(img, det_info):
    canvas = img.copy()
    for d in det_info:
        eta_pct = d.get('eta_pct', None)
        
        color = (0, 200, 0) # Verde Sano
        
        # Testo con l'efficienza invece della temperatura
        if eta_pct is not None:
            label = f"Sano | Eff: {eta_pct:.1f}%"
        else:
            label = "Sano | Eff: N/A"

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
                    
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(canvas, (tx, ty - th - 5), (tx + tw, ty + 5), (0, 0, 0), -1)
                    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    os.makedirs(EFF_DIR, exist_ok=True)
    
    from rfdetr import RFDETRSegLarge
    model = RFDETRSegLarge(pretrain_weights=WEIGHTS_PATH, num_classes=2)

    patch_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_patch.jpg")))
    print(f"\n[*] Avvio Calcolo Efficienza (Pannello Sano) su {len(patch_files)} patch...")
    
    # Inizializza Report CSV
    csv_path = os.path.join(EFF_DIR, "report_efficienza.csv")
    csv_fields = [
        "File_Patch", "ID_Pannello", "T_Media_Apparente_C", "T_Reale_K", 
        "Eta_Nominale_pct", "Eta_Reale_pct", "Potenza_Teorica_W", 
        "Potenza_Reale_W", "Perdita_Economica_EUR_Anno"
    ]
    csv_rows = []

    for i, path_patch in enumerate(patch_files):
        img_patch = cv2.imread(path_patch)
        if img_patch is None: continue
        
        path_drone = path_patch.replace("_patch.jpg", "_drone.jpg")
        img_drone = cv2.imread(path_drone)
        
        img_pil = Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
        results = model.predict(img_pil, threshold=0.30)

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
            
            eta_reale_pct = None

            if mask is not None and temp_matrix is not None and M_transform is not None:
                mask_u8 = (mask.astype(np.uint8)) * 255
                mask_warped = cv2.warpPerspective(mask_u8, M_transform, (d_w, d_h))
                
                if temp_matrix.shape[:2] != (d_h, d_w):
                    temp_matrix_resized = cv2.resize(temp_matrix, (d_w, d_h), interpolation=cv2.INTER_NEAREST)
                else:
                    temp_matrix_resized = temp_matrix

                pixel_termici = temp_matrix_resized[mask_warped > 127]
                
                if len(pixel_termici) > 0:
                    # Caso 1: Pannello Sano -> MEDIA MATEMATICA
                    t_app_c = np.mean(pixel_termici)
                    
                    # 1. Correzione Stefan-Boltzmann (restituisce Kelvin)
                    t_reale_k = applica_stefan_boltzmann(t_app_c, EPSILON, T_AMB_C)
                    
                    # 2. Calcolo Efficienza (Modello di Degradazione)
                    eta_reale = calcola_efficienza_reale(ETA_NOM, GAMMA, t_reale_k)
                    eta_reale_pct = eta_reale * 100.0
                    
                    # 3. Calcolo Potenze e Perdita
                    p_teo_w, p_real_w, p_persa_w, perdita_euro = calcola_potenze_e_perdite(eta_reale)
                    
                    # Salva nel report CSV
                    csv_rows.append({
                        "File_Patch": os.path.basename(path_patch),
                        "ID_Pannello": k + 1,
                        "T_Media_Apparente_C": round(t_app_c, 2),
                        "T_Reale_K": round(t_reale_k, 2),
                        "Eta_Nominale_pct": round(ETA_NOM * 100, 2),
                        "Eta_Reale_pct": round(eta_reale_pct, 2),
                        "Potenza_Teorica_W": round(p_teo_w, 2),
                        "Potenza_Reale_W": round(p_real_w, 2),
                        "Perdita_Economica_EUR_Anno": round(perdita_euro, 2)
                    })

            lista_det.append({
                'class_id': int(results.class_id[k]),
                'mask': mask,
                'eta_pct': eta_reale_pct
            })

        if lista_det:
            annotata = disegna_risultati_efficienza(img_patch, lista_det)
            out_name = os.path.basename(path_patch).replace("_patch", "_eff")
            cv2.imwrite(os.path.join(EFF_DIR, out_name), annotata)
            print(f"[{i+1}/{len(patch_files)}] Calcolata efficienza: {out_name}")

    # Salva il CSV
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[+] Report CSV salvato in: {csv_path}")

    print(f"[FINE] Analisi Efficienza completata. Patch annotate in: {EFF_DIR}")

if __name__ == "__main__":
    main()
