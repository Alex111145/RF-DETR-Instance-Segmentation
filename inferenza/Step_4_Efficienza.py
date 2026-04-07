#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import warnings
import subprocess
import urllib.request
import urllib.parse
import json
import math
import cv2
import csv
import numpy as np
from PIL import Image, ExifTags

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
# DATABASE PANNELLI E COSTANTI
# ==============================================================================
MAPPA_PANNELLI = {
    "1": {"tipo": "Monocristallino", "eta_nom": 0.20,  "gamma": -0.0037},
    "2": {"tipo": "Policristallino", "eta_nom": 0.165, "gamma": -0.0042},
}

EPSILON_VETRO = 0.90
G_IRR_STC     = 1000.0
T_STC_K       = 298.15

# ---> FILTRO AREA: ~1.5 m² <---
SOGLIA_AREA_PX = 10000 

# ==============================================================================
# INTERFACCIA UTENTE E API
# ==============================================================================
def chiedi_parametri_iniziali():
    print("\n" + "="*60)
    print("  CONFIGURAZIONE ANALISI TERMODINAMICA PV (IBRIDA)")
    print("="*60)
    print("\n  Tecnologia pannello:")
    print("    1) Monocristallino  (η=20.0%, γ=-0.37%/°C)")
    print("    2) Policristallino  (η=16.5%, γ=-0.42%/°C)")
    scelta = input("  Scelta [1/2, default 2]: ").strip() or "2"
    pannello = MAPPA_PANNELLI.get(scelta, MAPPA_PANNELLI["2"])
    
    print("\n  Dimensioni fisiche modulo:")
    w_inp = input("    Larghezza [default 1.67 m]: ").strip()
    h_inp = input("    Altezza   [default 1.0 m]: ").strip()
    w_mod = float(w_inp) if w_inp else 1.67
    h_mod = float(h_inp) if h_inp else 1.00
    area_modulo = w_mod * h_mod
    
    print(f"\n  [✓] Area calcolata: {area_modulo:.2f} m²")
    print("="*60 + "\n")
    
    return {
        "eta_nom": pannello["eta_nom"],
        "gamma": pannello["gamma"],
        "area": area_modulo,
        "tipo": pannello["tipo"]
    }

def estrai_gps_time(image_path):
    lat, lon, utc = None, None, None
    try:
        import re
        img = Image.open(image_path)
        
        xmp = img.info.get("xmp", b"").decode("utf-8", errors="ignore")
        def _get_xmp(tag):
            m = re.search(rf'drone-dji:{tag}="([^"]+)"', xmp)
            return m.group(1) if m else None
            
        lat_str = _get_xmp("GpsLatitude")
        lon_str = _get_xmp("GpsLongitude")
        utc = _get_xmp("UTCAtExposure") 
        
        if lat_str and lon_str:
            lat = float(lat_str)
            lon = float(lon_str)

        if lat is None or lon is None or utc is None:
            exif = img._getexif()
            if exif is not None:
                gps_info = {}
                datetime_orig = None
                
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == "GPSInfo":
                        for t in value:
                            sub_tag = ExifTags.GPSTAGS.get(t, t)
                            gps_info[sub_tag] = value[t]
                    elif tag == "DateTimeOriginal":
                        datetime_orig = value 
                
                def dms_to_decimal(dms, ref):
                    if not dms or not ref: return None
                    try:
                        d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
                        dec = d + m/60.0 + s/3600.0
                        return -dec if ref in ['S', 'W'] else dec
                    except Exception: return None
                
                if lat is None and "GPSLatitude" in gps_info:
                    lat = dms_to_decimal(gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N"))
                if lon is None and "GPSLongitude" in gps_info:
                    lon = dms_to_decimal(gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E"))
                if utc is None and datetime_orig:
                    utc = datetime_orig.replace(":", "-", 2).replace(" ", "T")

        return lat, lon, utc
    except Exception as e:
        return None, None, None

def get_openmeteo_tamb(lat, lon, utc_time):
    try:
        dt_part = utc_time[:19]
        date = dt_part[:10]
        hour = int(dt_part[11:13])
        
        print(f"[*] Interrogazione Open-Meteo per data {date} ore {hour} UTC...")
        params = urllib.parse.urlencode({
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "start_date": date, "end_date": date, "hourly": "temperature_2m"
        })
        url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            t_amb = data["hourly"]["temperature_2m"][hour]
            print(f"  [+] Open-Meteo T Ambiente: {t_amb:.1f} °C")
            return float(t_amb)
    except Exception as e:
        print(f"  [!] API Open-Meteo fallita. Uso default 25.0 °C.")
        return 25.0

# ==============================================================================
# MODELLO FISICO MATEMATICO E SDK
# ==============================================================================
def applica_stefan_boltzmann(t_app_c, eps, t_amb_c):
    t_app_k = t_app_c + 273.15
    t_amb_k = t_amb_c + 273.15
    valore_interno = (t_app_k**4 - (1 - eps) * t_amb_k**4) / eps
    return valore_interno**0.25 if valore_interno >= 0 else t_app_k

def calcola_efficienza_reale(eta_nom, gamma, t_reale_k):
    eta_reale = eta_nom * (1 + gamma * (t_reale_k - T_STC_K))
    return max(0.0, eta_reale)

def estrai_matrice_termica_dji(drone_path, sdk_dir):
    try:
        with open(drone_path, "rb") as f:
            raw_bytes = f.read()
        eoi_pos = raw_bytes.rfind(b"\xff\xd9")
        if eoi_pos != -1:
            thermal_blob = raw_bytes[eoi_pos + 2:]
            if len(thermal_blob) >= 640 * 512 * 2:
                arr_raw = np.frombuffer(thermal_blob[:640 * 512 * 2], dtype=np.uint16).reshape(512, 640)
                return arr_raw.astype(np.float32) / 100.0 - 273.15
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
        eta_rel_pct = d.get('eta_rel_pct', None)
        class_id = d.get('class_id', 1) 
        
        if class_id == 0 or (eta_rel_pct is not None and eta_rel_pct < 90.0):
            color = (0, 0, 255) # Rosso
            testo_stato = "Rotto"
        else:
            color = (0, 255, 0) # Verde
            testo_stato = "Sano"
            
        if eta_rel_pct is not None:
            label = f"{testo_stato} | Eff: {eta_rel_pct:.1f}%"
        else:
            label = f"{testo_stato} | Eff: N/A"

        if d['mask'] is not None:
            mask_u8 = (d['mask'].astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c_big = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c_big) >= SOGLIA_AREA_PX:
                    cv2.drawContours(canvas, [c_big], 0, color, 2)
                    M = cv2.moments(c_big)
                    tx, ty = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"]!=0 else (c_big[0][0][0], c_big[0][0][1])
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(canvas, (tx, ty - th - 5), (tx + tw, ty + 5), (0, 0, 0), -1)
                    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    os.makedirs(EFF_DIR, exist_ok=True)
    user_params = chiedi_parametri_iniziali()
    
    config_path = os.path.join(EFF_DIR, "config_analisi.json")
    with open(config_path, "w") as f:
        json.dump(user_params, f)
        
    patch_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_patch.jpg")))
    if not patch_files:
        print("[!] Nessuna patch trovata.")
        return

    print("[*] Estrazione metadati climatici (GPS/Time)...")
    primo_drone = patch_files[0].replace("_patch.jpg", "_drone.jpg")
    lat, lon, utc = estrai_gps_time(primo_drone)
    
    t_amb = 25.0
    if lat is not None and lon is not None and utc is not None:
        print(f"  [+] GPS trovato: Lat {lat:.4f}, Lon {lon:.4f} | Data/Ora: {utc}")
        t_amb = get_openmeteo_tamb(lat, lon, utc)
    else:
        print("  [!] Dati GPS non trovati nel file RJPEG. Uso T Ambiente = 25.0 °C.")

    from rfdetr import RFDETRSegLarge
    model = RFDETRSegLarge(pretrain_weights=WEIGHTS_PATH, num_classes=2)

    print(f"\n[*] FASE 1: Analisi logica ibrida e determinazione del Riferimento...")
    memoria_analisi = [] 
    max_eta_assoluta_sani = 0.0
    
    n_totale = len(patch_files)
    for i_patch, path_patch in enumerate(patch_files, 1):
        pct = int(i_patch / n_totale * 30)
        barra = "[" + "#" * pct + "-" * (30 - pct) + "]"
        print(f"\r  {barra} {i_patch}/{n_totale} - {os.path.basename(path_patch):<40}", end="", flush=True)

        img_patch = cv2.imread(path_patch)
        if img_patch is None: continue
        path_drone = path_patch.replace("_patch.jpg", "_drone.jpg")
        img_drone = cv2.imread(path_drone)
        
        img_pil = Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
        results = model.predict(img_pil, threshold=args.threshold)

        if results is None or len(results.xyxy) == 0:
            memoria_analisi.append({"patch": path_patch, "img": img_patch, "dets": []})
            continue

        temp_matrix = estrai_matrice_termica_dji(path_drone, SDK_DIR) if os.path.exists(path_drone) else None
        M_transform = calcola_omografia_patch_drone(img_patch, img_drone) if temp_matrix is not None else None

        d_h, d_w = img_drone.shape[:2] if img_drone is not None else (0,0)
        patch_dets = []

        for k in range(len(results.xyxy)):
            mask = results.mask[k] if results.mask is not None else None
            class_id = int(results.class_id[k]) 
            
            # FILTRO AREA GEOMETRICO
            if mask is None: continue
            mask_u8 = (mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            c_big = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c_big) < SOGLIA_AREA_PX:
                continue # E' spazzatura/tetto, salta questo poligono!
            
            eta_reale = None
            t_estratt_c = None
            
            if mask is not None and temp_matrix is not None and M_transform is not None:
                mask_warped = cv2.warpPerspective(mask_u8, M_transform, (d_w, d_h))
                if temp_matrix.shape[:2] != (d_h, d_w):
                    temp_matrix = cv2.resize(temp_matrix, (d_w, d_h), interpolation=cv2.INTER_NEAREST)
                    
                pixel_termici = temp_matrix[mask_warped > 127]
                
                if len(pixel_termici) > 0:
                    if class_id == 1 or class_id == 2:
                        t_estratt_c = np.mean(pixel_termici)
                    else:
                        t_estratt_c = np.max(pixel_termici)
                        
                    t_reale_k = applica_stefan_boltzmann(t_estratt_c, EPSILON_VETRO, t_amb)
                    eta_reale = calcola_efficienza_reale(user_params["eta_nom"], user_params["gamma"], t_reale_k)
                    
                    if (class_id == 1 or class_id == 2) and eta_reale > max_eta_assoluta_sani:
                        max_eta_assoluta_sani = eta_reale

            patch_dets.append({
                'class_id': class_id, 'mask': mask, 'eta_reale': eta_reale, 't_estratt_c': t_estratt_c
            })
            
        memoria_analisi.append({"patch": path_patch, "img": img_patch, "dets": patch_dets})

    print()  # Newline dopo la barra di avanzamento FASE 1

    if max_eta_assoluta_sani == 0.0:
        for elemento in memoria_analisi:
            for d in elemento["dets"]:
                if d['eta_reale'] is not None and d['eta_reale'] > max_eta_assoluta_sani:
                    max_eta_assoluta_sani = d['eta_reale']

    print(f"  [+] Pannello di Riferimento (100% Salute) trovato: Efficienza Assoluta {max_eta_assoluta_sani*100:.2f}%")

    print(f"\n[*] FASE 2: Generazione Output Annotati e Dati Grezzi...")
    csv_path = os.path.join(EFF_DIR, "dati_grezzi.csv")
    csv_fields = [
        "File_Patch", "ID_Pannello", "Classe_IA", "T_Estratta_C", "Eta_Assoluta_pct", 
        "Salute_Relativa_pct", "Potenza_Erogata_W"
    ]
    csv_rows = []

    n_elementi = len(memoria_analisi)
    for i_el, elemento in enumerate(memoria_analisi, 1):
        pct2 = int(i_el / n_elementi * 30)
        barra2 = "[" + "#" * pct2 + "-" * (30 - pct2) + "]"
        print(f"\r  {barra2} {i_el}/{n_elementi} - {os.path.basename(elemento['patch']):<40}", end="", flush=True)

        path_patch = elemento["patch"]
        img_patch = elemento["img"]
        dets = elemento["dets"]
        lista_draw = []

        for k, d in enumerate(dets):
            eta_reale = d['eta_reale']
            eta_rel_pct = None
            
            if eta_reale is not None:
                eta_rel_pct = (eta_reale / max_eta_assoluta_sani) * 100.0 if max_eta_assoluta_sani > 0 else 0.0
                p_attuale_w = G_IRR_STC * user_params["area"] * eta_reale
                
                classe_nome = "Sano" if (d['class_id'] == 1 or d['class_id'] == 2) else "Difettoso"
                
                csv_rows.append({
                    "File_Patch": os.path.basename(path_patch), "ID_Pannello": k + 1,
                    "Classe_IA": classe_nome,
                    "T_Estratta_C": round(d['t_estratt_c'], 2) if d['t_estratt_c'] else 0.0,
                    "Eta_Assoluta_pct": round(eta_reale * 100, 2), "Salute_Relativa_pct": round(eta_rel_pct, 2),
                    "Potenza_Erogata_W": round(p_attuale_w, 2)
                })

            lista_draw.append({
                'class_id': d['class_id'], 'mask': d['mask'],
                'eta_rel_pct': eta_rel_pct
            })

        if lista_draw:
            annotata = disegna_risultati_efficienza(img_patch, lista_draw)
            out_name = os.path.basename(path_patch).replace("_patch", "_eff")
            cv2.imwrite(os.path.join(EFF_DIR, out_name), annotata)

    print()  # Newline dopo la barra di avanzamento FASE 2

    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"[FINE] Analisi Completata. Lancia 5.py per mappa e report PDF!")

if __name__ == "__main__":
    main()
