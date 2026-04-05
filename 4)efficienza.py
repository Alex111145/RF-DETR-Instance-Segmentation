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
from datetime import datetime

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
COSTO_KWH     = 0.40
GIORNI_UTIL   = 300

# ==============================================================================
# INTERFACCIA UTENTE (CLI) E API
# ==============================================================================
def chiedi_parametri_iniziali():
    print("\n" + "="*60)
    print("  CONFIGURAZIONE ANALISI TERMODINAMICA PV (PANNELLI DANNEGGIATI)")
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
    except Exception:
        return None, None, None

def get_pvgis_esh(lat, lon):
    try:
        print(f"[*] Interrogazione PVGIS per coordinate ({lat:.4f}, {lon:.4f})...")
        url = f"https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?lat={lat}&lon={lon}&peakpower=1&loss=14&outputformat=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            e_annua = data["outputs"]["totals"]["fixed"]["E_y"]
            esh = e_annua / 365.0
            print(f"  [+] PVGIS ESH medio giornaliero: {esh:.2f} ore")
            return esh
    except Exception as e:
        print(f"  [!] API PVGIS fallita ({e}). Uso default 3.18 ore.")
        return 3.18

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
        print(f"  [!] API Open-Meteo fallita ({e}). Uso default 25.0 °C.")
        return 25.0

# ==============================================================================
# MODELLO FISICO MATEMATICO E REPORTISTICA
# ==============================================================================
def applica_stefan_boltzmann(t_app_c, eps, t_amb_c):
    t_app_k = t_app_c + 273.15
    t_amb_k = t_amb_c + 273.15
    valore_interno = (t_app_k**4 - (1 - eps) * t_amb_k**4) / eps
    return valore_interno**0.25 if valore_interno >= 0 else t_app_k

def calcola_efficienza_reale(eta_nom, gamma, t_reale_k):
    eta_reale = eta_nom * (1 + gamma * (t_reale_k - T_STC_K))
    return max(0.0, eta_reale)

def genera_report_pdf(csv_rows, pdf_path, user_params):
    """
    Legge i dati dal CSV calcolato in memoria e genera un PDF (via PIL)
    con le statistiche globali, le perdite e il resoconto economico.
    """
    # Creiamo un "foglio" A4 digitale (1240x1754 pixel a 150 DPI)
    w, h = 1240, 1754
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    tot_pannelli = len(csv_rows)
    if tot_pannelli == 0: return

    # Calcolo Medie e Somme (Filtriamo i danneggiati sotto il 90%)
    eta_media = sum([r["Salute_Relativa_pct"] for r in csv_rows]) / tot_pannelli
    danneggiati = [r for r in csv_rows if r["Salute_Relativa_pct"] < 90.0]
    num_danneggiati = len(danneggiati)
    
    # La potenza persa in W viene divisa per 1000 per ottenere i kW
    pot_persa_kw = sum([r["Potenza_Persa_W"] for r in danneggiati]) / 1000.0
    perdita_euro = sum([r["Mancato_Guadagno_EUR_Anno"] for r in danneggiati])
    
    # === DISEGNO TESTATA ===
    cv2.rectangle(canvas, (0, 0), (w, 150), (40, 40, 40), -1)
    cv2.putText(canvas, "REPORT TECNICO DI ISPEZIONE FOTOVOLTAICA UAV", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    y = 250
    def add_text(text, scale, color, thick, dy=50):
        nonlocal y
        cv2.putText(canvas, text, (80, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y += dy

    add_text(f"Data Elaborazione: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0.8, (100, 100, 100), 1, 80)
    
    add_text("PARAMETRI DI ANALISI", 1.1, (0, 0, 0), 2, 60)
    cv2.line(canvas, (80, y-40), (1160, y-40), (200, 200, 200), 2)
    add_text(f"- Tecnologia Moduli: {user_params['tipo']}", 0.9, (50, 50, 50), 1, 40)
    add_text(f"- Efficienza Nominale (STC): {user_params['eta_nom']*100:.1f}%", 0.9, (50, 50, 50), 1, 40)
    add_text(f"- Coefficiente di Temperatura: {user_params['gamma']*100:.2f}%/C", 0.9, (50, 50, 50), 1, 80)

    add_text("STATISTICHE GENERALI IMPIANTO", 1.1, (0, 0, 0), 2, 60)
    cv2.line(canvas, (80, y-40), (1160, y-40), (200, 200, 200), 2)
    add_text(f"- Totale Pannelli Analizzati dall'IA: {tot_pannelli}", 0.9, (50, 50, 50), 1, 40)
    add_text(f"- Efficienza Media dell'Impianto (SoH): {eta_media:.1f}%", 0.9, (50, 50, 50), 1, 40)
    add_text(f"- Pannelli Integri (SoH >= 90%): {tot_pannelli - num_danneggiati}", 0.9, (0, 150, 0), 2, 80)

    add_text("ANALISI DEL DEGRADO (ANOMALIE E HOTSPOT)", 1.1, (0, 0, 200), 2, 60)
    cv2.line(canvas, (80, y-40), (1160, y-40), (200, 200, 200), 2)
    add_text(f"- Pannelli Danneggiati (SoH < 90%): {num_danneggiati}", 0.9, (0, 0, 200), 2, 40)
    add_text(f"- Potenza Nominale Persa Complessiva: {pot_persa_kw:.2f} kW", 0.9, (50, 50, 50), 2, 40)
    add_text(f"- Mancato Guadagno Stimato: {perdita_euro:.2f} EUR / Anno", 0.9, (50, 50, 50), 2, 80)

    add_text("CONCLUSIONE DIAGNOSTICA", 1.1, (0, 0, 0), 2, 60)
    cv2.line(canvas, (80, y-40), (1160, y-40), (200, 200, 200), 2)
    if num_danneggiati > 0:
        add_text("Si raccomanda la manutenzione o sostituzione dei moduli termicamente", 0.9, (50, 50, 50), 1, 40)
        add_text("danneggiati per ripristinare la capacita produttiva nominale,", 0.9, (50, 50, 50), 1, 40)
        add_text(f"evitando una perdita cumulativa di {perdita_euro:.2f} Euro annui.", 0.9, (50, 50, 50), 1, 40)
    else:
        add_text("L'impianto si trova in condizioni operative ottimali. Nessun modulo", 0.9, (50, 50, 50), 1, 40)
        add_text("scende sotto la soglia critica del 90% di efficienza relativa.", 0.9, (50, 50, 50), 1, 40)

    # === DISEGNO PIE' DI PAGINA ===
    cv2.rectangle(canvas, (0, h-80), (w, h), (40, 40, 40), -1)
    cv2.putText(canvas, "Generato automaticamente tramite IA - Ispezione Termografica UAV", (50, h-35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Salva il canvas come vero file PDF multipiattaforma
    img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    img_pil.save(pdf_path, "PDF", resolution=150.0)


# ==============================================================================
# SDK DJI & ALLINEAMENTO
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
        
        # Colore Giallo/Rosso in base alle soglie per pannelli danneggiati
        if eta_rel_pct is not None:
            if eta_rel_pct >= 80.0:
                color = (0, 200, 255) # Giallo (BGR)
            else:
                color = (0, 0, 255)   # Rosso
                
            label = f"Rotto | Eff: {eta_rel_pct:.1f}%"
        else:
            color = (128, 128, 128)
            label = "Rotto | Eff: N/A"

        if d['mask'] is not None:
            mask_u8 = (d['mask'].astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c_big = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c_big) > 100:
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
    os.makedirs(EFF_DIR, exist_ok=True)
    
    user_params = chiedi_parametri_iniziali()
    
    patch_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_patch.jpg")))
    if not patch_files:
        print("[!] Nessuna patch trovata.")
        return

    # Estrazione Meteo ed ESH
    print("[*] Estrazione metadati climatici (GPS/Time)...")
    primo_drone = patch_files[0].replace("_patch.jpg", "_drone.jpg")
    lat, lon, utc = estrai_gps_time(primo_drone)
    
    t_amb = 25.0
    esh = 3.18
    if lat is not None and lon is not None and utc is not None:
        print(f"  [+] GPS trovato: Lat {lat:.4f}, Lon {lon:.4f} | Data/Ora: {utc}")
        t_amb = get_openmeteo_tamb(lat, lon, utc)
        esh = get_pvgis_esh(lat, lon)
    else:
        print("  [!] Dati GPS non trovati nel file RJPEG. Uso default.")

    # Caricamento AI
    from rfdetr import RFDETRSegLarge
    model = RFDETRSegLarge(pretrain_weights=WEIGHTS_PATH, num_classes=2)

    # ---------------------------------------------------------
    # FASE 1: Scansione globale (Riferimento = Pannello Sano / T.Media più bassa)
    # ---------------------------------------------------------
    print(f"\n[*] FASE 1: Analisi globale per stabilire il Pannello di Riferimento...")
    memoria_analisi = [] 
    max_eta_assoluta = 0.0
    
    for path_patch in patch_files:
        img_patch = cv2.imread(path_patch)
        if img_patch is None: continue
        path_drone = path_patch.replace("_patch.jpg", "_drone.jpg")
        img_drone = cv2.imread(path_drone)
        
        img_pil = Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
        results = model.predict(img_pil, threshold=0.30)

        if results is None or len(results.xyxy) == 0:
            memoria_analisi.append({"patch": path_patch, "img": img_patch, "dets": []})
            continue

        temp_matrix = estrai_matrice_termica_dji(path_drone, SDK_DIR) if os.path.exists(path_drone) else None
        M_transform = calcola_omografia_patch_drone(img_patch, img_drone) if temp_matrix is not None else None

        d_h, d_w = img_drone.shape[:2] if img_drone is not None else (0,0)
        patch_dets = []

        for k in range(len(results.xyxy)):
            mask = results.mask[k] if results.mask is not None else None
            eta_reale_max = None
            
            if mask is not None and temp_matrix is not None and M_transform is not None:
                mask_u8 = (mask.astype(np.uint8)) * 255
                mask_warped = cv2.warpPerspective(mask_u8, M_transform, (d_w, d_h))
                if temp_matrix.shape[:2] != (d_h, d_w):
                    temp_matrix = cv2.resize(temp_matrix, (d_w, d_h), interpolation=cv2.INTER_NEAREST)
                    
                pixel_termici = temp_matrix[mask_warped > 127]
                
                if len(pixel_termici) > 0:
                    # Calcola il riferimento sull'intero impianto usando la media per scartare gli hotspot
                    t_media_c = np.mean(pixel_termici)
                    t_reale_k_media = applica_stefan_boltzmann(t_media_c, EPSILON_VETRO, t_amb)
                    eta_reale_media = calcola_efficienza_reale(user_params["eta_nom"], user_params["gamma"], t_reale_k_media)
                    
                    if eta_reale_media > max_eta_assoluta:
                        max_eta_assoluta = eta_reale_media
                        
                    # Calcola l'efficienza del singolo pannello danneggiato usando la T. MAX
                    t_max_c = np.max(pixel_termici)
                    t_reale_k_max = applica_stefan_boltzmann(t_max_c, EPSILON_VETRO, t_amb)
                    eta_reale_max = calcola_efficienza_reale(user_params["eta_nom"], user_params["gamma"], t_reale_k_max)

            patch_dets.append({
                'class_id': int(results.class_id[k]), 'mask': mask, 'eta_reale': eta_reale_max,
                't_max': t_max_c if 't_max_c' in locals() else None
            })
            
        memoria_analisi.append({"patch": path_patch, "img": img_patch, "dets": patch_dets})

    print(f"  [+] Pannello di Riferimento (100% Salute) trovato: Efficienza Assoluta {max_eta_assoluta*100:.2f}%")

    # ---------------------------------------------------------
    # FASE 2: Calcolo Relativo, Economico e Generazione Output
    # ---------------------------------------------------------
    print(f"\n[*] FASE 2: Generazione Output Annotati e Report Economico...")
    csv_path = os.path.join(EFF_DIR, "report_efficienza.csv")
    csv_fields = [
        "File_Patch", "ID_Pannello", "T_Max_Apparente_C", "Eta_Assoluta_pct", 
        "Salute_Relativa_pct", "Potenza_Erogata_W", "Potenza_Persa_W", "Mancato_Guadagno_EUR_Anno"
    ]
    csv_rows = []

    for elemento in memoria_analisi:
        path_patch = elemento["patch"]
        img_patch = elemento["img"]
        dets = elemento["dets"]
        lista_draw = []

        for k, d in enumerate(dets):
            eta_reale = d['eta_reale']
            eta_rel_pct = None
            loss_euro = 0.0
            
            if eta_reale is not None:
                # Efficienza relativa basata sul picco termico vs Pannello migliore
                eta_rel_pct = (eta_reale / max_eta_assoluta) * 100.0 if max_eta_assoluta > 0 else 0.0
                
                # Calcolo Economico
                p_riferimento_w = G_IRR_STC * user_params["area"] * max_eta_assoluta
                p_attuale_w = G_IRR_STC * user_params["area"] * eta_reale
                
                p_persa_w = p_riferimento_w - p_attuale_w
                if p_persa_w < 0: p_persa_w = 0.0
                
                e_persa_kwh = (p_persa_w * esh) / 1000.0
                loss_euro = e_persa_kwh * GIORNI_UTIL * COSTO_KWH
                
                csv_rows.append({
                    "File_Patch": os.path.basename(path_patch), "ID_Pannello": k + 1,
                    "T_Max_Apparente_C": round(d['t_max'], 2) if d['t_max'] else 0.0,
                    "Eta_Assoluta_pct": round(eta_reale * 100, 2), "Salute_Relativa_pct": round(eta_rel_pct, 2),
                    "Potenza_Erogata_W": round(p_attuale_w, 2), "Potenza_Persa_W": round(p_persa_w, 2),
                    "Mancato_Guadagno_EUR_Anno": round(loss_euro, 2)
                })

            lista_draw.append({
                'class_id': d['class_id'], 'mask': d['mask'],
                'eta_rel_pct': eta_rel_pct
            })

        if lista_draw:
            annotata = disegna_risultati_efficienza(img_patch, lista_draw)
            out_name = os.path.basename(path_patch).replace("_patch", "_eff")
            cv2.imwrite(os.path.join(EFF_DIR, out_name), annotata)

    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[+] Report CSV salvato in: {csv_path}")
        
        # ---------------------------------------------------------
        # FASE 3: Generazione Report Tecnico in formato PDF
        # ---------------------------------------------------------
        pdf_path = os.path.join(EFF_DIR, "report_tecnico.pdf")
        genera_report_pdf(csv_rows, pdf_path, user_params)
        print(f"[+] Report PDF salvato in: {pdf_path}")

    print(f"[FINE] Analisi Completata. File salvati in: {EFF_DIR}")

if __name__ == "__main__":
    main()
