#!/usr/bin/env python3
import os
import glob
import re
import csv
import json
import warnings
import cv2
import numpy as np
import rasterio
import urllib.request
from PIL import Image, ExifTags
from datetime import datetime
from tqdm import tqdm

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "risultati_finali")


JSON_EFFICIENZA = os.path.join(OUTPUT_DIR, "efficienza_risultati", "efficienza_dati.json")
PATCH_IR_DIR    = os.path.join(BASE_DIR, "training_patches_ir")
IR_MOSAIC       = os.path.join(BASE_DIR, "ortomosaicoir.tif")
WEIGHTS_PATH    = os.path.join(BASE_DIR, "weights.pt")


MAPPA_IR_OUT_PATH        = os.path.join(OUTPUT_DIR, "mappa_efficienza_ir.tif")
MAPPA_DIFETTOSI_OUT_PATH = os.path.join(OUTPUT_DIR, "mappa_difettosi_ir.tif")
PDF_OUT_PATH             = os.path.join(OUTPUT_DIR, "report_tecnico.pdf")
CSV_UNICI                = os.path.join(OUTPUT_DIR, "report_pannelli_unici.csv")


COSTO_KWH_ACQUISTO = 0.40   # Quota autoconsumata
COSTO_KWH_VENDITA  = 0.10   # Quota immessa in rete (vendita GSE)
PCT_AUTOCONSUMO    = 0.40   # 40% autoconsumo, 60% immissione in rete
GIORNI_UTIL        = 300
SOGLIA_AREA_PX     = 10000 


C_PRIMARY = (159, 91, 0)   
C_SUCCESS = (80, 175, 76)  
C_WARNING = (0, 152, 255)
C_AMBER   = (0, 190, 255)
C_DANGER  = (54, 67, 244)  
C_TEXT    = (50, 50, 50)   
C_LIGHT   = (240, 245, 245) 


COLOR_VERDE  = (0, 255, 0)
COLOR_GIALLO = (255, 255, 0)
COLOR_ROSSO  = (255, 0, 0)


def estrai_gps_da_drone():
    """Legge la prima foto drone disponibile ed estrae le coordinate GPS dall'EXIF."""
    foto_dir = os.path.join(BASE_DIR, "foto_drone")
    if not os.path.isdir(foto_dir):
        return None, None
    for fname in sorted(os.listdir(foto_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg')):
            continue
        try:
            img = Image.open(os.path.join(foto_dir, fname))
            exif_data = img._getexif()
            if not exif_data:
                continue
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
            gps_raw = tags.get("GPSInfo", {})
            gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
            if "GPSLatitude" not in gps:
                continue
            def conv(coord, ref):
                d, m, s = coord
                v = float(d) + float(m) / 60 + float(s) / 3600
                return -v if ref in ('S', 'W') else v
            lat = conv(gps["GPSLatitude"],  gps["GPSLatitudeRef"])
            lon = conv(gps["GPSLongitude"], gps["GPSLongitudeRef"])
            return lat, lon
        except:
            continue
    return None, None


def get_pvgis_data(lat, lon):
    """
    Interroga PVGIS e restituisce (esh, giorni_utili).
    - esh          : ore equivalenti di pieno sole medie giornaliere (E_y / 365)
    - giorni_utili : giorni/anno con irraggiamento medio >= 1.5 kWh/m²/giorno
    """
    GIORNI_MESE = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    SOGLIA_HID  = 1.5   
    try:
        url = (f"https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?"
               f"lat={lat}&lon={lon}&peakpower=1&loss=14&outputformat=json")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        esh = data["outputs"]["totals"]["fixed"]["E_y"] / 365.0

        giorni_utili = 0
        for m in data["outputs"]["monthly"]["fixed"]:
            if m["H(i)_d"] >= SOGLIA_HID:
                giorni_utili += GIORNI_MESE[m["month"] - 1]

        return esh, giorni_utili
    except:
        return 3.18, 300

def testo_centrato(canvas, testo, cx, cy, colore, scala=0.55, spessore=1):
    (tw, th), _ = cv2.getTextSize(testo, cv2.FONT_HERSHEY_SIMPLEX, scala, spessore)
    tx, ty = cx - tw // 2, cy + th // 2
    cv2.putText(canvas, testo, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scala, (0,0,0), spessore + 2, cv2.LINE_AA)
    cv2.putText(canvas, testo, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scala, colore, spessore, cv2.LINE_AA)

def determina_colore(salute_pct, label):
    if label == "DIFETTOSO":
        return COLOR_ROSSO
    elif salute_pct < 90.0:
        return COLOR_GIALLO
    else:
        return COLOR_VERDE

def disegna_grafico_a_ciambella(canvas, cx, cy, r, eta_media_pct):
    cv2.circle(canvas, (cx+2, cy+5), r+2, (220, 220, 220), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r, (220, 220, 220), -1, cv2.LINE_AA)
    angolo = int(360 * (eta_media_pct / 100.0))
    cv2.ellipse(canvas, (cx, cy), (r, r), -90, 0, angolo, C_SUCCESS, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(r * 0.65), (255, 255, 255), -1, cv2.LINE_AA)
    label = f"{eta_media_pct:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
    cv2.putText(canvas, label, (cx - tw//2, cy + th//2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, C_TEXT, 4, cv2.LINE_AA)


def genera_report_pdf_a2a(dati, pdf_path):
    w, h = 1240, 1754 
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.rectangle(canvas, (0, 0), (w, 160), C_PRIMARY, -1)
    cv2.putText(canvas, "REPORT EFFICIENZA IMPIANTO FOTOVOLTAICO", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    
    col1_x, col2_x = 80, 650
    y_cursor = 250
    

    cv2.putText(canvas, "STATISTICHE IMPIANTO", (col1_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_TEXT, 3)
    y_cursor += 70
    rows = [
        ("Totale Moduli Rilevati", str(dati['tot_pannelli']), C_TEXT),
        ("Moduli Ottimali (Verde)", str(dati['tot_sani']), C_SUCCESS),
        ("Moduli Sporchi (Giallo)", str(dati['tot_degradati']), C_WARNING),
        ("Moduli Critici (Rosso)", str(dati['tot_rotti']), C_DANGER)
    ]
    for label, val, color in rows:
        cv2.putText(canvas, label, (col1_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
        cv2.putText(canvas, val, (col1_x + 400, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_cursor += 50


    disegna_grafico_a_ciambella(canvas, col2_x + 250, 420, 150, dati['eta_media_impianto'])


    y_cursor = 730
    box_h = 160
    cv2.rectangle(canvas, (col1_x, y_cursor), (w-80, y_cursor+box_h), C_LIGHT, -1)
    cv2.rectangle(canvas, (col1_x, y_cursor), (col1_x+10, y_cursor+box_h), C_DANGER, -1) 
    
    cv2.putText(canvas, "STIMA MANCATO GUADAGNO ANNUO", (col1_x + 40, y_cursor + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_TEXT, 2)
    

    cv2.putText(canvas, "Perdita Totale (Moduli Difettosi):", (col1_x + 40, y_cursor + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_DANGER, 2)
    

    testo_importo = f"EUR {dati['perdita_euro_totale']:.2f}"
    (text_width, _), _ = cv2.getTextSize(testo_importo, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    x_right_align = (w - 80) - text_width - 40 
    
    cv2.putText(canvas, testo_importo, (x_right_align, y_cursor + 115), cv2.FONT_HERSHEY_SIMPLEX, 1.4, C_DANGER, 3)

 
    y_cursor += box_h + 80
    cv2.putText(canvas, "TOP 5 MODULI CRITICI (DA SOSTITUIRE)", (col1_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_DANGER, 2)
    y_cursor += 50
    for wp in dati['worst_panels']:
        txt = f"ID #{wp['id']} - Salute: {wp['eta']:.1f}% - kWh persi/anno: {wp['kwh_persi']:.1f} - EUR {wp['euro_persi']:.2f}"
        cv2.putText(canvas, txt, (col1_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_TEXT, 1)
        y_cursor += 40


    cv2.rectangle(canvas, (0, h-60), (w, h), (40, 40, 40), -1)
    cv2.putText(canvas, "Generato tramite AI - Analisi Termografica Avanzata", (50, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(pdf_path, "PDF", resolution=150.0)

def main():
    print("\n" + "="*60 + "\n GENERAZIONE DIGITAL TWIN & REPORT FINALE \n" + "="*60)

    JSON_TERMICA = os.path.join(OUTPUT_DIR, "analisi_termica", "analisi_dati.json")
    if not os.path.exists(JSON_TERMICA):
        print("[!] ERRORE: Esegui prima lo Step 3.")
        return
    if not os.path.exists(JSON_EFFICIENZA):
        print("[!] ERRORE: Esegui prima lo Step 4.")
        return

    with open(JSON_TERMICA, "r") as f:
        db_step3 = json.load(f)
    with open(JSON_EFFICIENZA, "r") as f:
        db_step4 = json.load(f)

 
    src_ir = rasterio.open(IR_MOSAIC)
    ir_canvas  = np.transpose(src_ir.read([1,2,3]), (1,2,0)).copy()

    REG_DIR = os.path.join(OUTPUT_DIR, "registrazione_allineamento")
    pair_to_offset = {}
    for reg_file in glob.glob(os.path.join(REG_DIR, "pair*_tile_col_*_row_*.jpg")):
        m = re.search(r"pair(\d+)_tile_col_(\d+)_row_(\d+)", os.path.basename(reg_file))
        if m:
            pair_to_offset[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))

    lat_drone, lon_drone = estrai_gps_da_drone()
    if lat_drone is not None:
        print(f"[*] Coordinate volo: {lat_drone:.5f}°N, {lon_drone:.5f}°E — interrogo PVGIS...")
        esh, giorni_utili = get_pvgis_data(lat_drone, lon_drone)
        print(f"[*] ESH da PVGIS       : {esh:.2f} ore/giorno")
        print(f"[*] Giorni utili PVGIS : {giorni_utili} giorni/anno (mesi con H(i)_d >= 1.5 kWh/m²)")
    else:
        esh, giorni_utili = 3.18, 300
        print(f"[!] GPS non trovato nelle foto drone. Uso valori default: ESH={esh}, giorni={giorni_utili}")

    pannelli_globali = []

    for nome_patch, rilevamenti in tqdm(db_step4.items(), desc="Mappatura Geografica"):

        pair_num = int(re.search(r"pair(\d+)_", nome_patch).group(1))
        if pair_num not in pair_to_offset:
            continue
        c_off, r_off = pair_to_offset[pair_num]

        pannelli_step3 = db_step3.get(nome_patch, [])

        for k, d_json in enumerate(rilevamenti):
            if k >= len(pannelli_step3): break
            if d_json["salute"] == 0: continue
            points = pannelli_step3[k].get("points")
            if not points: continue
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            if len(pts) < 3: continue

            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)

            colore = determina_colore(d_json["salute"], d_json["label"])

            xs_ir = box[:,0,0] + c_off
            ys_ir = box[:,0,1] + r_off
            ir_cnt = np.array([list(zip(xs_ir, ys_ir))], dtype=np.int32)

            M_ir = cv2.moments(ir_cnt)
            cx_ir = int(M_ir["m10"]/M_ir["m00"]) if M_ir["m00"] != 0 else 0
            cy_ir = int(M_ir["m01"]/M_ir["m00"]) if M_ir["m00"] != 0 else 0

            if d_json["label"] == "DIFETTOSO":
                p_max = 350 
                p_persa = p_max * (1 - (d_json["salute"]/100))
                kwh_persi_anno = (p_persa / 1000) * esh * giorni_utili
                
              
                quota_autoconsumo = kwh_persi_anno * PCT_AUTOCONSUMO
                quota_rete = kwh_persi_anno * (1.0 - PCT_AUTOCONSUMO)
                
      
                euro_p = (quota_autoconsumo * COSTO_KWH_ACQUISTO) + (quota_rete * COSTO_KWH_VENDITA)
            else:
                kwh_persi_anno = 0.0
                euro_p = 0.0

            pannelli_globali.append({
                'ir_contour': ir_cnt, 'ir_centroid': (cx_ir, cy_ir),
                'eta': d_json["salute"],
                'color': colore,
                'kwh_persi': kwh_persi_anno,
                'euro_persi': euro_p, 'stato': d_json["label"]
            })


    if pannelli_globali:
        aree_ir = np.array([cv2.contourArea(p['ir_contour']) for p in pannelli_globali], dtype=np.float32)
        area_media = float(np.mean(aree_ir))
        print(f"[*] Area media pannello (IR): {area_media:.0f} px²  —  totale prima del filtro: {len(pannelli_globali)}")
        pannelli_globali = [p for p in pannelli_globali if cv2.contourArea(p['ir_contour']) >= area_media]
        print(f"[*] Pannelli dopo filtro area: {len(pannelli_globali)}")

  
    pannelli_globali.sort(key=lambda x: cv2.contourArea(x['ir_contour']), reverse=True)
    unici = []
    for p in pannelli_globali:
        if not any(cv2.pointPolygonTest(u['ir_contour'], p['ir_centroid'], False) >= 0 for u in unici):
            unici.append(p)
    unici.sort(key=lambda p: (p['ir_centroid'][1], p['ir_centroid'][0]))
    for i, p in enumerate(unici): p['id'] = i + 1

 
    with open(CSV_UNICI, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Stato", "Salute_%", "kWh_persi_anno", "Perdita_€_Anno"])
        for p in unici:
            label_txt = f"#{p['id']} {p['eta']:.0f}%"
           
            cv2.drawContours(ir_canvas, p['ir_contour'], -1, p['color'], 4)
            testo_centrato(ir_canvas, label_txt, p['ir_centroid'][0], p['ir_centroid'][1], p['color'])
            
            is_problematic = p['stato'] == "DIFETTOSO"
            kwh_csv = round(p['kwh_persi'], 2) if is_problematic else 0.0
            euro_csv = round(p['euro_persi'], 2) if is_problematic else 0.0
            
            writer.writerow([p['id'], p['stato'], round(p['eta'], 2), kwh_csv, euro_csv])

 
    profile_ir = src_ir.profile.copy()
    profile_ir.update(count=3)
    with rasterio.open(MAPPA_IR_OUT_PATH, 'w', **profile_ir) as dst:
        dst.write(np.transpose(ir_canvas, (2,0,1)))

    difettosi_canvas = np.transpose(src_ir.read([1,2,3]), (1,2,0)).copy()
    
  
    for p in unici:
        if p['stato'] == "DIFETTOSO":
            label_txt = f"#{p['id']} {p['eta']:.0f}%"
            cv2.drawContours(difettosi_canvas, [p['ir_contour']], -1, p['color'], 4)
            testo_centrato(difettosi_canvas, label_txt, p['ir_centroid'][0], p['ir_centroid'][1], p['color'])
            
    profile_dif = src_ir.profile.copy()
    profile_dif.update(count=3)
    with rasterio.open(MAPPA_DIFETTOSI_OUT_PATH, 'w', **profile_dif) as dst:
        dst.write(np.transpose(difettosi_canvas, (2,0,1)))

    perdita_difettosi = sum(p['euro_persi'] for p in unici if p['stato'] == "DIFETTOSO")
    
    dati_rep = {
        'tot_pannelli': len(unici),
        'tot_sani':      len([p for p in unici if p['stato'] != "DIFETTOSO" and p['eta'] >= 90]),
        'tot_degradati': len([p for p in unici if p['stato'] != "DIFETTOSO" and p['eta'] < 90]),
        'tot_rotti':     len([p for p in unici if p['stato'] == "DIFETTOSO"]),
        'eta_media_impianto': np.mean([p['eta'] for p in unici]) if unici else 0,
        'perdita_euro_difettosi': perdita_difettosi,
        'perdita_euro_totale':    perdita_difettosi,
        'worst_panels': sorted([p for p in unici if p['stato'] == "DIFETTOSO"], key=lambda x: x['eta'])[:5]
    }
    genera_report_pdf_a2a(dati_rep, PDF_OUT_PATH)

    print(f"\n[FINE] Mappa Efficienza IR: {MAPPA_IR_OUT_PATH}")
    print(f"[FINE] Mappa Difettosi: {MAPPA_DIFETTOSI_OUT_PATH}")
    print(f"[FINE] Report PDF: {PDF_OUT_PATH}")

if __name__ == "__main__":
    main()
