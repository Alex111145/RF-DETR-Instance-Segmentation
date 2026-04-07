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
from rasterio.warp import transform as transform_coords
from PIL import Image, ExifTags
from datetime import datetime
from tqdm import tqdm

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAZIONE PERCORSI E COSTANTI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "risultati_finali")

# File generati dagli step precedenti
JSON_EFFICIENZA = os.path.join(OUTPUT_DIR, "efficienza_risultati", "efficienza_dati.json")
PATCH_IR_DIR    = os.path.join(BASE_DIR, "training_patches_ir")
IR_MOSAIC       = os.path.join(BASE_DIR, "ortomosaicoir.tif")
RGB_MOSAIC      = os.path.join(BASE_DIR, "ortomosaicorgb.tif")
WEIGHTS_PATH    = os.path.join(BASE_DIR, "weights.pt")

# Output finali
MAPPA_OUT_PATH  = os.path.join(OUTPUT_DIR, "mappa_efficienza_rgb.tif")
PDF_OUT_PATH    = os.path.join(OUTPUT_DIR, "report_tecnico.pdf")
CSV_UNICI       = os.path.join(OUTPUT_DIR, "report_pannelli_unici.csv")

# Parametri Economici (Standard se non diversamente specificato)
COSTO_KWH       = 0.40
GIORNI_UTIL      = 300
SOGLIA_AREA_PX  = 10000 

# Palette Colori A2A (BGR)
C_PRIMARY = (159, 91, 0)   
C_SUCCESS = (80, 175, 76)  
C_WARNING = (0, 152, 255)  
C_DANGER  = (54, 67, 244)  
C_TEXT    = (50, 50, 50)   
C_LIGHT   = (240, 245, 245) 

COLOR_VERDE  = (0, 255, 0)
COLOR_GIALLO = (0, 200, 255)
COLOR_ROSSO  = (0, 0, 255)

# ==============================================================================
# FUNZIONI DI SUPPORTO
# ==============================================================================
def get_pvgis_esh(lat, lon):
    try:
        url = f"https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?lat={lat}&lon={lon}&peakpower=1&loss=14&outputformat=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data["outputs"]["totals"]["fixed"]["E_y"] / 365.0
    except: return 3.18

def determina_colore_rgb(eta_rel_pct):
    if eta_rel_pct >= 90.0: return COLOR_VERDE
    elif eta_rel_pct >= 80.0: return COLOR_GIALLO
    else: return COLOR_ROSSO

def disegna_grafico_a_ciambella(canvas, cx, cy, r, eta_media_pct):
    cv2.circle(canvas, (cx+2, cy+5), r+2, (220, 220, 220), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r, (220, 220, 220), -1, cv2.LINE_AA)
    angolo = int(360 * (eta_media_pct / 100.0))
    cv2.ellipse(canvas, (cx, cy), (r, r), -90, 0, angolo, C_SUCCESS, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(r * 0.65), (255, 255, 255), -1, cv2.LINE_AA)
    label = f"{eta_media_pct:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
    cv2.putText(canvas, label, (cx - tw//2, cy + th//2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, C_TEXT, 4, cv2.LINE_AA)

# ==============================================================================
# GENERAZIONE PDF
# ==============================================================================
def genera_report_pdf_a2a(dati, pdf_path):
    w, h = 1240, 1754 
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.rectangle(canvas, (0, 0), (w, 160), C_PRIMARY, -1)
    cv2.putText(canvas, "REPORT EFFICIENZA IMPIANTO FOTOVOLTAICO", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    
    col1_x, col2_x = 80, 650
    y_cursor = 250
    
    # Sezione Statistiche
    cv2.putText(canvas, "STATISTICHE IMPIANTO", (col1_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_TEXT, 2)
    y_cursor += 60
    rows = [
        ("Totale Moduli Rilevati", str(dati['tot_pannelli']), C_TEXT),
        ("Moduli Ottimali (Verde)", str(dati['tot_sani']), C_SUCCESS),
        ("Moduli Degradati (Giallo)", str(dati['tot_degradati']), C_WARNING),
        ("Moduli Critici (Rosso)", str(dati['tot_rotti']), C_DANGER)
    ]
    for label, val, color in rows:
        cv2.putText(canvas, label, (col1_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
        cv2.putText(canvas, val, (col1_x + 350, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_cursor += 45

    # Grafico Ciambella
    disegna_grafico_a_ciambella(canvas, col2_x + 250, 400, 140, dati['eta_media_impianto'])

    # Perdite Economiche
    y_cursor = 700
    cv2.rectangle(canvas, (col1_x, y_cursor), (w-80, y_cursor+150), C_LIGHT, -1)
    cv2.rectangle(canvas, (col1_x, y_cursor), (col1_x+10, y_cursor+150), C_WARNING, -1)
    cv2.putText(canvas, "STIMA MANCATO GUADAGNO ANNUO", (col1_x + 40, y_cursor + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_TEXT, 2)
    cv2.putText(canvas, f"€ {dati['perdita_euro']:.2f}", (col1_x + 40, y_cursor + 110), cv2.FONT_HERSHEY_SIMPLEX, 1.8, C_WARNING, 4)

    # Top 5 Moduli
    y_cursor += 220
    cv2.putText(canvas, "TOP 5 MODULI CRITICI (DA SOSTITUIRE)", (col1_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_DANGER, 2)
    y_cursor += 50
    for wp in dati['worst_panels']:
        txt = f"ID #{wp['id']} - Salute: {wp['eta']:.1f}% - Perdita Stimata: € {wp['euro_persi']:.2f}"
        cv2.putText(canvas, txt, (col1_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1)
        y_cursor += 40

    # Footer
    cv2.rectangle(canvas, (0, h-60), (w, h), (40, 40, 40), -1)
    cv2.putText(canvas, "Generato tramite AI - Analisi Termografica Avanzata", (50, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(pdf_path, "PDF", resolution=150.0)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("\n" + "="*60 + "\n GENERAZIONE DIGITAL TWIN & REPORT FINALE \n" + "="*60)

    if not os.path.exists(JSON_EFFICIENZA):
        print("[!] ERRORE: Esegui prima lo Step 4.")
        return

    with open(JSON_EFFICIENZA, "r") as f:
        db_step4 = json.load(f)
    
    # Inizializzazione Mosaici
    src_ir = rasterio.open(IR_MOSAIC)
    src_rgb = rasterio.open(RGB_MOSAIC)
    rgb_canvas = np.transpose(src_rgb.read([1,2,3]), (1,2,0)).copy()

    from rfdetr import RFDETRSegLarge
    model = RFDETRSegLarge(pretrain_weights=WEIGHTS_PATH, num_classes=3)
    
    lista_orig = sorted(glob.glob(os.path.join(PATCH_IR_DIR, "*.jpg")))
    esh = 3.18 # Default

    pannelli_globali = []

    for nome_patch, rilevamenti in tqdm(db_step4["analisi"].items(), desc="Mappatura Geografica"):
        # Recupero offset
        idx = int(re.search(r"pair(\d+)_", nome_patch).group(1)) - 1
        m_off = re.search(r"tile_col_(\d+)_row_(\d+)", os.path.basename(lista_orig[idx]))
        c_off, r_off = int(m_off.group(1)), int(m_off.group(2))

        # Maschere e Georeferenziazione
        p_img = cv2.imread(os.path.join(OUTPUT_DIR, "pair", nome_patch))
        res = model.predict(Image.fromarray(cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)), threshold=0.6)
        
        if res and len(res.xyxy) > 0:
            for k, d_json in enumerate(rilevamenti):
                if k >= len(res.xyxy): break
                mask = res.mask[k]
                cnts, _ = cv2.findContours((mask.astype(np.uint8)*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts: continue
                c_big = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c_big) < SOGLIA_AREA_PX: continue

                # Trasformazione coordinate IR -> RGB
                xs, ys = c_big[:,0,0] + c_off, c_big[:,0,1] + r_off
                e, n = rasterio.transform.xy(src_ir.transform, ys, xs)
                xr, yr = transform_coords(src_ir.crs, src_rgb.crs, e, n)
                rs, cs = rasterio.transform.rowcol(src_rgb.transform, xr, yr)
                
                m_cnt = np.array([list(zip(cs, rs))], dtype=np.int32)
                M = cv2.moments(m_cnt)
                cx, cy = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"] != 0 else (0,0)

                # Calcolo economico basato sull'efficienza del JSON
                p_max = 350 # Watt nominali stimati per pannello
                p_persa = p_max * (1 - (d_json["salute_relativa"]/100))
                euro_p = (p_persa / 1000) * esh * GIORNI_UTIL * COSTO_KWH

                pannelli_globali.append({
                    'contour': m_cnt, 'centroid': (cx, cy),
                    'eta': d_json["salute_relativa"], 
                    'color': determina_colore_rgb(d_json["salute_relativa"]),
                    'euro_persi': euro_p, 'stato': d_json["stato"]
                })

    # NMS (Rimozione duplicati)
    pannelli_globali.sort(key=lambda x: cv2.contourArea(x['contour']), reverse=True)
    unici = []
    for p in pannelli_globali:
        if not any(cv2.pointPolygonTest(u['contour'], p['centroid'], False) >= 0 for u in unici):
            unici.append(p)
    
    # Ordinamento e ID
    unici.sort(key=lambda p: (p['centroid'][1], p['centroid'][0]))
    for i, p in enumerate(unici): p['id'] = i + 1

    # Disegno e Salvataggio CSV
    with open(CSV_UNICI, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Stato", "Salute_%", "Perdita_€_Anno"])
        for p in unici:
            cv2.drawContours(rgb_canvas, p['contour'], -1, p['color'], 4)
            cv2.putText(rgb_canvas, f"#{p['id']}", p['centroid'], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            writer.writerow([p['id'], p['stato'], round(p['eta'], 2), round(p['euro_persi'], 2)])

    # Export Mappa
    with rasterio.open(MAPPA_OUT_PATH, 'w', **src_rgb.profile) as dst:
        dst.write(np.transpose(rgb_canvas, (2,0,1)))

    # Generazione Report PDF
    dati_rep = {
        'tot_pannelli': len(unici),
        'tot_sani': len([p for p in unici if p['eta'] >= 90]),
        'tot_degradati': len([p for p in unici if 80 <= p['eta'] < 90]),
        'tot_rotti': len([p for p in unici if p['eta'] < 80]),
        'eta_media_impianto': np.mean([p['eta'] for p in unici]) if unici else 0,
        'perdita_euro': sum([p['euro_persi'] for p in unici]),
        'worst_panels': sorted(unici, key=lambda x: x['eta'])[:5]
    }
    genera_report_pdf_a2a(dati_rep, PDF_OUT_PATH)

    print(f"\n[FINE] Digital Twin: {MAPPA_OUT_PATH}\n[FINE] Report PDF: {PDF_OUT_PATH}")

if __name__ == "__main__":
    main()
