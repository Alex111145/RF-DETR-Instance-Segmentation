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
MAPPA_OUT_PATH     = os.path.join(OUTPUT_DIR, "mappa_efficienza_rgb.tif")
MAPPA_IR_OUT_PATH  = os.path.join(OUTPUT_DIR, "mappa_efficienza_ir.tif")
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

# Colori in ordine RGB (canvas caricato da rasterio in RGB)
COLOR_VERDE  = (0, 255, 0)
COLOR_GIALLO = (255, 255, 0)
COLOR_ROSSO  = (255, 0, 0)

# ==============================================================================
# FUNZIONI DI SUPPORTO E CORREZIONE DERIVA
# ==============================================================================
def get_pvgis_esh(lat, lon):
    try:
        url = f"https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?lat={lat}&lon={lon}&peakpower=1&loss=14&outputformat=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data["outputs"]["totals"]["fixed"]["E_y"] / 365.0
    except: return 3.18

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

def correggi_deriva_locale(rgb_canvas, m_cnt):
    """
    Estrae una regione attorno alla maschera calcolata, identifica il vero pannello
    fotovoltaico (area scura) e fa "snappare" (sposta) la maschera per sovrapporla correttamente.
    """
    pts = m_cnt.reshape(-1, 2)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    
    w = x_max - x_min
    h = y_max - y_min
    if w < 10 or h < 10:
        return m_cnt
        
    cx, cy = x_min + w//2, y_min + h//2
    
    # Crea una ROI allargata per catturare il pannello anche se è sfalsato
    pad_x, pad_y = int(w * 0.8), int(h * 0.8) 
    h_canvas, w_canvas = rgb_canvas.shape[:2]
    
    x1 = max(0, cx - pad_x)
    y1 = max(0, cy - pad_y)
    x2 = min(w_canvas, cx + pad_x)
    y2 = min(h_canvas, cy + pad_y)
    
    roi = rgb_canvas[y1:y2, x1:x2]
    if roi.size == 0: return m_cnt
    
    # Isolamento dei pannelli (solitamente rettangoli scuri con bordi chiari)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Operazione morfologica per chiudere i buchi dentro ai pannelli
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    area_attesa = w * h
    miglior_centro = None
    min_dist = float('inf')
    roi_cx_local, roi_cy_local = cx - x1, cy - y1
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Il contorno deve essere comparabile alla grandezza calcolata dalla maschera
        if area_attesa * 0.4 < area < area_attesa * 1.6:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX_local = int(M["m10"] / M["m00"])
                cY_local = int(M["m01"] / M["m00"])
                
                dist = np.sqrt((cX_local - roi_cx_local)**2 + (cY_local - roi_cy_local)**2)
                if dist < min_dist:
                    min_dist = dist
                    miglior_centro = (cX_local, cY_local)
    
    if miglior_centro is not None:
        # Trasla la maschera al centro reale trovato
        shift_x = miglior_centro[0] - roi_cx_local
        shift_y = miglior_centro[1] - roi_cy_local
        
        # Sicurezza: limitiamo lo scostamento per evitare "scatti" verso altri tetti/strutture
        if abs(shift_x) < w and abs(shift_y) < h:
            shift_array = np.array([shift_x, shift_y], dtype=np.int32)
            return m_cnt + shift_array
            
    return m_cnt

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
    cv2.putText(canvas, f"EUR {dati['perdita_euro']:.2f}", (col1_x + 40, y_cursor + 110), cv2.FONT_HERSHEY_SIMPLEX, 1.8, C_WARNING, 4)

    # Top 5 Moduli
    y_cursor += 220
    cv2.putText(canvas, "TOP 5 MODULI CRITICI (DA SOSTITUIRE)", (col1_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_DANGER, 2)
    y_cursor += 50
    for wp in dati['worst_panels']:
        txt = f"ID #{wp['id']} - Salute: {wp['eta']:.1f}% - Perdita Stimata: EUR {wp['euro_persi']:.2f}"
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

    # Inizializzazione Mosaici
    src_ir = rasterio.open(IR_MOSAIC)
    src_rgb = rasterio.open(RGB_MOSAIC)
    rgb_canvas = np.transpose(src_rgb.read([1,2,3]), (1,2,0)).copy()
    ir_canvas  = np.transpose(src_ir.read([1,2,3]), (1,2,0)).copy()

    # Calcolo Scala da GeoTIFF (IR vs RGB)
    res_x_ir, res_y_ir = abs(src_ir.transform.a), abs(src_ir.transform.e)
    res_x_rgb, res_y_rgb = abs(src_rgb.transform.a), abs(src_rgb.transform.e)

    scale_x = res_x_ir / res_x_rgb
    scale_y = res_y_ir / res_y_rgb
    
    print(f"[*] Fattori di scala calcolati da GeoTIFF -> X: {scale_x:.2f}, Y: {scale_y:.2f}")

    # Costruisce mappa pair_N -> (col_offset, row_offset) dai file di registrazione
    REG_DIR = os.path.join(OUTPUT_DIR, "registrazione_allineamento")
    pair_to_offset = {}
    for reg_file in glob.glob(os.path.join(REG_DIR, "pair*_tile_col_*_row_*.jpg")):
        m = re.search(r"pair(\d+)_tile_col_(\d+)_row_(\d+)", os.path.basename(reg_file))
        if m:
            pair_to_offset[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))

    esh = 3.18 # Default

    pannelli_globali = []

    for nome_patch, rilevamenti in tqdm(db_step4.items(), desc="Mappatura Geografica"):
        # Recupero offset
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

            # Rettangolo orientato (minAreaRect) -> 4 punti
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)

            colore = determina_colore(d_json["salute"], d_json["label"])

            # Rettangolo in spazio IR (coordinate pixel dirette - NON TOCCATO)
            xs_ir = box[:,0,0] + c_off
            ys_ir = box[:,0,1] + r_off
            ir_cnt = np.array([list(zip(xs_ir, ys_ir))], dtype=np.int32)

            # 1. Trova il centroide in coordinate pixel IR
            M_ir = cv2.moments(ir_cnt)
            cx_ir = int(M_ir["m10"]/M_ir["m00"]) if M_ir["m00"] != 0 else 0
            cy_ir = int(M_ir["m01"]/M_ir["m00"]) if M_ir["m00"] != 0 else 0

            # 2. Converti centroide IR -> Geografico -> Pixel RGB
            lon, lat = rasterio.transform.xy(src_ir.transform, cy_ir, cx_ir)
            xr_geo, yr_geo = transform_coords(src_ir.crs, src_rgb.crs, [lon], [lat])
            row_rgb, col_rgb = rasterio.transform.rowcol(src_rgb.transform, xr_geo[0], yr_geo[0])
            cx, cy = int(col_rgb), int(row_rgb)

            # 3. Scala e posiziona la maschera per l'RGB
            rgb_cnt_pts = []
            for x, y in zip(xs_ir, ys_ir):
                new_x = int((x - cx_ir) * scale_x + cx)
                new_y = int((y - cy_ir) * scale_y + cy)
                rgb_cnt_pts.append([new_x, new_y])
            
            m_cnt = np.array([rgb_cnt_pts], dtype=np.int32)

            # 4. CORREZIONE DERIVA (SNAPPING SULL'IMMAGINE)
            m_cnt = correggi_deriva_locale(rgb_canvas, m_cnt)

            # Ricalcolo centroide corretto (utile per label e filtri)
            M_rgb = cv2.moments(m_cnt)
            cx_corr = int(M_rgb["m10"]/M_rgb["m00"]) if M_rgb["m00"] != 0 else cx
            cy_corr = int(M_rgb["m01"]/M_rgb["m00"]) if M_rgb["m00"] != 0 else cy

            # Calcolo economico basato sull'efficienza del JSON
            p_max = 350 # Watt nominali stimati per pannello
            p_persa = p_max * (1 - (d_json["salute"]/100))
            euro_p = (p_persa / 1000) * esh * GIORNI_UTIL * COSTO_KWH

            pannelli_globali.append({
                'contour': m_cnt, 'centroid': (cx_corr, cy_corr),
                'ir_contour': ir_cnt, 'ir_centroid': (cx_ir, cy_ir),
                'eta': d_json["salute"],
                'color': colore,
                'euro_persi': euro_p, 'stato': d_json["label"]
            })

    # Filtro area: calcola area media IR e rimuove maschere sotto la media
    if pannelli_globali:
        aree_ir = np.array([cv2.contourArea(p['ir_contour']) for p in pannelli_globali], dtype=np.float32)
        area_media = float(np.mean(aree_ir))
        print(f"[*] Area media pannello (IR): {area_media:.0f} px²  —  totale prima del filtro: {len(pannelli_globali)}")
        pannelli_globali = [p for p in pannelli_globali if cv2.contourArea(p['ir_contour']) >= area_media]
        print(f"[*] Pannelli dopo filtro area: {len(pannelli_globali)}")

    # NMS (Rimozione duplicati)
    pannelli_globali.sort(key=lambda x: cv2.contourArea(x['ir_contour']), reverse=True)
    unici = []
    for p in pannelli_globali:
        if not any(cv2.pointPolygonTest(u['ir_contour'], p['ir_centroid'], False) >= 0 for u in unici):
            unici.append(p)

    # Ordinamento e ID
    unici.sort(key=lambda p: (p['centroid'][1], p['centroid'][0]))
    for i, p in enumerate(unici): p['id'] = i + 1

    # Disegno e Salvataggio CSV
    with open(CSV_UNICI, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Stato", "Salute_%", "Perdita_€_Anno"])
        for p in unici:
            label_txt = f"#{p['id']} {p['eta']:.0f}%"
            # RGB
            cv2.drawContours(rgb_canvas, p['contour'], -1, p['color'], 4)
            testo_centrato(rgb_canvas, label_txt, p['centroid'][0], p['centroid'][1], p['color'])
            # IR
            cv2.drawContours(ir_canvas, p['ir_contour'], -1, p['color'], 4)
            testo_centrato(ir_canvas, label_txt, p['ir_centroid'][0], p['ir_centroid'][1], p['color'])
            writer.writerow([p['id'], p['stato'], round(p['eta'], 2), round(p['euro_persi'], 2)])

    # Export Mappa RGB
    profile = src_rgb.profile.copy()
    profile.update(count=3)
    with rasterio.open(MAPPA_OUT_PATH, 'w', **profile) as dst:
        dst.write(np.transpose(rgb_canvas, (2,0,1)))

    # Export Mappa IR
    profile_ir = src_ir.profile.copy()
    profile_ir.update(count=3)
    with rasterio.open(MAPPA_IR_OUT_PATH, 'w', **profile_ir) as dst:
        dst.write(np.transpose(ir_canvas, (2,0,1)))

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
