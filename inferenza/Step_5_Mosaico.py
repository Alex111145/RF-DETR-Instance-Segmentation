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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "risultati_finali")

JSON_EFFICIENZA = os.path.join(OUTPUT_DIR, "efficienza_risultati", "efficienza_dati.json")
PATCH_IR_DIR    = os.path.join(BASE_DIR, "training_patches_ir")
IR_MOSAIC       = os.path.join(BASE_DIR, "ortomosaicoir.tif")
RGB_MOSAIC      = os.path.join(BASE_DIR, "ortomosaicorgb.tif")

MAPPA_OUT_PATH        = os.path.join(OUTPUT_DIR, "mappa_efficienza_rgb.tif")
MAPPA_IR_OUT_PATH     = os.path.join(OUTPUT_DIR, "mappa_efficienza_ir.tif")
MAPPA_IR_DIFETTOSI    = os.path.join(OUTPUT_DIR, "mappa_ir_difettosi.tif")
PDF_OUT_PATH    = os.path.join(OUTPUT_DIR, "report_tecnico.pdf")
CSV_UNICI       = os.path.join(OUTPUT_DIR, "report_pannelli_unici.csv")

COSTO_KWH       = 0.40
GIORNI_UTIL      = 300

C_PRIMARY = (159, 91, 0)   
C_SUCCESS = (80, 175, 76)  
C_WARNING = (0, 152, 255)
C_DANGER  = (54, 67, 244)  
C_TEXT    = (50, 50, 50)   
C_LIGHT   = (240, 245, 245) 

COLOR_VERDE  = (0, 255, 0)
COLOR_GIALLO = (255, 255, 0)
COLOR_ROSSO  = (255, 0, 0)


def estrai_gps_da_drone():
    foto_dir = os.path.join(BASE_DIR, "foto_drone")
    if not os.path.isdir(foto_dir):
        return None, None
    for fname in sorted(os.listdir(foto_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg')): continue
        try:
            img = Image.open(os.path.join(foto_dir, fname))
            exif_data = img._getexif()
            if not exif_data: continue
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
            gps_raw = tags.get("GPSInfo", {})
            gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
            if "GPSLatitude" not in gps: continue
            def conv(coord, ref):
                d, m, s = coord
                v = float(d) + float(m) / 60 + float(s) / 3600
                return -v if ref in ('S', 'W') else v
            lat = conv(gps["GPSLatitude"],  gps["GPSLatitudeRef"])
            lon = conv(gps["GPSLongitude"], gps["GPSLongitudeRef"])
            return lat, lon
        except: continue
    return None, None

def get_pvgis_data(lat, lon):
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

ANGLE_GAP     = 15.0   
SPATIAL_GAP   = 800    
MIN_ZONE_SIZE = 12      

def roof_zone_mapping(pair_to_offset, db_pannelli):
    panels_data = [] 
    
    for img_name, panels in db_pannelli.items():
        m = re.search(r"pair(\d+)_", img_name)
        if not m: continue
        pair_num = int(m.group(1))
        if pair_num not in pair_to_offset: continue
            
        col_off, row_off = pair_to_offset[pair_num]
        
        for idx, panel in enumerate(panels):
            pts = panel.get("points")
            if not pts or len(pts) < 3: continue
                
            pts_arr = np.array(pts, dtype=np.float32)
            rect = cv2.minAreaRect(pts_arr)
            a = rect[2]
            w, h = rect[1]
            if w < h: a += 90.0
            a = a % 180.0
            
            M = cv2.moments(pts_arr)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + col_off
                cy = int(M["m01"] / M["m00"]) + row_off
            else:
                cx, cy = int(pts_arr[0][0]) + col_off, int(pts_arr[0][1]) + row_off
                
            panels_data.append({
                "id": (img_name, idx),
                "angle": a,
                "cx": cx,
                "cy": cy,
            })

    if not panels_data: return {}

    n = len(panels_data)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in tqdm(range(n), desc="Mappatura falde (Fase 1)"):
        for j in range(i + 1, n):
            pi, pj = panels_data[i], panels_data[j]
            ad = abs(pi["angle"] - pj["angle"])
            ad = min(ad, 180.0 - ad)
            if ad > ANGLE_GAP: continue
            dist_sq = (pi["cx"] - pj["cx"])**2 + (pi["cy"] - pj["cy"])**2
            if dist_sq <= SPATIAL_GAP**2:
                union(i, j)

    from collections import defaultdict
    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(panels_data[i])
        
    large_groups = []
    small_groups = []
    
    for comp in comps.values():
        if len(comp) >= MIN_ZONE_SIZE:
            large_groups.append(comp)
        else:
            small_groups.append(comp)

    if large_groups and small_groups:
        for sg in small_groups:
            best_dist = float('inf')
            best_idx = None
            
            for i, lg in enumerate(large_groups):
                for p_small in sg:
                    for p_large in lg:
                        d_sq = (p_small["cx"] - p_large["cx"])**2 + (p_small["cy"] - p_large["cy"])**2
                        if d_sq < best_dist:
                            best_dist = d_sq
                            best_idx = i
                            
            if best_idx is not None and best_dist <= (SPATIAL_GAP * 1.5)**2:
                large_groups[best_idx].extend(sg)
            else:
                large_groups.append(sg)
    elif not large_groups:
        large_groups = list(comps.values())

    sorted_comps = sorted(large_groups, key=len, reverse=True)

    result = {}
    for zone_id, comp in enumerate(sorted_comps, 1):
        for p in comp:
            result[p["id"]] = zone_id

    return result

def testo_centrato(canvas, testo, cx, cy, colore, scala=0.55, spessore=1):
    (tw, th), _ = cv2.getTextSize(testo, cv2.FONT_HERSHEY_SIMPLEX, scala, spessore)
    tx, ty = cx - tw // 2, cy + th // 2
    cv2.putText(canvas, testo, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scala, (0,0,0), spessore + 2, cv2.LINE_AA)
    cv2.putText(canvas, testo, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scala, colore, spessore, cv2.LINE_AA)

def determina_colore(salute_pct, label):
    if label == "DIFETTOSO": return COLOR_ROSSO
    elif salute_pct < 90.0: return COLOR_GIALLO
    else: return COLOR_VERDE

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
    pts = m_cnt.reshape(-1, 2)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    w = x_max - x_min
    h = y_max - y_min
    if w < 10 or h < 10: return m_cnt
    cx, cy = x_min + w//2, y_min + h//2
    pad_x, pad_y = int(w * 0.8), int(h * 0.8) 
    h_canvas, w_canvas = rgb_canvas.shape[:2]
    x1, y1 = max(0, cx - pad_x), max(0, cy - pad_y)
    x2, y2 = min(w_canvas, cx + pad_x), min(h_canvas, cy + pad_y)
    roi = rgb_canvas[y1:y2, x1:x2]
    if roi.size == 0: return m_cnt
    
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    area_attesa = w * h
    miglior_centro = None
    min_dist = float('inf')
    roi_cx_local, roi_cy_local = cx - x1, cy - y1
    for cnt in contours:
        area = cv2.contourArea(cnt)
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
        shift_x = miglior_centro[0] - roi_cx_local
        shift_y = miglior_centro[1] - roi_cy_local
        if abs(shift_x) < w and abs(shift_y) < h:
            return m_cnt + np.array([shift_x, shift_y], dtype=np.int32)
    return m_cnt

def genera_report_pdf_a2a(dati, pdf_path):
    w, h = 1240, 1754 
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.rectangle(canvas, (0, 0), (w, 160), C_PRIMARY, -1)
    cv2.putText(canvas, "REPORT EFFICIENZA IMPIANTO FOTOVOLTAICO", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    
    col1_x, col2_x = 80, 650
    y_cursor = 250
    
    cv2.putText(canvas, "STATISTICHE IMPIANTO", (col1_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_TEXT, 2)
    y_cursor += 60
    rows = [
        ("Totale Moduli Rilevati", str(dati['tot_pannelli']), C_TEXT),
        ("Moduli Ottimali (Verde)", str(dati['tot_sani']), C_SUCCESS),
        ("Moduli Sporchi (Giallo)", str(dati['tot_degradati']), C_WARNING),
        ("Moduli Critici (Rosso)", str(dati['tot_rotti']), C_DANGER)
    ]
    for label, val, color in rows:
        cv2.putText(canvas, label, (col1_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
        cv2.putText(canvas, val, (col1_x + 350, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_cursor += 45

    disegna_grafico_a_ciambella(canvas, col2_x + 250, 400, 140, dati['eta_media_impianto'])

    y_cursor = 700
    box_h = 300
    cv2.rectangle(canvas, (col1_x, y_cursor), (w-80, y_cursor+box_h), C_LIGHT, -1)
    cv2.rectangle(canvas, (col1_x, y_cursor), (col1_x+10, y_cursor+box_h), C_WARNING, -1)
    cv2.putText(canvas, "STIMA MANCATO GUADAGNO ANNUO", (col1_x + 40, y_cursor + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_TEXT, 2)

    cv2.putText(canvas, "kWh/anno persi", (col1_x + 370, y_cursor + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 1)
    cv2.putText(canvas, "EUR/anno *",      (col1_x + 530, y_cursor + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 1)

    cv2.putText(canvas, "Pannelli Difettosi (Rosso):", (col1_x + 40, y_cursor + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_DANGER, 1)
    cv2.putText(canvas, f"{dati['kwh_difettosi']:.1f}",          (col1_x + 370, y_cursor + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_DANGER, 2)
    cv2.putText(canvas, f"{dati['perdita_euro_difettosi']:.2f}", (col1_x + 530, y_cursor + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_DANGER, 2)

    cv2.putText(canvas, "Pannelli Sporchi (Giallo):", (col1_x + 40, y_cursor + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WARNING, 1)
    cv2.putText(canvas, f"{dati['kwh_sporchi']:.1f}",           (col1_x + 370, y_cursor + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_WARNING, 2)
    cv2.putText(canvas, f"{dati['perdita_euro_sporchi']:.2f}",  (col1_x + 530, y_cursor + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_WARNING, 2)

    cv2.putText(canvas, "* prezzo energia: 0.40 EUR/kWh", (col1_x + 40, y_cursor + 218), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_TEXT, 1)

    sep_x = 760
    cv2.line(canvas, (sep_x, y_cursor + 20), (sep_x, y_cursor + box_h - 20), (180, 180, 180), 2)
    tot_x = sep_x + 30
    cv2.putText(canvas, "TOTALE MANCATO GUADAGNO", (tot_x, y_cursor + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_TEXT, 1)
    cv2.putText(canvas, f"EUR {dati['perdita_euro_totale']:.2f}", (tot_x, y_cursor + 210), cv2.FONT_HERSHEY_SIMPLEX, 1.5, C_WARNING, 4)

    y_cursor += 370
    cv2.putText(canvas, "TOP 5 MODULI CRITICI (DA SOSTITUIRE)", (col1_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_DANGER, 2)
    y_cursor += 50
    for wp in dati['worst_panels']:
        txt = f"ID #{wp['id']} - Salute: {wp['eta']:.1f}% - kWh persi/anno: {wp['kwh_persi']:.1f} - EUR {wp['euro_persi']:.2f}/anno"
        cv2.putText(canvas, txt, (col1_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1)
        y_cursor += 40

    cv2.rectangle(canvas, (0, h-60), (w, h), (40, 40, 40), -1)
    cv2.putText(canvas, "Generato tramite AI - Analisi Termografica Avanzata", (50, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(pdf_path, "PDF", resolution=150.0)

def main():
    print("\n" + "="*60 + "\n GENERAZIONE DIGITAL TWIN & REPORT FINALE \n" + "="*60)

    JSON_TERMICA = os.path.join(OUTPUT_DIR, "analisi_termica", "analisi_dati.json")
    if not os.path.exists(JSON_TERMICA) or not os.path.exists(JSON_EFFICIENZA):
        print("[!] ERRORE: File JSON mancanti. Esegui gli step precedenti.")
        return

    with open(JSON_TERMICA, "r") as f: db_step3 = json.load(f)
    with open(JSON_EFFICIENZA, "r") as f: db_step4 = json.load(f)

    src_ir = rasterio.open(IR_MOSAIC)
    src_rgb = rasterio.open(RGB_MOSAIC)
    rgb_canvas = np.transpose(src_rgb.read([1,2,3]), (1,2,0)).copy()
    ir_canvas  = np.transpose(src_ir.read([1,2,3]), (1,2,0)).copy()

    scale_x = abs(src_ir.transform.a) / abs(src_rgb.transform.a)
    scale_y = abs(src_ir.transform.e) / abs(src_rgb.transform.e)
    
    REG_DIR = os.path.join(OUTPUT_DIR, "registrazione_allineamento")
    pair_to_offset = {}
    for f in glob.glob(os.path.join(REG_DIR, "pair*_tile_col_*_row_*.jpg")):
        m = re.search(r"pair(\d+)_tile_col_(\d+)_row_(\d+)", os.path.basename(f))
        if m: pair_to_offset[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))

    pair_to_zone = roof_zone_mapping(pair_to_offset, db_step3)

    lat_drone, lon_drone = estrai_gps_da_drone()
    if lat_drone is not None:
        esh, giorni_utili = get_pvgis_data(lat_drone, lon_drone)
    else:
        esh, giorni_utili = 3.18, 300

    pannelli_globali = []
    for nome_patch, rilevamenti in tqdm(db_step4.items(), desc="Mappatura Geografica"):
        pair_num = int(re.search(r"pair(\d+)_", nome_patch).group(1))
        if pair_num not in pair_to_offset: continue
        c_off, r_off = pair_to_offset[pair_num]
        pannelli_step3 = db_step3.get(nome_patch, [])

        for k, d_json in enumerate(rilevamenti):
            zone_id = pair_to_zone.get((nome_patch, k), 1)

            if k >= len(pannelli_step3) or d_json["salute"] == 0: continue
            points = pannelli_step3[k].get("points")
            if not points: continue
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            if len(pts) < 3: continue

            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)
            colore = determina_colore(d_json["salute"], d_json["label"])

            xs_ir, ys_ir = box[:,0,0] + c_off, box[:,0,1] + r_off
            ir_cnt = np.array([list(zip(xs_ir, ys_ir))], dtype=np.int32)

            M_ir = cv2.moments(ir_cnt)
            cx_ir = int(M_ir["m10"]/M_ir["m00"]) if M_ir["m00"] != 0 else 0
            cy_ir = int(M_ir["m01"]/M_ir["m00"]) if M_ir["m00"] != 0 else 0

            lon, lat = rasterio.transform.xy(src_ir.transform, cy_ir, cx_ir)
            xr_geo, yr_geo = transform_coords(src_ir.crs, src_rgb.crs, [lon], [lat])
            row_rgb, col_rgb = rasterio.transform.rowcol(src_rgb.transform, xr_geo[0], yr_geo[0])
            cx, cy = int(col_rgb), int(row_rgb)

            rgb_cnt_pts = [[int((x - cx_ir)*scale_x + cx), int((y - cy_ir)*scale_y + cy)] for x, y in zip(xs_ir, ys_ir)]
            m_cnt = np.array([rgb_cnt_pts], dtype=np.int32)
            m_cnt = correggi_deriva_locale(rgb_canvas, m_cnt)

            M_rgb = cv2.moments(m_cnt)
            cx_corr = int(M_rgb["m10"]/M_rgb["m00"]) if M_rgb["m00"] != 0 else cx
            cy_corr = int(M_rgb["m01"]/M_rgb["m00"]) if M_rgb["m00"] != 0 else cy

            p_persa = 350 * (1 - (d_json["salute"]/100))
            kwh_persi_anno = (p_persa / 1000) * esh * giorni_utili
            euro_p = kwh_persi_anno * COSTO_KWH

            pannelli_globali.append({
                'contour': m_cnt, 'centroid': (cx_corr, cy_corr),
                'ir_contour': ir_cnt, 'ir_centroid': (cx_ir, cy_ir),
                'eta': d_json["salute"], 'color': colore,
                'kwh_persi': kwh_persi_anno, 'euro_persi': euro_p, 
                'stato': d_json["label"], 'zona': zone_id
            })

    if pannelli_globali:
        aree_ir = np.array([cv2.contourArea(p['ir_contour']) for p in pannelli_globali], dtype=np.float32)
        area_media = float(np.mean(aree_ir))
        pannelli_globali = [p for p in pannelli_globali if cv2.contourArea(p['ir_contour']) >= area_media]

    pannelli_globali.sort(key=lambda x: cv2.contourArea(x['ir_contour']), reverse=True)
    unici = []
    for p in pannelli_globali:
        if not any(cv2.pointPolygonTest(u['ir_contour'], p['ir_centroid'], False) >= 0 for u in unici):
            unici.append(p)

    unici.sort(key=lambda p: (p['centroid'][1], p['centroid'][0]))
    for i, p in enumerate(unici): p['id'] = i + 1

    with open(CSV_UNICI, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Zona", "Stato", "Salute_%", "kWh_persi_anno", "Perdita_€_Anno"])
        for p in unici:
            label_txt = f"#{p['id']} {p['eta']:.0f}%"
            cv2.drawContours(rgb_canvas, p['contour'], -1, p['color'], 4)
            testo_centrato(rgb_canvas, label_txt, p['centroid'][0], p['centroid'][1], p['color'])
            cv2.drawContours(ir_canvas, p['ir_contour'], -1, p['color'], 4)
            testo_centrato(ir_canvas, label_txt, p['ir_centroid'][0], p['ir_centroid'][1], p['color'])
            is_prob = p['stato'] == "DIFETTOSO" or p['eta'] < 90
            writer.writerow([p['id'], p.get('zona', 1), p['stato'], round(p['eta'], 2), round(p['kwh_persi'], 2) if is_prob else "", round(p['euro_persi'], 2)])

    profile = src_rgb.profile.copy()
    profile.update(count=3)
    with rasterio.open(MAPPA_OUT_PATH, 'w', **profile) as dst: dst.write(np.transpose(rgb_canvas, (2,0,1)))

    profile_ir = src_ir.profile.copy()
    profile_ir.update(count=3)
    with rasterio.open(MAPPA_IR_OUT_PATH, 'w', **profile_ir) as dst: dst.write(np.transpose(ir_canvas, (2,0,1)))

    difettosi = [p for p in unici if p['stato'] == "DIFETTOSO"]
    if difettosi:
        with rasterio.open(IR_MOSAIC) as src_ir2: ir_diff = np.transpose(src_ir2.read([1, 2, 3]), (1, 2, 0)).copy()
        for p in difettosi:
            cv2.drawContours(ir_diff, p['ir_contour'], -1, COLOR_ROSSO, 4)
            testo_centrato(ir_diff, f"#{p['id']}", p['ir_centroid'][0], p['ir_centroid'][1], COLOR_ROSSO)
        with rasterio.open(MAPPA_IR_DIFETTOSI, 'w', **profile_ir) as dst: dst.write(np.transpose(ir_diff, (2, 0, 1)))
        print(f"[FINE] Mappa IR difettosi: {MAPPA_IR_DIFETTOSI}")

    dati_rep = {
        'tot_pannelli': len(unici),
        'tot_sani': len([p for p in unici if p['stato'] != "DIFETTOSO" and p['eta'] >= 90]),
        'tot_degradati': len([p for p in unici if p['stato'] != "DIFETTOSO" and p['eta'] < 90]),
        'tot_rotti': len(difettosi),
        'eta_media_impianto': np.mean([p['eta'] for p in unici]) if unici else 0,
        'kwh_difettosi': sum(p['kwh_persi'] for p in difettosi),
        'kwh_sporchi': sum(p['kwh_persi'] for p in unici if p['stato'] != "DIFETTOSO" and p['eta'] < 90),
        'perdita_euro_difettosi': sum(p['euro_persi'] for p in difettosi),
        'perdita_euro_sporchi': sum(p['euro_persi'] for p in unici if p['stato'] != "DIFETTOSO" and p['eta'] < 90),
        'perdita_euro_totale': sum(p['euro_persi'] for p in unici if p['eta'] < 90),
        'worst_panels': sorted(difettosi, key=lambda x: x['eta'])[:5]
    }
    genera_report_pdf_a2a(dati_rep, PDF_OUT_PATH)
    print(f"\n[FINE] Digital Twin: {MAPPA_OUT_PATH}\n[FINE] Report PDF: {PDF_OUT_PATH}")

if __name__ == "__main__":
    main()
