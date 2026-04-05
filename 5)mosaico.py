#!/usr/bin/env python3
import os
import sys
import glob
import re
import csv
import warnings
import cv2
import numpy as np
import rasterio
import json
import urllib.request
import urllib.parse
from rasterio.warp import transform as transform_coords
from PIL import Image, ExifTags
from datetime import datetime

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAZIONE PERCORSI E COSTANTI
# ==============================================================================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR           = os.path.dirname(CURRENT_SCRIPT_DIR)

WEIGHTS_PATH       = os.path.join(BASE_DIR, "weights.pt")
OUTPUT_DIR         = os.path.join(BASE_DIR, "risultati_finali")
PATCH_IR_DIR       = os.path.join(BASE_DIR, "training_patches_ir")
CSV_GREZZI         = os.path.join(OUTPUT_DIR, "efficienza_risultati", "dati_grezzi.csv")
CONFIG_PATH        = os.path.join(OUTPUT_DIR, "efficienza_risultati", "config_analisi.json")
MAPPA_OUT_PATH     = os.path.join(OUTPUT_DIR, "mappa_efficienza_rgb.tif")
PDF_OUT_PATH       = os.path.join(OUTPUT_DIR, "report_tecnico.pdf")
CSV_UNICI          = os.path.join(OUTPUT_DIR, "efficienza_risultati", "report_pannelli_unici.csv")

IR_MOSAIC  = os.path.join(BASE_DIR, "ortomosaicoir.tif")
RGB_MOSAIC = os.path.join(BASE_DIR, "ortomosaicorgb.tif")
if not os.path.exists(RGB_MOSAIC): 
    RGB_MOSAIC = os.path.join(BASE_DIR, "ortomosaiccrgb.tif")

G_IRR_STC     = 1000.0
COSTO_KWH     = 0.40
GIORNI_UTIL   = 300

# ---> FILTRO AREA: ~1.5 m² <---
SOGLIA_AREA_PX = 10000 

# Colori Stile A2A (in formato BGR per OpenCV)
C_PRIMARY = (159, 91, 0)   
C_SUCCESS = (80, 175, 76)  
C_WARNING = (0, 152, 255)  
C_DANGER  = (54, 67, 244)  
C_TEXT    = (50, 50, 50)   
C_LIGHT   = (240, 245, 245) 

# Colori per i poligoni
COLOR_VERDE  = (0, 255, 0)
COLOR_GIALLO = (0, 200, 255)
COLOR_ROSSO  = (0, 0, 255)

# ==============================================================================
# FUNZIONI DI UTILITA'
# ==============================================================================
def estrai_gps(image_path):
    try:
        import re
        img = Image.open(image_path)
        xmp = img.info.get("xmp", b"").decode("utf-8", errors="ignore")
        def _get_xmp(tag):
            m = re.search(rf'drone-dji:{tag}="([^"]+)"', xmp)
            return m.group(1) if m else None
        lat_str = _get_xmp("GpsLatitude")
        lon_str = _get_xmp("GpsLongitude")
        if lat_str and lon_str:
            return float(lat_str), float(lon_str)

        exif = img._getexif()
        if exif is not None:
            gps_info = {}
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    for t in value:
                        sub_tag = ExifTags.GPSTAGS.get(t, t)
                        gps_info[sub_tag] = value[t]
            def dms_to_decimal(dms, ref):
                if not dms or not ref: return None
                try:
                    d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
                    dec = d + m/60.0 + s/3600.0
                    return -dec if ref in ['S', 'W'] else dec
                except Exception: return None
            
            if "GPSLatitude" in gps_info:
                lat = dms_to_decimal(gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N"))
                lon = dms_to_decimal(gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E"))
                return lat, lon
        return None, None
    except Exception: return None, None

def get_pvgis_esh(lat, lon):
    try:
        url = f"https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?lat={lat}&lon={lon}&peakpower=1&loss=14&outputformat=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            e_annua = data["outputs"]["totals"]["fixed"]["E_y"]
            esh = e_annua / 365.0
            return esh
    except Exception:
        return 3.18

def mappa_pair_a_originale(patch_ir_dir):
    original_patches = sorted(glob.glob(os.path.join(patch_ir_dir, "*.jpg")))
    mapping = {}
    for i, orig_path in enumerate(original_patches):
        pair_name = f"pair{i+1}_patch.jpg"
        mapping[pair_name] = orig_path
    return mapping

def determina_colore_rgb(eta_rel_pct):
    if eta_rel_pct >= 90.0: return COLOR_VERDE
    elif eta_rel_pct >= 80.0: return COLOR_GIALLO
    else: return COLOR_ROSSO

# ==============================================================================
# REPORTISTICA PDF A2A-LIKE
# ==============================================================================
def disegna_grafico_a_ciambella(canvas, cx, cy, r, eta_media_pct):
    nom_pct = 100.0
    angolo_erogata = int(360 * (eta_media_pct / nom_pct))
    
    cv2.circle(canvas, (cx+2, cy+5), r+2, (220, 220, 220), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r, (220, 220, 220), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (r, r), -90, 0, angolo_erogata, C_SUCCESS, -1, cv2.LINE_AA)
    r_inner = int(r * 0.65)
    cv2.circle(canvas, (cx, cy), r_inner, (255, 255, 255), -1, cv2.LINE_AA)
    
    label = f"{eta_media_pct:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 1.8, 4)
    cv2.putText(canvas, label, (cx - tw//2, cy + th//2), font, 1.8, C_TEXT, 4, cv2.LINE_AA)

def genera_report_pdf_a2a(dati, pdf_path):
    w, h = 1240, 1754 
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    cv2.rectangle(canvas, (0, 0), (w, 160), C_PRIMARY, -1)
    cv2.putText(canvas, "REPORT EFFICIENZA IMPIANTO FOTOVOLTAICO", (50, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"Ispezione UAV Termografica | Data: {datetime.now().strftime('%d/%m/%Y')} | Rif: REP-{datetime.now().strftime('%y%m%d')}", 
                (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 1, cv2.LINE_AA)

    def add_section_title(text, x, y, color=C_PRIMARY):
        cv2.rectangle(canvas, (x, y-25), (x+15, y+5), color, -1)
        cv2.putText(canvas, text.upper(), (x+30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_TEXT, 2, cv2.LINE_AA)
        cv2.line(canvas, (x, y+15), (x+500, y+15), (200, 200, 200), 1)

    def add_row(label, value, x, y, val_color=C_TEXT, bold=False):
        thick = 2 if bold else 1
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(value), (x+350, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, val_color, thick, cv2.LINE_AA)
        cv2.line(canvas, (x, y+15), (x+500, y+15), (240, 240, 240), 1)

    col1_x = 80
    y_cursor = 250
    add_section_title("Parametri di Analisi", col1_x, y_cursor)
    y_cursor += 50
    add_row("Tecnologia Moduli", dati['tipo_pannello'], col1_x, y_cursor, bold=True)
    y_cursor += 40
    add_row("Ore Pieno Sole (ESH)", f"{dati['esh']:.2f} h/giorno", col1_x, y_cursor)
    
    y_cursor += 80
    add_section_title("Statistiche Impianto", col1_x, y_cursor)
    y_cursor += 50
    add_row("Totale Pannelli Mappati", str(dati['tot_pannelli']), col1_x, y_cursor, bold=True)
    y_cursor += 40
    add_row("Pannelli VERDI (>= 90%)", str(dati['tot_sani']), col1_x, y_cursor, val_color=C_SUCCESS, bold=True)
    y_cursor += 40
    add_row("Pannelli GIALLI (80-89%)", str(dati['tot_degradati']), col1_x, y_cursor, val_color=C_WARNING, bold=True)
    y_cursor += 40
    add_row("Pannelli ROSSI (< 80%)", str(dati['tot_rotti']), col1_x, y_cursor, val_color=C_DANGER, bold=True)

    col2_x = 650
    y_cursor = 250
    add_section_title("Efficienza Media Impianto", col2_x, y_cursor, color=C_SUCCESS)
    
    graph_cx, graph_cy = col2_x + 250, y_cursor + 200
    disegna_grafico_a_ciambella(canvas, graph_cx, graph_cy, 140, dati['eta_media_impianto'])
    
    y_legend = graph_cy + 180
    cv2.rectangle(canvas, (col2_x+50, y_legend-12), (col2_x+65, y_legend+3), C_SUCCESS, -1)
    cv2.putText(canvas, "Capacita Reale Erogata", (col2_x+80, y_legend), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1, cv2.LINE_AA)
    
    y_legend += 30
    cv2.rectangle(canvas, (col2_x+50, y_legend-12), (col2_x+65, y_legend+3), (220, 220, 220), -1)
    cv2.putText(canvas, "Perdita Termica / Degrado", (col2_x+80, y_legend), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1, cv2.LINE_AA)

    y_cursor = y_legend + 80
    add_section_title("Impatto Economico Annuo", col2_x, y_cursor, color=C_WARNING)
    y_cursor += 50
    add_row("Potenza Nominale Persa", f"{dati['pot_persa_kw']:.2f} kW", col2_x, y_cursor)
    
    y_cursor += 50
    box_rect = (col2_x, y_cursor, col2_x + 500, y_cursor + 120)
    cv2.rectangle(canvas, (box_rect[0], box_rect[1]), (box_rect[2], box_rect[3]), C_LIGHT, -1)
    cv2.rectangle(canvas, (box_rect[0], box_rect[1]), (box_rect[2], box_rect[3]), C_WARNING, 2)
    
    cv2.putText(canvas, f"- {dati['perdita_euro']:.2f} EUR", (box_rect[0]+120, box_rect[1]+65), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, C_WARNING, 3, cv2.LINE_AA)
    cv2.putText(canvas, "Mancato Guadagno Stimato Annuo", (box_rect[0]+110, box_rect[1]+100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1, cv2.LINE_AA)

    # === NUOVA SEZIONE: TOP MODULI CRITICI ===
    y_cursor += 160
    add_section_title("Top Moduli da Sostituire", col2_x, y_cursor, color=C_DANGER)
    y_cursor += 40
    
    if dati['worst_panels']:
        # Header Tabella Top 5
        cv2.putText(canvas, "ID", (col2_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 2)
        cv2.putText(canvas, "SoH", (col2_x + 80, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 2)
        cv2.putText(canvas, "Persa (W)", (col2_x + 180, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 2)
        cv2.putText(canvas, "Danno (EUR)", (col2_x + 320, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 2)
        cv2.line(canvas, (col2_x, y_cursor+15), (col2_x+500, y_cursor+15), (200, 200, 200), 1)
        y_cursor += 40
        
        for wp in dati['worst_panels']:
            cv2.putText(canvas, f"#{wp['id']}", (col2_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_DANGER, 2)
            cv2.putText(canvas, f"{wp['eta']:.1f}%", (col2_x + 80, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1)
            cv2.putText(canvas, f"{wp['p_persa_w']:.1f}", (col2_x + 180, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1)
            cv2.putText(canvas, f"- {wp['euro_persi']:.2f}", (col2_x + 320, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_DANGER, 2)
            cv2.line(canvas, (col2_x, y_cursor+15), (col2_x+500, y_cursor+15), (240, 240, 240), 1)
            y_cursor += 35
    else:
        cv2.putText(canvas, "Nessun modulo critico rilevato.", (col2_x, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_SUCCESS, 1)

    # === CONCLUSIONE DIAGNOSTICA (Spostata a sinistra in basso) ===
    y_cursor = 1150
    add_section_title("Conclusione Diagnostica", 80, y_cursor)
    
    y_cursor += 40
    box_h = 140
    if dati['tot_rotti'] + dati['tot_degradati'] > 0:
        box_color = (235, 235, 255) # Sfondo rossastro (BGR)
        line_color = C_DANGER
        t1 = f"ATTENZIONE: Rilevati {dati['tot_rotti']} moduli Rossi e {dati['tot_degradati']} moduli Gialli."
        t2 = f"Si raccomanda l'intervento tecnico. L'assenza di manutenzione comportera una perdita"
        t3 = f"cumulativa di {dati['perdita_euro']:.2f} Euro/anno a causa dei colli di bottiglia sulle stringhe."
    else:
        box_color = (235, 255, 235) # Sfondo verdino (BGR)
        line_color = C_SUCCESS
        t1 = "L'impianto risulta in ottime condizioni operative."
        t2 = "Tutti i moduli sono Verdi. Nessuna anomalia termica"
        t3 = "critica tale da compromettere la produzione."

    # Usa solo mezza pagina per la conclusione
    cv2.rectangle(canvas, (80, y_cursor), (600, y_cursor+box_h), box_color, -1)
    cv2.rectangle(canvas, (80, y_cursor), (86, y_cursor+box_h), line_color, -1) 
    
    cv2.putText(canvas, t1, (110, y_cursor+45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, t2, (110, y_cursor+80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 1, cv2.LINE_AA)
    if t3:
        cv2.putText(canvas, t3, (110, y_cursor+115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, h-60), (w, h), (40, 40, 40), -1)
    cv2.putText(canvas, "Generato tramite Intelligenza Artificiale - Ispezione Termografica UAV", (50, h-25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    img_pil.save(pdf_path, "PDF", resolution=150.0)

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print("\n" + "="*60)
    print("  GENERAZIONE DIGITAL TWIN E REPORT PDF AVANZATO")
    print("="*60)

    if not os.path.exists(IR_MOSAIC) or not os.path.exists(RGB_MOSAIC):
        print("[!] ERRORE: Mosaico IR o RGB non trovato.")
        return
        
    if not os.path.exists(CSV_GREZZI) or not os.path.exists(CONFIG_PATH):
        print("[!] ERRORE: File Dati o Config non trovati. Esegui prima 4_efficienza.py")
        return
        
    with open(CONFIG_PATH, "r") as f:
        user_params = json.load(f)

    eff_data = {}
    with open(CSV_GREZZI, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nome_patch = row["File_Patch"]
            id_pannello = int(row["ID_Pannello"])
            eta_rel = float(row["Salute_Relativa_pct"])
            if nome_patch not in eff_data:
                eff_data[nome_patch] = {}
            eff_data[nome_patch][id_pannello] = eta_rel

    from rfdetr import RFDETRSegLarge
    print("[*] Caricamento modello IA per estrazione vettoriale...")
    model = RFDETRSegLarge(pretrain_weights=WEIGHTS_PATH, num_classes=2)
    
    mappa_orig = mappa_pair_a_originale(PATCH_IR_DIR)

    print("[*] Apertura mosaici e coordinate geografiche...")
    src_ir = rasterio.open(IR_MOSAIC)
    ir_transform = src_ir.transform
    ir_crs = src_ir.crs

    src_rgb = rasterio.open(RGB_MOSAIC)
    rgb_transform = src_rgb.transform
    rgb_crs = src_rgb.crs
    rgb_profile = src_rgb.profile.copy()
    rgb_profile.update(count=3)

    rgb_img_data = src_rgb.read([1, 2, 3]) 
    rgb_canvas = np.transpose(rgb_img_data, (1, 2, 0)).copy()

    pair_files = list(eff_data.keys())
    pannelli_globali = [] 

    print(f"[*] Elaborazione geometria su {len(pair_files)} patch...")

    for i, pair_name in enumerate(pair_files):
        orig_path = mappa_orig.get(pair_name)
        if not orig_path: continue
        
        m = re.search(r"tile_col_(\d+)_row_(\d+)", os.path.basename(orig_path))
        if not m: continue
        patch_col_offset, patch_row_offset = int(m.group(1)), int(m.group(2))

        patch_path = os.path.join(OUTPUT_DIR, "pair", pair_name)
        img_patch = cv2.imread(patch_path)
        if img_patch is None: continue
        
        img_pil = Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
        results = model.predict(img_pil, threshold=0.60)
        
        if results is None or len(results.xyxy) == 0: continue

        for k in range(len(results.xyxy)):
            if (k + 1) not in eff_data[pair_name]: continue
            
            eta_rel = eff_data[pair_name][k + 1]
            color_rgb = determina_colore_rgb(eta_rel)
            mask = results.mask[k]
            
            if mask is None: continue
            mask_u8 = (mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            c_big = max(contours, key=cv2.contourArea)
            
            # FILTRO AREA GEOMETRICO
            if cv2.contourArea(c_big) < SOGLIA_AREA_PX:
                continue

            px_xs = c_big[:, 0, 0] + patch_col_offset
            px_ys = c_big[:, 0, 1] + patch_row_offset
            
            east_irs, north_irs = rasterio.transform.xy(ir_transform, px_ys, px_xs)
            xs_rgb, ys_rgb = transform_coords(ir_crs, rgb_crs, east_irs, north_irs)
            rows_rgb, cols_rgb = rasterio.transform.rowcol(rgb_transform, xs_rgb, ys_rgb)
            
            mapped_contour = np.array([list(zip(cols_rgb, rows_rgb))], dtype=np.int32)
            
            area_globale = cv2.contourArea(mapped_contour)
            M = cv2.moments(mapped_contour)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(mapped_contour)
                cx, cy = x + w//2, y + h//2
            
            pannelli_globali.append({
                'contour': mapped_contour, 'centroid': (cx, cy),
                'area': area_globale, 'eta': eta_rel, 'color': color_rgb
            })

    # Filtro NMS (Evita i doppioni)
    pannelli_globali.sort(key=lambda p: p['area'], reverse=True)
    pannelli_filtrati = []
    for nuovo_pan in pannelli_globali:
        duplicato = False
        cx, cy = nuovo_pan['centroid']
        for pan_salvato in pannelli_filtrati:
            if cv2.pointPolygonTest(pan_salvato['contour'], (cx, cy), measureDist=False) >= 0:
                duplicato = True
                break
        if not duplicato:
            pannelli_filtrati.append(nuovo_pan)

    # ==========================================================
    # ASSEGNAZIONE ID, CALCOLI E GENERAZIONE MAPPA / CSV
    # ==========================================================
    print(f"[*] Estrazione ESH e Calcolo Danni per singolo pannello...")
    esh = 3.18
    primo_originale = glob.glob(os.path.join(PATCH_IR_DIR, "*.jpg"))[0]
    lat, lon = estrai_gps(primo_originale)
    if lat and lon:
        esh = get_pvgis_esh(lat, lon)

    # Ordiniamo visivamente per posizione Y (così l'ID 1 è in alto a sinistra)
    pannelli_filtrati.sort(key=lambda p: (p['centroid'][1], p['centroid'][0]))

    for idx, pan in enumerate(pannelli_filtrati):
        pan['id'] = idx + 1
        
        # Calcolo perdite solo per Gialli e Rossi
        if pan['color'] == COLOR_GIALLO or pan['color'] == COLOR_ROSSO:
            p_max_w = G_IRR_STC * user_params["area"] * user_params["eta_nom"]
            p_reale_w = p_max_w * (pan['eta'] / 100.0)
            p_persa = p_max_w - p_reale_w
            euro = (p_persa / 1000.0) * esh * GIORNI_UTIL * COSTO_KWH
            
            pan['p_persa_w'] = p_persa
            pan['euro_persi'] = euro
            pan['stato'] = 'Degradato' if pan['color'] == COLOR_GIALLO else 'Rotto'
        else:
            pan['p_persa_w'] = 0.0
            pan['euro_persi'] = 0.0
            pan['stato'] = 'Sano'

    # Disegno Finale Ortomosaico con NUMERO ID
    for pan in pannelli_filtrati:
        cv2.drawContours(rgb_canvas, pan['contour'], -1, pan['color'], 4)
        
        # Nuova Etichetta: #ID | Efficienza
        label = f"#{pan['id']} | {pan['eta']:.1f}%"
        
        x, y, w, h = cv2.boundingRect(pan['contour'])
        cx, cy = pan['centroid']
        font_scale = max(0.55, w / 220.0) 
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(rgb_canvas, (cx - tw//2 - 5, cy - th - 5), (cx + tw//2 + 5, cy + 5), (0, 0, 0), -1)
        cv2.putText(rgb_canvas, label, (cx - tw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, pan['color'], 2, cv2.LINE_AA)

    print(f"[*] Salvataggio Mappa GeoTIFF in corso...")
    out_img_chw = np.transpose(rgb_canvas, (2, 0, 1))
    with rasterio.open(MAPPA_OUT_PATH, 'w', **rgb_profile) as dst:
        dst.write(out_img_chw)

    # Salvataggio CSV Dettagliato
    print(f"[*] Creazione Report CSV Pannelli Unici...")
    with open(CSV_UNICI, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID_Pannello", "Stato", "Efficienza_Relativa_pct", "Potenza_Persa_W", "Mancato_Guadagno_EUR"])
        for p in pannelli_filtrati:
            writer.writerow([p['id'], p['stato'], f"{p['eta']:.2f}", f"{p['p_persa_w']:.2f}", f"{p['euro_persi']:.2f}"])

    # ==========================================================
    # CREAZIONE PDF E TOP 5 PEGGIORI
    # ==========================================================
    print(f"[*] Generazione Report Tecnico PDF...")
    
    tot_pannelli = len(pannelli_filtrati)
    tot_verdi = len([p for p in pannelli_filtrati if p['color'] == COLOR_VERDE])
    tot_gialli = len([p for p in pannelli_filtrati if p['color'] == COLOR_GIALLO])
    tot_rossi = len([p for p in pannelli_filtrati if p['color'] == COLOR_ROSSO])

    somma_eta = sum([p['eta'] for p in pannelli_filtrati])
    eta_media_impianto = somma_eta / tot_pannelli if tot_pannelli > 0 else 0
    pot_persa_w_tot = sum([p['p_persa_w'] for p in pannelli_filtrati])
    pot_persa_kw = pot_persa_w_tot / 1000.0
    perdita_euro = sum([p['euro_persi'] for p in pannelli_filtrati])
    
    # Prepara Top 5 Peggiori Pannelli
    peggiori = [p for p in pannelli_filtrati if p['eta'] < 90.0]
    peggiori.sort(key=lambda x: x['eta']) # Ordina dal peggiore in su
    
    dati_report = {
        'tipo_pannello': user_params['tipo'],
        'eta_nominale_assoluta': user_params["eta_nom"] * 100,
        'esh': esh,
        'tot_pannelli': tot_pannelli,
        'tot_sani': tot_verdi,
        'tot_degradati': tot_gialli,
        'tot_rotti': tot_rossi,
        'eta_media_impianto': eta_media_impianto,
        'pot_persa_kw': pot_persa_kw,
        'perdita_euro': perdita_euro,
        'worst_panels': peggiori[:5] # Manda solo la top 5
    }
    
    genera_report_pdf_a2a(dati_report, PDF_OUT_PATH)

    print(f"\n[FINE] Elaborazione conclusa con successo!")
    print(f"  -> Mappa RGB salvata in: {MAPPA_OUT_PATH}")
    print(f"  -> Report CSV Unici in : {CSV_UNICI}")
    print(f"  -> Report PDF salvato in: {PDF_OUT_PATH}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
