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

IR_MOSAIC  = os.path.join(BASE_DIR, "ortomosaicoir.tif")
RGB_MOSAIC = os.path.join(BASE_DIR, "ortomosaicorgb.tif")
if not os.path.exists(RGB_MOSAIC): 
    RGB_MOSAIC = os.path.join(BASE_DIR, "ortomosaiccrgb.tif")

G_IRR_STC     = 1000.0
COSTO_KWH     = 0.40
GIORNI_UTIL   = 300

# ==============================================================================
# FUNZIONI DI UTILITA' E API PVGIS
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
    except Exception as e:
        print(f"  [!] API PVGIS fallita. Uso default 3.18 ore.")
        return 3.18

def mappa_pair_a_originale(patch_ir_dir):
    original_patches = sorted(glob.glob(os.path.join(patch_ir_dir, "*.jpg")))
    mapping = {}
    for i, orig_path in enumerate(original_patches):
        pair_name = f"pair{i+1}_patch.jpg"
        mapping[pair_name] = orig_path
    return mapping

def determina_colore_rgb(eta_rel_pct):
    """Soglie: Verde 90-100, Giallo 85-89.9, Rosso < 85"""
    if eta_rel_pct >= 90.0:
        return (0, 255, 0)       # Verde 
    elif eta_rel_pct >= 85.0:
        return (255, 200, 0)     # Giallo
    else:
        return (255, 0, 0)       # Rosso 

# ==============================================================================
# REPORTISTICA PDF A2A-LIKE
# ==============================================================================
def disegna_grafico_torta(canvas, cx, cy, r, eta_media_pct):
    """
    Grafico a torta: Verde per l'efficienza media relativa. Grigio per la perdita.
    """
    # L'efficienza relativa massima è il 100%
    nom_pct = 100.0
    persa_pct = nom_pct - eta_media_pct
    
    angolo_erogata = 360 * (eta_media_pct / nom_pct)
    angolo_persa = 360 * (persa_pct / nom_pct)
    
    start_angle = -90 
    
    end_angle = start_angle + angolo_erogata
    cv2.ellipse(canvas, (cx, cy), (r, r), 0, start_angle, end_angle, (100, 210, 50), -1, cv2.LINE_AA)
    
    start_angle = end_angle
    end_angle = start_angle + angolo_persa
    cv2.ellipse(canvas, (cx, cy), (r, r), 0, start_angle, end_angle, (230, 230, 230), -1, cv2.LINE_AA)

    label = f"{eta_media_pct:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    cv2.putText(canvas, label, (cx - tw//2, cy + th//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3, cv2.LINE_AA)

def genera_report_pdf(pannelli_filtrati, pdf_path, user_params, esh):
    w, h = 1240, 1754
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    tot_pannelli = len(pannelli_filtrati)
    if tot_pannelli == 0: return

    # CALCOLI GLOBALI BASATI SUI PANNELLI UNICI (Senza duplicati)
    somma_eta = sum([p['eta'] for p in pannelli_filtrati])
    eta_media_impianto = somma_eta / tot_pannelli
    
    # Determiniamo l'Efficienza Nominale Assoluta 
    eta_nominale_assoluta = user_params["eta_nom"] * 100
    
    # Filtriamo i pannelli sotto il 90% (Danneggiati)
    danneggiati = [p for p in pannelli_filtrati if p['eta'] < 90.0]
    num_danneggiati = len(danneggiati)
    
    # Calcolo Danni Economici (Potenza di riferimento = Pannello a 100% di SoH)
    pot_persa_w_tot = 0.0
    for p in danneggiati:
        # Troviamo l'efficienza assoluta che aveva questo pannello prima che calcolassimo il SoH
        # E_rel = E_ass / E_max -> E_ass = (E_rel * E_max) / 100
        # Poiché non abbiamo conservato E_max nel dizionario del filtro, applichiamo un approccio
        # diretto: Un SoH < 100% significa che manca un (100 - SoH)% di potenza.
        # P_max_teorica = G * Area * eta_nominale
        p_max_w = G_IRR_STC * user_params["area"] * user_params["eta_nom"]
        p_reale_w = p_max_w * (p['eta'] / 100.0)
        pot_persa_w_tot += (p_max_w - p_reale_w)
        
    pot_persa_kw = pot_persa_w_tot / 1000.0
    e_persa_kwh = pot_persa_kw * esh
    perdita_euro = e_persa_kwh * GIORNI_UTIL * COSTO_KWH
    
    # === DISEGNO TESTATA STILE A2A ===
    cv2.rectangle(canvas, (0, 0), (w, 180), (40, 40, 40), -1)
    
    cv2.putText(canvas, "REPORT TECNICO DI ISPEZIONE FOTOVOLTAICA UAV", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"Ispezione termografica eseguita il {datetime.now().strftime('%d/%m/%Y')}", (50, 135), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

    y = 280
    def add_section_header(text, icon_color=(0,0,0), dy=70):
        nonlocal y
        cv2.rectangle(canvas, (50, y-45), (70, y-25), icon_color, -1, cv2.LINE_AA)
        cv2.putText(canvas, text.upper(), (90, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
        y += dy

    def add_table_row(label, value, is_bold=False):
        nonlocal y
        thickness = 2 if is_bold else 1
        cv2.putText(canvas, label, (80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(value), (600, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), thickness, cv2.LINE_AA)
        cv2.line(canvas, (80, y+15), (1160, y+15), (230, 230, 230), 1)
        y += 50

    add_section_header("parametri di analisi")
    add_table_row("Tecnologia Moduli Fotovoltaici", user_params['tipo'])
    add_table_row("Efficienza Nominale (STC)", f"{eta_nominale_assoluta:.1f}%")
    add_table_row("Ore Equivalenti Pieno Sole (ESH)", f"{esh:.2f} h")
    y += 50 

    add_section_header("riepilogo efficienza impianto", (100, 210, 50))

    graph_cx, graph_cy = 620, 950
    graph_r = 200
    disegna_grafico_torta(canvas, graph_cx, graph_cy, graph_r, eta_media_impianto)
    
    y = graph_cy + graph_r + 60
    cv2.putText(canvas, "Fetta Verde: Efficienza Salute (SoH) media dell'intero impianto", (graph_cx - graph_r, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)
    y += 30
    cv2.putText(canvas, "Fetta Grigia: Degrado termico cumulativo riscontrato", (graph_cx - graph_r, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)
    y += 100 

    add_section_header("statistiche aggregate e degrado", (0, 0, 200)) 

    add_table_row("Totale Pannelli FISICI Analizzati", str(tot_pannelli), True)
    add_table_row("Efficienza Media dell'Impianto (SoH)", f"{eta_media_impianto:.1f}%", True)
    
    thickness_danneg = 2 if num_danneggiati > 0 else 1
    add_table_row("Pannelli Integri (SoH >= 90%)", f"{tot_pannelli - num_danneggiati}", False)
    add_table_row("Pannelli Danneggiati (SoH < 90%)", f"{num_danneggiati}", thickness_danneg)
    y += 30
    
    cv2.rectangle(canvas, (80, y-30), (1160, y+100), (250, 250, 250), -1) 
    add_table_row("Potenza Nominale Persa Complessiva", f"{pot_persa_kw:.2f} kW", True)
    add_table_row("Mancato Guadagno Stimato Annuo", f"{perdita_euro:.2f} EUR / Anno", True)
    y += 120

    add_section_header("conclusione diagnostica")
    
    if num_danneggiati > 0:
        cv2.putText(canvas, "Si raccomanda la manutenzione o sostituzione dei moduli termicamente", (80, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 1, cv2.LINE_AA)
        y += 40
        cv2.putText(canvas, "danneggiati per ripristinare la capacita produttiva nominale,", (80, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 1, cv2.LINE_AA)
        y += 40
        cv2.putText(canvas, f"evitando una perdita cumulativa di {perdita_euro:.2f} Euro annui.", (80, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "L'impianto si trova in condizioni operative ottimali. Nessun modulo", (80, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 1, cv2.LINE_AA)
        y += 40
        cv2.putText(canvas, "scende sotto la soglia critica del 90% di efficienza relativa.", (80, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, h-80), (w, h), (40, 40, 40), -1)
    cv2.putText(canvas, "Generato automaticamente tramite IA - Ispezione UAV Isolare - Dati Filtrati Anti-Duplicazione", (50, h-35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    img_pil.save(pdf_path, "PDF", resolution=150.0)

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print("\n" + "="*60)
    print("  GENERAZIONE DIGITAL TWIN & REPORT PDF")
    print("="*60)

    if not os.path.exists(IR_MOSAIC) or not os.path.exists(RGB_MOSAIC):
        print("[!] ERRORE: Mosaico IR o RGB non trovato.")
        return
        
    if not os.path.exists(CSV_GREZZI) or not os.path.exists(CONFIG_PATH):
        print("[!] ERRORE: File Dati o Config non trovati. Esegui prima 4.py")
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
    print("[*] Caricamento modello IA...")
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
        results = model.predict(img_pil, threshold=0.30)
        
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

    # Filtro NMS
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

    # Disegno Finale Ortomosaico
    for pan in pannelli_filtrati:
        cv2.drawContours(rgb_canvas, pan['contour'], -1, pan['color'], 4)
        label = f"{pan['eta']:.1f}%"
        x, y, w, h = cv2.boundingRect(pan['contour'])
        cx, cy = pan['centroid']
        font_scale = max(0.7, w / 180.0) 
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(rgb_canvas, (cx - tw//2 - 5, cy - th - 5), (cx + tw//2 + 5, cy + 5), (0, 0, 0), -1)
        cv2.putText(rgb_canvas, label, (cx - tw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, pan['color'], 2, cv2.LINE_AA)

    print(f"[*] Salvataggio Mappa GeoTIFF in corso...")
    out_img_chw = np.transpose(rgb_canvas, (2, 0, 1))
    with rasterio.open(MAPPA_OUT_PATH, 'w', **rgb_profile) as dst:
        dst.write(out_img_chw)

    print(f"[*] Generazione Report PDF...")
    # Estraiamo ESH chiedendo le coordinate
    esh = 3.18
    primo_originale = glob.glob(os.path.join(PATCH_IR_DIR, "*.jpg"))[0]
    lat, lon = estrai_gps(primo_originale)
    if lat and lon:
        esh = get_pvgis_esh(lat, lon)
        
    genera_report_pdf(pannelli_filtrati, PDF_OUT_PATH, user_params, esh)

    print(f"\n[FINE] Elaborazione conclusa con successo!")
    print(f"  -> Mappa RGB salvata in: {MAPPA_OUT_PATH}")
    print(f"  -> Report PDF salvato in: {PDF_OUT_PATH}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
