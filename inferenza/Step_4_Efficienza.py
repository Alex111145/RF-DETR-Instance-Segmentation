
import os
import re
import json
import struct
import cv2
import numpy as np
from PIL import Image, ExifTags
from tqdm import tqdm


BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TERM_DIR        = os.path.join(BASE_DIR, "risultati_finali", "analisi_termica")
PAIR_DIR        = os.path.join(BASE_DIR, "risultati_finali", "pair")
EFF_DIR         = os.path.join(BASE_DIR, "risultati_finali", "efficienza_risultati")
FOTO_DRONE_DIR  = os.path.join(BASE_DIR, "foto_drone")

ETA_NOMINAL = 0.165
GAMMA       = -0.0042
EPSILON     = 0.90
T_AMB       = 25.0

# Raggio della finestra locale (in frazione dell'estensione totale del campo).
# 0.25 = ogni pannello viene confrontato con i vicini entro il 25% della dimensione del campo.
# Abbassa il valore se est/sud sono molto ravvicinati; alzalo se il campo è piccolo.
FRAZIONE_RAGGIO_LOCALE = 0.25


def estrai_metadati_da_drone():
    """
    Legge la prima foto drone disponibile ed estrae dal MakerNote DJI:
      - GPS (lat, lon)
      - T_amb  (tag 0x2002) = temperatura ambiente impostata in DJI Pilot (°C)
      - Emissività (tag 0x2004) = emissività superficiale impostata in DJI Pilot
    """
    if not os.path.isdir(FOTO_DRONE_DIR):
        return None, None, None, None
    for fname in sorted(os.listdir(FOTO_DRONE_DIR)):
        if not fname.lower().endswith(('.jpg', '.jpeg')):
            continue
        try:
            img = Image.open(os.path.join(FOTO_DRONE_DIR, fname))
            exif_data = img._getexif()
            if not exif_data:
                continue

            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}

            # GPS
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

            # Parametri termici dal MakerNote DJI (IFD little-endian, tag tipo FLOAT=11)
            #   0x2002 = T_amb        (°C)
            #   0x2004 = Emissività   (0.0 – 1.0)
            t_amb    = None
            epsilon  = None
            maker = exif_data.get(37500, b'')
            if len(maker) >= 14:
                num_entries = struct.unpack_from('<H', maker, 0)[0]
                for i in range(min(num_entries, 50)):
                    off = 2 + i * 12
                    if off + 12 > len(maker):
                        break
                    tag, typ, cnt = struct.unpack_from('<HHI', maker, off)
                    if typ == 11 and cnt == 1:
                        val = struct.unpack_from('<f', maker, off + 8)[0]
                        if tag == 0x2002:
                            t_amb   = val
                        elif tag == 0x2004:
                            epsilon = val

            return lat, lon, t_amb, epsilon
        except:
            continue
    return None, None, None, None


def scegli_tecnologia():
    print("\n" + "="*55)
    print(" SELEZIONE TECNOLOGIA PANNELLO FOTOVOLTAICO")
    print("="*55)
    print(" 1. Policristallino      (η_nom=16.5%,  γ=-0.42%/°C)")
    print(" 2. Monocristallino PERC (η_nom=20.0%,  γ=-0.37%/°C)")
    print("="*55)
    while True:
        scelta = input(" Inserisci la scelta [1/2]: ").strip()
        if scelta == "1":
            print(" [OK] Parametri Policristallino selezionati.\n")
            return 0.165, -0.0042
        elif scelta == "2":
            print(" [OK] Parametri Monocristallino PERC selezionati.\n")
            return 0.20, -0.0037
        else:
            print(" [!] Scelta non valida. Inserire 1 o 2.")


def parse_pair_num(img_name):
    """Estrae il numero pair dal nome file (es: pair3_patch.jpg → 3)."""
    m = re.search(r"pair(\d+)_", img_name)
    return int(m.group(1)) if m else None


def carica_pair_to_offset():
    """
    Legge i file nella cartella registrazione_allineamento e ricostruisce
    la mappa pair_N → (col_off, row_off), identica a quella usata in Step 5.
    """
    import glob
    reg_dir = os.path.join(BASE_DIR, "risultati_finali", "registrazione_allineamento")
    mapping = {}
    for f in glob.glob(os.path.join(reg_dir, "pair*_tile_col_*_row_*.jpg")):
        m = re.search(r"pair(\d+)_tile_col_(\d+)_row_(\d+)", os.path.basename(f))
        if m:
            mapping[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    return mapping


def eta_rif_locale(col, row, sani_map):
    """
    Riferimento di efficienza locale per un pannello in posizione (col, row).

    Logica:
    - Cerca tutti i pannelli sani (non hotspot) entro una finestra spaziale
      pari a FRAZIONE_RAGGIO_LOCALE * estensione_campo in X e Y.
    - Restituisce il MAX di efficienza in quella finestra: corrisponde al
      pannello più freddo (più pulito) della stessa zona di esposizione.
    - Se la finestra è vuota (zona isolata) usa il max globale come fallback.

    Effetto: pannelli sud caldi ma uniformi → riferimento = miglior pannello
    sud → appaiono sani. Pannello sporco (più caldo dei vicini) → efficienza
    inferiore al riferimento locale → flaggato correttamente.
    """
    if not sani_map:
        return ETA_NOMINAL

    cols = [c for c, r, _ in sani_map]
    rows = [r for _, r, _ in sani_map]

    span_x = max(cols) - min(cols) if len(set(cols)) > 1 else 0
    span_y = max(rows) - min(rows) if len(set(rows)) > 1 else 0

    dx = span_x * FRAZIONE_RAGGIO_LOCALE if span_x > 0 else float('inf')
    dy = span_y * FRAZIONE_RAGGIO_LOCALE if span_y > 0 else float('inf')

    vicini = [e for c, r, e in sani_map
              if abs(c - col) <= dx and abs(r - row) <= dy]

    return max(vicini) if vicini else max(e for _, _, e in sani_map)


def calcola_efficienza_termodinamica(t_c):
    if t_c is None or t_c == 0: return 0.0
    try:
        t_k = t_c + 273.15
        t_amb_k = T_AMB + 273.15
        t_reale_k = (( (t_k**4) - (1 - EPSILON) * (t_amb_k**4) ) / EPSILON)**0.25
        delta_t = (t_reale_k - 273.15) - 25.0
        return max(0.0, ETA_NOMINAL * (1 + GAMMA * delta_t))
    except: return 0.0

def main():
    global ETA_NOMINAL, GAMMA, T_AMB, EPSILON
    ETA_NOMINAL, GAMMA = scegli_tecnologia()

    # Lettura parametri termici dal MakerNote DJI delle foto drone
    lat, lon, t_amb, epsilon = estrai_metadati_da_drone()

    print("\n[*] Parametri termici estratti dal MakerNote DJI:")
    if lat is not None:
        print(f"    Coordinate volo : {lat:.5f}°N, {lon:.5f}°E")

    if t_amb is not None and t_amb != 0.0:
        T_AMB = round(float(t_amb), 2)
        print(f"    T_amb           : {T_AMB:.1f}°C  (dal metadato drone)")
    else:
        print(f"    T_amb           : {T_AMB:.1f}°C  (default — non trovata nei metadati)")

    if epsilon is not None and 0.0 < epsilon <= 1.0:
        EPSILON = round(float(epsilon), 4)
        print(f"    Emissività (ε)  : {EPSILON:.4f}  (dal metadato drone)")
    else:
        print(f"    Emissività (ε)  : {EPSILON:.4f}  (default — non trovata nei metadati)")
    print()

    os.makedirs(EFF_DIR, exist_ok=True)
    json_in = os.path.join(TERM_DIR, "analisi_dati.json")
    
    if not os.path.exists(json_in):
        print(f"[!] Errore: Manca {json_in}.")
        return

    with open(json_in, "r") as f:
        db_termico = json.load(f)

    # Mappa pair_N → (col_off, row_off) dalla cartella di registrazione
    pair_to_offset = carica_pair_to_offset()
    print(f"[*] Offset spaziali caricati: {len(pair_to_offset)} pair trovati nella registrazione")

    # ── Prima passata: mappa spaziale dei pannelli sani ──────────────────────
    # Raccoglie (col_off, row_off, eta_assoluta) per ogni pannello non-hotspot.
    # Il MAX locale di questa mappa diventa il riferimento per la zona corrispondente,
    # separando automaticamente cluster con esposizioni diverse (est / sud / ecc.).
    sani_map = []
    for img_name, rilevamenti in db_termico.items():
        pair_num = parse_pair_num(img_name)
        if pair_num not in pair_to_offset:
            continue
        col_off, row_off = pair_to_offset[pair_num]
        for d in rilevamenti:
            if d.get("class_id") != 1:
                t = d.get("temp_media")
                if t is not None:
                    sani_map.append((col_off, row_off,
                                     calcola_efficienza_termodinamica(t)))

    max_eta_globale = max(e for _, _, e in sani_map) if sani_map else ETA_NOMINAL

    n_zone = len(set((c, r) for c, r, _ in sani_map))
    print(f"[*] Riferimento locale attivo: {len(sani_map)} pannelli sani "
          f"in {n_zone} posizioni — finestra {int(FRAZIONE_RAGGIO_LOCALE*100)}% del campo")

    dati_step4 = {}

    for img_name, rilevamenti in tqdm(db_termico.items(), desc="Calcolo Efficienza"):
        img_path = os.path.join(PAIR_DIR, img_name)
        canvas = cv2.imread(img_path)
        if canvas is None: continue

        # Riferimento locale per questa patch
        pair_num = parse_pair_num(img_name)
        if pair_num in pair_to_offset and sani_map:
            col_off, row_off = pair_to_offset[pair_num]
            max_eta_rif = eta_rif_locale(col_off, row_off, sani_map)
        else:
            max_eta_rif = max_eta_globale

        overlay = canvas.copy()
        analisi_patch = []

        for i, d in enumerate(rilevamenti):
            t_rif = d.get("temp_utilizzata")
            if t_rif is None:
                t_rif = d.get("temp_max") if d.get("class_id") == 1 else d.get("temp_media")

            if t_rif is None:
                t_rif = 0.0
                label_temp = "N/A"
            else:
                label_temp = f"{round(t_rif, 1)}C"

            eta_ass = calcola_efficienza_termodinamica(t_rif)
            salute_rel = min(100.0, (eta_ass / max_eta_rif * 100)) if max_eta_rif > 0 else 0
            
            if d.get("class_id") == 1:
                color = (0, 0, 255)    # Rosso  – hotspot / difettoso
            elif salute_rel < 90:
                color = (0, 255, 255)  # Giallo – sano ma degradato (<90%)
            else:
                color = (0, 255, 0)    # Verde  – ottimale
            
            if 'points' in d and d['points']:
                pts = np.array(d['points'], dtype=np.int32)
             
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(canvas, [pts], True, color, 2)

                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
               
                    cX, cY = pts[0][0], pts[0][1]

                label = f"P{i+1}: {salute_rel:.1f}% ({label_temp})"
                
              
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.45
                thickness = 1
                (w, h), _ = cv2.getTextSize(label, font, scale, thickness)
         
                text_pos = (cX - w // 2, cY + h // 2)

                
                cv2.putText(canvas, label, text_pos, font, scale, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(canvas, label, text_pos, font, scale, (255,255,255), thickness, cv2.LINE_AA)
      

            analisi_patch.append({
                "id": i + 1,
                "label": d.get("label"),
                "salute": round(salute_rel, 2),
                "temp": round(t_rif, 2) if t_rif != 0.0 else None
            })

        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
        dati_step4[img_name] = analisi_patch
        cv2.imwrite(os.path.join(EFF_DIR, img_name.replace(".jpg", "_efficienza.jpg")), canvas)

    with open(os.path.join(EFF_DIR, "efficienza_dati.json"), "w") as f:
        json.dump(dati_step4, f, indent=4)
    
    print(f"\n[OK] Analisi completata. Etichette centrate salvate in {EFF_DIR}")

if __name__ == "__main__":
    main()
