
import os
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

    lista_eta_sani = []
    for img_name, rilevamenti in db_termico.items():
        for d in rilevamenti:
            if d.get("class_id") != 1:
                t = d.get("temp_media")
                if t is not None:
                    lista_eta_sani.append(calcola_efficienza_termodinamica(t))
    
    max_eta_rif = max(lista_eta_sani) if lista_eta_sani else ETA_NOMINAL

    dati_step4 = {}
    
    for img_name, rilevamenti in tqdm(db_termico.items(), desc="Calcolo Efficienza"):
        img_path = os.path.join(PAIR_DIR, img_name)
        canvas = cv2.imread(img_path)
        if canvas is None: continue
        
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
