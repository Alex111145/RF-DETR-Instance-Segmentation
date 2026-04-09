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
    m = re.search(r"pair(\d+)_", img_name)
    return int(m.group(1)) if m else None


def carica_pair_to_offset():
    import glob
    reg_dir = os.path.join(BASE_DIR, "risultati_finali", "registrazione_allineamento")
    mapping = {}
    for f in glob.glob(os.path.join(reg_dir, "pair*_tile_col_*_row_*.jpg")):
        m = re.search(r"pair(\d+)_tile_col_(\d+)_row_(\d+)", os.path.basename(f))
        if m:
            mapping[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    return mapping


ANGLE_GAP     = 15.0   
SPATIAL_GAP   = 800    
MIN_ZONE_SIZE = 8      

def roof_zone_mapping(pair_to_offset, db_pannelli):
    """
    1. Union-Find base (Distanza + Angolo)
    2. ASSORBIMENTO CONVEX HULL: Le zone interne a un'altra falda vengono assorbite forzatamente.
    """
    panels_data = [] 
    
    for img_name, panels in db_pannelli.items():
        m = re.search(r"pair(\d+)_", img_name)
        if not m: continue
        pair_num = int(m.group(1))
        if pair_num not in pair_to_offset: continue
            
        col_off, row_off = pair_to_offset[pair_num]
        
        for idx, panel in enumerate(panels):
            pts = panel.get("points")
            if not pts or len(pts) < 3:
                continue
                
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

    if not panels_data:
        return {}

    n = len(panels_data)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    # 1. Clustering iniziale
    for i in tqdm(range(n), desc="Mappatura falde (Fase 1)"):
        for j in range(i + 1, n):
            pi, pj = panels_data[i], panels_data[j]
            ad = abs(pi["angle"] - pj["angle"])
            ad = min(ad, 180.0 - ad)
            if ad > ANGLE_GAP:
                continue
                
            dist_sq = (pi["cx"] - pj["cx"])**2 + (pi["cy"] - pj["cy"])**2
            if dist_sq <= SPATIAL_GAP**2:
                union(i, j)

    from collections import defaultdict
    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(panels_data[i])
        
    groups = list(comps.values())

    # 2. Post-Processing: Assorbimento in base al Convex Hull (Nessuna area dentro un'altra)
    merged_any = True
    while merged_any:
        merged_any = False
        groups.sort(key=len, reverse=True)
        merged_indices = set()
        
        for i in range(len(groups)):
            if i in merged_indices: continue
            compA = groups[i]
            if len(compA) < 3: continue
            
            # Crea l'inviluppo convesso della falda grande
            ptsA = np.array([[p["cx"], p["cy"]] for p in compA], dtype=np.float32)
            hullA = cv2.convexHull(ptsA)
            
            for j in range(i + 1, len(groups)):
                if j in merged_indices: continue
                compB = groups[j]
                
                inside_count = 0
                for pB in compB:
                    # Verifica se il pannello B è dentro (o a contatto) della falda A
                    dist = cv2.pointPolygonTest(hullA, (float(pB["cx"]), float(pB["cy"])), measureDist=True)
                    if dist >= -250: # Tolleranza di margine per i bordi interni
                        inside_count += 1
                        
                # Se la maggior parte della zona B cade dentro la zona A, assorbila tutta
                if inside_count >= len(compB) * 0.5:
                    compA.extend(compB)
                    merged_indices.add(j)
                    merged_any = True
                    
        groups = [groups[k] for k in range(len(groups)) if k not in merged_indices]

    # 3. Pulizia finale: assorbi micro-aree rimanenti alla più vicina
    final_groups = []
    small_groups = []
    
    for g in groups:
        if len(g) >= MIN_ZONE_SIZE:
            final_groups.append(g)
        else:
            small_groups.append(g)
            
    if final_groups and small_groups:
        for sg in small_groups:
            best_dist = float('inf')
            best_idx = None
            for i, fg in enumerate(final_groups):
                for p in sg:
                    for fp in fg:
                        d_sq = (p["cx"] - fp["cx"])**2 + (p["cy"] - fp["cy"])**2
                        if d_sq < best_dist:
                            best_dist = d_sq
                            best_idx = i
            if best_idx is not None and best_dist <= (SPATIAL_GAP * 2)**2:
                final_groups[best_idx].extend(sg)
            else:
                final_groups.append(sg)
    elif not final_groups:
        final_groups = groups

    sorted_comps = sorted(final_groups, key=len, reverse=True)

    result = {}
    print(f"\n[*] Zone (falde) rilevate: {len(sorted_comps)}")
    for zone_id, comp in enumerate(sorted_comps, 1):
        angoli = [p["angle"] for p in comp]
        print(f"    Zona {zone_id}: {len(comp)} pannelli, angolo medio = {np.mean(angoli):.1f}°")
        for p in comp:
            result[p["id"]] = zone_id

    return result


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

    pair_to_offset = carica_pair_to_offset()
    pair_to_zone = roof_zone_mapping(pair_to_offset, db_termico)

    # ── Prima passata: efficienza pannelli sani per zona ─────────────────────
    sani_per_zona = {}
    for img_name, rilevamenti in db_termico.items():
        for i, d in enumerate(rilevamenti):
            zone_id  = pair_to_zone.get((img_name, i), 1)
            if d.get("class_id") != 1:
                t = d.get("temp_media")
                if t is not None:
                    sani_per_zona.setdefault(zone_id, []).append(
                        calcola_efficienza_termodinamica(t))

    max_eta_per_zona = {z: max(etas) for z, etas in sani_per_zona.items()}
    max_eta_globale  = max(max_eta_per_zona.values()) if max_eta_per_zona else ETA_NOMINAL

    for z in sorted(max_eta_per_zona):
        print(f"    Zona {z}: {len(sani_per_zona[z])} pannelli sani, "
              f"max_eta = {max_eta_per_zona[z]:.4f}")

    dati_step4 = {}

    for img_name, rilevamenti in tqdm(db_termico.items(), desc="Calcolo Efficienza"):
        img_path = os.path.join(PAIR_DIR, img_name)
        canvas = cv2.imread(img_path)
        if canvas is None: continue

        overlay = canvas.copy()
        analisi_patch = []

        for i, d in enumerate(rilevamenti):
            zone_id     = pair_to_zone.get((img_name, i), 1)
            max_eta_rif = max_eta_per_zona.get(zone_id, max_eta_globale)

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
                color = (0, 0, 255)
            elif salute_rel < 90:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            
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
                "temp": round(t_rif, 2) if t_rif != 0.0 else None,
                "zona": int(zone_id)
            })

        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
        dati_step4[img_name] = analisi_patch
        cv2.imwrite(os.path.join(EFF_DIR, img_name.replace(".jpg", "_efficienza.jpg")), canvas)

    with open(os.path.join(EFF_DIR, "efficienza_dati.json"), "w") as f:
        json.dump(dati_step4, f, indent=4)
    
    print(f"\n[OK] Analisi completata. Etichette centrate salvate in {EFF_DIR}")

if __name__ == "__main__":
    main()
