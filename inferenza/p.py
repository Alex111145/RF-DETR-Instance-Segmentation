#!/usr/bin/env python3
"""
debug_zone.py — Visualizza le falde del tetto sull'ortomosaico IR.
Le zone sono maschere morfologiche che seguono la forma reale dei blocchi di pannelli.
"""
import os
import re
import glob
import json
import cv2
import numpy as np
import rasterio
from collections import defaultdict
from tqdm import tqdm

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR      = os.path.join(BASE_DIR, "risultati_finali")
IR_MOSAIC    = os.path.join(BASE_DIR, "ortomosaicoir.tif")
PAIR_DIR     = os.path.join(OUT_DIR, "pair")
REG_DIR      = os.path.join(OUT_DIR, "registrazione_allineamento")
JSON_TERMICA = os.path.join(OUT_DIR, "analisi_termica", "analisi_dati.json")
OUT_IMAGE    = os.path.join(OUT_DIR, "debug_zone_analizzate.jpg")

SCALA_OUTPUT = 0.25
ANGLE_GAP    = 15.0   # gradi — max differenza d'angolo nella stessa falda
SPATIAL_GAP  = 800    # pixel — max distanza tra patch della stessa falda
PATCH_SIZE   = 504    # pixel — dimensione patch IR

PALETTE = [
    (0, 220, 0),
    (0, 140, 255),
    (255, 50, 50),
    (0, 220, 220),
    (220, 0, 220),
    (50, 200, 150),
    (255, 180, 0),
    (180, 0, 255),
    (0, 100, 255),
    (255, 100, 0),
]


def roof_zone_mapping(pair_to_offset, db_pannelli):
    """
    Determina le zone delle falde del tetto a livello di SINGOLO PANNELLO 
    combinando angolazione e prossimità spaziale tramite Union-Find.
    """
    panels_data = [] 
    
    for img_name, panels in db_pannelli.items():
        m = re.search(r"pair(\d+)_", img_name)
        if not m: continue
        pair_num = int(m.group(1))
        
        if pair_num not in pair_to_offset:
            continue
            
        col_off, row_off = pair_to_offset[pair_num]
        
        for idx, panel in enumerate(panels):
            pts = panel.get("points")
            if not pts or len(pts) < 3:
                continue
                
            pts_arr = np.array(pts, dtype=np.float32)
            rect = cv2.minAreaRect(pts_arr)
            a = rect[2]
            w, h = rect[1]
            if w < h:
                a += 90.0
            a = a % 180.0  # CORRETTO: calcolo effettivo su 180 gradi
            
            M = cv2.moments(pts_arr)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + col_off
                cy = int(M["m01"] / M["m00"]) + row_off
            else:
                cx, cy = int(pts_arr[0][0]) + col_off, int(pts_arr[0][1]) + row_off
                
            panels_data.append({
                "id": (img_name, idx),
                "pair": pair_num,
                "angle": a,
                "cx": cx,
                "cy": cy
            })

    if not panels_data:
        print("[*] Nessun pannello trovato — zona unica.")
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

    for i in tqdm(range(n), desc="Mappatura falde (Union-Find)"):
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
        
    sorted_comps = sorted(comps.values(), key=len, reverse=True)

    result = {}
    print(f"\n[*] Zone (falde) rilevate: {len(sorted_comps)}")
    
    for zone_id, comp in enumerate(sorted_comps, 1):
        angoli = [p["angle"] for p in comp]
        print(f"    Zona {zone_id}: {len(comp)} pannelli, angolo medio = {np.mean(angoli):.1f}°")
        for p in comp:
            result[p["id"]] = zone_id

    return result


def disegna_maschere_zone(canvas, pair_to_offset, pair_to_zone, db_pannelli):
    """
    Per ogni zona dipinge i contorni REALI dei pannelli (dal JSON Step 3)
    shiftati nella posizione del mosaico, poi chiusura morfologica e contorno.
    """
    H, W = canvas.shape[:2]

    # Maschera per zona + contatore pannelli
    zone_masks   = defaultdict(lambda: np.zeros((H, W), dtype=np.uint8))
    zone_counts  = defaultdict(int)

    for img_name, panels in db_pannelli.items():
        m = re.search(r"pair(\d+)_", img_name)
        if not m:
            continue
        pair_num = int(m.group(1))
        if pair_num not in pair_to_offset:
            continue
        col_off, row_off = pair_to_offset[pair_num]

        for idx, panel in enumerate(panels):
            zone_id = pair_to_zone.get((img_name, idx), 1)
            pts = panel.get("points")
            if not pts or len(pts) < 3:
                continue
            pts_arr = np.array(pts, dtype=np.int32) + np.array([col_off, row_off])
            cv2.fillPoly(zone_masks[zone_id], [pts_arr], 255)
            zone_counts[zone_id] += 1

    # Kernel MORFOLOGICO DIMINUITO: da (80, 80) a (35, 35) per evitare overlap eccessivo
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))

    for zone_id in sorted(zone_masks.keys()):
        mask  = zone_masks[zone_id]
        color = PALETTE[(zone_id - 1) % len(PALETTE)]

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        overlay = canvas.copy()
        cv2.fillPoly(overlay, contours, color)
        cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)
        cv2.drawContours(canvas, contours, -1, color, 5)

        c_big = max(contours, key=cv2.contourArea)
        Mc = cv2.moments(c_big)
        if Mc["m00"] != 0:
            cx = int(Mc["m10"] / Mc["m00"])
            cy = int(Mc["m01"] / Mc["m00"])
        else:
            x, y, bw, bh = cv2.boundingRect(c_big)
            cx, cy = x + bw // 2, y + bh // 2

        label = f"Zona #{zone_id}  ({zone_counts[zone_id]} pannelli)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.putText(canvas, label, (cx - tw // 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(canvas, label, (cx - tw // 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

    return canvas


def main():
    with rasterio.open(IR_MOSAIC) as src:
        canvas = np.transpose(src.read([1, 2, 3]), (1, 2, 0)).copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    H, W = canvas.shape[:2]
    print(f"[*] Mosaico IR: {W}×{H} px")

    pair_to_offset = {}
    for f in glob.glob(os.path.join(REG_DIR, "pair*_tile_col_*_row_*.jpg")):
        m = re.search(r"pair(\d+)_tile_col_(\d+)_row_(\d+)", os.path.basename(f))
        if m:
            pair_to_offset[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    print(f"[*] Pair trovati: {len(pair_to_offset)}")

    if not os.path.exists(JSON_TERMICA):
        print(f"[!] JSON termico non trovato: {JSON_TERMICA}")
        print("[!] Esegui prima Step_3_Temperatura.py")
        return

    with open(JSON_TERMICA) as f:
        db_pannelli = json.load(f)

    pair_to_zone = roof_zone_mapping(pair_to_offset, db_pannelli)

    canvas = disegna_maschere_zone(canvas, pair_to_offset, pair_to_zone, db_pannelli)

    if SCALA_OUTPUT != 1.0:
        nw = int(W * SCALA_OUTPUT)
        nh = int(H * SCALA_OUTPUT)
        canvas = cv2.resize(canvas, (nw, nh), interpolation=cv2.INTER_AREA)

    cv2.imwrite(OUT_IMAGE, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[OK] Salvato: {OUT_IMAGE}  ({canvas.shape[1]}×{canvas.shape[0]} px)")


if __name__ == "__main__":
    main()
