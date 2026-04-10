#!/usr/bin/env python3
import os
import glob
import re
import argparse
import json
import warnings
import cv2
import numpy as np
import rasterio

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAZIONE PERCORSI E COSTANTI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "risultati_finali")

JSON_TERMICA = os.path.join(OUTPUT_DIR, "analisi_termica", "analisi_dati.json")
JSON_EFFICIENZA = os.path.join(OUTPUT_DIR, "efficienza_risultati", "efficienza_dati.json")
IR_MOSAIC = os.path.join(BASE_DIR, "ortomosaicoir.tif")

# Colori BGR per disegno (stessi dello Step 5)
COLOR_VERDE     = (0, 255, 0)
COLOR_GIALLO    = (255, 255, 0)
COLOR_ARANCIONE = (255, 140, 0)
COLOR_ROSSO     = (255, 0, 0)
COLOR_VIOLA     = (200, 0, 255)

def testo_centrato(canvas, testo, cx, cy, colore, scala=0.6, spessore=2):
    """Disegna il testo centrato con un bordo nero per renderlo sempre leggibile."""
    (tw, th), _ = cv2.getTextSize(testo, cv2.FONT_HERSHEY_SIMPLEX, scala, spessore)
    tx, ty = cx - tw // 2, cy + th // 2
    cv2.putText(canvas, testo, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scala, (0,0,0), spessore + 2, cv2.LINE_AA)
    cv2.putText(canvas, testo, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scala, colore, spessore, cv2.LINE_AA)

def main():
    print("\n" + "="*60)
    print(" GENERATORE MAPPA PER ID SPECIFICI ")
    print("="*60 + "\n")

    # 1. Lettura Argomenti
    parser = argparse.ArgumentParser(description="Estrai pannelli specifici dal mosaico IR.")
    parser.add_argument("--ids", nargs='+', type=int, help="Lista degli ID da cercare (es. --ids 15 42 108)")
    args = parser.parse_args()

    ids_da_cercare = args.ids
    if not ids_da_cercare:
        input_ids = input("Inserisci gli ID dei pannelli da cercare (separati da spazio): ")
        if not input_ids.strip():
            print("[!] Nessun ID inserito. Uscita.")
            return
        ids_da_cercare = [int(x) for x in input_ids.split() if x.isdigit()]

    if not ids_da_cercare:
        print("[!] Formato ID non valido.")
        return

    print(f"[*] Cerco i seguenti ID: {ids_da_cercare}")

    # 2. Controllo File Necessari
    if not os.path.exists(JSON_TERMICA) or not os.path.exists(JSON_EFFICIENZA) or not os.path.exists(IR_MOSAIC):
        print("[!] ERRORE: File JSON o Mosaico non trovati. Esegui prima gli step precedenti.")
        return

    with open(JSON_TERMICA, "r") as f: db_step3 = json.load(f)
    with open(JSON_EFFICIENZA, "r") as f: db_step4 = json.load(f)

    # 3. Mappatura Offset
    REG_DIR = os.path.join(OUTPUT_DIR, "registrazione_allineamento")
    pair_to_offset = {}
    for reg_file in glob.glob(os.path.join(REG_DIR, "pair*_tile_col_*_row_*.jpg")):
        m = re.search(r"pair(\d+)_tile_col_(\d+)_row_(\d+)", os.path.basename(reg_file))
        if m: pair_to_offset[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))

    # 4. Ricostruzione Spaziale dei Pannelli (Stessa logica esatta dello Step 5)
    pannelli_globali = []
    for nome_patch, rilevamenti in db_step4.items():
        pair_num = int(re.search(r"pair(\d+)_", nome_patch).group(1))
        if pair_num not in pair_to_offset: continue
        c_off, r_off = pair_to_offset[pair_num]
        pannelli_step3 = db_step3.get(nome_patch, [])

        for k, d_json in enumerate(rilevamenti):
            if k >= len(pannelli_step3) or d_json["salute"] == 0: continue
            points = pannelli_step3[k].get("points")
            if not points: continue
            
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)

            xs_ir = box[:,0,0] + c_off
            ys_ir = box[:,0,1] + r_off
            ir_cnt = np.array([list(zip(xs_ir, ys_ir))], dtype=np.int32)

            M_ir = cv2.moments(ir_cnt)
            cx_ir = int(M_ir["m10"]/M_ir["m00"]) if M_ir["m00"] != 0 else 0
            cy_ir = int(M_ir["m01"]/M_ir["m00"]) if M_ir["m00"] != 0 else 0

            label_originale = d_json["label"]
            eta_originale = d_json["salute"]
            
            # Replicazione logica Stati
            if label_originale == "DIFETTOSO":
                t_max = pannelli_step3[k].get("t_max")
                t_ref = pannelli_step3[k].get("t_ref")
                area_ratio = pannelli_step3[k].get("area_ratio", 0.0)
                
                delta_t = (t_max - t_ref) if (t_max is not None and t_ref is not None) else 0 
                
                if delta_t > 10:
                    if area_ratio >= 0.15 or delta_t >= 40:
                        stato_finale, eta_finale, colore = "CRITICO (Più Hotspot)", 34.0, COLOR_VIOLA
                    else:
                        stato_finale, eta_finale, colore = "GRAVISSIMO (1 Hotspot)", 67.0, COLOR_ROSSO
                else:
                    stato_finale, eta_finale, colore = "GRAVE (Calore Diffuso)", eta_originale, COLOR_ARANCIONE
            else:
                eta_finale = eta_originale
                if eta_finale < 90.0:
                    stato_finale, colore = "DEGRADATO", COLOR_GIALLO
                else:
                    stato_finale, colore = "SANO", COLOR_VERDE

            pannelli_globali.append({
                'ir_contour': ir_cnt, 'ir_centroid': (cx_ir, cy_ir),
                'eta': eta_finale, 'color': colore, 'stato': stato_finale
            })

    # Filtro Area e NMS (Rimozione duplicati sovrapposti)
    if pannelli_globali:
        aree_ir = np.array([cv2.contourArea(p['ir_contour']) for p in pannelli_globali], dtype=np.float32)
        area_media = float(np.mean(aree_ir))
        pannelli_globali = [p for p in pannelli_globali if cv2.contourArea(p['ir_contour']) >= area_media]

    pannelli_globali.sort(key=lambda x: cv2.contourArea(x['ir_contour']), reverse=True)
    unici = []
    for p in pannelli_globali:
        if not any(cv2.pointPolygonTest(u['ir_contour'], p['ir_centroid'], False) >= 0 for u in unici):
            unici.append(p)

    # Ordinamento Spaziale e Assegnazione ID
    unici.sort(key=lambda p: (p['ir_centroid'][1], p['ir_centroid'][0]))
    for i, p in enumerate(unici): 
        p['id'] = i + 1

    # 5. Filtraggio in base agli ID richiesti dall'utente
    pannelli_filtrati = [p for p in unici if p['id'] in ids_da_cercare]

    if not pannelli_filtrati:
        print("[!] Nessun pannello trovato con gli ID specificati.")
        return

    # 6. Disegno e Salvataggio
    print(f"[*] Trovati {len(pannelli_filtrati)} pannelli. Generazione mappa in corso...")
    
    src_ir = rasterio.open(IR_MOSAIC)
    canvas = np.transpose(src_ir.read([1,2,3]), (1,2,0)).copy()

    for p in pannelli_filtrati:
        label_txt = f"ID #{p['id']} | {p['stato']} | {p['eta']:.0f}%"
        # Disegna il contorno molto spesso per renderlo ben visibile da lontano
        cv2.drawContours(canvas, [p['ir_contour']], -1, p['color'], 6)
        
        # Disegna un mirino/cerchio attorno al centroide per evidenziarlo ulteriormente
        cv2.circle(canvas, p['ir_centroid'], 40, p['color'], 3)
        
        testo_centrato(canvas, label_txt, p['ir_centroid'][0], p['ir_centroid'][1] - 50, p['color'])
        
        print(f"    -> Evidenziato ID #{p['id']} [{p['stato']}]")

    # Nome file dinamico basato sugli ID (es. mappa_ricerca_ID_15_42.tif)
    str_ids = "_".join([str(x) for x in ids_da_cercare])
    out_name = f"mappa_ricerca_ID_{str_ids}.tif"
    
    # Se la stringa è troppo lunga, la accorcia
    if len(out_name) > 50: out_name = "mappa_ricerca_ID_multipli.tif"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    profile_ir = src_ir.profile.copy()
    profile_ir.update(count=3)
    with rasterio.open(out_path, 'w', **profile_ir) as dst:
        dst.write(np.transpose(canvas, (2,0,1)))

    print(f"\n[FINE] Mappa salvata con successo in: {out_path}")

if __name__ == "__main__":
    main()
