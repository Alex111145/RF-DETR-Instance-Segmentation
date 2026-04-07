#!/usr/bin/env python3
import os
import json
import csv
import cv2
import numpy as np
from tqdm import tqdm

# ==============================================================================
# CONFIGURAZIONE PERCORSI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TERM_DIR = os.path.join(BASE_DIR, "risultati_finali", "analisi_termica")
EFF_DIR  = os.path.join(BASE_DIR, "risultati_finali", "efficienza_risultati")

# PARAMETRI FISICI
ETA_NOMINAL = 0.18    
GAMMA       = -0.0035 
EPSILON     = 0.90    
T_AMB       = 25.0    
T_STC       = 25.0    

# ==============================================================================
# FUNZIONI DI CALCOLO
# ==============================================================================
def calcola_efficienza_singola(t_c):
    if t_c is None: return 0.0
    t_app_k = t_c + 273.15
    t_amb_k = T_AMB + 273.15
    # Correzione Stefan-Boltzmann
    t_reale_k = (( (t_app_k**4) - (1 - EPSILON) * (t_amb_k**4) ) / EPSILON)**0.25
    t_reale_c = t_reale_k - 273.15
    delta_t = t_reale_c - T_STC
    return max(0.0, ETA_NOMINAL * (1 + GAMMA * delta_t))

# ==============================================================================
# MAIN PROCESS
# ==============================================================================
def main():
    os.makedirs(EFF_DIR, exist_ok=True)
    json_input_path = os.path.join(TERM_DIR, "analisi_dati.json")

    if not os.path.exists(json_input_path):
        print(f"[!] Errore: Il file {json_input_path} non esiste.")
        return

    with open(json_input_path, "r") as f:
        database = json.load(f)

    # --- FASE 1: RICERCA RIFERIMENTO ---
    lista_eta_sani = []
    for patch_file in database:
        for det in database[patch_file]:
            if det["class_id"] != 1 and det["temp_media"] is not None:
                lista_eta_sani.append(calcola_efficienza_singola(det["temp_media"]))
    
    max_eta_riferimento = max(lista_eta_sani) if lista_eta_sani else ETA_NOMINAL

    # --- FASE 2: ELABORAZIONE E SALVATAGGIO ---
    csv_path = os.path.join(EFF_DIR, "report_pannelli_dettagliato.csv")
    json_output_path = os.path.join(EFF_DIR, "efficienza_dati.json") # FILE PER STEP 5
    
    dati_finali_per_step5 = {}

    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["File_Immagine", "ID_Pannello", "Stato_IA", "Temp_C", "Eff_Assoluta", "Salute_Relativa_pct"])

        print(f"[*] Calcolo efficienza e generazione dati per Step 5...")
        for nome_img, rilevamenti in tqdm(database.items(), desc="Processing"):
            
            img_thermal_path = os.path.join(TERM_DIR, nome_img.replace(".jpg", "_thermal.jpg"))
            canvas = cv2.imread(img_thermal_path)
            
            analisi_patch = []

            for i, d in enumerate(rilevamenti):
                # Se DIFETTOSO (1) -> usa Temp MAX, altrimenti MEDIA
                t_rif = d["temp_max"] if d["class_id"] == 1 else d["temp_media"]
                
                eta_pannello = calcola_efficienza_singola(t_rif)
                salute_rel = (eta_pannello / max_eta_riferimento * 100) if max_eta_riferimento > 0 else 0
                
                # Prepara dati per JSON (Step 5)
                analisi_patch.append({
                    "id_pannello": i + 1,
                    "stato": d["label"],
                    "class_id": d["class_id"],
                    "temp_riferimento": round(t_rif, 2),
                    "efficienza_assoluta": round(eta_pannello, 4),
                    "salute_relativa": round(salute_rel, 2)
                })

                # Scrittura CSV
                writer.writerow([nome_img, i+1, d["label"], round(t_rif, 2), round(eta_pannello, 4), round(salute_rel, 2)])

                # Overlay grafico
                if canvas is not None:
                    color = (0, 0, 255) if d["class_id"] == 1 else (0, 255, 0)
                    cv2.putText(canvas, f"P{i+1}: {salute_rel:.1f}%", (10, 30 + (i*25)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Aggiungi al database JSON
            dati_finali_per_step5[nome_img] = analisi_patch

            if canvas is not None:
                cv2.imwrite(os.path.join(EFF_DIR, nome_img.replace(".jpg", "_efficienza.jpg")), canvas)

    # --- SALVATAGGIO JSON PER STEP 5 ---
    with open(json_output_path, "w") as f_json:
        json.dump({
            "metadata": {
                "eta_riferimento_impianto": round(max_eta_riferimento, 4),
                "parametri": {"t_amb": T_AMB, "epsilon": EPSILON, "gamma": GAMMA}
            },
            "analisi": dati_finali_per_step5
        }, f_json, indent=4)

    print(f"\n[FINE] Analisi completata.")
    print(f" -> JSON per Step 5: {json_output_path}")
    print(f" -> Report CSV: {csv_path}")

if __name__ == "__main__":
    main()
