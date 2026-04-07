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

# PARAMETRI FISICI (Modificabili)
ETA_NOMINAL = 0.18    # Efficienza nominale 18%
GAMMA       = -0.0035 # Coefficiente di temperatura -0.35%/°C
EPSILON     = 0.90    # Emissività vetro
T_AMB       = 25.0    # Temperatura ambiente stimata
T_STC       = 25.0    # Temp standard di test (STC)

# ==============================================================================
# FUNZIONI DI CALCOLO
# ==============================================================================
def calcola_efficienza_singola(t_c):
    """Calcola l'efficienza reale di un singolo pannello basata sulla sua temperatura."""
    if t_c is None: return 0.0
    
    # 1. Correzione Stefan-Boltzmann (Temp Apparente -> Temp Reale Silicio)
    t_app_k = t_c + 273.15
    t_amb_k = T_AMB + 273.15
    t_reale_k = (( (t_app_k**4) - (1 - EPSILON) * (t_amb_k**4) ) / EPSILON)**0.25
    
    # 2. Delta Temperatura rispetto a STC (25°C)
    t_reale_c = t_reale_k - 273.15
    delta_t = t_reale_c - T_STC
    
    # 3. Formula Efficienza: n = n_nom * (1 + gamma * delta_t)
    eta_reale = ETA_NOMINAL * (1 + GAMMA * delta_t)
    return max(0.0, eta_reale)

# ==============================================================================
# MAIN PROCESS
# ==============================================================================
def main():
    os.makedirs(EFF_DIR, exist_ok=True)
    json_path = os.path.join(TERM_DIR, "analisi_dati.json")

    if not os.path.exists(json_path):
        print(f"[!] Errore: Il file {json_path} non esiste. Esegui prima lo Step 3.")
        return

    with open(json_path, "r") as f:
        database = json.load(f)

    # --- FASE 1: RICERCA RIFERIMENTO OTTIMALE ---
    # Cerchiamo il valore massimo di efficienza tra tutti i pannelli SANI (class_id != 1)
    # per avere un parametro di "Salute 100%" dell'impianto in quel momento.
    lista_eta_sani = []
    for patch_file in database:
        for det in database[patch_file]:
            if det["class_id"] != 1 and det["temp_media"] is not None:
                lista_eta_sani.append(calcola_efficienza_singola(det["temp_media"]))
    
    max_eta_riferimento = max(lista_eta_sani) if lista_eta_sani else ETA_NOMINAL

    # --- FASE 2: ANALISI SINGOLO PANNELLO ED EXPORT ---
    csv_path = os.path.join(EFF_DIR, "report_pannelli_dettagliato.csv")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header del CSV
        writer.writerow([
            "File_Immagine", "ID_Pannello", "Stato_IA", 
            "Temp_Riferimento_C", "Eff_Assoluta", "Salute_Relativa_pct"
        ])

        print(f"[*] Calcolo efficienza per ogni pannello...")
        for nome_img, rilevamenti in tqdm(database.items(), desc="Calcolo"):
            
            # Carichiamo l'immagine termica per aggiungere i dati finali
            img_thermal_path = os.path.join(TERM_DIR, nome_img.replace(".jpg", "_thermal.jpg"))
            canvas = cv2.imread(img_thermal_path)

            for i, d in enumerate(rilevamenti):
                # LOGICA DI SCELTA TEMPERATURA:
                # Se DIFETTOSO (1) -> usa la Temp MAX (punto critico)
                # Se SANO (0, 2) -> usa la Temp MEDIA (funzionamento globale)
                t_rif = d["temp_max"] if d["class_id"] == 1 else d["temp_media"]
                
                # Calcolo efficienza del singolo pannello
                eta_pannello = calcola_efficienza_singola(t_rif)
                
                # Calcolo salute relativa rispetto al miglior pannello sano trovato
                salute_rel = (eta_pannello / max_eta_riferimento * 100) if max_eta_riferimento > 0 else 0
                
                # Scrittura dati nel CSV
                writer.writerow([
                    nome_img, i+1, d["label"], 
                    round(t_rif, 2), round(eta_pannello, 4), round(salute_rel, 2)
                ])

                # Overlay grafico sull'immagine
                if canvas is not None:
                    color = (0, 0, 255) if d["class_id"] == 1 else (0, 255, 0)
                    testo = f"P{i+1}: {salute_rel:.1f}%"
                    # Posizionamento dinamico del testo per ogni pannello (semplificato)
                    cv2.putText(canvas, testo, (10, 30 + (i*25)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Salvataggio immagine finale con efficienze per pannello
            if canvas is not None:
                out_name = nome_img.replace(".jpg", "_efficienza.jpg")
                cv2.imwrite(os.path.join(EFF_DIR, out_name), canvas)

    print(f"\n[FINE] Analisi completata per tutti i singoli pannelli.")
    print(f" -> Report: {csv_path}")
    print(f" -> Immagini: {EFF_DIR}")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
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

# PARAMETRI FISICI (Modificabili)
ETA_NOMINAL = 0.18    # Efficienza nominale 18%
GAMMA       = -0.0035 # Coefficiente di temperatura -0.35%/°C
EPSILON     = 0.90    # Emissività vetro
T_AMB       = 25.0    # Temperatura ambiente stimata
T_STC       = 25.0    # Temp standard di test (STC)

# ==============================================================================
# FUNZIONI DI CALCOLO
# ==============================================================================
def calcola_efficienza_singola(t_c):
    """Calcola l'efficienza reale di un singolo pannello basata sulla sua temperatura."""
    if t_c is None: return 0.0
    
    # 1. Correzione Stefan-Boltzmann (Temp Apparente -> Temp Reale Silicio)
    t_app_k = t_c + 273.15
    t_amb_k = T_AMB + 273.15
    t_reale_k = (( (t_app_k**4) - (1 - EPSILON) * (t_amb_k**4) ) / EPSILON)**0.25
    
    # 2. Delta Temperatura rispetto a STC (25°C)
    t_reale_c = t_reale_k - 273.15
    delta_t = t_reale_c - T_STC
    
    # 3. Formula Efficienza: n = n_nom * (1 + gamma * delta_t)
    eta_reale = ETA_NOMINAL * (1 + GAMMA * delta_t)
    return max(0.0, eta_reale)

# ==============================================================================
# MAIN PROCESS
# ==============================================================================
def main():
    os.makedirs(EFF_DIR, exist_ok=True)
    json_path = os.path.join(TERM_DIR, "analisi_dati.json")

    if not os.path.exists(json_path):
        print(f"[!] Errore: Il file {json_path} non esiste. Esegui prima lo Step 3.")
        return

    with open(json_path, "r") as f:
        database = json.load(f)

    # --- FASE 1: RICERCA RIFERIMENTO OTTIMALE ---
    # Cerchiamo il valore massimo di efficienza tra tutti i pannelli SANI (class_id != 1)
    # per avere un parametro di "Salute 100%" dell'impianto in quel momento.
    lista_eta_sani = []
    for patch_file in database:
        for det in database[patch_file]:
            if det["class_id"] != 1 and det["temp_media"] is not None:
                lista_eta_sani.append(calcola_efficienza_singola(det["temp_media"]))
    
    max_eta_riferimento = max(lista_eta_sani) if lista_eta_sani else ETA_NOMINAL

    # --- FASE 2: ANALISI SINGOLO PANNELLO ED EXPORT ---
    csv_path = os.path.join(EFF_DIR, "report_pannelli_dettagliato.csv")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header del CSV
        writer.writerow([
            "File_Immagine", "ID_Pannello", "Stato_IA", 
            "Temp_Riferimento_C", "Eff_Assoluta", "Salute_Relativa_pct"
        ])

        print(f"[*] Calcolo efficienza per ogni pannello...")
        for nome_img, rilevamenti in tqdm(database.items(), desc="Calcolo"):
            
            # Carichiamo l'immagine termica per aggiungere i dati finali
            img_thermal_path = os.path.join(TERM_DIR, nome_img.replace(".jpg", "_thermal.jpg"))
            canvas = cv2.imread(img_thermal_path)

            for i, d in enumerate(rilevamenti):
                # LOGICA DI SCELTA TEMPERATURA:
                # Se DIFETTOSO (1) -> usa la Temp MAX (punto critico)
                # Se SANO (0, 2) -> usa la Temp MEDIA (funzionamento globale)
                t_rif = d["temp_max"] if d["class_id"] == 1 else d["temp_media"]
                
                # Calcolo efficienza del singolo pannello
                eta_pannello = calcola_efficienza_singola(t_rif)
                
                # Calcolo salute relativa rispetto al miglior pannello sano trovato
                salute_rel = (eta_pannello / max_eta_riferimento * 100) if max_eta_riferimento > 0 else 0
                
                # Scrittura dati nel CSV
                writer.writerow([
                    nome_img, i+1, d["label"], 
                    round(t_rif, 2), round(eta_pannello, 4), round(salute_rel, 2)
                ])

                # Overlay grafico sull'immagine
                if canvas is not None:
                    color = (0, 0, 255) if d["class_id"] == 1 else (0, 255, 0)
                    testo = f"P{i+1}: {salute_rel:.1f}%"
                    # Posizionamento dinamico del testo per ogni pannello (semplificato)
                    cv2.putText(canvas, testo, (10, 30 + (i*25)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Salvataggio immagine finale con efficienze per pannello
            if canvas is not None:
                out_name = nome_img.replace(".jpg", "_efficienza.jpg")
                cv2.imwrite(os.path.join(EFF_DIR, out_name), canvas)

    print(f"\n[FINE] Analisi completata per tutti i singoli pannelli.")
    print(f" -> Report: {csv_path}")
    print(f" -> Immagini: {EFF_DIR}")

if __name__ == "__main__":
    main()
