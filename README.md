# Ispezione Fotovoltaico UAV — Pipeline Termografica AI

Sistema di analisi automatica di impianti fotovoltaici tramite ispezione UAV con doppia telecamera (IR + RGB).
Utilizza un modello AI di segmentazione (RF-DETR) per identificare i pannelli e calcola l'efficienza termodinamica di ciascun modulo.

---

## Struttura del Progetto

```
Yolo/
├── inferenza/
│   ├── 1.py                  # Step 1: Allineamento patch IR ↔ foto drone
│   ├── 2.py                  # Step 2: Inferenza AI (rilevamento pannelli)
│   ├── 3.py                  # Step 3: Analisi termica per pannello
│   ├── 4.py                  # Step 4: Calcolo efficienza termodinamica
│   └── 5.py                  # Step 5: Digital Twin + Report PDF + Mappa GeoTIFF
├── foto_drone/               # Foto RJPEG del drone DJI (IR + dati GPS)
├── training_patches_ir/      # Tile 512×512 estratte dall'ortomosaico IR
├── ortomosaicoir.tif         # Ortomosaico infrarosso georeferenziato
├── ortomosaicorgb.tif        # Ortomosaico RGB georeferenziato
├── weights.pt                # Pesi del modello RF-DETR addestrato
├── sdk/
│   ├── linux/                # Binari DJI SDK per Linux (dji_irp, libdirp.so, ecc.)
│   └── windows/              # Binari DJI SDK per Windows
├── risultati_finali/         # Output generati automaticamente dagli script
│   ├── pair/                 # Coppie patch_IR + foto_drone originali
│   ├── registrazione_allineamento/  # Immagini affiancate per controllo visivo
│   ├── inferenza_pannelli/   # Patch annotate con box/maschere AI
│   ├── analisi_termica/      # Patch annotate con temperatura per pannello
│   ├── efficienza_risultati/ # CSV dati grezzi + immagini efficienza
│   │   ├── dati_grezzi.csv
│   │   └── config_analisi.json
│   ├── mappa_efficienza_rgb.tif  # Ortomosaico RGB con poligoni colorati
│   ├── report_pannelli_unici.csv # CSV finale con ID, stato, perdite per pannello
│   └── report_tecnico.pdf        # Report PDF stile professionale
└── requirements.txt
```

---

## Prerequisiti

- Python **3.9 – 3.11**
- Linux (raccomandato) o Windows
- GPU opzionale ma consigliata per l'inferenza AI

---

## Creazione dell'Ambiente Virtuale

```bash
# 1. Crea l'ambiente virtuale nella cartella del progetto
python3 -m venv .venv

# 2. Attivalo (Linux/macOS)
source .venv/bin/activate

# 3. Attivalo (Windows)
.venv\Scripts\activate

# 4. Aggiorna pip
pip install --upgrade pip

# 5. Installa le dipendenze
pip install -r requirements.txt
```

> **Nota su rfdetr:** se il pacchetto non è su PyPI nella versione richiesta, installalo da sorgente o da wheel fornita dal progetto:
> ```bash
> pip install rfdetr --extra-index-url <url-repo-privato>
> ```

---

## Dati di Input Necessari

Prima di avviare la pipeline assicurati di avere nella cartella `Yolo/`:

| File / Cartella | Descrizione |
|---|---|
| `ortomosaicoir.tif` | Ortomosaico IR georeferenziato (GeoTIFF) |
| `ortomosaicorgb.tif` oppure `ortomosaiccrgb.tif` | Ortomosaico RGB georeferenziato |
| `training_patches_ir/*.jpg` | Tile IR (512×512) con nome `tile_col_X_row_Y.jpg` |
| `foto_drone/*.JPG` | Foto RJPEG originali del drone DJI con GPS EXIF |
| `weights.pt` | Pesi del modello RF-DETR (2 classi: Sano/Difettoso) |

---

## Esecuzione della Pipeline

Gli script vanno eseguiti **in sequenza** dalla cartella `inferenza/`:

```bash
cd Yolo/inferenza
```

### Step 1 — Allineamento IR ↔ Drone
```bash
python 1.py
```
**Cosa fa:** Legge le tile IR da `training_patches_ir/`, calcola le coordinate geografiche del centro di ogni tile tramite il CRS dell'ortomosaico IR, trova la foto drone più vicina per GPS (EXIF), applica feature matching ORB + omografia per allinearle visivamente. Salva:
- `risultati_finali/pair/pairN_patch.jpg` — copia della tile IR (bit-a-bit)
- `risultati_finali/pair/pairN_drone.jpg` — copia della foto drone (bit-a-bit)
- `risultati_finali/registrazione_allineamento/` — immagini affiancate per verifica visiva

**Librerie chiave:** `rasterio` (CRS/transform), `OpenCV` (ORB, BFMatcher, findHomography), `PIL` (lettura EXIF GPS)

---

### Step 2 — Inferenza AI (Rilevamento Pannelli)
```bash
python 2.py [--threshold 0.30]
```
**Cosa fa:** Carica il modello `RFDETRSegLarge` con i pesi `weights.pt`. Per ogni `pairN_patch.jpg` esegue la segmentazione con soglia di confidenza configurabile. Disegna i contorni delle maschere (o bounding box se la maschera non è disponibile) con colore verde (Sano) o rosso (Difettoso). Salva i risultati annotati in `risultati_finali/inferenza_pannelli/`.

**Librerie chiave:** `rfdetr` (modello AI), `OpenCV` (findContours, drawContours, moments), `PIL` (conversione BGR→RGB per il modello)

---

### Step 3 — Analisi Termica per Pannello
```bash
python 3.py [--threshold 0.30]
```
**Cosa fa:** Combina inferenza AI + estrazione dati termici dalle foto RJPEG DJI. Usa due strategie per i dati termici:
1. **RAW parsing binario**: cerca il blob di dati termici dopo il marker JPEG `0xFFD9`, decodifica come `uint16` in Kelvin → °C
2. **Fallback CLI SDK DJI** (`sdk/linux/dji_irp`): invoca il tool DJI per generare un file `.raw` temporaneo

Per ogni pannello rilevato, proietta la maschera AI nello spazio della foto drone tramite omografia ORB, campiona la matrice termica nella ROI e calcola la temperatura media. Annota ogni pannello con la temperatura in °C.

**Librerie chiave:** `rfdetr`, `OpenCV` (warpPerspective, ORB), `numpy` (frombuffer, mean), `subprocess` (SDK DJI)

---

### Step 4 — Calcolo Efficienza Termodinamica
```bash
python 4.py
```
**Cosa fa:** Script **interattivo** — chiede tipo pannello (Monocristallino/Policristallino) e dimensioni fisiche del modulo. Poi:
1. Recupera la temperatura ambiente reale via API [Open-Meteo](https://open-meteo.com/) usando GPS + timestamp EXIF della prima foto
2. Applica la legge di **Stefan-Boltzmann** per correggere l'emissività del vetro (ε=0.90)
3. Calcola l'efficienza assoluta con il modello lineare: `η_reale = η_nom × (1 + γ × ΔT)` (coefficiente termico γ)
4. Normalizza ogni pannello rispetto al pannello migliore (= 100% salute)
5. Salva `dati_grezzi.csv` e `config_analisi.json` in `efficienza_risultati/`

**Librerie chiave:** `rfdetr`, `OpenCV`, `numpy`, `urllib` (API meteo), `csv`, `json`

---

### Step 5 — Digital Twin, Mappa GeoTIFF e Report PDF
```bash
python 5.py
```
**Cosa fa:** Legge i risultati del Step 4 e genera i tre output finali:

1. **Mappa GeoTIFF** (`mappa_efficienza_rgb.tif`): riproietta i contorni dei pannelli dallo spazio pixel IR alle coordinate geografiche dell'ortomosaico RGB usando `rasterio.warp.transform` (conversione CRS IR→RGB). Disegna poligoni colorati sull'ortomosaico RGB:
   - Verde: efficienza ≥ 90%
   - Giallo: efficienza 80–89%
   - Rosso: efficienza < 80%
   Applica NMS geometrico (point-in-polygon) per eliminare duplicati.

2. **CSV Pannelli Unici** (`report_pannelli_unici.csv`): ID univoco, stato, efficienza, potenza persa (W), mancato guadagno (€/anno) per ogni pannello

3. **Report PDF** (`report_tecnico.pdf`): documento A4 generato con OpenCV + PIL che include grafico a ciambella dell'efficienza media, statistiche impianto, impatto economico (calcolato con dati ESH da [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/)) e top 5 moduli critici.

**Librerie chiave:** `rasterio` (CRS transform, GeoTIFF write), `rfdetr`, `OpenCV` (rendering PDF), `PIL` (save PDF), `urllib` (PVGIS API)

---

## Parametri Configurabili

Puoi modificare le costanti in cima a ogni script:

| Costante | File | Valore Default | Descrizione |
|---|---|---|---|
| `SOGLIA_AREA_PX` | `4.py`, `5.py` | `10000` px | Area minima maschera (filtra rumore) |
| `G_IRR_STC` | `5.py` | `1000.0 W/m²` | Irraggiamento condizioni standard |
| `COSTO_KWH` | `5.py` | `0.40 €` | Prezzo energia per calcolo economico |
| `GIORNI_UTIL` | `5.py` | `300` giorni | Giorni di produzione annua stimati |
| `EPSILON_VETRO` | `4.py` | `0.90` | Emissività del vetro fotovoltaico |

---

## Output Finali

| File | Descrizione |
|---|---|
| `risultati_finali/mappa_efficienza_rgb.tif` | Ortomosaico RGB georeferenziato con pannelli colorati per stato |
| `risultati_finali/report_pannelli_unici.csv` | Tabella con ID, stato, efficienza e perdite economiche per pannello |
| `risultati_finali/report_tecnico.pdf` | Report PDF professionale con grafici e analisi economica |

---

## Connessione Internet

Gli script `4.py` e `5.py` effettuano chiamate a:
- **Open-Meteo** — temperatura ambiente storica alla data del volo
- **PVGIS (EU JRC)** — ore equivalenti di sole (ESH) per la posizione GPS

Se non c'è connessione internet, vengono usati valori di default (`T_amb = 25°C`, `ESH = 3.18 h/giorno`).

---

## Dipendenze

| Libreria | Uso |
|---|---|
| `opencv-python` | Computer vision (ORB, omografia, contorni, rendering annotazioni) |
| `numpy` | Operazioni su matrici termiche e maschere |
| `rasterio` | Lettura/scrittura GeoTIFF, trasformazioni CRS geografiche |
| `Pillow` | Lettura EXIF/XMP, conversione immagini, salvataggio PDF |
| `rfdetr` | Modello AI RF-DETR per segmentazione istanza dei pannelli |
