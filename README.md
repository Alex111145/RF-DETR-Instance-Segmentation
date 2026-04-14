# Ispezione Fotovoltaico UAV — Pipeline Termografica AI

Sistema di analisi automatica di impianti fotovoltaici tramite ispezione UAV con doppia telecamera (IR + RGB). Utilizza un modello AI di segmentazione (RF-DETR) per identificare i pannelli e calcola l'efficienza termodinamica di ciascun modulo.

---

## Struttura del Progetto

```
RF-DETR-Instance-Segmentation/
├── addestramento/
│   ├── Step_0_patch.py           # Step 0: Taglio ortomosaico IR in patch 504×504 (GUI interattiva)
│   └── my_thesis_data/           # Patch estratte pronte per il caricamento su Roboflow
├── inferenza/
│   ├── Step_1_registrazione.py   # Step 1: Allineamento patch IR ↔ foto drone
│   ├── Step_2_Inferenza.py       # Step 2: Inferenza AI (rilevamento pannelli)
│   ├── Step_3_Temperatura.py     # Step 3: Analisi termica per pannello
│   ├── Step_4_Efficienza.py      # Step 4: Calcolo efficienza termodinamica
│   └── Step_5_Mosaico.py         # Step 5: Digital Twin + Report PDF + Mappa GeoTIFF
├── foto_drone/                   # Foto RJPEG del drone DJI (IR + dati GPS)
├── training_patches_ir/          # Tile 640×640 estratte dall'ortomosaico IR (pipeline standard)
├── ortomosaicoir.tif             # Ortomosaico infrarosso georeferenziato (inferenza)
├── ortomosaico.tif               # Ortomosaico infrarosso georeferenziato (addestramento)
├── weights.pt                    # Pesi del modello RF-DETR addestrato
├── sdk/
│   ├── linux/                    # Binari DJI SDK per Linux (dji_irp, libdirp.so, ecc.)
│   └── windows/                  # Binari DJI SDK per Windows
├── risultati_finali/             # Output generati automaticamente dagli script
│   ├── pair/                     # Coppie patch_IR + foto_drone originali
│   ├── registrazione_allineamento/  # Immagini affiancate per controllo visivo
│   ├── inferenza_pannelli/       # Patch annotate con box/maschere AI
│   ├── analisi_termica/          # Patch annotate con temperatura per pannello
│   ├── efficienza_risultati/     # CSV dati grezzi + immagini efficienza
│   │   ├── dati_grezzi.csv
│   │   └── config_analisi.json
│   ├── mappa_pannelli_difettosi.tif  # Ortomosaico IR con poligoni colorati
│   ├── report_pannelli_unici.csv    # CSV finale con ID, stato, perdite per pannello
│   └── report_tecnico.pdf           # Report PDF stile professionale
└── requirements.txt
```

---

## Prerequisiti

- Python **3.9 – 3.11**
- Linux (raccomandato) o Windows
- GPU opzionale ma consigliata per l'inferenza AI

---

## Setup SDK DJI (solo Linux)

Prima di avviare la pipeline, rendi eseguibile il binario DJI e registra le librerie condivise nel sistema:

```bash
# Rendi eseguibile il tool di estrazione termica
chmod +x /root/RF-DETR-Instance-Segmentation/sdk/linux/dji_irp

# Copia le librerie .so nella directory di sistema
cp /root/RF-DETR-Instance-Segmentation/sdk/linux/*.so* /usr/lib/

# Aggiorna la cache del linker
ldconfig
```

> Questi passaggi sono necessari per gli Step 3 e 4. Se non vengono eseguiti, gli script ricadono automaticamente sul parsing binario RAW (Strategia 1), che funziona per la maggior parte dei sensori DJI standard (640×512).

---

## Creazione dell'Ambiente Virtuale

```bash
# 1. Crea l'ambiente virtuale nella cartella del progetto
python3 -m venv .env

# 2. Attivalo (Linux/macOS)
source .env/bin/activate

# 3. Attivalo (Windows)
.env\Scripts\activate

# 4. Aggiorna pip
pip install --upgrade pip

# 5. Installa le dipendenze
pip install -r RF-DETR-Instance-Segmentation/requirements.txt
```

> **Nota su rfdetr:** se il pacchetto non è su PyPI nella versione richiesta, installalo da sorgente o da wheel fornita dal progetto:
> ```bash
> pip install rfdetr --extra-index-url <url-repo-privato>
> ```

---

# PARTE 1 — ADDESTRAMENTO

Questa sezione descrive come preparare il dataset di patch IR, caricarlo su Roboflow per l'annotazione e addestrare il modello RF-DETR.

## Dati di Input Necessari per l'Addestramento

Prima di avviare lo Step 0 assicurati di avere nella cartella `RF-DETR-Instance-Segmentation/`:

| File / Cartella | Descrizione |
|---|---|
| `ortomosaico.tif` | Ortomosaico IR georeferenziato (GeoTIFF) da usare per il taglio delle patch |

---

## Step 0 — Taglio Ortomosaico IR in Patch

```bash
cd RF-DETR-Instance-Segmentation/addestramento
python Step_0_patch.py
```

**Cosa fa:**

Carica l'ortomosaico `ortomosaico.tif` dalla cartella radice del progetto e lo riduce al 15% per la preview (in modo da visualizzarlo anche su mosaici di grandi dimensioni). Apre una finestra OpenCV dove si cliccano **4 punti** per definire un'area poligonale di interesse. Una volta confermata la selezione, calcola il bounding rect del poligono e genera una maschera binaria pixel-level: solo le patch il cui **centro cade all'interno del poligono** vengono ritagliate. Le patch con meno del 20% di pixel non-neri (bordi neri del mosaico) vengono scartate automaticamente. I file vengono salvati nella cartella `my_thesis_data/` con nome `tile_col_X_row_Y.jpg`, dove X e Y sono gli offset in pixel — indispensabili negli step successivi per riproiettare le coordinate.

**Comandi interattivi durante la selezione:**

| Tasto | Azione |
|---|---|
| Click sinistro | Aggiunge un punto (massimo 4) |
| `r` | Resetta tutti i punti e ricomincia |
| `c` | Conferma la selezione (solo con 4 punti) |
| `ESC` | Esce senza salvare |

**Parametri interni configurabili (costanti in cima allo script):**

| Costante | Valore Default | Descrizione |
|---|---|---|
| `TILE_SIZE` | `504` px | Dimensione delle patch quadrate estratte |
| `OVERLAP` | `0.30` | Sovrapposizione tra patch adiacenti (30%) |
| `IGNORE_EMPTY_THRESHOLD` | `0.20` | Soglia minima pixel non-neri per tenere una patch |
| `INPUT_IMAGE_PATH` | `../ortomosaico.tif` | Path del mosaico di input |
| `OUTPUT_DIR` | `../my_thesis_data/` | Cartella di destinazione delle patch |

**Output generato:**

| File | Descrizione |
|---|---|
| `my_thesis_data/tile_col_X_row_Y.jpg` | Patch 504×504 px ritagliata dal mosaico, con offset X, Y in pixel |

**Librerie chiave:** `OpenCV` (imread, resize, fillPoly, boundingRect, imwrite), `numpy`

---

## Caricamento su Roboflow e Annotazione

Una volta generate le patch in `my_thesis_data/`, il flusso di lavoro consigliato per costruire il dataset di addestramento è il seguente:

**1. Crea un progetto su Roboflow**

Accedi a [roboflow.com](https://roboflow.com) e crea un nuovo progetto di tipo **Instance Segmentation** (non Object Detection), con 2 classi: `Sano` e `Difettoso`.

**2. Carica le patch**

Trascina l'intera cartella `my_thesis_data/` nell'interfaccia di upload di Roboflow, oppure usa la CLI:

```bash
pip install roboflow
```

```python
from roboflow import Roboflow
rf = Roboflow(api_key="LA_TUA_API_KEY")
project = rf.workspace("tuo-workspace").project("fotovoltaico-ir")
project.upload("../my_thesis_data/")
```

**3. Annota i pannelli**

Usa lo strumento di segmentazione poligonale di Roboflow per disegnare le maschere dei pannelli su ciascuna patch. Assegna la classe `Sano` ai pannelli termicamente uniformi e `Difettoso` a quelli con anomalie termiche visibili (hotspot, stringhe interrotte, ombreggiature).

**4. Genera il dataset e scarica i pesi**

Applica le augmentation desiderate (flip, rotazione, variazione di luminosità) e genera la versione del dataset. Esporta in formato **COCO JSON** o **YOLOv8 Segmentation** compatibile con RF-DETR. Al termine dell'addestramento, scarica il file `weights.pt` e posizionalo nella cartella radice `RF-DETR-Instance-Segmentation/`.

---

# PARTE 2 — INFERENZA

Questa sezione descrive come eseguire la pipeline completa di analisi termica su un impianto fotovoltaico a partire dalle foto del drone e dall'ortomosaico IR, usando il modello già addestrato.

## Dati di Input Necessari per l'Inferenza

Prima di avviare la pipeline assicurati di avere nella cartella `RF-DETR-Instance-Segmentation/`:

| File / Cartella | Descrizione |
|---|---|
| `ortomosaicoir.tif` | Ortomosaico IR georeferenziato (GeoTIFF) |
| `training_patches_ir/*.jpg` | Tile IR (512×512) con nome `tile_col_X_row_Y.jpg` |
| `foto_drone/*.JPG` | Foto RJPEG originali del drone DJI con GPS EXIF |
| `weights.pt` | Pesi del modello RF-DETR (2 classi: Sano/Difettoso) |

---

## Esecuzione della Pipeline

Gli script vanno eseguiti **in sequenza** dalla cartella `inferenza/`:

```bash
cd RF-DETR-Instance-Segmentation/inferenza
```

---

### Step 1 — Allineamento IR ↔ Drone

```bash
python Step_1_registrazione.py
```

**Cosa fa:** Legge le tile IR da `training_patches_ir/`, calcola le coordinate geografiche del centro di ogni tile tramite il CRS dell'ortomosaico IR, trova la foto drone più vicina per GPS (EXIF), applica feature matching ORB + omografia per allinearle visivamente. Salva:

- `risultati_finali/pair/pairN_patch.jpg` — copia della tile IR (bit-a-bit)
- `risultati_finali/pair/pairN_drone.jpg` — copia della foto drone (bit-a-bit)
- `risultati_finali/registrazione_allineamento/` — immagini affiancate per verifica visiva

**Librerie chiave:** `rasterio` (CRS/transform), `OpenCV` (ORB, BFMatcher, findHomography), `PIL` (lettura EXIF GPS)

---

### Step 2 — Inferenza AI (Rilevamento Pannelli)

```bash
python Step_2_Inferenza.py
python Step_2_Inferenza.py --threshold 0.30
```

**Cosa fa:** Carica il modello `RFDETRSegLarge` con i pesi `weights.pt`. Per ogni `pairN_patch.jpg` esegue la segmentazione con soglia di confidenza configurabile. Disegna i contorni delle maschere (o bounding box se la maschera non è disponibile) con colore verde (Sano) o rosso (Difettoso). Salva i risultati annotati in `risultati_finali/inferenza_pannelli/`.

**Librerie chiave:** `rfdetr` (modello AI), `OpenCV` (findContours, drawContours, moments), `PIL` (conversione BGR→RGB per il modello)

---

### Step 3 — Analisi Termica per Pannello

```bash
python Step_3_Temperatura.py
python Step_3_Temperatura.py --threshold 0.30
```

**Cosa fa:** Combina inferenza AI + estrazione dati termici dalle foto RJPEG DJI. Usa due strategie per i dati termici:

1. **RAW parsing binario**: cerca il blob di dati termici dopo il marker JPEG `0xFFD9`, decodifica come `uint16` in Kelvin → °C
2. **Fallback CLI SDK DJI** (`sdk/linux/dji_irp`): invoca il tool DJI per generare un file `.raw` temporaneo

Per ogni pannello rilevato, proietta la maschera AI nello spazio della foto drone tramite omografia ORB, campiona la matrice termica nella ROI e calcola la temperatura media. Annota ogni pannello con la temperatura in °C.

**Librerie chiave:** `rfdetr`, `OpenCV` (warpPerspective, ORB), `numpy` (frombuffer, mean), `subprocess` (SDK DJI)

---

### Step 4 — Calcolo Efficienza Termodinamica

```bash
python Step_4_Efficienza.py
python Step_4_Efficienza.py --threshold 0.30
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
python Step_5_Mosaico.py
```

**Cosa fa:** Legge i risultati del Step 4 e genera i tre output finali:

1. **Mappa GeoTIFF** (`mappa_pannelli_difettosi.tif`): riproietta i contorni dei pannelli dallo spazio pixel IR alle coordinate geografiche usando `rasterio.warp.transform`. Disegna poligoni colorati sull'ortomosaico IR:
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
| `SOGLIA_AREA_PX` | `Step_4_Efficienza.py`, `Step_5_Mosaico.py` | `10000` px | Area minima maschera (filtra rumore) |
| `G_IRR_STC` | `Step_5_Mosaico.py` | `1000.0 W/m²` | Irraggiamento condizioni standard |
| `COSTO_KWH` | `Step_5_Mosaico.py` | `0.40 €` | Prezzo energia per calcolo economico |
| `GIORNI_UTIL` | `Step_5_Mosaico.py` | `300` giorni | Giorni di produzione annua stimati |
| `EPSILON_VETRO` | `Step_4_Efficienza.py` | `0.90` | Emissività del vetro fotovoltaico |

---

## Output Finali

| File | Descrizione |
|---|---|
| `risultati_finali/mappa_pannelli_difettosi.tif` | Ortomosaico IR georeferenziato con pannelli colorati per stato |
| `risultati_finali/report_pannelli_unici.csv` | Tabella con ID, stato, efficienza e perdite economiche per pannello |
| `risultati_finali/report_tecnico.pdf` | Report PDF professionale con grafici e analisi economica |

---

## Connessione Internet

Gli script `Step_4_Efficienza.py` e `Step_5_Mosaico.py` effettuano chiamate a:

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
