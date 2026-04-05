
"""
taglia_mosaico_training.py
==========================
Taglia l'ortomosaico in patch 640x640 per il training di MaskDINO.
Supporta selezione interattiva dell'area (poligono) o taglio completo.

Uso:
    python taglia_mosaico_training.py                    # con GUI selezione area
    python taglia_mosaico_training.py --no-gui           # taglia tutto il mosaico
    python taglia_mosaico_training.py --input /path/ortomosaico.tif --output /path/patches
"""

import cv2
import os
import sys
import argparse
import numpy as np

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

def get_base_dir():
    candidates = [
        os.path.expanduser("~/Desktop/3DTARGHET"),
        os.path.expanduser("~/Desktop/Uni/Triennale/Tesi/python/tagliomosaico"),
        os.path.expanduser("~/DinoMask"),
        r"C:\Users\dell\Desktop\Tesi\python\tagliomosaico",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()

TILE_SIZE   = 640    # dimensione patch — NON modificare, deve corrispondere al training
OVERLAP     = 0.20   # 20% overlap tra patch adiacenti
MIN_CONTENT = 0.20   # scarta patch con meno del 20% di pixel non-neri

# ==============================================================================
# GUI — SELEZIONE AREA CON 4 PUNTI
# ==============================================================================

_points = []

def _click(event, x, y, flags, img):
    if event == cv2.EVENT_LBUTTONDOWN and len(_points) < 4:
        _points.append((x, y))
        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
        if len(_points) > 1:
            cv2.line(img, _points[-2], _points[-1], (0, 255, 0), 2)
        if len(_points) == 4:
            cv2.line(img, _points[3], _points[0], (0, 255, 0), 2)
            cv2.putText(img, "Premi C per confermare", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Seleziona area — 4 punti | R=reset | C=conferma | ESC=taglia tutto", img)


def select_area_gui(full_img):
    """
    Mostra il mosaico ridotto e chiede all'utente di cliccare 4 punti.
    Ritorna la maschera poligonale in coordinate originali, oppure None (taglia tutto).
    """
    global _points
    _points = []

    h, w = full_img.shape[:2]
    scale = min(0.15, 1400 / max(w, h))
    preview = cv2.resize(full_img, (int(w * scale), int(h * scale)))
    clone = preview.copy()

    win = "Seleziona area — 4 punti | R=reset | C=conferma | ESC=taglia tutto"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _click, preview)
    cv2.putText(preview, "Clicca 4 punti sull'area di interesse", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow(win, preview)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            _points = []
            preview[:] = clone[:]
            cv2.putText(preview, "Clicca 4 punti sull'area di interesse", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow(win, preview)
        elif key == ord('c') and len(_points) == 4:
            cv2.destroyAllWindows()
            # Riporta punti alla scala originale
            pts = np.array(
                [(int(p[0] / scale), int(p[1] / scale)) for p in _points], np.int32
            )
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            return mask, pts
        elif key == 27:  # ESC → taglia tutto
            cv2.destroyAllWindows()
            print("  ESC premuto → taglio dell'intero mosaico")
            return None, None

# ==============================================================================
# TAGLIO PATCH
# ==============================================================================

def taglia(full_img, output_dir, mask=None, pts=None):
    """
    Taglia full_img in patch TILE_SIZE×TILE_SIZE con overlap OVERLAP.
    Se mask è fornita, processa solo le patch il cui centro è dentro il poligono.
    """
    h, w = full_img.shape[:2]
    step = max(1, int(TILE_SIZE * (1 - OVERLAP)))

    # Bounding box dell'area da processare
    if pts is not None:
        x_s, y_s, w_box, h_box = cv2.boundingRect(pts)
        x_e = min(x_s + w_box, w)
        y_e = min(y_s + h_box, h)
    else:
        x_s, y_s, x_e, y_e = 0, 0, w, h

    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    skipped_vuote = 0
    skipped_fuori = 0

    total_y = len(range(y_s, y_e - TILE_SIZE + 1, step))
    total_x = len(range(x_s, x_e - TILE_SIZE + 1, step))
    print(f"  Griglia stimata: {total_x} × {total_y} = {total_x * total_y} patch candidate")

    for row, y in enumerate(range(y_s, y_e - TILE_SIZE + 1, step)):
        for col, x in enumerate(range(x_s, x_e - TILE_SIZE + 1, step)):

            # Controlla che il centro della patch sia dentro il poligono
            cx, cy = x + TILE_SIZE // 2, y + TILE_SIZE // 2
            if mask is not None and mask[cy, cx] == 0:
                skipped_fuori += 1
                continue

            patch = full_img[y: y + TILE_SIZE, x: x + TILE_SIZE]

            # Scarta patch quasi vuote (bordi neri del mosaico)
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
            content = cv2.countNonZero(gray) / (TILE_SIZE * TILE_SIZE)
            if content < MIN_CONTENT:
                skipped_vuote += 1
                continue

            # Nome con offset codificato → compatibile con coordinate_mapper.py
            fname = f"tile_col_{x}_row_{y}.jpg"
            cv2.imwrite(os.path.join(output_dir, fname), patch,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        # Progress ogni 10 righe
        if (row + 1) % 10 == 0:
            print(f"  ... riga {row+1}/{total_y}  patch salvate: {saved}")

    return saved, skipped_vuote, skipped_fuori

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    global TILE_SIZE, OVERLAP  # <-- SPOSTATO QUI IN ALTO

    parser = argparse.ArgumentParser(description="Taglia ortomosaico in patch 640x640 per training")
    parser.add_argument("--input",   "-i", default=os.path.join(BASE_DIR, "ortomosaico.tif"),
                        help="Path ortomosaico input")
    parser.add_argument("--output",  "-o", default=os.path.join(BASE_DIR, "training_patches"),
                        help="Cartella output patch")
    parser.add_argument("--no-gui",  action="store_true",
                        help="Salta la selezione GUI e taglia tutto il mosaico")
    parser.add_argument("--tile",    type=int, default=TILE_SIZE,  help="Dimensione patch (default: 640)")
    parser.add_argument("--overlap", type=float, default=OVERLAP,  help="Overlap 0.0–0.9 (default: 0.20)")
    args = parser.parse_args()

    TILE_SIZE = args.tile
    OVERLAP   = args.overlap

    print(f"\n{'='*55}")
    print(f"  Taglia Mosaico → Patch Training")
    print(f"{'='*55}")
    print(f"  Input    : {args.input}")
    print(f"  Output   : {args.output}")
    print(f"  Tile     : {TILE_SIZE}×{TILE_SIZE} px")
    print(f"  Overlap  : {int(OVERLAP*100)}%")
    print(f"  Step     : {int(TILE_SIZE*(1-OVERLAP))} px")
    print(f"{'='*55}\n")

    if not os.path.exists(args.input):
        print(f"❌ File non trovato: {args.input}")
        sys.exit(1)

    print("🔄 Caricamento mosaico...")
    img = cv2.imread(args.input)
    if img is None:
        print("❌ Impossibile leggere il file (formato non supportato da OpenCV?)")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"   Dimensioni: {w}×{h} px  ({w*h/1e6:.1f} Mpx)")

    # Selezione area
    mask, pts = None, None
    if not args.no_gui:
        print("\n📌 Selezione area di interesse (chiudi la finestra con ESC per tagliare tutto)...")
        mask, pts = select_area_gui(img)
    else:
        print("  Modalità --no-gui: taglio dell'intero mosaico")

    # Taglio
    print(f"\n✂️  Taglio in corso...")
    saved, skip_vuote, skip_fuori = taglia(img, args.output, mask, pts)

    print(f"\n{'='*55}")
    print(f"  ✅ Patch salvate      : {saved}")
    print(f"  ⬛ Scartate (vuote)   : {skip_vuote}")
    if mask is not None:
        print(f"  🔲 Scartate (fuori)  : {skip_fuori}")
    print(f"  📁 Output             : {args.output}")
    print(f"{'='*55}")
    print(f"\n➡️  Prossimo step: carica le patch su Roboflow / Label Studio per annotarle")


if __name__ == "__main__":
    main()
