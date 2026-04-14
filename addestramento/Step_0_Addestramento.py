import cv2
import os
import glob
import random
import shutil
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "ortomosaico.tif")
OUTPUT_DIR = os.path.join(BASE_DIR, "my_thesis_data")


TILE_SIZE = 504
OVERLAP = 0.30
IGNORE_EMPTY_THRESHOLD = 0.2

points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
          
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(param, points[-2], points[-1], (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(param, points[3], points[0], (0, 255, 0), 2)
            cv2.imshow("SELEZIONE 4 PUNTI", param)



def main():
    global points
    print(f" Cartella di lavoro: {BASE_DIR}")

    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"ERRORE: File {INPUT_IMAGE_PATH} non trovato!")
        return

    print(f" Caricamento mosaico (potrebbe richiedere tempo)...")
    full_img = cv2.imread(INPUT_IMAGE_PATH)
    if full_img is None: 
        print(" Impossibile leggere l'immagine.")
        return

    h_orig, w_orig = full_img.shape[:2]
    scale = 0.15 
    preview_img = cv2.resize(full_img, (int(w_orig * scale), int(h_orig * scale)))
    clone = preview_img.copy()


    print("\n1️ Clicca su 4 punti per definire l'area.")
    print("   Premi 'r' per resettare o 'c' per confermare dopo i 4 punti.")
    
    window_name = "SELEZIONE 4 PUNTI"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event, preview_img)

    while True:
        cv2.imshow(window_name, preview_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"): # Reset
            points = []
            preview_img = clone.copy()
            cv2.setMouseCallback(window_name, click_event, preview_img)
        elif key == ord("c") and len(points) == 4: # Confirm
            break
        elif key == 27: # ESC
            return

    cv2.destroyAllWindows()

  
    pts_orig = np.array([(int(p[0]/scale), int(p[1]/scale)) for p in points], np.int32)
    x_s, y_s, w_box, h_box = cv2.boundingRect(pts_orig)
    x_e, y_e = x_s + w_box, y_s + h_box

    mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_orig], 255)


    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
        print(f" Creata cartella: {OUTPUT_DIR}")

    step = int(TILE_SIZE * (1 - OVERLAP))
    saved_count = 0

    print(f"\n Inizio taglio patch {TILE_SIZE}x{TILE_SIZE}...")
    
    for y in range(y_s, min(y_e, h_orig) - TILE_SIZE + 1, step):
        for x in range(x_s, min(x_e, w_orig) - TILE_SIZE + 1, step):
            
    
            center_x, center_y = x + TILE_SIZE // 2, y + TILE_SIZE // 2
            if mask[center_y, center_x] == 0:
                continue

            patch = full_img[y : y + TILE_SIZE, x : x + TILE_SIZE]
            
   
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if (cv2.countNonZero(gray) / (TILE_SIZE**2)) >= IGNORE_EMPTY_THRESHOLD:
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"tile_col_{x}_row_{y}.jpg"), patch)
                saved_count += 1

    print(f"\nCOMPLETATO! Tutte le {saved_count} patch sono in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
