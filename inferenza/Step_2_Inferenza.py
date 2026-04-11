
import os
import glob
import argparse
import warnings
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")


CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR           = os.path.dirname(CURRENT_SCRIPT_DIR)

WEIGHTS_PATH  = os.path.join(BASE_DIR, "weights.pt")
OUTPUT_DIR    = os.path.join(BASE_DIR, "risultati_finali")
INPUT_DIR     = os.path.join(OUTPUT_DIR, "pair")
INFERENCE_DIR = os.path.join(OUTPUT_DIR, "inferenza_pannelli")

COLOR_MAP = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 255, 0)}
NAME_MAP  = {0: "SANO", 1: "DIFETTOSO", 2: "SANO"}


def stima_batch_size(vram_free_mb: int) -> int:
    """Stima batch_size ottimale in base alla VRAM libera dopo il caricamento del modello."""

    vram_per_img_mb = 500
    reserved_mb     = 1000   # margine di sicurezza
    available_mb    = max(0, vram_free_mb - reserved_mb)
    return max(1, min(available_mb // vram_per_img_mb, 32))


def load_image(path):
    """Carica immagine; restituisce (path, img_bgr, img_pil) oppure None."""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None
   
    img_pil = Image.fromarray(img_bgr[:, :, ::-1])
    return path, img_bgr, img_pil


def save_image(out_path, canvas):
    cv2.imwrite(out_path, canvas)


def disegna_rilevamento(img, det_info):
    canvas = img.copy()
    for d in det_info:
        cid        = d['class_id']
        score      = d['score']
        class_name = NAME_MAP.get(cid, f"ID:{cid}")
        color      = COLOR_MAP.get(cid, (255, 255, 255))
        label      = f"{class_name} {score:.0%}"
        mask       = d['mask']

        if mask is not None:
            mask_u8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c_big = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c_big) > 100:
                    cv2.drawContours(canvas, [c_big], 0, color, 2)
                    M = cv2.moments(c_big)
                    if M["m00"] != 0:
                        tx, ty = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    else:
                        tx, ty = c_big[0][0][0], c_big[0][0][1]
                    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        else:
            x1, y1, x2, y2 = d['xyxy']
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=INPUT_DIR)
    parser.add_argument("--output",     default=INFERENCE_DIR)
    parser.add_argument("--threshold",  type=float, default=0.50)
    parser.add_argument("--batch-size", type=int,   default=0,
                        help="Batch size GPU (0 = auto da VRAM libera)")
    parser.add_argument("--prefetch",   type=int,   default=4,
                        help="Thread per pre-caricare immagini in background")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERRORE: Cartella {args.input} non trovata.")
        return

    os.makedirs(args.output, exist_ok=True)

 
    if torch.cuda.is_available():
        props         = torch.cuda.get_device_properties(0)
        vram_total_mb = props.total_memory // 1024 ** 2
        print(f"[*] GPU : {props.name}")
        print(f"[*] VRAM: {vram_total_mb} MB totali")
    else:
        vram_total_mb = 0
        print("[!] CUDA non disponibile — uso CPU (lento)")

    # ── Carica modello ────────────────────────────────────────────────────
    from rfdetr import RFDETRSegLarge
    print("[*] Caricamento modello (3 classi)...")
    model = RFDETRSegLarge(pretrain_weights=WEIGHTS_PATH, num_classes=3)


    if torch.cuda.is_available():
        vram_used_mb = torch.cuda.memory_allocated(0) // 1024 ** 2
        vram_free_mb = vram_total_mb - vram_used_mb
        print(f"[*] VRAM dopo modello: {vram_free_mb} MB liberi ({vram_used_mb} MB usati dal modello)")
    else:
        vram_free_mb = 0

    batch_size = args.batch_size if args.batch_size > 0 else stima_batch_size(vram_free_mb)
    print(f"[*] Batch size: {batch_size}")


    print(f"[*] Compilazione JIT float16 (batch={batch_size}) — attendi...")
    model.optimize_for_inference(
        compile=True,
        batch_size=batch_size,
        dtype=torch.float16,
    )
    print("[*] Modello ottimizzato.")

    files = sorted(glob.glob(os.path.join(args.input, "*_patch.jpg")))
    print(f"[*] Analisi di {len(files)} patch...")

    save_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="save")

    with ThreadPoolExecutor(max_workers=args.prefetch, thread_name_prefix="load") as load_pool:
        futures = [load_pool.submit(load_image, p) for p in files]

   
        batches = [futures[i : i + batch_size] for i in range(0, len(futures), batch_size)]

        pbar = tqdm(total=len(files), desc="Inferenza")
        for batch_futs in batches:
            loaded = [f.result() for f in batch_futs]
            loaded = [x for x in loaded if x is not None]
            if not loaded:
                pbar.update(len(batch_futs))
                continue

            n_real = len(loaded)

 
            if n_real < batch_size:
                pad     = [loaded[-1]] * (batch_size - n_real)
                pil_in  = [x[2] for x in loaded + pad]
            else:
                pil_in  = [x[2] for x in loaded]

            results_batch = model.predict(pil_in, threshold=args.threshold)
           
            if not isinstance(results_batch, list):
                results_batch = [results_batch]

            for idx in range(n_real):
                path, img_bgr, _ = loaded[idx]
                results = results_batch[idx]

                has_mask  = hasattr(results, 'mask') and results.mask is not None
                lista_det = []
                if results is not None and len(results.xyxy) > 0:
                    for k in range(len(results.xyxy)):
                        lista_det.append({
                            'class_id': int(results.class_id[k]),
                            'score':    float(results.confidence[k]),
                            'xyxy':     results.xyxy[k].astype(int),
                            'mask':     results.mask[k] if has_mask else None,
                        })

                if lista_det:
                    annotata = disegna_rilevamento(img_bgr, lista_det)
                    out_path = os.path.join(args.output, f"det_{os.path.basename(path)}")
                    save_pool.submit(save_image, out_path, annotata)

            pbar.update(len(batch_futs))
        pbar.close()

    save_pool.shutdown(wait=True)
    print(f"\n[FINE] Elaborazione completata. Risultati in: {args.output}")


if __name__ == "__main__":
    main()
