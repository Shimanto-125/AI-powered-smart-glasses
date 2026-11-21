import os
from pathlib import Path
import cv2
import pytesseract
from pytesseract import Output

ROOT = Path(__file__).resolve().parent.parent
TESSDATA_DIR = ROOT / "models" / "tessdata"

if TESSDATA_DIR.exists():
    tess_prefix = str(TESSDATA_DIR.resolve()).replace("\\", "/") + "/"
    os.environ["TESSDATA_PREFIX"] = tess_prefix

def get_available_languages():
    try:
        return pytesseract.get_languages()
    except Exception:
        return ["eng"]

def preprocess_for_ocr(bgr):
    """Enhance contrast and sharpness for camera frames."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def read_text(bgr, lang="eng"):
    """Extract all text from image."""
    try:
        proc = preprocess_for_ocr(bgr)
        text = pytesseract.image_to_string(proc, lang=lang).strip()
        if not text:
            # Fallback to adaptive threshold
            thresh = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
            text = pytesseract.image_to_string(thresh, lang=lang).strip()
        return text
    except Exception as e:
        print("read_text error:", e)
        if "ben" in lang:
            try:
                return pytesseract.image_to_string(preprocess_for_ocr(bgr), lang="eng").strip()
            except Exception:
                pass
        return ""

def detect_and_draw_text(bgr, lang="eng", min_conf=25):
    """
    Detects text in frame, draws bounding boxes around words, 
    and returns (annotated_bgr, word_list, full_text_string).
    """
    out = bgr.copy()
    try:
        proc = preprocess_for_ocr(bgr)
        data = pytesseract.image_to_data(proc, lang=lang, output_type=Output.DICT)
        
        words = []
        n_boxes = len(data.get("text", []))
        for i in range(n_boxes):
            text = str(data["text"][i]).strip()
            try:
                conf = int(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1
                
            if conf >= min_conf and len(text) > 1:
                words.append(text)
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                # Draw high-visibility cyan bounding box for text
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 255), 2)
                cv2.putText(
                    out,
                    text,
                    (x, max(18, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 255),
                    2,
                )
        
        full_text = " ".join(words)
        if not full_text:
            full_text = read_text(bgr, lang=lang)
            
        return out, words, full_text
    except Exception as e:
        print("detect_and_draw_text error:", e)
        return out, [], ""
