"""
AI Smart Glasses - Real-Time Live Webcam Detection & Voice Feedback
Run via: python live_detect.py
Controls:
 - 'q' : Quit
 - 's' : Speak current scene immediately
 - 'a' : Toggle Auto-Voice guidance on/off
 - '+' / '-' : Adjust detection confidence threshold
"""

import argparse
import time
import cv2
from pathlib import Path
from ultralytics import YOLO

from config import BASE_MODEL, BEST_WEIGHTS, CONFIDENCE
from src.ocr import detect_and_draw_text
from src.speech import is_speaking, speak
from src.vision import analyze


def parse_args():
    parser = argparse.ArgumentParser(description="Live Camera Object Detection")
    default_m = str(BEST_WEIGHTS) if BEST_WEIGHTS.exists() else BASE_MODEL
    parser.add_argument("--model", type=str, default=default_m, help="Path to YOLO model (.pt)")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold (default: 0.20)")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    return parser.parse_args()


def open_camera_by_index(index):
    """Try to open the requested camera index across DirectShow and MSMF."""
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        try:
            cap = cv2.VideoCapture(int(index), backend)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"[*] Successfully connected to Camera {index} via backend {backend}")
                    return cap
                cap.release()
        except Exception:
            pass
    return None


def main():
    args = parse_args()
    model_path = args.model
    print(f"[*] Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    print(f"[*] Opening camera index {args.cam}...")
    cap = open_camera_by_index(args.cam)
    if cap is None:
        print(f"\n[!] Could not open Camera {args.cam}. Trying other available cameras...")
        for fallback_idx in [0, 1, 2]:
            if fallback_idx != args.cam:
                cap = open_camera_by_index(fallback_idx)
                if cap is not None:
                    print(f"[*] Switched to working Camera {fallback_idx}")
                    break

    if cap is None:
        print("\n" + "!" * 60)
        print("[!] ERROR: Could not open any webcam (tried index 0, 1, 2).")
        print("    Possible causes:")
        print("    1. Another application (like Streamlit or browser) is currently using the camera.")
        print("    2. Windows Camera privacy settings are blocking camera access.")
        print("    3. Webcam is disconnected or driver is disabled.")
        print("!" * 60 + "\n")
        return

    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 55)
    print(" 👓 AI SMART GLASSES - LIVE DETECTION STARTED")
    print(" - Press 'q' to Quit")
    print(" - Press 's' to Speak current scene immediately")
    print(" - Press 'a' to Toggle Auto-Voice guidance on/off")
    print("=" * 55 + "\n")

    auto_speak = True
    auto_ocr = False
    speech_interval = 4.0
    last_speak_time = 0.0
    last_ocr_time = 0.0
    prev_time = time.time()
    current_conf = args.conf
    cached_ocr_text = ""

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[!] Warning: Lost camera frame.")
            break

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time > prev_time else 0.0
        prev_time = current_time

        # Run detection and scene analysis
        out, dets, msgs = analyze(frame, model, conf=current_conf, focal_px=None)

        # Run Auto-OCR if enabled
        if auto_ocr and (current_time - last_ocr_time >= 3.0):
            out, words, text_found = detect_and_draw_text(frame)
            if text_found:
                cached_ocr_text = text_found
                if auto_speak and not is_speaking():
                    print(f"[🗣️ OCR Voice]: {text_found}")
                    speak(f"Text reads: {text_found}", async_mode=True)
            last_ocr_time = current_time

        # Draw overlays
        cv2.putText(
            out,
            f"FPS: {fps:.1f} | Detections: {len(dets)} | Conf: {current_conf:.2f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        voice_status = f"Voice: {'ON' if auto_speak else 'OFF (a)'} | OCR: {'ON' if auto_ocr else 'OFF (o)'} | Read Now: 't'"
        color = (0, 255, 255) if (auto_speak or auto_ocr) else (180, 180, 180)
        cv2.putText(
            out,
            voice_status,
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        if cached_ocr_text:
            cv2.putText(
                out,
                f"OCR: {cached_ocr_text[:35]}...",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 190, 0),
                2,
            )

        # Auto speech feedback
        if auto_speak and msgs and (current_time - last_speak_time >= speech_interval):
            if not is_speaking():
                spoken = ". ".join(msgs[:3])
                print(f"[🔊 Audio]: {spoken}")
                speak(spoken, lang="bn", async_mode=True)
                last_speak_time = current_time

        cv2.imshow("AI Smart Glasses - Live Camera Feed", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            if msgs:
                spoken = ". ".join(msgs[:3])
                print(f"[🔊 Manual Audio]: {spoken}")
                speak(spoken, lang="bn", async_mode=True)
                last_speak_time = current_time
            else:
                print("[🔊 Manual Audio]: কোনো পরিচিত বস্তু নেই।")
                speak("কোনো পরিচিত বস্তু সামনে নেই", lang="bn", async_mode=True)
        elif key == ord("t"):
            # Trigger immediate OCR on current frame
            print("[📖 Reading text from camera frame...]")
            _, words, text = detect_and_draw_text(frame)
            if text:
                print(f"[📖 OCR Found]: {text}")
                cached_ocr_text = text
                speak(f"লেখা পাওয়া গেছে: {text}", lang="bn", async_mode=True)
            else:
                print("[📖 OCR]: কোনো স্পষ্ট লেখা দেখা যাচ্ছে না।")
                speak("কোনো স্পষ্ট লেখা পাওয়া যায়নি", lang="bn", async_mode=True)
        elif key == ord("o"):
            auto_ocr = not auto_ocr
            print(f"[*] Auto-OCR is now: {'ON' if auto_ocr else 'OFF'}")
        elif key == ord("a"):
            auto_speak = not auto_speak
            print(f"[*] Auto-voice is now: {'ON' if auto_speak else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Live camera stopped.")


if __name__ == "__main__":
    main()
