import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from config import BASE_MODEL, BEST_WEIGHTS, CONFIDENCE
from src.ocr import detect_and_draw_text, get_available_languages, read_text
from src.speech import is_speaking, make_audio, speak
from src.vision import analyze
from src.weather import current_weather

st.set_page_config(page_title="AI Smart Glasses", page_icon="👓", layout="wide")
st.title("👓 AI-Powered Smart Glasses")
st.caption("Academic prototype: real-time object awareness, OCR and voice feedback")
st.warning("Prototype only. Never rely on these predictions for crossing roads or avoiding hazards.")

# Sidebar: Model Selection & Cache Management
st.sidebar.header("⚙️ Model & Detection Settings")

available_models = {}
if BEST_WEIGHTS.exists():
    available_models["🎯 Custom Trained Model (models/best.pt)"] = str(BEST_WEIGHTS)
available_models["⚡ Official Pretrained YOLO11 (yolo11n.pt)"] = BASE_MODEL

selected_model_label = st.sidebar.selectbox(
    "Choose Active Model / সক্রিয় মডেল:",
    list(available_models.keys()),
    index=0,
    key="sidebar_model_select",
)
selected_model_path = available_models[selected_model_label]

# Global Confidence Slider (applies to both Live and Upload modes)
global_conf = st.sidebar.slider(
    "Confidence Threshold / কনফিডেন্স মাত্রা",
    min_value=0.05,
    max_value=0.90,
    value=0.20,
    step=0.05,
    key="sidebar_conf_slider",
    help="কম্পিউটার ভিশন ডিটেকশনের থ্রেশহোল্ড। মডেল কম ডিটেক্ট করলে এটি ০.১০ বা ০.১৫-এ নামিয়ে দিন।",
)

if st.sidebar.button("🔄 Clear Cache & Reload Model", key="sidebar_clear_cache_btn"):
    st.cache_resource.clear()
    st.sidebar.success("ক্যাশ ক্লিয়ার হয়েছে এবং মডেল রিলোড হয়েছে!")
    st.rerun()


@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


try:
    model = load_yolo_model(selected_model_path)
    st.sidebar.success(f"মডেল লোড হয়েছে: `{selected_model_path}`")
    st.sidebar.info(f"মোট ক্লাস: {len(model.names)} টি")
except Exception as e:
    st.error(f"মডেল লোড করতে সমস্যা: {e}")
    st.stop()


def open_specific_camera(index):
    """Attempt to open ONLY the user-selected camera index across multiple backends with warm-up."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_V4L2, cv2.CAP_ANY]
    for backend in backends:
        try:
            cap = cv2.VideoCapture(int(index), backend)
            if cap.isOpened():
                # Allow camera sensor a few frames to initialize/warm up
                for _ in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        return cap
                    time.sleep(0.05)
                cap.release()
        except Exception:
            pass
    return None


def scan_available_cameras(max_check=4):
    """Scan and return list of working camera indices."""
    active = []
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_V4L2, cv2.CAP_ANY]
    for i in range(max_check):
        for backend in backends:
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    for _ in range(2):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            active.append(i)
                            cap.release()
                            break
                        time.sleep(0.05)
                    if i in active:
                        break
                    cap.release()
            except Exception:
                pass
    return active


# Mode selector: Live Camera vs Image Upload / Snapshot
mode = st.radio(
    "Select Mode / মোড বেছে নিন:",
    ["🎥 Live Camera (রিয়েল-টাইম লাইভ ক্যামেরা, অবজেক্ট ও লেখা পড়া)", "📸 Snapshot & Upload (ছবি আপলোড / ফটো ও OCR)"],
    horizontal=True,
    key="main_mode_radio",
)

if mode == "🎥 Live Camera (রিয়েল-টাইম লাইভ ক্যামেরা, অবজেক্ট ও লেখা পড়া)":
    st.subheader("🎥 Real-Time Live Detection, OCR & Voice Feedback")
    st.info(
        "💡 **ক্লাউড বনাম লোকাল ব্যবহার তথ্য:**\n"
        "- **Streamlit Cloud (অনলাইনে):** ক্লাউড হোস্টিং দূরবর্তী সার্ভারে চলে, তাই সার্ভার সরাসরি আপনার ল্যাপটপের ইউএসবি ক্যামেরা এক্সেস করতে পারে না। অনলাইনে ক্যামেরা টেস্ট করতে উপরের **'📸 Snapshot & Upload'** মোডে যান (সেখানে ব্রাউজার ক্যামেরা সরাসরি কাজ করে)।\n"
        "- **লোকাল পিসি / স্মার্ট গ্লাস হার্ডওয়্যার:** লাইভ ভিডিও স্ট্রিম চালানোর জন্য আপনার পিসির টার্মিনালে লোকালভাবে চালান: `streamlit run app.py` বা `python live_detect.py`।"
    )
    st.write("ক্যামেরা অন করলে স্বয়ংক্রিয়ভাবে অবজেক্ট ডিটেক্ট করবে এবং অপশন অন থাকলে চারপাশের লেখাও (OCR) পড়বে।")

    # Camera selector & scan
    c_cam_col1, c_cam_col2 = st.columns([2, 1])
    with c_cam_col1:
        camera_options = {
            "Camera 0 (ল্যাপটপের নিজস্ব ক্যামেরা)": 0,
            "Camera 1 (এক্সটার্নাল ইউএসবি ওয়েবক্যাম / স্মার্ট গ্লাস)": 1,
            "Camera 2 (সেকেন্ডারি ডিভাইস)": 2,
            "Camera 3": 3,
        }
        selected_cam_name = st.selectbox(
            "ক্যামেরা সোর্স নির্বাচন করুন (Webcam / Laptop Camera):",
            list(camera_options.keys()),
            index=0,
            key="camera_source_select",
        )
        selected_camera_idx = camera_options[selected_cam_name]

    with c_cam_col2:
        st.write("")
        st.write("")
        if st.button("🔍 Scan Cameras (ক্যামেরা চেক করুন)", key="btn_scan_cameras"):
            with st.spinner("ক্যামেরা স্ক্যান করা হচ্ছে..."):
                active_cams = scan_available_cameras()
                if active_cams:
                    st.success(f"সক্রিয় ক্যামেরা পাওয়া গেছে: {active_cams}")
                else:
                    st.warning("কোনো ক্যামেরা পাওয়া যায়নি। ইউএসবি কানেকশন চেক করুন।")

    c_conf1, c_conf2, c_conf3, c_conf4 = st.columns([1, 1, 1, 1])
    with c_conf1:
        focal = st.number_input("Calibrated focal length (0 to disable distance)", min_value=0.0, value=0.0, step=50.0, key="live_focal")
    with c_conf2:
        conf_live = st.slider("Live Confidence", min_value=0.05, max_value=0.90, value=global_conf, step=0.05, key="live_conf_slider")
    with c_conf3:
        live_ocr_lang = st.selectbox("OCR Language / লেখার ভাষা", ["eng", "ben", "ben+eng"], index=0, key="live_ocr_lang_select")
    with c_conf4:
        speech_interval = st.number_input("ভয়েস বিরতি (সেকেন্ড)", min_value=2, max_value=15, value=4, step=1, key="live_speech_interval")

    c_tog1, c_tog2, c_tog3, c_tog4 = st.columns(4)
    with c_tog1:
        start_live = st.toggle("🔴 Start Live Camera / ক্যামেরা চালু করুন", value=False, key="live_start_toggle")
    with c_tog2:
        enable_voice = st.checkbox("🔊 Object Voice (অবজেক্ট বলুন)", value=True, key="live_enable_voice")
    with c_tog3:
        enable_live_ocr = st.checkbox("📖 Live Text (লেখা পড়ুন)", value=True, key="live_enable_ocr")
    with c_tog4:
        speak_ocr_voice = st.checkbox("🗣️ Read Text Aloud (ভয়েসে শোনান)", value=True, key="live_speak_ocr")

    frame_window = st.empty()
    status_box = st.empty()
    ocr_box = st.empty()
    audio_status = st.empty()

    if start_live:
        cap = open_specific_camera(selected_camera_idx)
        if cap is None:
            st.error(
                f"❌ **Camera {selected_camera_idx} ({selected_cam_name}) খোলা সম্ভব হয়নি!**\n\n"
                "🌐 **আপনি কি এটি Streamlit Cloud (অনলাইন ওয়েবসাইট)-এ ব্যবহার করছেন?**\n"
                "- Streamlit Cloud একটি রিমোট সার্ভারে চলে, যেখানে কোনো ফিজিক্যাল ক্যামেরা থাকে না। পাইথনের `OpenCV` রিমোট ক্লাউড থেকে আপনার ল্যাপটপের ইউএসবি ক্যামেরা পড়তে পারে না।\n"
                "- **ক্লাউডে সমাধান:** উপরের মোড থেকে **'📸 Snapshot & Upload'** সিলেক্ট করুন এবং **'Or capture image with browser camera'** ব্যবহার করুন। এটি আপনার ল্যাপটপ বা মোবাইলের ব্রাউজার ক্যামেরা অন করে সঙ্গে সঙ্গে ডিটেকশন ও ভয়েস ফিডব্যাক দেবে!\n\n"
                "💻 **আপনি যদি আপনার নিজের ল্যাপটপে লোকালভাবে চালিয়ে থাকেন:**\n"
                "1. আপনার ইউএসবি ওয়েবক্যামটি কম্পিউটারে ঠিকমতো লাগানো আছে কি না চেক করুন।\n"
                "2. অন্য কোনো অ্যাপ (যেমন Zoom, Google Meet, Teams, বা Windows Camera App) ক্যামেরাটি ব্যবহার করে থাকলে তা বন্ধ করুন।\n"
                "3. উইন্ডোজ সেটিংস থেকে `Settings -> Privacy & Security -> Camera`-তে Desktop Apps-এর জন্য ক্যামেরা এক্সেস চালু আছে কি না দেখুন।"
            )
        else:
            status_box.info(f"🎥 লাইভ ক্যামেরা চালু আছে ({selected_cam_name})... বন্ধ করতে উপরের টগলটি অফ করুন।")
            last_speech_time = 0.0
            last_ocr_time = 0.0
            latest_ocr_text = ""
            last_spoken_ocr_text = ""

            while start_live:
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.warning("ক্যামেরা থেকে ফ্রেম পাওয়া যায়নি।")
                    break

                current_time = time.time()

                # Run YOLO Object Detection on raw frame
                out, dets, msgs = analyze(frame, model, conf_live, focal or None)

                # Run Live OCR periodically if enabled (runs on clean frame every 2.0s)
                if enable_live_ocr and (current_time - last_ocr_time >= 2.0):
                    _, ocr_words, found_text = detect_and_draw_text(frame, lang=live_ocr_lang)
                    if found_text:
                        latest_ocr_text = found_text
                        # If Read Text Aloud is enabled and this text is newly detected or interval passed
                        if speak_ocr_voice and (found_text != last_spoken_ocr_text or (current_time - last_speech_time >= speech_interval)):
                            if not is_speaking():
                                is_bn = any("\u0980" <= c <= "\u09ff" for c in found_text)
                                speech_phrase = f"লেখা দেখা যাচ্ছে: {found_text}" if is_bn else f"Text: {found_text}"
                                speak(speech_phrase, lang="bn" if is_bn else "en", async_mode=True)
                                last_spoken_ocr_text = found_text
                                last_speech_time = current_time
                                audio_status.success(f"🗣️ লেখা পড়ে শোনানো হচ্ছে: **{found_text}**")
                    last_ocr_time = current_time

                # If text was found, draw text notification banner on live frame
                if latest_ocr_text:
                    cv2.rectangle(out, (10, out.shape[0] - 40), (out.shape[1] - 10, out.shape[0] - 10), (30, 30, 30), -1)
                    cv2.putText(
                        out,
                        f"TEXT: {latest_ocr_text[:45]}",
                        (15, out.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 220, 255),
                        2,
                    )

                # Convert BGR frame to RGB for Streamlit display
                frame_window.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                # Update Detections Status
                if msgs:
                    spoken_text = ". ".join(msgs[:3])
                    status_box.markdown("**শনাক্ত বস্তু:** " + " | ".join(msgs))

                    # Trigger voice feedback if enabled, interval passed, and not already speaking
                    if enable_voice and (current_time - last_speech_time >= speech_interval):
                        if not is_speaking():
                            speak(spoken_text, lang="bn", async_mode=True)
                            last_speech_time = current_time
                            audio_status.success(f"🔊 বলা হচ্ছে: {spoken_text}")
                else:
                    status_box.caption("কোনো পরিচিত বস্তু সামনে নেই। (কনফিডেন্স থ্রেশহোল্ড কমিয়ে দেখতে পারেন)")
                    if current_time - last_speech_time > speech_interval * 2:
                        audio_status.empty()

                # Update OCR Status Box
                if enable_live_ocr:
                    if latest_ocr_text:
                        ocr_box.markdown(f"📖 **পঠিত লেখা (Live OCR):** `{latest_ocr_text}`")
                    else:
                        ocr_box.caption("📖 ক্যামেরার সামনে কোনো স্পষ্ট লেখা ধরা পড়েনি। (লেখাটি ক্যামেরার সোজা ও আলোতে ধরুন)")

            cap.release()
            status_box.success("ক্যামেরা বন্ধ করা হয়েছে।")
            audio_status.empty()
            ocr_box.empty()

else:
    st.subheader("📸 Snapshot & Upload Mode")
    
    col_upload_opts1, col_upload_opts2 = st.columns([1, 1])
    with col_upload_opts1:
        focal_snap = st.number_input("Calibrated focal length in pixels (0 disables distance)", min_value=0.0, value=0.0, key="focal_snap")
        conf_snap = st.slider("Detection Confidence / কনফিডেন্স", min_value=0.05, max_value=0.90, value=global_conf, step=0.05, key="conf_snap")
    with col_upload_opts2:
        lang = st.selectbox("OCR Language / লেখার ভাষা", ["eng", "ben", "ben+eng"], index=0, key="ocr_lang_select")

    up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="file_uploader_snap")
    cam = st.camera_input("Or capture image with browser camera", key="camera_input_snap")
    f = cam or up

    if f:
        rgb = np.array(Image.open(f).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        out, dets, msgs = analyze(bgr, model, conf_snap, focal_snap or None)

        c1, c2 = st.columns(2)
        c1.image(rgb, caption="Input", use_container_width=True)
        c2.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption=f"Detections (Found: {len(dets)})", use_container_width=True)

        st.subheader("Detection output")
        if dets:
            st.dataframe(dets, use_container_width=True)
        else:
            st.warning(f"বর্তমান কনফিডেন্স থ্রেশহোল্ড ({conf_snap:.2f})-এ কোনো অবজেক্ট শনাক্ত হয়নি। উপরের কনফিডেন্স স্লাইডারটি ০.১০ বা ০.১৫-এ নামিয়ে দেখুন, অথবা সাইডবার থেকে Pretrained YOLO11 মডেলটি সিলেক্ট করুন।")

        spoken = ". ".join(msgs) if msgs else "কোনো পরিচিত object শনাক্ত হয়নি।"
        st.write("Audio:", spoken)

        if st.button("Generate voice feedback", key="btn_generate_voice"):
            try:
                speak(spoken, lang="bn", async_mode=True)
                a = make_audio(spoken, "bn")
                st.audio(a, format="audio/mp3") if a else None
            except Exception as e:
                st.error(f"Speech error: {e}")

        st.subheader("OCR — Read environmental text")
        if st.button("Read text from image", key="btn_read_ocr"):
            try:
                txt = read_text(bgr, lang)
                if txt:
                    st.success(f"**শনাক্তকৃত লেখা:** {txt}")
                    speak(txt, lang="bn" if "ben" in lang else "en", async_mode=True)
                    a = make_audio(txt, "bn" if "ben" in lang else "en")
                    st.audio(a, format="audio/mp3")
                else:
                    st.warning("স্পষ্ট কোনো লেখা পাওয়া যায়নি। ছবিটি আরও স্পষ্ট বা আলোতে তুলে চেষ্টা করুন।")
            except Exception as e:
                st.error(f"OCR/Tesseract setup issue: {e}")

st.divider()
st.subheader("Optional weather API")
lat = st.number_input("Latitude", value=23.8103, format="%.5f", key="weather_lat")
lon = st.number_input("Longitude", value=90.4125, format="%.5f", key="weather_lon")
if st.button("Get current weather", key="btn_weather"):
    try:
        st.json(current_weather(lat, lon))
    except Exception as e:
        st.error(str(e))

st.caption("Hosted web apps cannot control your physical cellular modem or AirPods. Browser audio uses the device's OS/browser output routing.")
