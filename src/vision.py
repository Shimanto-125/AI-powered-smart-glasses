import cv2

# Bengali class name translations for natural voice announcements
BN_NAMES = {
    "person": "মানুষ",
    "bicycle": "বাইসাইকেল",
    "car": "গাড়ি",
    "motorcycle": "মোটরসাইকেল",
    "airplane": "উড়োজাহাজ",
    "bus": "বাস",
    "train": "ট্রেন",
    "truck": "ট্রাক",
    "boat": "নৌকা",
    "traffic light": "ট্রাফিক লাইট",
    "fire hydrant": "ফায়ার হাইড্র্যান্ট",
    "stop sign": "স্টপ সাইন",
    "parking meter": "পার্কিং মিটার",
    "bench": "বেঞ্চ",
    "bird": "পাখি",
    "cat": "বিড়াল",
    "dog": "কুকুর",
    "horse": "ঘোড়া",
    "sheep": "ভেড়া",
    "cow": "গরু",
    "elephant": "হাতি",
    "bear": "ভালুক",
    "zebra": "জেব্রা",
    "giraffe": "জিরাফ",
    "backpack": "ব্যাগ",
    "umbrella": "ছাতা",
    "handbag": "হ্যান্ডব্যাগ",
    "tie": "টাই",
    "suitcase": "স্যুটকেস",
    "frisbee": "ফ্রিসবি",
    "skis": "স্কি",
    "snowboard": "স্নোবোর্ড",
    "sports ball": "বল",
    "kite": "ঘুড়ি",
    "baseball bat": "ব্যাট",
    "baseball glove": "গ্লাভস",
    "skateboard": "স্কেটবোর্ড",
    "surfboard": "সার্ফবোর্ড",
    "tennis racket": "র‌্যাকেট",
    "bottle": "বোতল",
    "wine glass": "গ্লাস",
    "cup": "কাপ",
    "fork": "কাঁটাচামচ",
    "knife": "ছুরি",
    "spoon": "চামচ",
    "bowl": "বাটি",
    "banana": "কলা",
    "apple": "আপেল",
    "sandwich": "স্যান্ডউইচ",
    "orange": "কমলা",
    "broccoli": "ব্রকলি",
    "carrot": "গাজর",
    "hot dog": "হট ডগ",
    "pizza": "পিজ্জা",
    "donut": "ডোনাট",
    "cake": "কেক",
    "chair": "চেয়ার",
    "couch": "সোফা",
    "potted plant": "গাছের টব",
    "bed": "বিছানা",
    "dining table": "ডাইনিং টেবিল",
    "toilet": "টয়লেট",
    "tv": "টিভি",
    "laptop": "ল্যাপটপ",
    "mouse": "মাউস",
    "remote": "রিমোট",
    "keyboard": "কিবোর্ড",
    "cell phone": "মোবাইল ফোন",
    "microwave": "মাইক্রোওয়েভ",
    "oven": "ওভেন",
    "toaster": "টোস্টার",
    "sink": "সিঙ্ক",
    "refrigerator": "ফ্রিজ",
    "book": "বই",
    "clock": "ঘড়ি",
    "vase": "ফুলদানি",
    "scissors": "কাঁচি",
    "teddy bear": "টেডি বিয়ার",
    "hair drier": "হেয়ার ড্রায়ার",
    "toothbrush": "টুথব্রাশ",
}

def direction(x1, x2, width):
    c = (x1 + x2) / 2
    return "বাম পাশে" if c < width * 0.35 else ("ডান পাশে" if c > width * 0.65 else "সামনে")

def distance_m(box_width_px, real_width_m, focal_px):
    if box_width_px <= 0 or not focal_px:
        return None
    return focal_px * real_width_m / box_width_px

def light_color(crop):
    # Experimental HSV heuristic only; NOT reliable traffic-signal recognition.
    if crop is None or crop.size == 0:
        return "অজানা"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = (s > 90) & (v > 100)
    counts = {
        "লাল": int((((h < 10) | (h > 170)) & m).sum()),
        "হলুদ": int((((h >= 15) & (h <= 35)) & m).sum()),
        "সবুজ": int((((h >= 36) & (h <= 90)) & m).sum()),
    }
    k = max(counts, key=counts.get)
    return k if counts[k] >= 8 else "অজানা"

def analyze(frame, model, conf=0.35, focal_px=None):
    H, W = frame.shape[:2]
    out = frame.copy()
    found = []
    msgs = []
    
    for r in model.predict(frame, conf=conf, verbose=False):
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            score = float(box.conf[0])
            name = model.names.get(cls, str(cls))
            bn_name = BN_NAMES.get(name.lower(), name)
            dir_str = direction(x1, x2, W)
            msg = f"{bn_name} {dir_str}"
            
            widths = {"person": 0.50, "car": 1.8, "bus": 2.5, "truck": 2.5, "bicycle": 0.6, "motorcycle": 0.8}
            d = distance_m(x2 - x1, widths[name], focal_px) if name in widths and focal_px else None
            if d:
                msg += f", আনুমানিক {d:.1f} মিটার"
            if name == "traffic light":
                state = light_color(frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)])
                msg += f", রং {state}"
                
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                out,
                f"{name} {score:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 0),
                2,
            )
            found.append({"object": name, "bangla": bn_name, "confidence": round(score, 3), "message": msg})
            msgs.append(msg)
            
    return out, found, list(dict.fromkeys(msgs))
