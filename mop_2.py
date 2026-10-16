"""Green building camera dashboard and three-hall digital twin."""
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    import cv2
except ImportError:
    cv2 = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "telemetry.json"
MODEL_FILE = ROOT / "yolov8n.pt"
TELEMETRY_LOCK = threading.Lock()
DEFAULT_HALLS = [
    {"name": "Hall A - Atrium", "occupancy": 4, "capacity": 40, "lights_on": 8, "light_count": 12, "ac_units": 2, "ac_kw": 3.2, "temperature_c": 23.5, "source": "Simulation"},
    {"name": "Hall B - Studio", "occupancy": 12, "capacity": 30, "lights_on": 10, "light_count": 10, "ac_units": 2, "ac_kw": 2.8, "temperature_c": 24.8, "source": "Simulation"},
    {"name": "Hall C - Forum", "occupancy": 0, "capacity": 60, "lights_on": 6, "light_count": 16, "ac_units": 3, "ac_kw": 4.5, "temperature_c": 22.1, "source": "Simulation"},
]

st.set_page_config(page_title="Verdant · Smart Campus Towers", page_icon="◼", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@400;500;600;700&display=swap');
.stApp{background:radial-gradient(circle at 50% 45%,rgba(0,170,255,.12),#070A10 70%);color:#E2E8F0;font-family:Inter,sans-serif}.block-container{max-width:1920px;padding:5.5rem 1.25rem 2rem!important}
[data-testid="stHeader"]{height:3.5rem;background:#070A10;z-index:1000}[data-testid="stSidebar"]{background:rgba(8,13,22,.96);border-right:1px solid rgba(56,189,248,.16)}
div[data-testid="stVerticalBlockBorderWrapper"]{background:rgba(12,18,28,.72);backdrop-filter:blur(12px);border:1px solid rgba(56,189,248,.22);border-radius:12px;box-shadow:inset 2px 2px 0 rgba(0,229,255,.08),0 8px 28px rgba(0,0,0,.25);padding:8px;margin-bottom:10px}
.hud-title{display:flex;align-items:center;justify-content:space-between;gap:8px;border-bottom:1px solid rgba(56,189,248,.16);padding:5px 3px 10px;margin-bottom:8px;color:#E2E8F0;font-weight:700;font-size:13px}.hud-title-main{display:flex;align-items:center;gap:8px}.hud-dot{height:7px;width:7px;border-radius:50%;background:#00E5FF;box-shadow:0 0 9px #00E5FF}.hud-actions{color:#6B8194;font-size:11px;letter-spacing:4px}.hud-kpi{font-family:Orbitron,Inter,sans-serif;color:#DDF8FF;font-size:23px;font-weight:600;text-shadow:0 0 12px rgba(0,229,255,.35)}.hud-label{color:#7F94A8;font-size:9px;letter-spacing:.12em;text-transform:uppercase}.hud-note{color:#7990A5;font-size:10px}.hud-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.hud-grid>div{padding:8px;border:1px solid rgba(56,189,248,.13);border-radius:8px;background:rgba(10,16,26,.45)}
.bms-nav{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;background:rgba(12,18,28,.84);backdrop-filter:blur(12px);border:1px solid rgba(56,189,248,.22);border-radius:10px;padding:11px 16px;margin-bottom:12px}.bms-brand{font-family:Orbitron,Inter,sans-serif;font-size:14px;color:#DDF8FF;display:flex;align-items:center;gap:7px;letter-spacing:.04em}.bms-brand-mark{display:inline-grid;place-items:center;background:#00AEEF;color:white;width:27px;height:27px;border-radius:5px;font-weight:800;box-shadow:0 0 16px rgba(0,229,255,.35)}.bms-nav-right{display:flex;align-items:center;gap:10px;flex-wrap:wrap;color:#9BB0C2;font-size:10px}.bms-pill{border:1px solid rgba(56,189,248,.22);border-radius:999px;background:rgba(8,15,25,.8);padding:6px 10px;font-size:9px;color:#B8D4E6}.bms-live{background:rgba(0,229,255,.1);color:#00E5FF;border-color:#00E5FF}.bms-temp{font-size:14px;font-weight:800;color:#E2E8F0}
div[data-testid="stMetric"]{background:rgba(10,16,26,.5);border:1px solid rgba(56,189,248,.14);padding:8px;border-radius:8px;box-shadow:none}div[data-testid="stMetricLabel"]{color:#8197AA;font-size:9px}div[data-testid="stMetricValue"]{color:#E2F8FF;font-weight:700;font-size:15px}
div.stButton>button{background:rgba(8,24,38,.86);border:1px solid rgba(0,229,255,.3);color:#CFF7FF;border-radius:7px;font-size:10px}div.stButton>button:hover{border-color:#00E5FF;color:white;box-shadow:0 0 12px rgba(0,229,255,.2)}
[data-testid="stRadio"] div[role="radiogroup"]{gap:5px;flex-wrap:wrap}[data-testid="stRadio"] div[role="radiogroup"] label{border:1px solid rgba(56,189,248,.18);border-radius:999px;padding:4px 8px;background:rgba(10,16,26,.55)}[data-testid="stRadio"] div[role="radiogroup"] label p{font-size:10px}[data-testid="stPlotlyChart"]{background:transparent}hr{border-color:rgba(56,189,248,.16)}
@media(max-width:1050px){.block-container{padding:4.5rem .7rem 1.5rem!important}.hud-kpi{font-size:18px}}
</style>""", unsafe_allow_html=True)

def load_data():
    try:
        payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("halls"), list):
            by_name = {h.get("name"): h for h in payload["halls"]}
            halls = [dict(base, **by_name.get(base["name"], {})) for base in DEFAULT_HALLS]
            for hall in halls:
                hall.setdefault("lights_commanded_on", int(hall.get("lights_on", 0)))
                hall.setdefault("lights_working", int(hall.get("lights_on", 0)))
                hall["lights_on"] = int(hall["lights_working"])
            return halls
    except (OSError, json.JSONDecodeError):
        pass
    return [dict(h) for h in DEFAULT_HALLS]

def save_data(halls, power=None, live_session=None):
    payload = {"updated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "halls": halls}
    if power is not None:
        payload["power"] = power
    if live_session is not None:
        payload["live_session"] = live_session
    with TELEMETRY_LOCK:
        temp = DATA_FILE.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(DATA_FILE)

def record_live_vision(hall_name, people, light_count, detect_lights_enabled):
    """Persist the newest live-camera detections without touching Streamlit state."""
    with TELEMETRY_LOCK:
        try:
            payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        halls = payload.get("halls")
        if not isinstance(halls, list):
            halls = [dict(hall) for hall in DEFAULT_HALLS]
        hall = next((item for item in halls if item.get("name") == hall_name), None)
        if hall is None:
            return
        if people is not None:
            hall["occupancy"] = min(int(people), int(hall.get("capacity", people)))
            hall["source"] = "CameraVision Â· YOLO"
        if detect_lights_enabled:
            working = min(int(light_count), int(hall.get("light_count", light_count)))
            hall["lights_working"] = working
            hall["lights_on"] = working
            hall["light_detection"] = "OpenCV bright-region estimate"
        hall["captured_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        payload["updated_at"] = hall["captured_at"]
        payload["halls"] = halls
        temp = DATA_FILE.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(DATA_FILE)

def detect_lights(frame, brightness_threshold=205, saturation_threshold=110):
    """Find compact, near-white bright regions that stand out from their surroundings."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    # Requiring low saturation rejects bright colored clothing, signs, and paint.
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, int(brightness_threshold)], dtype=np.uint8),
        np.array([180, int(saturation_threshold), 255], dtype=np.uint8),
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        ratio = w / max(h, 1)
        extent = area / max(w * h, 1)
        if area < max(30, frame_area * 0.000015) or area > frame_area * 0.008:
            continue
        if not (0.2 <= ratio <= 5.0) or extent < 0.35:
            continue

        # Compare the candidate with a small surrounding ring so uniformly
        # bright walls/windows do not become fixture detections.
        pad = max(3, round(max(w, h) * 0.2))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        roi_value = value[y1:y2, x1:x2]
        roi_mask = np.zeros(roi_value.shape, dtype=np.uint8)
        cv2.drawContours(roi_mask, [contour], -1, 255, thickness=cv2.FILLED, offset=(-x1, -y1))
        inner = roi_value[roi_mask > 0]
        surrounding = roi_value[roi_mask == 0]
        if inner.size == 0 or surrounding.size == 0:
            continue
        if float(inner.mean()) - float(surrounding.mean()) < 12:
            continue
        detections.append((x, y, w, h))
    return detections

def draw_detection_grid(frame, person_boxes, light_boxes, rows, columns, hall_name="", people_count=None):
    """Overlay per-cell person and bright-light counts on a camera frame."""
    height, width = frame.shape[:2]
    rows = max(1, int(rows))
    columns = max(1, int(columns))
    people_by_cell = [[0 for _ in range(columns)] for _ in range(rows)]
    lights_by_cell = [[0 for _ in range(columns)] for _ in range(rows)]

    def cell_for(x, y):
        column = min(columns - 1, max(0, int(x * columns / max(width, 1))))
        row = min(rows - 1, max(0, int(y * rows / max(height, 1))))
        return row, column

    for x1, y1, x2, y2 in person_boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        row, column = cell_for((x1 + x2) / 2, (y1 + y2) / 2)
        people_by_cell[row][column] += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 230, 120), 2)
        cv2.putText(frame, "PERSON", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 230, 120), 1)

    for x, y, box_width, box_height in light_boxes:
        row, column = cell_for(x + box_width / 2, y + box_height / 2)
        lights_by_cell[row][column] += 1
        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (0, 220, 255), 2)
        cv2.putText(frame, "LIGHT", (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

    cell_width, cell_height = width / columns, height / rows
    for column in range(1, columns):
        x = int(column * cell_width)
        cv2.line(frame, (x, 0), (x, height), (100, 190, 130), 1)
    for row in range(1, rows):
        y = int(row * cell_height)
        cv2.line(frame, (0, y), (width, y), (100, 190, 130), 1)

    for row in range(rows):
        for column in range(columns):
            x = int(column * cell_width) + 6
            y = int(row * cell_height) + 20
            people_label = str(people_by_cell[row][column]) if people_count is not None else "?"
            label = f"P:{people_label}  L:{lights_by_cell[row][column]}"
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 250, 240), 2, cv2.LINE_AA)

    total_people = str(sum(map(sum, people_by_cell))) if people_count is not None else "?"
    total_lights = sum(map(sum, lights_by_cell))
    summary = f"{hall_name}  People: {total_people}  Working lights: {total_lights}".strip()
    cv2.rectangle(frame, (0, 0), (width, min(34, height)), (20, 35, 28), -1)
    cv2.putText(frame, summary, (8, min(24, height - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 248, 238), 1, cv2.LINE_AA)
    return frame

class LiveGridDetector:
    """Keep WebRTC playback responsive while analyzing only the newest queued frame."""
    def __init__(self, hall_name, detect_lights_enabled, brightness_threshold, saturation_threshold, grid_rows, grid_columns):
        self.hall_name = hall_name
        self.detect_lights_enabled = detect_lights_enabled
        self.brightness_threshold = brightness_threshold
        self.saturation_threshold = saturation_threshold
        self.grid_rows = grid_rows
        self.grid_columns = grid_columns
        self.frames = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.worker_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = None
        self.model = None
        self.last_submit = 0.0
        self.last_frame_at = time.monotonic()
        self.detections = {"people": None, "person_boxes": [], "light_boxes": []}

    @property
    def config(self):
        return (self.hall_name, self.detect_lights_enabled, self.brightness_threshold, self.saturation_threshold, self.grid_rows, self.grid_columns)

    @property
    def stopped(self):
        return self.stop_event.is_set()

    def stop(self):
        self.stop_event.set()

    def _start_worker(self):
        with self.worker_lock:
            if self.worker is None:
                self.worker = threading.Thread(target=self._analyze_frames, name="live-grid-detector", daemon=True)
                self.worker.start()

    def _load_model(self):
        try:
            model = YOLO(str(MODEL_FILE)) if YOLO is not None else None
        except Exception:
            model = None
        with self.lock:
            self.model = model

    def _analyze_frames(self):
        if YOLO is not None:
            threading.Thread(target=self._load_model, name="live-yolo-loader", daemon=True).start()

        while not self.stop_event.is_set():
            if time.monotonic() - self.last_frame_at > 5.0:
                self.stop_event.set()
                break
            try:
                frame = self.frames.get(timeout=0.1)
            except queue.Empty:
                continue
            people = None
            person_boxes = []
            with self.lock:
                model = self.model
            if model is not None:
                try:
                    result = model.predict(frame, classes=[0], imgsz=416, verbose=False)[0]
                    people = len(result.boxes)
                    person_boxes = result.boxes.xyxy.tolist()
                except Exception:
                    with self.lock:
                        self.model = None
            light_boxes = detect_lights(frame, self.brightness_threshold, self.saturation_threshold) if self.detect_lights_enabled and cv2 is not None and np is not None else []
            with self.lock:
                self.detections = {"people": people, "person_boxes": person_boxes, "light_boxes": light_boxes}
            record_live_vision(self.hall_name, people, len(light_boxes), self.detect_lights_enabled)

    def process(self, video_frame):
        import av

        if cv2 is None or np is None:
            return video_frame
        frame = video_frame.to_ndarray(format="bgr24")
        self.last_frame_at = time.monotonic()
        height, width = frame.shape[:2]
        if width > 960:
            frame = cv2.resize(frame, (960, max(1, round(height * 960 / width))), interpolation=cv2.INTER_AREA)
        if not self.stopped:
            self._start_worker()
            now = time.monotonic()
            if now - self.last_submit >= 0.4:
                self.last_submit = now
                try:
                    self.frames.put_nowait(frame.copy())
                except queue.Full:
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.frames.put_nowait(frame.copy())
                    except queue.Full:
                        pass
        with self.lock:
            detections = dict(self.detections)
        draw_detection_grid(
            frame,
            detections["person_boxes"],
            detections["light_boxes"],
            self.grid_rows,
            self.grid_columns,
            self.hall_name,
            detections["people"],
        )
        return av.VideoFrame.from_ndarray(frame, format="bgr24")

def classify_switchboard(frame, roi_percent, switch_count, rows, on_position):
    """Estimate rocker states from the brighter half of each switch crop."""
    height, width = frame.shape[:2]
    xp, yp, wp, hp = roi_percent
    x1, y1 = int(width * xp / 100), int(height * yp / 100)
    x2 = min(width, int(width * (xp + wp) / 100))
    y2 = min(height, int(height * (yp + hp) / 100))
    if x2 <= x1 or y2 <= y1:
        return None
    current = frame[y1:y2, x1:x2]
    if current.shape[1] < 2 or current.shape[0] < 2:
        return None
    rows = max(1, min(int(rows), switch_count))
    columns = (switch_count + rows - 1) // rows
    current_width = current.shape[1] // columns
    current_height = current.shape[0] // rows
    if min(current_width, current_height) < 2:
        return None
    states = []
    for index in range(switch_count):
        row, column = divmod(index, columns)
        cx1, cx2 = column * current_width, current.shape[1] if column == columns - 1 else (column + 1) * current_width
        cy1, cy2 = row * current_height, current.shape[0] if row == rows - 1 else (row + 1) * current_height
        patch = cv2.cvtColor(current[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2GRAY)
        top_mean = float(patch[:patch.shape[0] // 2].mean())
        bottom_mean = float(patch[patch.shape[0] // 2:].mean())
        difference = top_mean - bottom_mean
        if abs(difference) < 4.0:
            states.append("unknown")
        else:
            upper_is_on = on_position == "Upper"
            states.append("on" if (difference > 0) == upper_is_on else "off")
    return states

def hall_lighting_fault(hall):
    if hall.get("switchboard_vision_enabled") and hall.get("switchboard_calibrated"):
        return bool(hall.get("switchboard_fault_confirmed", False))
    return int(hall.get("lights_commanded_on", hall.get("lights_on", 0))) > int(hall.get("lights_working", hall.get("lights_on", 0)))

@st.cache_resource
def load_yolo_model():
    return YOLO(str(MODEL_FILE))

def local_recommendations(halls):
    out = []
    for h in halls:
        name = h["name"].split(" Â· ")[-1]
        commanded = int(h.get("lights_commanded_on", h.get("lights_on", 0)))
        working = int(h.get("lights_working", h.get("lights_on", 0)))
        if hall_lighting_fault(h):
            out.append({"priority":"High", "hall":name, "action":"Inspect the lighting circuit and failed fixtures in this hall; check lamp, driver, and switch connections, then retest the zone.", "reason":f"{commanded} lights are commanded on, but camera vision detects only {working} working lights ({commanded - working} possible failures)."})
        if h["occupancy"] == 0 and (h["lights_on"] or h["ac_kw"] > 0):
            out.append({"priority":"High", "hall":name, "action":"Unoccupied hall: switch off the lighting zone and set HVAC to standby.", "reason":f"No people detected while {h['lights_on']} lights and {h['ac_kw']:.1f} kW of AC are active."})
        elif h["occupancy"] > 0 and h["ac_kw"] > 0 and h["temperature_c"] <= 26:
            out.append({"priority":"Low", "hall":name, "action":"Review the AC schedule and stage active units to match occupancy while maintaining the comfort setpoint.", "reason":f"{h['ac_units']} AC units draw {h['ac_kw']:.1f} kW for {h['occupancy']} occupants at {h['temperature_c']:.1f} Â°C."})
        elif h["occupancy"] and h["lights_on"] > h["occupancy"] * 0.5:
            out.append({"priority":"Medium", "hall":name, "action":"Dim or switch off unused lighting rows; keep task lighting available.", "reason":f"{h['lights_on']} lights are active for {h['occupancy']} occupants."})
        if h["occupancy"] and h["temperature_c"] < 21:
            out.append({"priority":"Medium", "hall":name, "action":"Raise the cooling setpoint slightly and review comfort after 10 minutes.", "reason":f"Room temperature is {h['temperature_c']:.1f} Â°C with occupants present."})
        elif h["occupancy"] and h["temperature_c"] > 26:
            out.append({"priority":"Medium", "hall":name, "action":"Check airflow and gradually increase cooling while monitoring comfort.", "reason":f"Room temperature is {h['temperature_c']:.1f} Â°C with occupants present."})
    if not out:
        out.append({"priority":"Low", "hall":"All halls", "action":"Current resource use looks aligned with occupancy. Continue monitoring trends.", "reason":"No obvious occupancy and resource mismatch was detected."})
    return out

def genai_recommendations(halls, api_key=None):
    key = (api_key or "").strip() or os.getenv("GROQ_API_KEY")
    if not key:
        # Streamlit deployments commonly provide credentials through
        # .streamlit/secrets.toml instead of the server process environment.
        try:
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            key = None
    if not key:
        return None, "Configure GROQ_API_KEY in the Streamlit server environment or .streamlit/secrets.toml, or enter a Groq key in the advisor."
    # Prefer a model the current key can actually access. Explicit GROQ_MODEL
    # still wins for deployments that intentionally pin a model.
    model = os.getenv("GROQ_MODEL")
    if not model:
        available_models, _ = check_groq_access(key)
        if available_models:
            model_priority = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3-32b", "llama-3.3-70b-versatile"]
            model = next((candidate for candidate in model_priority if candidate in available_models), None)
            if not model:
                model = available_models[0]
    model = model or "openai/gpt-oss-20b"
    body = {"model": model, "messages":[{"role":"system","content":"You are a building energy advisor. For every hall, analyze occupancy, lights commanded on versus lights the camera detects as working, lighting power, AC unit count and AC power, and temperature. If commanded lights exceed detected working lights, clearly flag a possible lighting fault and recommend safe checks for the fixture, lamp, driver, wiring, or circuit; do not assert a specific cause without evidence. Also give actionable HVAC/AC and energy recommendations while preserving comfort. Return a JSON object with a recommendations array; each item has priority, hall, action, reason. Never claim to control equipment or invent measured savings."},{"role":"user","content":json.dumps(halls)}],"response_format":{"type":"json_object"},"temperature":0.3}
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization":"Bearer " + key,"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            result = json.loads(response.read())
        parsed = json.loads(result["choices"][0]["message"]["content"])
        recs = parsed if isinstance(parsed, list) else parsed.get("recommendations", [])
        return recs, None
    except urllib.error.HTTPError as exc:
        # Groq returns a structured JSON message for permission/model errors.
        # Surface that message so a valid key is not mistaken for an auth failure.
        try:
            raw_error = exc.read().decode("utf-8", errors="replace").strip()
            error_body = json.loads(raw_error) if raw_error else {}
            error_info = error_body.get("error", {}) if isinstance(error_body, dict) else {}
            detail = error_info.get("message") or error_info.get("code") or error_info.get("type") or raw_error
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            detail = None
        detail = str(detail or "Groq did not include additional details.")[:300]
        detail = detail.replace(key, "[redacted]")
        if exc.code == 403:
            detail += " Check that this model is enabled for the Groq project that issued the key."
        elif exc.code == 401:
            detail += " Verify the Groq API key is valid and active."
        return None, f"Groq rejected the request (HTTP {exc.code}): {detail} Showing local recommendations instead."
    except Exception as exc:
        return None, f"Groq request failed ({exc}). Showing local recommendations instead."

def check_groq_access(api_key=None):
    """Check whether Groq accepts this key and return model IDs visible to it."""
    key = (api_key or "").strip() or os.getenv("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            key = None
    if not key:
        return None, "Enter a Groq API key first."
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": "Bearer " + key},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            payload = json.loads(response.read())
        models = sorted({item.get("id", "") for item in payload.get("data", []) if item.get("id")})
        if not models:
            return [], "Groq accepted the key, but returned no model IDs for this project."
        return models, None
    except urllib.error.HTTPError as exc:
        try:
            message = exc.read().decode("utf-8", errors="replace").strip()
            parsed = json.loads(message) if message else {}
            detail = parsed.get("error", {}).get("message") if isinstance(parsed, dict) else None
            detail = detail or message
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            detail = None
        detail = str(detail or "Groq returned no diagnostic details.")[:250].replace(key, "[redacted]")
        return None, f"Groq key check failed (HTTP {exc.code}): {detail}"
    except Exception as exc:
        return None, f"Could not reach Groq's model endpoint: {type(exc).__name__}."

def hall_floor_svg(hall, hall_index):
    """Render a light architectural floor plan from current hall telemetry."""
    positions = [(x, y) for y in (75, 112, 149, 186) for x in (82, 146, 210, 274, 338)]
    people = "".join(f'<g transform="translate({x} {y})"><circle r="7" fill="#10B981"/><path d="M-5 9 Q0 4 5 9 L7 20 L-7 20Z" fill="#10B981"/></g>' for x,y in positions[:min(int(hall["occupancy"]),len(positions))])
    furniture = "".join(f'<rect x="{x}" y="{y}" width="38" height="15" rx="3" fill="#242832" stroke="#3A3F4B"/>' for x,y in ((112,89),(240,126),(112,163),(304,89)))
    lit=int(hall.get("lights_working",hall["lights_on"])); commanded=int(hall.get("lights_commanded_on",lit))
    lights="".join(f'<circle cx="{74+i*64}" cy="42" r="6" fill="{"#F59E0B" if i<lit else "#EF4444" if i<commanded else "#CBD5E1"}"/>' for i in range(min(int(hall["light_count"]),5)))
    hvac="#0EA5E9" if float(hall["ac_kw"])>0 else "#94A3B8"
    return f'''<svg viewBox="0 0 420 220" role="img" aria-label="{hall['name']} floor plan" style="width:100%;display:block">
    <rect x="18" y="20" width="384" height="184" rx="8" fill="#171A20" stroke="#454B57" stroke-width="2"/><path d="M18 54H402" stroke="#2A2E37"/>
    <text x="31" y="43" fill="#A3AAB7" font-size="10">LIGHT FIXTURES</text>{lights}{furniture}{people}
    <circle cx="365" cy="184" r="6" fill="{hvac}"/><text x="376" y="188" fill="#A3AAB7" font-size="10">AC</text>
    <circle cx="35" cy="184" r="5" fill="#10B981"/><text x="45" y="188" fill="#A3AAB7" font-size="10">{hall['occupancy']} / {hall['capacity']} PEOPLE</text>
    <text x="286" y="43" fill="#A3AAB7" font-size="10">{hall['temperature_c']:.1f}°C</text></svg>'''

def card_header(icon,title):
    st.markdown(f'<div class="bms-card-header"><span class="bms-card-icon">{icon}</span><span class="bms-card-title">{title}</span><span class="bms-card-menu">⋮</span></div>',unsafe_allow_html=True)

def card_footer():
    pass

halls=load_data()
with st.sidebar:
    st.markdown("### CameraVision")
    st.caption("Continuous browser camera, snapshots, and image upload controls.")
    selected=st.selectbox("Assign camera to hall",[h["name"] for h in halls])
    source=st.radio("Camera input",["Live camera feed","Take a snapshot","Upload an image"])
    pause_camera=st.toggle("Pause camera analysis",False,help="Pause vision inference while retaining the camera control.")
    light_watts=st.number_input("Rated watts per light",1,500,40,5)
    tariff=st.number_input("Electricity cost per kWh",0.0,10.0,0.18,0.01,format="%.2f")
    detect_lights_enabled=st.checkbox("Detect active lights in camera frame",True)
    brightness_threshold=st.slider("Light brightness threshold",150,250,205,disabled=not detect_lights_enabled)
    saturation_threshold=st.slider("Maximum light color saturation",20,180,110,5,disabled=not detect_lights_enabled,help="Lower this to reject more bright colored objects.")
    grid_columns=st.slider("Detection grid columns",2,8,4)
    grid_rows=st.slider("Detection grid rows",2,8,3)
    detect_switchboard=st.checkbox("Read switchboard state with camera vision",False)
    if detect_switchboard:
        st.caption("Frame the switchboard and configure its region.")
        roi_x=st.slider("Board left (%)",0,95,10,key="board_x")
        roi_y=st.slider("Board top (%)",0,95,10,key="board_y")
        roi_w=st.slider("Board width (%)",5,100,30,key="board_w")
        roi_h=st.slider("Board height (%)",5,100,35,key="board_h")
        switch_count=st.number_input("Switch count",1,24,4,1,key="board_switch_count")
        switch_rows=st.number_input("Switchboard rows",1,4,1,1,key="board_switch_rows")
        on_position=st.selectbox("Brighter half when switch is ON",["Upper","Lower"])
    else:
        roi_x,roi_y,roi_w,roi_h=0,0,100,100
        switch_count,switch_rows,on_position=1,1,"Upper"
    frame=None
    if source!="Live camera feed":
        image=st.camera_input("Capture room frame") if source=="Take a snapshot" else st.file_uploader("Choose a camera frame",type=["jpg","jpeg","png"])
        if image is not None and cv2 is not None and np is not None:
            frame=cv2.imdecode(np.frombuffer(image.getvalue(),dtype=np.uint8),cv2.IMREAD_COLOR)
def city_twin_figure(halls):
    fig=go.Figure()
    def box(x0,y0,z0,dx,dy,dz,color,opacity):
        x1,y1,z1=x0+dx,y0+dy,z0+dz
        fig.add_trace(go.Mesh3d(x=[x0,x1,x1,x0,x0,x1,x1,x0],y=[y0,y0,y1,y1,y0,y0,y1,y1],z=[z0,z0,z0,z0,z1,z1,z1,z1],i=[0,0,4,4,0,0,1,1,2,2,3,3],j=[1,2,5,6,1,5,2,6,3,7,0,4],k=[2,3,6,7,5,4,6,5,7,6,4,7],color=color,opacity=opacity,flatshading=True,hoverinfo="skip",showlegend=False))
    sites=[]
    for x in (-23,-14):
        for y in (-22,-10,2,14,26): sites.append((x,y))
    for x in (43,53):
        for y in (-22,-10,2,14,26): sites.append((x,y))
    for x in (0,10,20,30): sites.extend(((x,-22),(x,34)))
    sites.extend(((-23,34),(53,34),(-6,-22),(36,-22)))
    for i,(x,y) in enumerate(sites[:32]):
        box(x,y,0,6.5,6.5,3.5+(i*7%6)*1.1,"#1E2430",.65)
    for x in range(-25,66,5):
        fig.add_trace(go.Scatter3d(x=[x,x],y=[-28,39],z=[-.08,-.08],mode="lines",line={"color":"rgba(56,75,99,.22)","width":1},hoverinfo="skip",showlegend=False))
    for y in range(-25,40,5):
        fig.add_trace(go.Scatter3d(x=[-27,66],y=[y,y],z=[-.08,-.08],mode="lines",line={"color":"rgba(56,75,99,.22)","width":1},hoverinfo="skip",showlegend=False))
    roads=[([-2,38,38,-2,-2],[-2,-2,29,29,-2],"#F43F5E",4),([-7,43,43,-7,-7],[-7,-7,34,34,-7],"#FACC15",2),([3,33,33,3,3],[3,3,24,24,3],"#00E5FF",3)]
    for xs,ys,color,width in roads:
        fig.add_trace(go.Scatter3d(x=xs,y=ys,z=[.02]*len(xs),mode="lines",line={"color":color,"width":width},hoverinfo="skip",showlegend=False))
    wings=[(2,2),(20,2),(11,14)]
    for idx,(hall,(x0,y0)) in enumerate(zip(halls,wings)):
        x1,y1,z1=x0+12,y0+8,4.8
        fault=hall_lighting_fault(hall)
        waste=int(hall["occupancy"])==0 and float(hall.get("total_power_kw",0))>0
        glow="#F43F5E" if fault else "#F59E0B" if waste else "#00B4D8"
        box(x0,y0,0,12,8,z1,glow,.28)
        ex=[x0,x1,x1,x0,x0,None,x0,x1,None,x1,x1,None,x1,x0,None,x0,x0]
        ey=[y0,y0,y1,y1,y0,None,y0,y0,None,y0,y1,None,y1,y1,None,y1,y0]
        ez=[0,0,0,0,0,None,0,z1,None,0,z1,None,z1,z1,None,z1,0]
        fig.add_trace(go.Scatter3d(x=ex,y=ey,z=ez,mode="lines",line={"color":"#00F0FF","width":4},hoverinfo="skip",showlegend=False))
        for z in (1.6,3.2):
            for gx in (x0+4,x0+8):
                fig.add_trace(go.Scatter3d(x=[gx,gx],y=[y0,y1],z=[z,z],mode="lines",line={"color":"rgba(0,240,255,.62)","width":2},hoverinfo="skip",showlegend=False))
            fig.add_trace(go.Scatter3d(x=[x0,x1],y=[y0+4,y0+4],z=[z,z],mode="lines",line={"color":"rgba(0,240,255,.62)","width":2},hoverinfo="skip",showlegend=False))
            fig.add_trace(go.Scatter3d(x=[x0,x1,x1,x0,x0],y=[y0,y0,y1,y1,y0],z=[z]*5,mode="lines",line={"color":"rgba(0,240,255,.72)","width":2},hoverinfo="skip",showlegend=False))
        count=min(int(hall["occupancy"]),18)
        fig.add_trace(go.Scatter3d(x=[x0+1+(i%6)*1.8 for i in range(count)],y=[y0+1+(i//6)*1.8 for i in range(count)],z=[.3]*count,mode="markers",marker={"color":"#4ADE80" if hall.get("source","").startswith("CameraVision") else "#38BDF8","size":6},name="Occupants" if idx==0 else None,showlegend=idx==0,hovertemplate="Occupant<extra></extra>"))
        n=min(int(hall.get("light_count",0)),8)
        working=int(hall.get("lights_working",hall.get("lights_on",0)))
        commanded=int(hall.get("lights_commanded_on",hall.get("lights_on",0)))
        colors=["#FACC15" if i<working else "#F43F5E" if i<commanded else "#64748B" for i in range(n)]
        fig.add_trace(go.Scatter3d(x=[x0+1+i*10/max(n-1,1) for i in range(n)],y=[y0+1]*n,z=[z1+.15]*n,mode="markers",marker={"color":colors,"size":7,"symbol":"square"},name="Fixtures" if idx==0 else None,showlegend=idx==0,hovertemplate="Ceiling fixture<extra></extra>"))
        fig.add_trace(go.Scatter3d(x=[x1-1.1],y=[y1-1],z=[.5],mode="markers",marker={"color":"#A855F7","size":9,"symbol":"diamond"},name="AC units" if idx==0 else None,showlegend=idx==0))
        fig.add_trace(go.Scatter3d(x=[(x0+x1)/2],y=[y0-.8],z=[z1+.25],mode="text",text=[hall["name"]],textfont={"color":"#B8EEFF","size":11},hoverinfo="skip",showlegend=False))
        fig.add_trace(go.Scatter3d(x=[x0+.4,x0+.4],y=[y0+.4,y0+.4],z=[0,z1],mode="lines",line={"color":glow,"width":7},hoverinfo="skip",showlegend=False))
    fig.update_layout(template="plotly_dark",height=590,margin={"l":10,"r":10,"t":25,"b":10},paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font={"color":"#A9C2D6","size":9},legend={"orientation":"h","y":1.02,"x":.02,"bgcolor":"rgba(0,0,0,0)"},scene={"bgcolor":"rgba(0,0,0,0)","xaxis":{"range":[-27,66],"visible":False},"yaxis":{"range":[-28,39],"visible":False},"zaxis":{"range":[0,13],"visible":False},"aspectmode":"manual","aspectratio":{"x":2.1,"y":1.5,"z":.75},"camera":{"eye":{"x":1.45,"y":1.65,"z":1.05}}})
    return fig

def hud_title(title,icon="◈"):
    st.markdown(f"<div class='hud-title'><span class='hud-title-main'><span class='hud-dot'></span>{icon} {title}</span><span class='hud-actions'>📄 ↻ ⬇</span></div>",unsafe_allow_html=True)

live_enabled=False
with st.sidebar:
    st.divider()
    st.markdown("### Hall sensor inputs")
    st.caption("Edit values to simulate lighting, HVAC, occupancy, and temperature telemetry.")
    with st.form("sensor_form"):
        edited=[]
        for i,hall in enumerate(halls):
            st.markdown(f"**{hall['name']}**")
            c1,c2=st.columns(2)
            occ=c1.number_input("People",min_value=0,max_value=int(hall["capacity"]),value=min(int(hall["occupancy"]),int(hall["capacity"])),key=f"occ{i}")
            commanded=c2.number_input("Lights switched on",min_value=0,max_value=int(hall["light_count"]),value=int(hall.get("lights_commanded_on",hall["lights_on"])),key=f"cmdlights{i}")
            working=st.number_input("Working lights detected",min_value=0,max_value=int(hall["light_count"]),value=int(hall.get("lights_working",hall["lights_on"])),key=f"working_lights{i}",help="Camera vision updates this count when detection is enabled.")
            c3,c4=st.columns(2)
            kw=c3.number_input("AC load (kW)",min_value=0.0,max_value=100.0,value=float(hall["ac_kw"]),step=0.1,key=f"kw{i}")
            temp=c4.number_input("Temp (°C)",min_value=10.0,max_value=40.0,value=float(hall["temperature_c"]),step=0.1,key=f"temp{i}")
            update={**hall,"occupancy":occ,"lights_on":working,"lights_commanded_on":commanded,"lights_working":working,"ac_kw":kw,"temperature_c":temp}
            if commanded!=int(hall.get("lights_commanded_on",hall["lights_on"])):
                update["switchboard_vision_enabled"]=False
                update["switchboard_calibrated"]=False
            edited.append(update)
        if st.form_submit_button("Save hall telemetry",use_container_width=True):
            halls=edited
            save_data(halls)

# Compute loads and make stable stage slots so the central twin stays centered.
for hall in halls:
    hall["lights_working"]=int(hall.get("lights_working",hall["lights_on"]))
    hall["lights_commanded_on"]=int(hall.get("lights_commanded_on",hall["lights_on"]))
    hall["lighting_power_kw"]=hall["lights_working"]*float(light_watts)/1000
    hall["ac_power_kw"]=float(hall["ac_kw"])
    hall["total_power_kw"]=hall["lighting_power_kw"]+hall["ac_power_kw"]

header_slot=st.empty()
left_col,center_col,right_col=st.columns([1.05,2.3,1.05],gap="small")
with left_col:
    left_power_slot=st.container()
    left_compare_slot=st.container()
    left_zone_slot=st.container()
with right_col:
    right_fixture_slot=st.container()
    right_hall_slot=st.container()
    right_ai_slot=st.container()

with center_col.container(border=True):
    hud_title("CYBER-CITY DIGITAL TWIN","⌖")
    view_mode=st.radio("Campus view",["3D Holographic City Twin","2D Blueprint Floor Plan","Live Camera HUD"],horizontal=True,label_visibility="collapsed",key="hud_view_mode")
    frame_preview=st.empty()
    if view_mode=="3D Holographic City Twin":
        if go is None: st.info("Install Plotly to display the 3D city twin.")
        else: st.plotly_chart(city_twin_figure(halls),use_container_width=True,config={"displayModeBar":False},key="city_twin")
    elif view_mode=="2D Blueprint Floor Plan":
        bcols=st.columns(3,gap="small")
        for col,hall in zip(bcols,halls):
            with col:
                st.markdown(f"**{hall['name']}**")
                st.markdown(hall_floor_svg(hall,0),unsafe_allow_html=True)
    else:
        if source!="Live camera feed":
            st.caption("Choose Live camera feed in the sidebar to start the browser camera.")
        else:
            try:
                from streamlit_webrtc import webrtc_streamer
            except ImportError:
                st.error("Continuous camera requires streamlit-webrtc.")
            else:
                config=(selected,bool(detect_lights_enabled),int(brightness_threshold),int(saturation_threshold),int(grid_rows),int(grid_columns))
                detector=st.session_state.get("live_grid_detector")
                if detector is None or detector.config!=config or detector.stopped:
                    if detector is not None: detector.stop()
                    detector=LiveGridDetector(*config)
                    st.session_state["live_grid_detector"]=detector
                def live_video_callback(video_frame):
                    return video_frame if pause_camera else detector.process(video_frame)
                live_context=webrtc_streamer(key="mop-continuous-camera",media_stream_constraints={"video":{"width":{"ideal":960},"height":{"ideal":540},"frameRate":{"ideal":30,"max":30}},"audio":False},video_frame_callback=live_video_callback)
                live_enabled=bool(live_context.state.playing)
                st.caption("Analysis paused · video remains live." if pause_camera else "Continuous camera with latest-frame YOLO and OpenCV grid analysis.")

detector=st.session_state.get("live_grid_detector")
if source!="Live camera feed" or (view_mode!="Live Camera HUD" and st.session_state.get("live_webrtc_playing",False)):
    if detector is not None: detector.stop()
    if source!="Live camera feed": st.session_state.pop("live_grid_detector",None)
was_playing=bool(st.session_state.get("live_webrtc_playing",False))
if was_playing and not live_enabled and detector is not None: detector.stop()
st.session_state["live_webrtc_playing"]=live_enabled

if source!="Live camera feed" and frame is not None and cv2 is not None and np is not None and not pause_camera:
    try:
        idx=next(i for i,h in enumerate(halls) if h["name"]==selected)
        if YOLO is not None:
            result=load_yolo_model().predict(frame,classes=[0],imgsz=640,verbose=False)[0]
            count=len(result.boxes)
            person_boxes=result.boxes.xyxy.tolist()
            annotated=result.plot()
        else:
            count=None
            person_boxes=[]
            annotated=frame.copy()
            st.warning("YOLO is unavailable; people counting is paused.")
        light_boxes=detect_lights(frame,brightness_threshold,saturation_threshold) if detect_lights_enabled else []
        if detect_switchboard:
            switch_states=classify_switchboard(frame,(roi_x,roi_y,roi_w,roi_h),int(switch_count),int(switch_rows),on_position)
            if switch_states is not None:
                confirmed=min(switch_states.count("on"),int(halls[idx]["light_count"]))
                uncertain=switch_states.count("unknown")
                halls[idx].update({"lights_commanded_on":confirmed,"switchboard_on_count":confirmed,"switchboard_off_count":switch_states.count("off"),"switchboard_uncertain_count":uncertain,"switchboard_calibrated":True,"switchboard_vision_enabled":True})
                working=min(len(light_boxes),int(halls[idx]["light_count"])) if detect_lights_enabled else int(halls[idx].get("lights_working",halls[idx]["lights_on"]))
                halls[idx]["switchboard_fault_confirmed"]=confirmed>working and uncertain==0
                st.session_state[f"cmdlights{idx}"]=confirmed
            else:
                halls[idx].update({"switchboard_vision_enabled":True,"switchboard_calibrated":True,"switchboard_fault_confirmed":False,"switchboard_uncertain_count":int(switch_count)})
        if count is not None:
            halls[idx]["occupancy"]=count
            halls[idx]["source"]="CameraVision · YOLO"
            st.session_state[f"occ{idx}"]=min(count,int(halls[idx]["capacity"]))
        if detect_lights_enabled:
            halls[idx]["lights_working"]=min(len(light_boxes),int(halls[idx]["light_count"]))
            halls[idx]["lights_on"]=halls[idx]["lights_working"]
            halls[idx]["light_detection"]="OpenCV bright-region estimate"
            st.session_state[f"working_lights{idx}"]=halls[idx]["lights_working"]
        draw_detection_grid(annotated,person_boxes,light_boxes,grid_rows,grid_columns,selected,count)
        frame_preview.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB),caption=f"CameraVision · {selected}",use_container_width=True)
    except Exception as exc:
        st.error(f"Could not process camera frame: {exc}")
elif source!="Live camera feed" and frame is not None and pause_camera and view_mode=="Live Camera HUD" and cv2 is not None:
    frame_preview.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),caption=f"CameraVision paused · {selected}",use_container_width=True)

for hall in halls:
    hall["lights_working"]=int(hall.get("lights_working",hall["lights_on"]))
    hall["lights_commanded_on"]=int(hall.get("lights_commanded_on",hall["lights_on"]))
    hall["lighting_power_kw"]=hall["lights_working"]*float(light_watts)/1000
    hall["ac_power_kw"]=float(hall["ac_kw"])
    hall["total_power_kw"]=hall["lighting_power_kw"]+hall["ac_power_kw"]
current_power_kw=sum(h["total_power_kw"] for h in halls)
now=time.time()
meter=st.session_state.setdefault("live_energy_meter",{"active":False,"started_at":None,"ended_at":None,"last_tick":None,"last_power_kw":0.0,"energy_kwh":0.0,"cost":0.0,"elapsed_seconds":0.0})
if live_enabled and not meter["active"]:
    meter.update({"active":True,"started_at":datetime.now().astimezone().isoformat(timespec="seconds"),"ended_at":None,"last_tick":now,"last_power_kw":current_power_kw,"energy_kwh":0.0,"cost":0.0,"elapsed_seconds":0.0})
elif meter["active"]:
    elapsed=max(0.0,now-float(meter["last_tick"] or now))
    interval=float(meter["last_power_kw"])*elapsed/3600
    meter["energy_kwh"]+=interval
    meter["cost"]+=interval*float(tariff)
    meter["elapsed_seconds"]+=elapsed
    meter["last_tick"]=now
    if live_enabled: meter["last_power_kw"]=current_power_kw
    else:
        meter["active"]=False
        meter["ended_at"]=datetime.now().astimezone().isoformat(timespec="seconds")
        meter["last_power_kw"]=0.0

occupants=sum(int(h["occupancy"]) for h in halls)
lights_on=sum(int(h["lights_working"]) for h in halls)
cmd_total=sum(int(h["lights_commanded_on"]) for h in halls)
broken=sum(max(0,int(h["lights_commanded_on"])-int(h["lights_working"])) for h in halls)
standby=sum(max(0,int(h["light_count"])-int(h["lights_commanded_on"])) for h in halls)
active_ac=sum(int(h["ac_units"]) if float(h["ac_kw"])>0 else 0 for h in halls)
capacity_total=sum(int(h["capacity"]) for h in halls)
lighting_load=sum(float(h["lighting_power_kw"]) for h in halls)
ac_load=sum(float(h["ac_power_kw"]) for h in halls)
total_load=lighting_load+ac_load
mean_temp=sum(float(h["temperature_c"]) for h in halls)/max(1,len(halls))
co2_est=400+occupants*35

ph=st.session_state.setdefault("power_history",[])
ph.append({"time":datetime.now().strftime("%H:%M:%S"),"power":total_load})
st.session_state["power_history"]=ph[-15:]
oh=st.session_state.setdefault("occupancy_history",[])
oh.append({"time":datetime.now().strftime("%H:%M:%S"),**{h["name"]:int(h["occupancy"]) for h in halls}})
st.session_state["occupancy_history"]=oh[-48:]
st.session_state["occ_history"]=st.session_state["occupancy_history"]
clock_text=datetime.now().strftime("%d %b %Y · %H:%M:%S")
header_slot.markdown(f"""<div class="bms-nav"><div class="bms-brand"><span class="bms-brand-mark">V</span><b>VERDANT</b><span style="color:#547189">/</span> CYBER-CITY COMMAND HUD</div><div class="bms-nav-right"><span class="bms-pill">SMART CAMPUS · 3 WINGS</span><span class="bms-temp">{mean_temp:.1f}°C</span><span class="bms-pill">{clock_text}</span><span>⌖ Bengaluru</span><span class="bms-pill {'bms-live' if live_enabled and not pause_camera else ''}">● {'CAMERA PAUSED' if live_enabled and pause_camera else 'CAMERA LIVE' if live_enabled else 'CAMERA STANDBY'}</span></div></div>""",unsafe_allow_html=True)

live_record={"active":meter["active"],"started_at":meter["started_at"],"ended_at":meter["ended_at"],"elapsed_seconds":round(meter["elapsed_seconds"],2),"energy_kwh":round(meter["energy_kwh"],6),"cost":round(meter["cost"],6),"tariff_per_kwh":float(tariff)}
power_record={"lighting_kw":round(lighting_load,4),"ac_kw":round(ac_load,4),"total_kw":round(total_load,4),"light_watts_each":int(light_watts),"tariff_per_kwh":float(tariff)}
save_data(halls,power_record,live_record)

def style(fig,height=220):
    fig.update_layout(template="plotly_dark",height=height,margin={"l":10,"r":10,"t":25,"b":10},paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font={"color":"#8EA7BA","size":9})
    return fig

with left_power_slot.container(border=True):
    hud_title("Real-Time Power Demand (kW)","ϟ")
    if go is not None:
        fig=go.Figure()
        hist=st.session_state["power_history"]
        fig.add_trace(go.Scatter(x=[p["time"] for p in hist],y=[p["power"] for p in hist],mode="lines",line={"color":"#00E5FF","width":3,"shape":"spline"},fill="tozeroy",fillcolor="rgba(0,229,255,0.22)",name="Load"))
        fig.add_hline(y=total_load,line_dash="dash",line_color="#FACC15",annotation_text=f"{total_load:.2f} kW")
        fig=style(fig,205)
        fig.update_layout(showlegend=False,xaxis={"showgrid":False},yaxis={"title":"kW","gridcolor":"#1E293B","rangemode":"tozero"})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="hud_power")
    st.markdown(f"<div class='hud-kpi'>{total_load:.2f} <span class='hud-label'>kW LIVE</span></div><div class='hud-note'>15-point rolling campus demand</div>",unsafe_allow_html=True)

with left_compare_slot.container(border=True):
    hud_title("Hall Equipment & Load Comparison","▤")
    if go is not None:
        names=[h["name"].split(" - ")[0] for h in halls]
        fig=go.Figure()
        fig.add_bar(x=names,y=[h["lighting_power_kw"] for h in halls],name="Lighting",marker_color="#00B4D8")
        fig.add_bar(x=names,y=[h["ac_power_kw"] for h in halls],name="AC",marker_color="#A3E635")
        fig.add_bar(x=names,y=[h["total_power_kw"] for h in halls],name="Total",marker_color="#818CF8")
        fig.add_bar(x=names,y=[h["total_power_kw"]*.96 for h in halls],name="Daily reference",marker_color="rgba(148,163,184,.4)")
        fig.add_trace(go.Scatter(x=names,y=[h["total_power_kw"]*1.08 for h in halls],mode="lines+markers",name="Trend",line={"color":"#FACC15","width":2,"dash":"dash"}))
        fig=style(fig,235)
        fig.update_layout(barmode="group",legend={"orientation":"h","y":1.18,"font":{"size":8}},xaxis={"showgrid":False},yaxis={"title":"kW","gridcolor":"#1E293B","rangemode":"tozero"})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="hud_compare")

with left_zone_slot.container(border=True):
    hud_title("Zone Power & Capacity Distribution","◉")
    if go is not None:
        fig=go.Figure(go.Barpolar(theta=[h["name"].split(" - ")[0] for h in halls],r=[h["total_power_kw"] for h in halls],customdata=[[h["capacity"]] for h in halls],marker={"color":["#00B4D8","#2DD4BF","#818CF8"],"line":{"color":"#BDF8FF","width":1}},opacity=.9,hovertemplate="%{theta}<br>%{r:.2f} kW<br>Capacity %{customdata[0]}<extra></extra>"))
        fig.update_layout(template="plotly_dark",height=220,margin={"l":10,"r":10,"t":25,"b":10},paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font={"color":"#8EA7BA","size":9},polar={"bgcolor":"rgba(0,0,0,0)","radialaxis":{"gridcolor":"#1E293B"},"angularaxis":{"gridcolor":"#1E293B"}})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="hud_zone")
    st.markdown(f"<div class='hud-note'>Occupancy {occupants}/{capacity_total} · Lights {lighting_load:.2f} kW · AC {ac_load:.2f} kW</div>",unsafe_allow_html=True)

with right_fixture_slot.container(border=True):
    hud_title("Fixture & Switchboard Inspection Status","⚡")
    if go is not None:
        labels=["Working Lights","Commanded ON","Failed / Broken","Standby / Off","Active AC Units"]
        values=[lights_on,cmd_total,broken,standby,active_ac]
        fig=go.Figure(go.Pie(labels=labels,values=values,hole=.58,sort=False,marker={"colors":["#00B4D8","#2DD4BF","#F43F5E","#64748B","#A3E635"],"line":{"color":"#09111C","width":2}},textinfo="label+value",textposition="outside",textfont={"color":"#B8D4E6","size":8},showlegend=False))
        fig.update_layout(template="plotly_dark",height=260,margin={"l":10,"r":10,"t":25,"b":10},paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font={"color":"#8EA7BA","size":8},showlegend=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="hud_fixtures")
    st.markdown(f"<div class='hud-note'>Working {lights_on} · faults {broken} · off {standby} · AC {active_ac}</div>",unsafe_allow_html=True)

with right_hall_slot.container(border=True):
    hud_title("Hall Efficiency & Occupancy Breakdown","▥")
    if go is not None:
        names=[h["name"].split(" - ")[0] for h in halls]
        fig=go.Figure()
        fig.add_bar(x=names,y=[h["occupancy"] for h in halls],name="Active Occupants",marker_color="#00B4D8")
        fig.add_bar(x=names,y=[h["lights_working"] for h in halls],name="Working Lights",marker_color="#10B981")
        fig.add_bar(x=names,y=[h["ac_units"] if h["ac_kw"]>0 else 0 for h in halls],name="AC Units",marker_color="#6366F1")
        fig=style(fig,220)
        fig.update_layout(barmode="stack",legend={"orientation":"h","y":1.18,"font":{"size":8}},xaxis={"showgrid":False},yaxis={"gridcolor":"#1E293B","rangemode":"tozero"})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="hud_efficiency")

with right_ai_slot.container(border=True):
    hud_title("AI Energy Advisor & Work Order Distribution","◈")
    ai_cols=st.columns([1,1])
    use_genai=ai_cols[0].toggle("Use Groq GenAI",False,key="hud_genai",help="Provide a Groq key for this session, or configure GROQ_API_KEY in the server environment / Streamlit secrets.")
    generate=ai_cols[1].button("Generate recommendations",use_container_width=True,key="hud_recommend")
    api_key=st.text_input("Groq API key (this session only)",type="password",key="hud_groq_key",label_visibility="collapsed",placeholder="Paste a new Groq API key",help="Kept in this Streamlit session only; it is not written to telemetry.json or project files.") if use_genai else None
    if use_genai and st.button("Check Groq access", key="hud_groq_check", help="Validates the key with Groq and lists models available to this project."):
        accessible_models, access_error = check_groq_access(api_key)
        if access_error:
            st.warning(access_error)
        elif accessible_models:
            st.success("Groq accepted the key. Models available to this project: " + ", ".join(accessible_models[:12]))
            if len(accessible_models) > 12:
                st.caption(f"And {len(accessible_models) - 12} more. Set GROQ_MODEL to a listed model if the advisor's default is restricted.")
        else:
            st.info("Groq accepted the key, but did not return any models for this project.")
    if generate:
        if use_genai:
            recommendations,error=genai_recommendations(halls,api_key)
            if error: st.warning(error)
            if recommendations is None: recommendations=local_recommendations(halls)
        else: recommendations=local_recommendations(halls)
        st.session_state["recommendations"]=recommendations
    recs=st.session_state.get("recommendations",local_recommendations(halls))
    counts={"High Priority Faults":0,"Unoccupied HVAC Waste":0,"AC Staging":0,"Comfort Optimization":0}
    for rec in recs:
        text=(str(rec.get("action",""))+" "+str(rec.get("reason",""))).lower()
        if "fault" in text or "light" in text or str(rec.get("priority","")).lower()=="high": counts["High Priority Faults"]+=1
        elif "unoccupied" in text or "waste" in text: counts["Unoccupied HVAC Waste"]+=1
        elif "ac" in text or "hvac" in text: counts["AC Staging"]+=1
        else: counts["Comfort Optimization"]+=1
    if go is not None and recs:
        fig=go.Figure(go.Pie(labels=list(counts),values=list(counts.values()),hole=.45,marker={"colors":["#F43F5E","#F59E0B","#00B4D8","#A855F7"]},textinfo="label+value",textposition="inside",textfont={"color":"#E2E8F0","size":8}))
        fig.update_layout(template="plotly_dark",height=180,margin={"l":10,"r":10,"t":25,"b":10},paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font={"color":"#8EA7BA","size":8},showlegend=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="hud_advisor_pie")
    with st.container(height=220,border=False):
        for rec in recs:
            st.markdown(f"<div class='alert-row'><span class='alert-ring'>{'🔴' if str(rec.get('priority','')).lower()=='high' else '🟡'}</span><div><b>{rec.get('priority','Review')} · {rec.get('action','Review resource use')}</b><div class='alert-hall'>{rec.get('hall','Campus')}</div><div class='alert-meta'>{rec.get('reason','Based on current telemetry.')}</div></div></div>",unsafe_allow_html=True)
        for hall in halls:
            if hall_lighting_fault(hall):
                st.markdown(f"<div class='alert-row'><span class='alert-ring'>🔺</span><div><b>Lighting fault · {max(0,hall['lights_commanded_on']-hall['lights_working'])} fixture(s) not detected</b><div class='alert-hall'>{hall['name']}</div><div class='alert-meta'>CameraVision · OpenCV · Priority HIGH</div></div></div>",unsafe_allow_html=True)
            if hall["occupancy"]==0 and hall["total_power_kw"]>0:
                st.markdown(f"<div class='alert-row'><span class='alert-ring'>⚡</span><div><b>Unoccupied load waste · {hall['total_power_kw']:.2f} kW</b><div class='alert-hall'>{hall['name']}</div><div class='alert-meta'>Live telemetry · Priority HIGH</div></div></div>",unsafe_allow_html=True)

st.markdown("<div class='hud-title' style='justify-content:center'><span class='hud-title-main'><span class='hud-dot'></span>CAMPUS TELEMETRY GAUGE DOCK</span></div>",unsafe_allow_html=True)
gauge_cols=st.columns(6,gap="small")
gauge_data=[("Temperature",mean_temp,40,"°C","#F97316"),("Occupancy %",occupants/max(capacity_total,1)*100,100,"%","#2DD4BF"),("Est. CO₂",400+occupants*35,2000," ppm","#A855F7"),("Live Power",total_load,25," kW","#EAB308"),("Light Health",lights_on/max(cmd_total,1)*100,100,"%","#00E5FF"),("Session Cost",meter["cost"],max(1.0,float(meter["cost"])*1.4),"","#8B5CF6")]
for col,(title,value,maximum,suffix,color) in zip(gauge_cols,gauge_data):
    with col.container(border=True):
        if go is not None:
            fig=go.Figure(go.Indicator(mode="gauge+number",value=value,title={"text":title,"font":{"size":10,"color":"#94AEC2"}},number={"font":{"size":19,"color":"#DDF8FF"},"suffix":suffix,"valueformat":".1f"},gauge={"shape":"angular","axis":{"range":[0,maximum],"tickwidth":0,"tickcolor":"#526477","tickfont":{"size":7,"color":"#71879A"},"nticks":4},"bar":{"color":color,"thickness":.28},"bgcolor":"#262B36","borderwidth":0,"steps":[{"range":[0,maximum],"color":"#262B36"}]}))
            fig.update_layout(template="plotly_dark",height=160,margin={"l":10,"r":10,"t":25,"b":10},paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key=f"hud_gauge_{title}")
        else: st.metric(title,f"{value:.1f}{suffix}")

with st.expander("Inspect shared telemetry JSON"):
    st.json({"updated_at":datetime.now().astimezone().isoformat(timespec="seconds"),"halls":halls,"power":power_record,"live_session":meter})
    st.caption(f"Persistent data file: {DATA_FILE.name}.")
