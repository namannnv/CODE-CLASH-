"""Green building camera dashboard and three-hall digital twin."""
import json
import os
import queue
import threading
import time
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

def genai_recommendations(halls):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        # Streamlit deployments commonly provide credentials through
        # .streamlit/secrets.toml instead of the server process environment.
        try:
            key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            key = None
    if not key:
        return None, "Configure OPENAI_API_KEY in the Streamlit server environment or .streamlit/secrets.toml, then restart the app."
    body = {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "messages":[{"role":"system","content":"You are a building energy advisor. For every hall, analyze occupancy, lights commanded on versus lights the camera detects as working, lighting power, AC unit count and AC power, and temperature. If commanded lights exceed detected working lights, clearly flag a possible lighting fault and recommend safe checks for the fixture, lamp, driver, wiring, or circuit; do not assert a specific cause without evidence. Also give actionable HVAC/AC and energy recommendations while preserving comfort. Return a JSON object with a recommendations array; each item has priority, hall, action, reason. Never claim to control equipment or invent measured savings."},{"role":"user","content":json.dumps(halls)}],"response_format":{"type":"json_object"},"temperature":0.3}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization":"Bearer " + key,"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            result = json.loads(response.read())
        parsed = json.loads(result["choices"][0]["message"]["content"])
        recs = parsed if isinstance(parsed, list) else parsed.get("recommendations", [])
        return recs, None
    except Exception as exc:
        return None, f"GenAI request failed ({exc}). Showing local recommendations instead."

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
live_enabled=False
header_slot=st.empty()
dash_cols=st.columns(4,gap="small")
gallery_col,energy_col,history_col,advisor_col=dash_cols
with gallery_col.container(border=True):
    card_header("⌂","Gallery & Digital Twin")
    view_mode=st.radio("Gallery view",["3D Twin","2D Floor Plan","Camera HUD"],horizontal=True,label_visibility="collapsed",key="gallery_view_mode")
    st.toggle("⤢ Expand Fullscreen Twin",value=bool(st.session_state.get("expand_twin",False)),key="expand_twin")
    frame_preview=st.empty()
    if view_mode=="2D Floor Plan":
        floor_hall=next(h for h in halls if h["name"]==selected)
        st.markdown(f"<div style='border:1px solid #2A2E37;border-radius:7px;padding:6px;background:#171A20;color:#E2E8F0'><b style='font-size:12px'>{floor_hall['name']}</b>{hall_floor_svg(floor_hall,0)}</div>",unsafe_allow_html=True)
        st.caption("Green · occupancy  |  Amber · working light  |  Red · fault  |  Blue · active AC")
    elif view_mode=="3D Twin":
        if go is None: st.info("Install Plotly to display the spatial twin.")
        else:
            figure=go.Figure()
            for index,hall in enumerate(halls):
                x0=index*13.0; x1=x0+9; y0=0; y1=8; z1=3.2
                figure.add_trace(go.Mesh3d(x=[x0,x1,x1,x0],y=[y0,y0,y1,y1],z=[0]*4,i=[0,0],j=[1,2],k=[2,3],color="#242832",opacity=1,showlegend=False))
                figure.add_trace(go.Scatter3d(x=[x0,x1,x1,x0,x0,None,x0,x1,None,x1,x1,None,x1,x0,None,x0,x0],y=[y0,y0,y1,y1,y0,None,y0,y0,None,y0,y1,None,y1,y1,None,y1,y0],z=[0]*5+[None]+[0,z1,None]+[0,z1,None]+[z1,z1,None]+[z1,0],mode="lines",line={"color":"#94A3B8","width":3},showlegend=False))
                n=min(int(hall["occupancy"]),20)
                figure.add_trace(go.Scatter3d(x=[x0+1+(i%5)*1.6 for i in range(n)],y=[1+(i//5)*1.4 for i in range(n)],z=[.2]*n,mode="markers",marker={"color":"#10B981","size":6},name="Occupants" if index==0 else None,showlegend=index==0))
                n=min(int(hall["light_count"]),8); working=int(hall.get("lights_working",hall["lights_on"])); commanded=int(hall.get("lights_commanded_on",working))
                colors=["#F59E0B" if i<working else "#EF4444" if i<commanded else "#CBD5E1" for i in range(n)]
                figure.add_trace(go.Scatter3d(x=[x0+1+i*7/max(n-1,1) for i in range(n)],y=[7.5]*n,z=[z1]*n,mode="markers",marker={"color":colors,"size":7,"symbol":"square"},name="Lights" if index==0 else None,showlegend=index==0))
                figure.add_trace(go.Scatter3d(x=[x0+8.2],y=[.6],z=[.4],mode="markers",marker={"color":"#0EA5E9","size":9,"symbol":"diamond"},name="AC" if index==0 else None,showlegend=index==0))
                figure.add_trace(go.Scatter3d(x=[x0+4.5],y=[8.5],z=[3.4],mode="text",text=[hall["name"]],textfont={"color":"#334155","size":10},showlegend=False))
            figure.update_layout(template="plotly_dark",height=255,margin={"l":0,"r":0,"t":0,"b":0},paper_bgcolor="#1A1D24",plot_bgcolor="#1A1D24",scene={"bgcolor":"#1A1D24","xaxis":{"visible":False},"yaxis":{"visible":False},"zaxis":{"visible":False},"aspectmode":"manual","aspectratio":{"x":2.5,"y":1,"z":.5}},legend={"orientation":"h","y":1.05})
            st.plotly_chart(figure,use_container_width=True,config={"displayModeBar":False},key="spatial_twin")
    else:
        if source!="Live camera feed":
            st.info("Choose Live camera feed in the CameraVision sidebar to start the continuous stream.")
        else:
            try:
                from streamlit_webrtc import webrtc_streamer
            except ImportError:
                st.error("Continuous video requires streamlit-webrtc.")
            else:
                detector_config=(selected,bool(detect_lights_enabled),int(brightness_threshold),int(saturation_threshold),int(grid_rows),int(grid_columns))
                detector=st.session_state.get("live_grid_detector")
                if detector is None or detector.config!=detector_config or detector.stopped:
                    if detector is not None: detector.stop()
                    detector=LiveGridDetector(*detector_config); st.session_state["live_grid_detector"]=detector
                def live_video_callback(video_frame):
                    return video_frame if pause_camera else detector.process(video_frame)
                if cv2 is None or np is None: st.warning("Install OpenCV and NumPy to enable grid detection.")
                live_context=webrtc_streamer(key="mop-continuous-camera",media_stream_constraints={"video":{"width":{"ideal":960},"height":{"ideal":540},"frameRate":{"ideal":30,"max":30}},"audio":False},video_frame_callback=live_video_callback)
                live_enabled=bool(live_context.state.playing)
                st.caption("Analysis paused · video remains live." if pause_camera else "Continuous browser camera stream · occupancy and lighting grids.")
    card_footer()

detector=st.session_state.get("live_grid_detector")
if source!="Live camera feed" or (view_mode!="Camera HUD" and st.session_state.get("live_webrtc_playing",False)):
    if detector is not None: detector.stop()
    if source!="Live camera feed": st.session_state.pop("live_grid_detector",None)
was_playing=bool(st.session_state.get("live_webrtc_playing",False))
if was_playing and not live_enabled and detector is not None: detector.stop()
st.session_state["live_webrtc_playing"]=live_enabled

if frame is not None and cv2 is not None and np is not None and not pause_camera:
    try:
        idx=next(i for i,h in enumerate(halls) if h["name"]==selected)
        if YOLO is not None:
            result=load_yolo_model().predict(frame,classes=[0],imgsz=640,verbose=False)[0]
            count=len(result.boxes); person_boxes=result.boxes.xyxy.tolist(); annotated=result.plot()
        else:
            count=None; person_boxes=[]; annotated=frame.copy()
            st.warning("YOLO is unavailable; person counting is paused.")
        light_boxes=detect_lights(frame,brightness_threshold,saturation_threshold) if detect_lights_enabled else []
        switch_states=None
        if detect_switchboard:
            switch_states=classify_switchboard(frame,(roi_x,roi_y,roi_w,roi_h),int(switch_count),int(switch_rows),on_position)
            if switch_states is not None:
                confirmed=min(switch_states.count("on"),int(halls[idx]["light_count"])); uncertain=switch_states.count("unknown")
                halls[idx].update({"lights_commanded_on":confirmed,"switchboard_on_count":confirmed,"switchboard_off_count":switch_states.count("off"),"switchboard_uncertain_count":uncertain,"switchboard_calibrated":True,"switchboard_vision_enabled":True})
                working=min(len(light_boxes),int(halls[idx]["light_count"])) if detect_lights_enabled else int(halls[idx].get("lights_working",halls[idx]["lights_on"]))
                halls[idx]["switchboard_fault_confirmed"]=confirmed>working and uncertain==0
                st.session_state[f"cmdlights{idx}"]=confirmed
            else:
                halls[idx].update({"switchboard_vision_enabled":True,"switchboard_calibrated":True,"switchboard_fault_confirmed":False,"switchboard_uncertain_count":int(switch_count)})
        if count is not None:
            halls[idx]["occupancy"]=count; halls[idx]["source"]="CameraVision · YOLO"; st.session_state[f"occ{idx}"]=min(count,int(halls[idx]["capacity"]))
        if detect_lights_enabled:
            halls[idx]["lights_working"]=min(len(light_boxes),int(halls[idx]["light_count"])); halls[idx]["lights_on"]=halls[idx]["lights_working"]; halls[idx]["light_detection"]="OpenCV bright-region estimate"; st.session_state[f"working_lights{idx}"]=halls[idx]["lights_working"]
        halls[idx]["captured_at"]=datetime.now().astimezone().isoformat(timespec="seconds")
        draw_detection_grid(annotated,person_boxes,light_boxes,grid_rows,grid_columns,selected,count)
        frame_preview.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB),caption=f"CameraVision · {selected}",use_container_width=True)
        st.caption(f"{count if count is not None else 'YOLO unavailable'} people · {len(light_boxes)} working light regions")
    except Exception as exc:
        st.error(f"Could not process frame: {exc}")
elif frame is not None and pause_camera and cv2 is not None:
    frame_preview.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),caption=f"CameraVision paused · {selected}",use_container_width=True)

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
            hall_update={**hall,"occupancy":occ,"lights_on":working,"lights_commanded_on":commanded,"lights_working":working,"ac_kw":kw,"temperature_c":temp}
            if commanded!=int(hall.get("lights_commanded_on",hall["lights_on"])):
                hall_update["switchboard_vision_enabled"]=False; hall_update["switchboard_calibrated"]=False
            edited.append(hall_update)
        if st.form_submit_button("Save hall telemetry",use_container_width=True):
            halls=edited
            save_data(halls)
            st.success("Hall telemetry saved to telemetry.json")

for hall in halls:
    hall["lights_working"]=int(hall.get("lights_working",hall["lights_on"]))
    hall["lights_commanded_on"]=int(hall.get("lights_commanded_on",hall["lights_on"]))
    hall["lighting_power_kw"]=hall["lights_working"]*float(light_watts)/1000
    hall["ac_power_kw"]=float(hall["ac_kw"]); hall["total_power_kw"]=hall["lighting_power_kw"]+hall["ac_power_kw"]
current_power_kw=sum(h["total_power_kw"] for h in halls); now=time.time()
meter=st.session_state.setdefault("live_energy_meter",{"active":False,"started_at":None,"ended_at":None,"last_tick":None,"last_power_kw":0.0,"energy_kwh":0.0,"cost":0.0,"elapsed_seconds":0.0})
if live_enabled and not meter["active"]:
    meter.update({"active":True,"started_at":datetime.now().astimezone().isoformat(timespec="seconds"),"ended_at":None,"last_tick":now,"last_power_kw":current_power_kw,"energy_kwh":0.0,"cost":0.0,"elapsed_seconds":0.0})
elif meter["active"]:
    elapsed=max(0.0,now-float(meter["last_tick"] or now)); interval=float(meter["last_power_kw"])*elapsed/3600
    meter["energy_kwh"]+=interval; meter["cost"]+=interval*float(tariff); meter["elapsed_seconds"]+=elapsed; meter["last_tick"]=now
    if live_enabled: meter["last_power_kw"]=current_power_kw
    else: meter["active"]=False; meter["ended_at"]=datetime.now().astimezone().isoformat(timespec="seconds"); meter["last_power_kw"]=0.0

occupants=sum(int(h["occupancy"]) for h in halls); lights_on=sum(int(h["lights_on"]) for h in halls)
lighting_load=sum(float(h["lighting_power_kw"]) for h in halls); ac_load=sum(float(h["ac_power_kw"]) for h in halls); total_load=lighting_load+ac_load
mean_temp=sum(float(h["temperature_c"]) for h in halls)/max(1,len(halls)); active_lights=sum(h["lights_on"]>0 for h in halls)
broken=sum(max(0,int(h["lights_commanded_on"])-int(h["lights_working"])) for h in halls)
standby=sum(max(0,int(h["light_count"])-int(h["lights_commanded_on"])) for h in halls)
active_ac=sum(int(h["ac_units"]) if float(h["ac_kw"])>0 else 0 for h in halls)
total_assets=sum(int(h["light_count"])+int(h["ac_units"]) for h in halls)
if "occupancy_history" not in st.session_state: st.session_state["occupancy_history"]=[]
st.session_state["occupancy_history"].append({"time":datetime.now().strftime("%H:%M:%S"),**{h["name"]:int(h["occupancy"]) for h in halls}})
st.session_state["occupancy_history"]=st.session_state["occupancy_history"][-48:]
st.session_state["occ_history"]=st.session_state["occupancy_history"]
clock_text=datetime.now().strftime("%d %b %Y · %H:%M:%S")
header_slot.markdown(f"""<div class="bms-nav"><div class="bms-brand"><span class="bms-brand-mark">V</span><b>verdant<span style="color:#D9381E">*</span></b><span style="color:#777F8D">|</span> Smart Campus Towers</div><div class="bms-nav-right"><span class="bms-pill">Verdant Twin ▾</span><span class="bms-temp">{mean_temp:.1f}°C</span><span class="bms-pill" id="bms-clock">{clock_text}</span><span>⌖ Bengaluru</span><span class="bms-pill {'bms-live' if live_enabled and not pause_camera else ''}">● {'CAMERA PAUSED' if live_enabled and pause_camera else 'CAMERA LIVE' if live_enabled else 'CAMERA STANDBY'}</span></div></div><script>(function(){{const tick=()=>{{const el=document.getElementById('bms-clock');if(el)el.textContent=new Intl.DateTimeFormat('en-IN',{{day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}}).format(new Date());}};tick();setInterval(tick,1000);}})();</script>""",unsafe_allow_html=True)
live_record={"active":meter["active"],"started_at":meter["started_at"],"ended_at":meter["ended_at"],"elapsed_seconds":round(meter["elapsed_seconds"],2),"energy_kwh":round(meter["energy_kwh"],6),"cost":round(meter["cost"],6),"tariff_per_kwh":float(tariff)}
power_record={"lighting_kw":round(lighting_load,4),"ac_kw":round(ac_load,4),"total_kw":round(total_load,4),"light_watts_each":int(light_watts),"tariff_per_kwh":float(tariff)}
save_data(halls,power_record,live_record)

with energy_col.container(border=True):
    card_header("ϟ","Energy Consumption")
    baseline=max(total_load*24,.01); energy=float(meter["energy_kwh"]); change=(energy-baseline)/baseline*100
    color="#DC2626" if change>0 else "#16A34A"
    st.markdown(f'<div class="hero-metric">{energy:.3f}</div><div class="hero-sub">kWh - TODAY</div><div class="positive-badge">▲ 7.74%</div>',unsafe_allow_html=True)
    mini=st.columns(3); mini[0].metric("LAST DAY",f"{baseline:.1f} kWh"); mini[1].metric("LIVE LOAD",f"{total_load:.2f} kW"); mini[2].metric("SESSION COST",f"${meter['cost']:.4f}")
    st.caption("¹ Simulated 24-hour baseline · session meter runs while camera is live.")
    card_footer()

with history_col.container(border=True):
    card_header("⚡","Energy Sources")
    if go is not None:
        chart=go.Figure(go.Pie(labels=[h["name"].split(" - ")[0] for h in halls],values=[h["total_power_kw"] for h in halls],hole=.42,marker={"colors":["#0D8BF2","#10B981","#F59E0B"]},textinfo="label+percent",textfont={"size":9,"color":"#E2E8F0"}))
        chart.update_layout(template="plotly_dark",height=200,margin={"l":4,"r":4,"t":4,"b":4},showlegend=False,paper_bgcolor="#1A1D24",plot_bgcolor="#1A1D24")
        st.plotly_chart(chart,use_container_width=True,config={"displayModeBar":False},key="energy_sources")
    else: st.info("Install Plotly to view energy sources.")
    st.markdown(f"<div style='font-size:9px;color:#9AA2AF'><span style='color:#0D8BF2'>●</span> HVAC / AC ({ac_load:.2f} kW) &nbsp; <span style='color:#10B981'>●</span> Active lighting ({lighting_load:.2f} kW)<br><span style='color:#F59E0B'>●</span> Tariff (${tariff:.2f}/kWh)</div>",unsafe_allow_html=True)
    st.caption("Per-hall live power consumption")
    card_footer()

with gallery_col.container(border=True):
    card_header("👥","Total Occupancy")
    if go is not None:
        chart=go.Figure(); history=st.session_state["occupancy_history"]; colors=["#A3E635","#38BDF8","#F43F5E"]
        for i,h in enumerate(halls):
            chart.add_trace(go.Scatter(x=[r["time"] for r in history],y=[r[h["name"]] for r in history],name=h["name"].split(" - ")[0],mode="lines+markers",line={"color":colors[i],"width":2},marker={"size":4}))
        chart.update_layout(template="plotly_dark",height=220,margin={"l":4,"r":4,"t":4,"b":0},paper_bgcolor="#1A1D24",plot_bgcolor="#1A1D24",font={"color":"#AEB5C2","size":9},legend={"orientation":"h","y":1.1,"font":{"size":9}},xaxis={"showgrid":False},yaxis={"gridcolor":"#2A2E37","rangemode":"tozero"})
        st.plotly_chart(chart,use_container_width=True,config={"displayModeBar":False},key="occupancy_history")
    else: st.info("Install Plotly to view occupancy history.")
    st.caption("One hall occupancy sample is appended per rerun.")
    card_footer()

with energy_col.container(border=True):
    card_header("⚡","Energy Consumption History")
    if go is not None:
        chart=go.Figure(); names=[h["name"].split(" - ")[0] for h in halls]; loads=[h["total_power_kw"] for h in halls]
        chart.add_bar(x=names,y=loads,name="Live kW",marker_color="#8CE62E",text=[f"{v:.2f}" for v in loads],textposition="outside")
        chart.add_bar(x=names,y=[v*.96 for v in loads],name="Daily benchmark",marker_color="#4E623A",text=[f"{v*.96:.2f}" for v in loads],textposition="outside")
        chart.update_layout(template="plotly_dark",height=220,barmode="group",margin={"l":4,"r":4,"t":10,"b":0},paper_bgcolor="#1A1D24",plot_bgcolor="#1A1D24",font={"color":"#AEB5C2","size":9},legend={"orientation":"h","y":1.12,"font":{"size":9}},yaxis={"title":"kW","gridcolor":"#2A2E37"},xaxis={"showgrid":False})
        st.plotly_chart(chart,use_container_width=True,config={"displayModeBar":False},key="energy_history")
    else: st.info("Install Plotly to display energy history.")
    st.caption("Live kW demand compared with a simulated daily benchmark.")
    card_footer()

with history_col.container(border=True):
    card_header("🌱","Energy Carbon Emissions")
    carbon_tonnes=total_load*.00082
    st.markdown(f'<div class="hero-metric">{carbon_tonnes:.2f}</div><div class="hero-sub">TONNES - TODAY</div><div class="positive-badge">▼ 7.74% VS LAST DAY</div>',unsafe_allow_html=True)
    mini=st.columns(3); mini[0].metric("LAST DAY",f"{carbon_tonnes:.2f} t"); mini[1].metric("7 D AVG",f"{carbon_tonnes:.2f} t"); mini[2].metric("30 D AVG",f"{carbon_tonnes:.2f} t")
    st.caption("Estimated from live load at 0.82 kg CO₂ per kWh.")
    card_footer()

with gallery_col.container(border=True):
    card_header("▤","Meeting Rooms")
    rows=[]
    for hall in halls:
        fault=hall_lighting_fault(hall); waste=hall["occupancy"]==0 and hall["total_power_kw"]>0
        status_color="#F59E0B" if waste else "#F43F5E" if fault else "#10B981"
        hall_id=hall["name"].upper().replace(" - ","-").replace(" ","-")
        rows.append(f"<tr><td>{hall_id}</td><td><span style='color:{status_color}'>●</span> {hall['lights_working']}/{hall['lights_commanded_on']}</td><td><span style='color:{'#10B981' if hall['ac_kw']>0 else '#666B75'}'>●</span> {int(hall['ac_units']) if hall['ac_kw']>0 else 0}</td><td>{hall['occupancy']}/{hall['capacity']}</td></tr>")
    rows.append(f"<tr class='summary-row'><td>CAMPUS TOTAL</td><td>{sum(h['lights_working'] for h in halls)} working</td><td>{sum(int(h['ac_units']) if h['ac_kw']>0 else 0 for h in halls)} active</td><td>{occupants}/{sum(h['capacity'] for h in halls)}</td></tr>")
    st.markdown("<table class='dark-table'><thead><tr><th>FLOOR / HALL</th><th>LIGHTS ON</th><th>AC ACTIVE</th><th>OCCUPANCY</th></tr></thead><tbody>"+"".join(rows)+"</tbody></table>",unsafe_allow_html=True)
    card_footer()

with energy_col.container(border=True):
    card_header("🌡️","AQI & Thermal Telemetry")
    hall=next(h for h in halls if h["name"]==selected); co2=400+int(hall["occupancy"])*35
    st.markdown(f"""<div class="diagnostic-grid">
    <div><small style="color:#8CE62E">TEMPERATURE</small><b>{hall['temperature_c']:.1f} °C</b></div>
    <div><small style="color:#F59E0B">CO₂ EST.</small><b>{co2} ppm</b></div>
    <div><small style="color:#EF4444">HUMIDITY</small><b>— %</b></div>
    <div><small style="color:#8CE62E">LIGHTING</small><b>{hall['lighting_power_kw']:.2f} kW</b></div>
    <div><small style="color:#F59E0B">AC LOAD</small><b>{hall['ac_power_kw']:.2f} kW</b></div>
    <div><small style="color:#8B5CF6">BRIGHTNESS THR.</small><b>{brightness_threshold}</b></div></div>""",unsafe_allow_html=True)
    st.caption(f"{selected} - Live Sensor Fusion · CO₂ estimated; humidity sensor unavailable.")
    card_footer()

with history_col.container(border=True):
    card_header("⚠️","Switchboard & Fixture Health")
    commanded=sum(int(h["lights_commanded_on"]) for h in halls); working=sum(int(h["lights_working"]) for h in halls)
    failed=sum(max(0,int(h["lights_commanded_on"])-int(h["lights_working"])) for h in halls)
    mode="ON" if any(h.get("switchboard_vision_enabled") and h.get("switchboard_calibrated") for h in halls) else "SIM"
    health=working/max(commanded,1)*100; active_units=sum(int(h["ac_units"]) if h["ac_kw"]>0 else 0 for h in halls)
    st.markdown(f"""<div class="diagnostic-grid">
    <div><small>COMMANDED ON</small><b>{commanded}</b></div><div><small style="color:#8CE62E">WORKING LIGHTS</small><b>{working}</b></div>
    <div><small style="color:#F43F5E">FAILED LIGHTS</small><b>{failed}</b></div><div><small>SWITCH CV MODE</small><b>{mode}</b></div>
    <div><small style="color:#8CE62E">FIXTURE HEALTH</small><b>{health:.0f}%</b></div><div><small>AC UNITS</small><b>{active_units} Active</b></div></div>""",unsafe_allow_html=True)
    st.caption("Optical Switchboard & Ceiling Verification")
    card_footer()

with advisor_col.container(border=True):
    card_header("🔔","Alerts")
    ai_col,refresh_col=st.columns([1,1])
    use_genai=ai_col.toggle("Use OpenAI GenAI",False,help="Set OPENAI_API_KEY in the Streamlit server environment or .streamlit/secrets.toml.")
    generate=refresh_col.button("Refresh AI",use_container_width=True)
    if generate:
        if use_genai:
            recommendations,error=genai_recommendations(halls)
            if error: st.warning(error)
            if recommendations is None: recommendations=local_recommendations(halls)
        else: recommendations=local_recommendations(halls)
        st.session_state["recommendations"]=recommendations
    recs=st.session_state.get("recommendations",local_recommendations(halls))
    feed=[]
    for rec in recs:
        action=rec.get("action","Review resource use"); feed.append({"title":action,"hall":rec.get("hall","Campus"),"reason":rec.get("reason","Based on current telemetry."),"priority":str(rec.get("priority","Review")).upper(),"kind":"fault" if "light" in (action+rec.get("reason","")).lower() else "warn"})
    for hall in halls:
        if hall_lighting_fault(hall): feed.insert(0,{"title":f"Lighting fault · {max(0,hall['lights_commanded_on']-hall['lights_working'])} fixture(s) not detected","hall":hall["name"],"reason":"Lights are commanded on but fewer working fixtures were detected.","priority":"HIGH","kind":"fault"})
        if hall["occupancy"]==0 and (hall["lights_on"]>0 or hall["ac_kw"]>0): feed.insert(0,{"title":f"Unoccupied resource waste · {hall['total_power_kw']:.2f} kW","hall":hall["name"],"reason":"Lighting or HVAC remains active while no occupants are detected.","priority":"HIGH","kind":"warn"})
        if hall["temperature_c"]<21 or hall["temperature_c"]>26: feed.insert(0,{"title":f"Temperature outside comfort band · {hall['temperature_c']:.1f}°C","hall":hall["name"],"reason":"Review the room setpoint and HVAC airflow.","priority":"MEDIUM","kind":"warn"})
    if not feed: feed=[{"title":"Systems operating within expected range","hall":"Smart Campus","reason":"No active faults or resource mismatches were detected.","priority":"LOW","kind":"warn"}]
    unique=[]; seen=set()
    for item in feed:
        key=(item["title"],item["hall"])
        if key not in seen: seen.add(key); unique.append(item)
    with st.container(height=230,border=False):
        for item in unique:
            source_label=next((h.get("source","Simulation") for h in halls if h["name"]==item["hall"]),"Campus telemetry")
            st.markdown(f"<div class='alert-row'><span class='alert-ring'>⭕</span><div><b>{item['title']}</b><div class='alert-hall'>{item['hall']}</div><div class='alert-meta'>a few seconds ago | {item['priority']} Priority | {item['reason']}</div></div></div>",unsafe_allow_html=True)
    card_footer()

with advisor_col.container(border=True):
    card_header("▥","Booked Desks")
    if go is not None:
        names=[h["name"].split(" - ")[0] for h in halls]; chart=go.Figure()
        chart.add_bar(x=names,y=[h["capacity"] for h in halls],name="Desks Available",marker_color="#84CC16")
        chart.add_bar(x=names,y=[h["occupancy"] for h in halls],name="Desks Occupied",marker_color="#10B981")
        chart.update_layout(template="plotly_dark",height=205,barmode="group",margin={"l":4,"r":4,"t":8,"b":0},paper_bgcolor="#1A1D24",plot_bgcolor="#1A1D24",font={"color":"#AEB5C2","size":9},legend={"orientation":"h","y":1.1,"font":{"size":9}},yaxis={"title":"Desks","gridcolor":"#2A2E37"},xaxis={"showgrid":False})
        st.plotly_chart(chart,use_container_width=True,config={"displayModeBar":False},key="booked_desks")

with advisor_col.container(border=True):
    card_header("⚠️","Fault Reports")
    fault_items=[]
    for hall in halls:
        cmd=int(hall["lights_commanded_on"]); work=int(hall["lights_working"])
        if hall_lighting_fault(hall): fault_items.append((f"Lighting Circuit Fault - {max(0,cmd-work)} switched-on light(s) not working as intended",hall["name"],"CameraVision · OpenCV"))
        if hall["occupancy"]==0 and hall["total_power_kw"]>0: fault_items.append((f"Resource Mismatch - {hall['total_power_kw']:.2f} kW active with 0 occupants",hall["name"],"CameraVision · Live telemetry"))
    if not fault_items: st.markdown("<div class='clear-state'>✓ No open lighting or HVAC fault reports</div>",unsafe_allow_html=True)
    for title,hall_name,source_name in fault_items:
        st.markdown(f"<div class='fault-ticket'><span class='fault-triangle'>🔺</span><div><b>{title}</b><div class='ticket-meta'>By: {source_name} | {hall_name} | Live</div></div></div>",unsafe_allow_html=True)

with st.expander("⤢ Expanded Fullscreen Twin",expanded=bool(st.session_state.get("expand_twin",False))):
    full_mode=st.radio("Twin view",["3D Twin","2D Floor Plan"],horizontal=True,key="fullscreen_twin_mode")
    if full_mode=="3D Twin":
        if go is not None:
            figure=go.Figure()
            for index,hall in enumerate(halls):
                x0=index*13.0; x1=x0+9; y0=0; y1=8; z1=3.2
                figure.add_trace(go.Mesh3d(x=[x0,x1,x1,x0],y=[y0,y0,y1,y1],z=[0]*4,i=[0,0],j=[1,2],k=[2,3],color="#242832",opacity=.9,showlegend=False))
                figure.add_trace(go.Scatter3d(x=[x0,x1,x1,x0,x0,None,x0,x1,None,x1,x0,None,x0,x0],y=[y0,y0,y1,y1,y0,None,y0,y0,None,y0,y0,None,y0,y0],z=[0,0,0,0,0,None,0,z1,None,z1,z1,None,z1,0],mode="lines",line={"color":"#4B5260","width":4},showlegend=False))
                n=min(int(hall["occupancy"]),24)
                figure.add_trace(go.Scatter3d(x=[x0+1+(i%5)*1.6 for i in range(n)],y=[1+(i//5)*1.25 for i in range(n)],z=[.2]*n,mode="markers",marker={"color":"#10B981","size":7},name="People" if index==0 else None,showlegend=index==0))
                figure.add_trace(go.Scatter3d(x=[x0+4.5],y=[8.4],z=[3.3],mode="text",text=[hall["name"]],textfont={"color":"#E2E8F0"},showlegend=False))
            figure.update_layout(template="plotly_dark",height=600,paper_bgcolor="#0D0F12",scene={"bgcolor":"#0D0F12","xaxis":{"visible":False},"yaxis":{"visible":False},"zaxis":{"visible":False},"aspectmode":"manual","aspectratio":{"x":2.5,"y":1,"z":.55}})
            st.plotly_chart(figure,use_container_width=True,key="fullscreen_twin")
    else:
        cols=st.columns(3,gap="small")
        for col,hall in zip(cols,halls):
            with col.container(border=True):
                st.markdown(f"**{hall['name']}**")
                st.markdown(hall_floor_svg(hall,0),unsafe_allow_html=True)

with st.expander("Inspect shared telemetry JSON"):
    st.json({"updated_at":datetime.now().astimezone().isoformat(timespec="seconds"),"halls":halls,"power":power_record,"live_session":meter})
    st.caption(f"Persistent data file: {DATA_FILE.name}.")

