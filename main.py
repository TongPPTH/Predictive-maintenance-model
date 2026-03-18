import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_file(path):
    col1, col2, col3, col4 = [], [], [], []

    with open(path, 'r') as f:
        for line in f:

            # ข้าม header
            if "Time" in line or "----" in line:
                continue

            parts = line.strip().split()

            if len(parts) < 8:
                continue

            try:
                # แยก 4 คอลัมน์
                col1.append([float(parts[0]), float(parts[1])])
                col2.append([float(parts[2]), float(parts[3])])
                col3.append([float(parts[4]), float(parts[5])])
                col4.append([float(parts[6]), float(parts[7])])
            except:
                continue

    # 🔥 ต่อข้อมูลตามลำดับคอลัมน์
    signal = np.array(col1 + col2 + col3 + col4)

    print("shape:", signal.shape)

    return signal

def get_machine_from_file(path):
    with open(path, 'r') as f:
        for line in f:
            if "Equipment" in line:
                return line.split("Equipment:")[1].strip()

    # ❗ ไม่ใช้ UNKNOWN → แจ้ง error
    raise ValueError(f"❌ หา Equipment ไม่เจอในไฟล์: {path}")

def debug_file(path):
    print("===== ", path, "=====")
    with open(path, 'r') as f:
        for i in range(15):
            print(f.readline().strip())

def extract_features(signal):
    amp = signal[:, 1]

    features = [
        np.mean(amp),
        np.std(amp),
        np.max(amp),
        np.min(amp),
        np.sqrt(np.mean(amp**2))  # RMS
    ]

    return features

def analyze_iso(signal):
    time = signal[:, 0] / 1000.0  # ms → s
    amp_g = signal[:, 1]

    # 1. g → m/s^2
    acc = amp_g * 9.81

    # 2. หา dt
    dt = np.mean(np.diff(time))

    # 3. integrate → velocity
    velocity = np.cumsum(acc * dt)

    # 4. RMS
    rms = np.sqrt(np.mean(velocity**2))

    # 5. แปลงเป็น mm/s
    rms_mm = rms * 1000

    # 6. classify
    if rms_mm < 2.8:
        status = "GOOD"
    elif rms_mm < 7.1:
        status = "WARNING"
    else:
        status = "DANGER"

    return rms_mm, status

def encode_status(status):
    if status == "GOOD":
        return 0
    elif status == "WARNING":
        return 1
    else:
        return 2

def extract_header_info(path):
    with open(path, 'r') as f:
        for line in f:
            if "Date/Time" in line and "Amplitude" in line:
                # แยกส่วน
                parts = line.split("Amplitude:")

                date_part = parts[0].replace("Date/Time:", "").strip()
                amp_part = parts[1].strip()

                # แปลง datetime
                try:
                    dt = datetime.strptime(date_part, "%d-%b-%y %H:%M:%S")
                except Exception as e:
                    print("❌ parse date error:", e)
                    dt = None

                return dt, amp_part

    raise ValueError(f"❌ หา Date/Time ไม่เจอในไฟล์: {path}")

X = []
y = []
machines = []

files = [
    "A_CH-06 A_NAA_1490__Jun24.txt",
    "A_CH-06 A_NAA_1490__Oct24.txt",
    "A_CH-06 A_NAA_1490__Sep24.txt",
    "A_Cooling Pump OAH 02_M1H_1480_Oct24.txt",
    "A_Cooling Pump OAH 02_M1H_1480_Sep24.txt",
    "A_Jockey pump_M1A_2925__Jun24.txt",
    "A_Jockey pump_M1A_2925__Oct24.txt",
    "A_Jockey pump_M1A_2925__Sep24.txt"
]

for f in files:
    # debug ดู header (เปิดใช้ถ้าสงสัย)
    # debug_file(f)

    signal = load_file(f)
    machine_name = get_machine_from_file(f)
    
    feat = extract_features(signal)

    rms, status = analyze_iso(signal)
    dt, amp_unit = extract_header_info(f)

    print("Datetime:", dt)
    print("Amplitude Unit:", amp_unit)
    print(machine_name, "→", rms, "mm/s →", status)
    print()

    X.append(feat)
    y.append(encode_status(status))
    machines.append(machine_name)


X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# predict
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "vibration_model.pkl")
print("✅ โหลดโมเดลสำเร็จ")