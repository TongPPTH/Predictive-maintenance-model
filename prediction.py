import joblib
import numpy as np

def predict_file(path, model):
    signal = load_file(path)
    feat = extract_features(signal)

    feat = np.array(feat).reshape(1, -1)

    pred = model.predict(feat)[0]

    return decode_label(pred)

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

def decode_label(label):
    if label == 0:
        return "GOOD"
    elif label == 1:
        return "WARNING"
    else:
        return "DANGER"

model = joblib.load("vibration_model.pkl")

result = predict_file("A_Jockey pump_M1A_2925__Sep24.txt", model)

print("Result:", result)

print("🔍 ผลวิเคราะห์:", result)