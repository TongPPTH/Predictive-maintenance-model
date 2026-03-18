import numpy as np
import matplotlib.pyplot as plt

def load_signal(path):
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


def plot_raw_signal(path):
    signal = load_signal(path)

    time = signal[:, 0]
    amp = signal[:, 1]

    # เริ่มที่ 0
    time = time - time[0]

    plt.figure()
    plt.plot(time, amp)

    plt.title("Correct Ordered Signal")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (G)")

    plt.grid()
    plt.show()



# -------------------------------
# 3. MAIN
# -------------------------------
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
    print("\n===== PROCESSING =====")
    plot_raw_signal(f)