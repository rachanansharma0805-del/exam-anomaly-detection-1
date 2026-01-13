import cv2
import pandas as pd
from collections import defaultdict

# ==============================
# CONFIG
# ==============================
INPUT_VIDEO = r"C:\Users\rachana sharma\exam-hall-anomaly\tests\Test3\processed_output_week3.mp4"
ANOMALY_CSV = r"C:\Users\rachana sharma\exam-hall-anomaly\tests\Test3\anomaly_log.csv"
OUTPUT_VIDEO = r"C:\Users\rachana sharma\exam-hall-anomaly\tests\Test3\anomaly_events.mp4"
BUFFER = 15  # frames before & after anomaly for context

# ==============================
# LOAD ANOMALY LOG
# ==============================
df = pd.read_csv(ANOMALY_CSV)

# Map frame -> list of anomalies
frame_anomalies = defaultdict(list)
for _, row in df.iterrows():
    for f in range(int(row["frame"])-BUFFER, int(row["frame"])+BUFFER+1):
        frame_anomalies[f].append(f'{row["type"]} (ID {row["person_id"]})')

# ==============================
# VIDEO SETUP
# ==============================
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise IOError("Cannot open video")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    FPS,
    (W, H)
)

# ==============================
# PROCESS VIDEO
# ==============================
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id in frame_anomalies:
        # Overlay all anomaly types for this frame
        for i, text in enumerate(frame_anomalies[frame_id]):
            cv2.putText(frame, text, (10, 30 + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        out.write(frame)

    frame_id += 1

cap.release()
out.release()
print(f"✅ Anomaly video saved: {OUTPUT_VIDEO}")
