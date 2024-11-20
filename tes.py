import cv2
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO('best.pt')  # Pastikan path menuju 'best.pt' benar

# Inisialisasi webcam
cap = cv2.VideoCapture(0)  # 0 untuk kamera default laptop

while True:
    # Capture frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengakses kamera!")
        break

    # Lakukan deteksi objek
    results = model(frame)

    # Draw bounding boxes dan labels pada frame
    annotated_frame = results.plot()

    # Tampilkan hasil deteksi
    cv2.imshow("Deteksi Objek", annotated_frame)

    # Keluar jika tekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
