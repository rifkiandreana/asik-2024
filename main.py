import os
import base64
from io import BytesIO
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Inisialisasi Flask app
app = Flask(__name__)

# Inisialisasi model YOLO
detector = YOLO('best.pt')  # Memuat model YOLO yang sudah dilatih

# Direktori untuk menyimpan gambar
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Warna untuk setiap kelas (sesuaikan dengan jumlah kelas yang Anda miliki)
class_colors = {
    '0': (0, 255, 0),  # Hijau untuk Healthy Seaweed
    '1': (255, 0, 0),   # Merah untuk Kerak Bryozoan
    '2': (0, 0, 255)    # Biru untuk ice-ice
}

# Route untuk halaman utama
@app.route("/")
def home():
    return render_template("index.html")

# Rute untuk halaman utama dengan upload gambar dan deteksi webcam
@app.route('/deteksipenyakit')
def deteksipenyakit():
    return render_template('deteksipenyakit.html')

# Rute untuk menerima gambar dan melakukan deteksi objek
@app.route('/submit', methods=['POST'])
def submit_data():
    data = request.json
    image_data = data.get('image')

    if not image_data:
        return jsonify({'status': 'error', 'message': 'Data tidak lengkap'}), 400

    try:
        # Decode base64 image
        image_str = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_str)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")  # Pastikan format RGB
        
        # Run YOLO detection
        results = detector.predict(image, conf=0.25)

        # Buat gambar untuk menggambar bounding box
        draw = ImageDraw.Draw(image)
        
        bounding_boxes = []
        
        # Load font for larger text (you can change the font and size)
        font = ImageFont.truetype("arial.ttf", 20)  # Set a larger font size

        # Loop through all results
        for result in results:
            for box in result.boxes:
                # Koordinat bounding box
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                class_id = int(box.cls[0])  # ID kelas objek
                
                # Mengambil nama kelas berdasarkan class_id
                class_name = detector.model.names.get(class_id, "Unknown")
                
                # Pilih warna berdasarkan nama kelas
                color = class_colors.get(str(class_id), (255, 255, 255))  # Default ke putih jika kelas tidak dikenali

                # Menggambar bounding box dengan warna yang sesuai
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Menggambar nama kelas dengan font besar
                text_bbox = draw.textbbox((x1, y1 - 10), class_name, font=font)  # Get bounding box of the text
                text_width = text_bbox[2] - text_bbox[0]  # Calculate width from bbox
                text_height = text_bbox[3] - text_bbox[1]  # Calculate height from bbox

                # Menambahkan nama kelas di atas kotak dengan font yang lebih besar
                draw.text((x1, y1 - text_height - 5), class_name, fill=color, font=font)

                # Menyimpan informasi bounding box
                bounding_boxes.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'class_id': class_id,
                    'class_name': class_name
                })

        # Konversi gambar dengan bounding box ke base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'status': 'success',
            'message': 'Data berhasil diterima',
            'bounding_boxes': bounding_boxes,
            'image': img_str,  # Gambar hasil deteksi dikirim sebagai base64
            'detected': True    # Flag to indicate that detection has occurred
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Route untuk halaman 'Estimasi'
@app.route("/estimasipanen")
def about():
    return render_template("estimasipanen.html")

# Route untuk halaman 'Penyakit'
@app.route("/penyakitrumputlaut")
def penyakit():
    return render_template("penyakitrumputlaut.html")

# Route untuk halaman 'Peta'
@app.route("/persebaran")
def peta():
    return render_template("peta.html")

if __name__ == '__main__':
    app.run(debug=True)
