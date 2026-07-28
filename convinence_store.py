import socket
import struct
import pickle
import cv2
import threading
from flask import Flask, Response, jsonify, render_template_string
from ultralytics import YOLO

model = YOLO(r"C:\Users\IOT16\Desktop\test_yolo2\best.pt")
model.to("cuda")  # GPU 사용

PORT = 9999
CLASS_NAMES = ["cola", "gum", "snack"]  # data.yaml의 클래스 순서와 일치해야 함

# 여러 스레드에서 공유할 상태
latest_frame = None
latest_status = {name: {"count": 0, "status": "unknown"} for name in CLASS_NAMES}
lock = threading.Lock()


def get_stock_status(count):
    if count >= 2:
        return "normal"
    elif count == 1:
        return "low"      # 재고 부족
    else:
        return "empty"     # 재고 소진


def socket_receiver():
    global latest_frame, latest_status

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', PORT))
    server_socket.listen(1)
    print("Jetson 연결 대기중...")

    conn, addr = server_socket.accept()
    print(f"연결됨: {addr}")

    data = b""
    payload_size = struct.calcsize("Q")

    while True:
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                return
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        encoded_frame = pickle.loads(frame_data)
        frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)

        results = model.predict(source=frame, conf=0.5, device='cuda' ,verbose=False)
        annotated = results[0].plot()

        # 클래스별 detection 개수 세기
        counts = {name: 0 for name in CLASS_NAMES}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in counts:
                counts[cls_name] += 1

        status = {}
        for name in CLASS_NAMES:
            status[name] = {
                "count": counts[name],
                "status": get_stock_status(counts[name])
            }

        with lock:
            latest_frame = annotated
            latest_status = status


def generate_mjpeg():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>편의점 재고 관리 시스템</title>
    <meta charset="utf-8">
    <style>
        body { font-family: sans-serif; text-align: center; background: #f5f5f5; }
        h1 { margin-top: 20px; }
        #video { border: 3px solid #333; margin-top: 10px; }
        .status-panel { display: flex; justify-content: center; gap: 20px; margin-top: 20px; }
        .card {
            padding: 20px 40px; border-radius: 12px; font-size: 20px; font-weight: bold;
            color: white; min-width: 150px;
        }
        .normal { background-color: #4CAF50; }
        .low { background-color: #FF9800; }
        .empty { background-color: #F44336; }
    </style>
</head>
<body>
    <h1>편의점 재고 관리 시스템</h1>
    <img id="video" src="/video_feed" width="800">
    <div class="status-panel" id="status-panel"></div>

    <script>
        async function updateStatus() {
            const res = await fetch('/stock_status');
            const data = await res.json();
            const panel = document.getElementById('status-panel');
            panel.innerHTML = '';

            const labelMap = {cola: "콜라", gum: "껌", snack: "과자"};
            const msgMap = {
                normal: "정상",
                low: "재고 부족 - 채워주세요",
                empty: "재고 소진 - 채워주세요"
            };

            for (const [name, info] of Object.entries(data)) {
                const div = document.createElement('div');
                div.className = 'card ' + info.status;
                div.innerHTML = `${labelMap[name]}<br>${info.count}개<br>${msgMap[info.status]}`;
                panel.appendChild(div);
            }
        }
        setInterval(updateStatus, 1000);
        updateStatus();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stock_status')
def stock_status():
    with lock:
        return jsonify(latest_status)


if __name__ == '__main__':
    threading.Thread(target=socket_receiver, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
