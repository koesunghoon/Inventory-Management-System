import cv2
import socket
import struct
import pickle

# PC IP/PORT 
PC_IP = "10.10.16.10"
PORT = 9999

cap = cv2.VideoCapture(0)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((PC_IP, PORT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 JPEG로 압축 (전송량 줄이기)
    _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    data = pickle.dumps(encoded)

    # 데이터 크기 먼저 전송 후 실제 데이터 전송
    message = struct.pack("Q", len(data)) + data
    client_socket.sendall(message)

cap.release()
client_socket.close()