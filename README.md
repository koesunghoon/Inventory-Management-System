# 편의점 재고 관리 시스템 (AI 기반 실시간 재고 모니터링)

Jetson(또는 웹캠 클라이언트)에서 촬영한 영상을 PC 서버로 전송하고, YOLOv8 객체 탐지 모델로 상품(콜라·껌·과자) 재고 수량을 실시간으로 파악해 웹 대시보드에 표시하는 시스템입니다.

## 목차
- [개요](#개요)
- [시스템 구조](#시스템-구조)
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 실행](#설치-및-실행)
- [모델 학습](#모델-학습)
- [실행 화면](#실행-화면)
- [향후 개선 사항](#향후-개선-사항)

## 개요
매대에 진열된 상품을 카메라로 실시간 모니터링하여 재고가 부족하거나 소진되었을 때 자동으로 감지하는 프로젝트입니다. 클라이언트(카메라 장치)가 영상을 소켓 통신으로 서버에 전송하면, 서버는 YOLOv8 모델로 객체를 탐지하고 결과를 Flask 웹 페이지에 실시간 스트리밍 및 재고 상태로 표시합니다.

## 시스템 구조
```
[클라이언트: 웹캠]                [서버: PC/GPU]
convinence_store_client.py  --->  convinence_store.py
   - 웹캠 프레임 캡처                - 소켓으로 프레임 수신
   - JPEG 인코딩                    - YOLOv8 추론 (GPU)
   - 소켓(TCP)으로 전송              - 클래스별 개수 집계 → 재고 상태 판정
                                    - Flask로 MJPEG 스트리밍 + REST API 제공
                                            |
                                            v
                                   웹 브라우저 대시보드
                                   (영상 + 실시간 재고 카드)
```

- 통신: TCP 소켓, `struct`로 프레임 크기 헤더 전송 후 `pickle` 직렬화된 프레임 전송
- 추론: Ultralytics YOLOv8, CUDA GPU 사용
- 서빙: Flask (`/video_feed`: MJPEG 스트림, `/stock_status`: JSON API)

## 주요 기능
- 실시간 영상 스트리밍 (MJPEG)
- 상품별(콜라/껌/과자) 탐지 개수 실시간 집계
- 재고 상태 3단계 자동 판정
  - `normal` (2개 이상): 정상
  - `low` (1개): 재고 부족
  - `empty` (0개): 재고 소진
- 1초 주기로 웹 대시보드 자동 갱신 (Fetch API 폴링)

## 기술 스택
| 구분 | 사용 기술 |
|---|---|
| 객체 탐지 | YOLOv8 (Ultralytics), Roboflow (데이터셋 관리/라벨링) |
| 서버 | Python, Flask, OpenCV |
| 통신 | TCP Socket, pickle, struct |
| 학습 환경 | Google Colab (T4 GPU) |
| 프론트엔드 | HTML/CSS/JavaScript (Flask 템플릿 내 인라인) |

## 프로젝트 구조
```
.
├── convinence_store.py          # 서버: 소켓 수신 + YOLO 추론 + Flask 웹 서버
├── convinence_store_client.py   # 클라이언트: 웹캠 캡처 + 서버 전송
└── convinence_store_2.ipynb     # 모델 학습 노트북 (Colab, Roboflow 데이터셋 + YOLOv8 학습)
```

## 설치 및 실행

### 요구 사항
- Python 3.8+
- CUDA 지원 GPU (서버 측)
- 서버·클라이언트가 동일 네트워크에서 통신 가능해야 함

### 1. 서버 (PC)
```bash
pip install ultralytics flask opencv-python
```
`convinence_store.py`에서 아래 항목을 환경에 맞게 수정합니다.
- `model = YOLO(r"본인 학습 결과 best.pt 경로")`
- 필요 시 `CLASS_NAMES` 순서를 `data.yaml`의 클래스 순서와 일치시키기

```bash
python convinence_store.py
```
서버 실행 후 `http://<서버IP>:5000` 접속 시 대시보드 확인 가능

### 2. 클라이언트 (카메라 장치)
```bash
pip install opencv-python
```
`convinence_store_client.py`에서 서버 IP를 수정합니다.
```python
PC_IP = "서버의 실제 IP"
```
```bash
python convinence_store_client.py
```

## 모델 학습
`convinence_store_2.ipynb`는 Google Colab 환경에서 실행하도록 작성되어 있습니다.
1. Roboflow에서 데이터셋 다운로드 (`workspace: graduationproject-1c9dr`, `project: convinence-store-ynkbv`)
2. YOLOv8n 기반 전이학습 (`epochs=100`, `imgsz=512`)
3. 검증 및 confusion matrix 확인

> ⚠️ 노트북에 Roboflow API 키가 하드코딩되어 있습니다. 공개 저장소에 올리기 전 반드시 키를 제거하고 환경 변수나 `.env` 파일로 분리하세요.

## 실행 화면
_(웹 대시보드 스크린샷 또는 시연 GIF를 여기에 추가하세요)_

## 향후 개선 사항
- 다중 클라이언트(여러 매대 카메라) 동시 지원
- 재고 부족 시 알림(Slack/문자) 연동
- 탐지 로그 DB 저장 및 재고 이력 통계
