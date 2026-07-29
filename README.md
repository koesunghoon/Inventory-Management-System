# 무인매장 재고 감지 시스템 (AI 기반 실시간 재고 모니터링)

Jetson(또는 웹캠 클라이언트)에서 촬영한 영상을 PC 서버로 전송하고, YOLOv8 객체 탐지 모델로 상품(콜라·껌·과자) 재고 수량을 실시간으로 파악해 웹 대시보드에 표시하는 시스템입니다.

## 목차
- [개요](#개요)
- [팀 구성](#팀-구성)
- [시스템 구조](#시스템-구조)
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 실행](#설치-및-실행)
- [모델 학습](#모델-학습)
- [성능 및 제약사항](#성능-및-제약사항)
- [트러블슈팅](#트러블슈팅)
- [실행 화면](#실행-화면)
- [향후 개선 사항](#향후-개선-사항)

## 개요
매대에 진열된 상품을 카메라로 실시간 모니터링하여 재고가 부족하거나 소진되었을 때 자동으로 감지하는 프로젝트입니다. 클라이언트(카메라 장치)가 영상을 소켓 통신으로 서버에 전송하면, 서버는 YOLOv8 모델로 객체를 탐지하고 결과를 Flask 웹 페이지에 실시간 스트리밍 및 재고 상태로 표시합니다.

## 팀 구성
| 이름 | 역할 |
|---|---|
| 김재민 | Roboflow 라벨링, 데이터(영상) 촬영 (공동), 영상 편집 |
| 고성훈 | Roboflow 라벨링, 데이터(영상) 촬영 (공동), 그 외 서버·클라이언트 개발 및 모델 학습 등 |

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

> ⚠️ **보안 참고**: `pickle`은 신뢰할 수 없는 데이터를 역직렬화할 경우 임의 코드 실행으로 이어질 수 있습니다. 사내망/폐쇄망 데모용으로는 무방하지만, 외부에 노출되는 환경이라면 `pickle` 대신 JPEG 바이트를 직접 전송하거나 `msgpack` 등으로 대체하는 것을 권장합니다.

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
├── convinence_store_2.ipynb     # 모델 학습 노트북 (Colab, Roboflow 데이터셋 + YOLOv8 학습)
└── requirements.txt              # 의존성 목록
```

## 설치 및 실행

### 요구 사항
- Python 3.8+
- CUDA 지원 GPU (서버 측)
- 서버·클라이언트가 동일 네트워크에서 통신 가능해야 함 (같은 공유기/서브넷 권장)

### 0. requirements.txt (신규)
```
ultralytics
flask
opencv-python
```

### 1. 서버 (PC)
```bash
pip install -r requirements.txt
```
`convinence_store.py`에서 아래 항목을 환경에 맞게 수정합니다.
- `model = YOLO(r"본인 학습 결과 best.pt 경로")`
- 필요 시 `CLASS_NAMES` 순서를 `data.yaml`의 클래스 순서와 반드시 일치시키기 (순서가 다르면 라벨이 뒤바뀌어 표시됩니다)

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

## 성능 및 제약사항
`convinence_store_2.ipynb`의 `model.val()` 결과(Colab T4 GPU 기준)입니다.

| 항목 | 값 |
|---|---|
| mAP50 (전체) | 0.995 |
| mAP50-95 (전체) | 0.954 |
| Precision / Recall | 1.0 / 1.0 |
| 클래스별 mAP50-95 | cola 0.981, gum 0.889, snack 0.992 |
| 모델 추론 속도 | 전처리 3.3ms + 추론 4.2ms + 후처리 2.1ms ≈ 이미지당 9.6ms (T4 GPU 기준, 약 104 FPS) |
| 모델 | YOLOv8n, 파라미터 약 300만 개, 8.1 GFLOPs |
| 학습 시간 | 100 epoch, 약 0.176시간(약 10.6분, T4 GPU) |
| 학습 데이터셋 규모 | Train 427 / Valid 122 / Test 61 (총 610장) |
| 알려진 한계 | 조명 변화, 상품 겹침, 각도에 따른 오탐 등 |

> ⚠️ 위 추론 속도는 Colab T4 GPU에서 모델 자체만 검증한 수치입니다. 실제 서비스 파이프라인(소켓 통신 + Flask 서빙 포함)의 종단간(end-to-end) FPS/지연시간은 별도로 측정되지 않았습니다.

## 트러블슈팅
개발 중 겪었던 문제와 해결 과정입니다. **실제로 겪으신 이슈로 바꿔 채워주세요** — 아래는 이 스택(소켓+pickle+YOLOv8+Flask)에서 흔히 발생하는 항목들로 예시를 잡아둔 템플릿입니다.

| 증상 | 원인 | 해결 |
|---|---|---|
| 클라이언트-서버 소켓 연결 실패 | 서버 IP를 잘못 입력했거나, 방화벽이 해당 포트를 차단 | 서버 `ipconfig`/`ifconfig`로 실제 IP 확인, 방화벽 포트 개방 또는 테스트 시 임시로 방화벽 해제 |
| 탐지 결과 클래스명이 실제 상품과 다르게 표시됨 | `CLASS_NAMES` 리스트 순서와 학습 시 `data.yaml`의 클래스 순서가 불일치 | 학습 노트북에서 클래스 순서를 export 하거나, `best.pt`의 `model.names`를 그대로 사용하도록 코드 수정 |
| 서버에서 GPU를 인식하지 못하고 CPU로 추론되어 매우 느림 | CUDA/torch 버전 불일치, 또는 GPU 드라이버 미설치 | `torch.cuda.is_available()`로 확인 후 CUDA 버전에 맞는 PyTorch 재설치 |
| 웹캠 프레임이 점점 밀리며(delay) 지연이 누적됨 | 소켓 송신 속도가 서버 추론 속도보다 빨라 버퍼에 프레임이 쌓임 | 클라이언트에서 프레임을 일정 주기로만 전송(sleep 추가)하거나, 최신 프레임만 유지하고 오래된 프레임은 버리는 큐 구조로 변경 |
| Flask 대시보드가 새로고침 시 간헐적으로 멈춤/깨짐 | MJPEG 스트림과 REST API가 동일 스레드에서 처리되어 블로킹 발생 | Flask를 멀티스레드 모드(`threaded=True`)로 실행하거나 스트리밍/추론 로직을 별도 스레드로 분리 |
| Roboflow 데이터셋 다운로드 시 인증 오류 | API 키 만료 또는 워크스페이스 권한 문제 | Roboflow 대시보드에서 새 API 키 발급 후 `.env`로 재설정 |

## 실행 화면
![실행 화면](./assets/최종영상.gif)

## 향후 개선 사항
- 다중 클라이언트(여러 매대 카메라) 동시 지원
- 재고 부족 시 알림(Slack/문자) 연동
- 탐지 로그 DB 저장 및 재고 이력 통계
