# 머신러닝 YOLO 학습 과제: 객체 감지 및 추적

직접 수집한 3개 부류(Lego car, Woowang, toy burger) 이미지로 YOLOv11 Small을 학습하고,
학습된 모델과 ByteTrack으로 동영상 속 물체를 감지·추적하며 이동 궤적을 그린 프로젝트입니다.

## 결과

| 모델 | mAP (테스트 세트) | 추론 지연 |
| --- | --- | --- |
| YOLOv11 Small | 100% | 31.9ms |

결과 영상: [result.mp4](result.mp4)

## 파일 구성

| 파일 | 내용 |
| --- | --- |
| `best.pt` | Roboflow에서 학습한 YOLOv11 Small 모델 |
| `track_video.py` | 감지 + 추적 + 이동 궤적 코드 |
| `yolo_tracking_final.ipynb` | 모델 다운로드부터 추적까지의 Colab 노트북 |
| `result.mp4` | 감지·추적 결과 5초 영상 |
| `trajectory.csv` | 프레임별 물체 ID·부류·중심 좌표 |

## 실행 방법

```bash
pip install -r requirements.txt
python track_video.py --model best.pt --video my_video.mp4
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--model` | `best.pt` | 학습된 모델 파일 |
| `--video` | (필수) | 입력 동영상 |
| `--seconds` | `5` | 처리할 길이(초) |
| `--conf` | `0.5` | 신뢰도 임계값 |

## 처리 과정

1. YOLOv11로 프레임마다 물체를 감지
2. ByteTrack으로 같은 물체에 같은 추적 ID를 부여
3. 물체별 중심점의 이동 경로를 선으로 그리고 좌표를 `trajectory.csv`로 저장
4. 결과를 `result.mp4`로 저장
