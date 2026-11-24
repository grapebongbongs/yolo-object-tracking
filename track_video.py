"""
YOLOv11 객체 감지 + ByteTrack 추적 + 이동 궤적 그리기

- 학습된 모델(best.pt)로 동영상의 앞부분을 처리하여
  경계상자, 부류, 신뢰도, 추적 ID, 이동 궤적을 그린 영상(result.mp4)과
  프레임별 좌표 기록(trajectory.csv)을 만든다.
- 실행 예: python track_video.py --model best.pt --video my_video.mp4
"""
import argparse
import csv
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 감지 + 추적 + 이동 궤적")
    parser.add_argument("--model", default="best.pt", help="학습된 모델 파일")
    parser.add_argument("--video", required=True, help="입력 동영상 파일")
    parser.add_argument("--output", default="result.mp4", help="결과 동영상 파일")
    parser.add_argument("--csv", default="trajectory.csv", help="궤적 기록 파일")
    parser.add_argument("--seconds", type=float, default=5, help="처리할 길이(초)")
    parser.add_argument("--conf", type=float, default=0.5, help="신뢰도 임계값")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"동영상을 열 수 없습니다: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * args.seconds)

    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tracks = defaultdict(list)   # 추적 ID별 중심점 기록
    rows = []
    frame_idx = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # 감지 + ByteTrack 추적 (persist=True: 프레임 간 ID 유지)
        r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                        conf=args.conf, verbose=False)[0]
        img = r.plot()   # 경계상자, 부류, 신뢰도, ID 표시

        if r.boxes.id is not None:
            boxes = r.boxes.xywh.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            for (cx, cy, bw, bh), tid, c in zip(boxes, ids, clss):
                tracks[tid].append((int(cx), int(cy)))
                rows.append([frame_idx, round(frame_idx / fps, 3), tid,
                             model.names[c], int(cx), int(cy)])
                # 이동 궤적 그리기
                pts = np.array(tracks[tid], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 255), thickness=3)

        out.write(img)
        frame_idx += 1

    cap.release()
    out.release()

    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_sec", "track_id", "class", "center_x", "center_y"])
        writer.writerows(rows)

    print(f"완료: {frame_idx}프레임 ({frame_idx / fps:.1f}초), 추적된 물체 수 {len(tracks)}")
    print(f"결과 영상: {args.output}, 궤적 기록: {args.csv}")


if __name__ == "__main__":
    main()
