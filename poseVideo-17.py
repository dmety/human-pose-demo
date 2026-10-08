from ultralytics import YOLO
import cv2
import torch

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = YOLO('yolo11n-pose.pt').to(device)

# 打开摄像头
cap = cv2.VideoCapture(0)

# COCO格式的17个关键点骨架连接配置
SKELETON = [
    (0, 1), (0, 2),  # 鼻子->左右眼
    (1, 3), (2, 4),  # 眼->耳
    (0, 5), (0, 6),  # 鼻子->左右肩
    (5, 7), (7, 9),  # 右臂
    (6, 8), (8, 10),  # 左臂
    (5, 6),  # 肩部连接
    (5, 11), (6, 12),  # 肩->胯
    (11, 12),  # 胯部连接
    (11, 13), (13, 15),  # 右腿
    (12, 14), (14, 16)   # 左腿
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推理（提高置信度要求）
    results = model(frame, conf=0.7, verbose=False)

    # 绘制每个检测到的人体
    for person in results[0].keypoints.data.cpu().numpy():
        # 绘制关键点
        for kpt in person:
            x, y, conf = kpt
            if conf > 0.5:  # 只绘制置信度足够高的关键点
                cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)

        # 绘制骨架
        for start, end in SKELETON:
            # 检查两个点是否存在且置信度足够
            if start < len(person) and end < len(person):
                if person[start][2] > 0.5 and person[end][2] > 0.5:
                    pt1 = (int(person[start][0]), int(person[start][1]))
                    pt2 = (int(person[end][0]), int(person[end][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    cv2.imshow('Standard 17 Keypoints Pose Estimation', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
