import cv2
import torch
from ultralytics import YOLO
import numpy as np

# 加载模型
model = YOLO('yolov8n-pose.pt').to('cuda' if torch.cuda.is_available() else 'cpu')

# 读取图片
image_path = "img/1.jpg"  # 图片路径

# 保存结果
output_path = "output/output.jpg"
# 配置参数
CONFIG = {
    "keypoint_radius": 32,  # 关键点大小
    "keypoint_color": (0, 0, 255),  # 红色(BGR)
    "skeleton_color": (0, 255, 0),  # 绿色(BGR)
    "skeleton_thickness": 26,  # 骨架线宽
    "conf_threshold": 0.6  # 关键点置信度阈值
}

# 15关键点定义
KEYPOINT_MAPPING = {
    0: 0,  # 头部中心 -> 鼻子
    1: None,  # 颈部(需计算)
    2: 6,  # 右肩
    3: 8,  # 右手肘
    4: 10,  # 右手腕
    5: 5,  # 左肩
    6: 7,  # 左手肘
    7: 9,  # 左手腕
    8: 12,  # 右胯
    9: 14,  # 右膝
    10: 16,  # 右脚踝
    11: 11,  # 左胯
    12: 13,  # 左膝
    13: 15,  # 左脚踝
    14: None  # 肚脐(需计算)
}

# 骨架连接关系
SKELETON = [
    (0, 1),  # 头-颈
    (1, 2), (1, 5),  # 颈-肩
    (2, 5),  # 肩-肩
    (2, 3), (3, 4),  # 右臂
    (5, 6), (6, 7),  # 左臂
    (8, 9), (9, 10),  # 右腿
    (11, 12), (12, 13),  # 左腿
    (2, 8), (5, 11),  # 躯干连接
    (8, 11),  # 胯部连接
    (1, 14), (8, 14), (11, 14)  # 肚脐连接
]


def calculate_body_points(person_kpts):
    """计算颈部/肚脐等衍生关键点"""
    calculated = {}

    # 计算颈部(两肩中点)
    if 2 in person_kpts and 5 in person_kpts:
        right_shoulder = person_kpts[2]
        left_shoulder = person_kpts[5]
        calculated[1] = (
            (right_shoulder[0] + left_shoulder[0]) / 2,
            (right_shoulder[1] + left_shoulder[1]) / 2
        )

    # 计算肚脐(肩胯中点偏下)
    if all(k in person_kpts for k in [2, 5, 8, 11]):
        shoulder_center = (
            (person_kpts[2][0] + person_kpts[5][0]) / 2,
            (person_kpts[2][1] + person_kpts[5][1]) / 2
        )
        hip_center = (
            (person_kpts[8][0] + person_kpts[11][0]) / 2,
            (person_kpts[8][1] + person_kpts[11][1]) / 2
        )
        calculated[14] = (
            0.5 * shoulder_center[0] + 0.5 * hip_center[0],
            0.6 * shoulder_center[1] + 0.4 * hip_center[1]  # 更靠近胯部
        )

    return calculated


def draw_pose(image, keypoints):
    """在图像上绘制关键点和骨架"""
    # 绘制骨架
    for start, end in SKELETON:
        if start in keypoints and end in keypoints:
            pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
            pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
            cv2.line(image, pt1, pt2,
                     CONFIG["skeleton_color"],
                     CONFIG["skeleton_thickness"])

    # 绘制关键点
    for kpt_id, (x, y) in keypoints.items():
        cv2.circle(image, (int(x), int(y)),
                   CONFIG["keypoint_radius"],
                   CONFIG["keypoint_color"], -1)

        # 显示关键点ID（调试用，正式使用可注释掉）
        cv2.putText(image, str(kpt_id), (int(x) + 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)



    return image



image = cv2.imread(image_path)
if image is None:
    print(f"错误：无法加载图片 {image_path}")
    exit()

# 执行推理
results = model(image, conf=0.7, verbose=False)

# 处理每个人体
for person in results[0].keypoints.data.cpu().numpy():
    # 转换关键点
    raw_points = {}
    for new_idx, orig_idx in KEYPOINT_MAPPING.items():
        if orig_idx is not None and orig_idx < len(person):
            if person[orig_idx][2] > CONFIG["conf_threshold"]:
                raw_points[new_idx] = person[orig_idx][:2]

    # 计算衍生关键点
    calculated_points = calculate_body_points(raw_points)

    # 合并关键点
    all_points = {**raw_points, **calculated_points}

    # 绘制姿态
    image = draw_pose(image, all_points)

# 显示结果
# cv2.imshow("Pose Detection", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


cv2.imwrite(output_path, image)
print(f"结果已保存到: {output_path}")
