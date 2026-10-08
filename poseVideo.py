from ultralytics import YOLO
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = YOLO('yolo11n-pose.pt').to(device)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 自定义15关键点映射
KEYPOINT_MAPPING = {
    0: 0,  # 头部中心（鼻子）
    1: None,  # 颈部
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
    14: None  # 肚脐
}

# 骨架连接配置（优化版）
SKELETON = [
    (0, 1),  # 头->颈
    (1, 2), (1, 5), (2, 5),  # 躯干上部
    (2, 3), (3, 4),  # 右臂
    (5, 6), (6, 7),  # 左臂
    (8, 9), (9, 10),  # 右腿
    (11, 12), (12, 13),  # 左腿
    (2, 8), (5, 11), (8, 11),  # 核心连接
    (1, 14), (8, 14), (11, 14)  # 肚脐连接
]

# 配置参数
CONFIG = {
    "text_scale": 1,
    "text_thickness": 2
}


def is_double_hand_on_hip_simple(keypoints):
    """双手叉腰检测"""
    # 检查必需的关键点是否存在
    required_points = [2, 5, 4, 7]  # 右肩、左肩、右手腕、左手腕
    if not all(point in keypoints for point in required_points):
        return False

    # 获取关键点坐标
    left_wrist = np.array(keypoints[7])
    right_wrist = np.array(keypoints[4])

    # 计算身体宽度（肩宽）
    body_width = abs(keypoints[2][0] - keypoints[5][0])

    # 计算身体高度（从肩膀到肚脐）
    if 14 in keypoints and 2 in keypoints:
        body_height = abs(keypoints[2][1] - keypoints[14][1])
    else:
        # 如果没有肚脐点，使用默认高度
        body_height = abs(keypoints[2][1] - keypoints[8][1]) * 0.6 if 8 in keypoints else 200

    # 计算手腕之间的距离和垂直差异
    wrist_dist = np.linalg.norm(left_wrist - right_wrist)
    y_diff = abs(left_wrist[1] - right_wrist[1])

    # 设置阈值
    wrist_dist_threshold = body_width * 1.5
    y_diff_threshold = body_height * 0.3

    # 判断是否叉腰
    return wrist_dist < wrist_dist_threshold and y_diff < y_diff_threshold


def put_chinese_text(img, text, position, font_size=24, color=(0, 255, 255)):
    """绘制中文文本"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_text_with_background(img, text, position, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """带背景的文本绘制"""
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                   CONFIG["text_scale"], CONFIG["text_thickness"])
    x, y = position
    cv2.rectangle(img, (x, y - text_height - 5),
                  (x + text_width, y + 5), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                CONFIG["text_scale"], text_color, CONFIG["text_thickness"])
    return img


def calculate_body_center(shoulders, hips):
    """智能计算肚脐位置（躯干中心偏下）"""
    # 计算躯干中心（肩部和胯部的加权平均）
    body_center_x = (shoulders[0] + hips[0]) / 2
    body_center_y = shoulders[1] * 0.4 + hips[1] * 0.6
    return body_center_x, body_center_y


def get_valid_keypoints(person_kpts, conf_threshold=0.5):
    """转换关键点并进行严格验证"""
    valid = {}

    # 先收集原始关键点（过滤低置信度）
    raw_points = {}
    for new_idx, orig_idx in KEYPOINT_MAPPING.items():
        if orig_idx is not None and orig_idx < len(person_kpts):
            if person_kpts[orig_idx][2] > conf_threshold:
                raw_points[new_idx] = person_kpts[orig_idx][:2]

    # 计算必需的关键点（颈部）
    if 2 in raw_points and 5 in raw_points:
        neck = (
            (raw_points[2][0] + raw_points[5][0]) / 2,
            (raw_points[2][1] + raw_points[5][1]) / 2
        )
        valid[1] = neck

    # 计算肚脐位置（更精确的方式）
    if 2 in raw_points and 5 in raw_points and 8 in raw_points and 11 in raw_points:
        # 取两肩中点
        shoulder_center = (
            (raw_points[2][0] + raw_points[5][0]) / 2,
            (raw_points[2][1] + raw_points[5][1]) / 2
        )
        # 取两胯中点
        hip_center = (
            (raw_points[8][0] + raw_points[11][0]) / 2,
            (raw_points[8][1] + raw_points[11][1]) / 2
        )
        # 肚脐位于肩胯之间偏下位置（约40%从肩膀开始）
        valid[14] = (
            shoulder_center[0] * 0.4 + hip_center[0] * 0.6,
            shoulder_center[1] * 0.4 + hip_center[1] * 0.6
        )

    # 合并其他已验证的关键点
    for k in raw_points:
        valid[k] = raw_points[k]

    return valid


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推理（提高置信度要求）
    results = model(frame, conf=0.7, verbose=False)

    hands_on_hip_detected = False

    # 绘制每个检测到的人体
    for person in results[0].keypoints.data.cpu().numpy():
        valid_kpts = get_valid_keypoints(person, conf_threshold=0.6)

        # 绘制关键点
        for kpt_id, (x, y) in valid_kpts.items():
            cv2.circle(frame, (int(x), int(y)), 8, (255, 255, 255), -1)

        # 安全绘制骨架
        for (start, end) in SKELETON:
            if start in valid_kpts and end in valid_kpts:
                pt1 = (int(valid_kpts[start][0]), int(valid_kpts[start][1]))
                pt2 = (int(valid_kpts[end][0]), int(valid_kpts[end][1]))
                # 最终确认不是(0,0)
                if all(p != (0, 0) for p in [pt1, pt2]):
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # 检测双手叉腰姿势
        if is_double_hand_on_hip_simple(valid_kpts):
            hands_on_hip_detected = True
            # 在手腕位置绘制提示
            if 4 in valid_kpts and 7 in valid_kpts:
                text_pos = (int((valid_kpts[4][0] + valid_kpts[7][0]) / 2),
                            int((valid_kpts[4][1] + valid_kpts[7][1]) / 2) - 40)
                frame = put_chinese_text(frame, "双手叉腰", text_pos)

    cv2.imshow('Accurate Pose Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
