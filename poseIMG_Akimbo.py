import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 加载模型
model = YOLO('yolo11n-pose.pt').to('cuda' if torch.cuda.is_available() else 'cpu')

# 读取图片
image_path = "img/1.jpg"  # 图片路径
output_path = "output/1.jpg"  # 输出路径

# 配置参数
CONFIG = {
    "keypoint_radius": 32,          # 关键点半径
    "keypoint_color": (255, 255, 255),  # 红色关键点
    "skeleton_color": (0, 255, 0),  # 绿色骨架
    "skeleton_thickness": 26,       # 加大骨架粗细
    "conf_threshold": 0.6,          # 提高置信度阈值
    "text_scale": 1.5,              # 加大文本尺寸
    "text_color": (255, 255, 255),  # 白色文本
    "text_thickness": 3,            # 加粗文本
    "text_bg_color": (0, 0, 0),     # 黑色背景
    "text_offset_y": -50,           # 调整文本垂直位置
    "hand_circle_size": 20,         # 加大手圈半径
    "hand_circle_thickness": 4      # 加大手圈粗细
}

# 关键点定义
KEYPOINT_NAMES = {
    0: "鼻子",
    1: "颈部",
    2: "右肩",
    3: "右肘",
    4: "右腕",
    5: "左肩",
    6: "左肘",
    7: "左腕",
    8: "右胯",
    9: "右膝",
    10: "右脚踝",
    11: "左胯",
    12: "左膝",
    13: "左脚踝",
    14: "肚脐"
}

KEYPOINT_MAPPING = {
    0: 0,  # 鼻子
    1: None,  # 颈部(计算得出)
    2: 6,  # 右肩
    3: 8,  # 右肘
    4: 10,  # 右腕
    5: 5,  # 左肩
    6: 7,  # 左肘
    7: 9,  # 左腕
    8: 12,  # 右胯
    9: 14,  # 右膝
    10: 16,  # 右脚踝
    11: 11,  # 左胯
    12: 13,  # 左膝
    13: 15,  # 左脚踝
    14: None  # 肚脐(计算得出)
}

# 骨架连接
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
        calculated[1] = (
            (person_kpts[2][0] + person_kpts[5][0]) / 2,
            (person_kpts[2][1] + person_kpts[5][1]) / 2
        )
    # 计算肚脐(躯干中心偏下)
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
            0.6 * shoulder_center[1] + 0.4 * hip_center[1]
        )
    return calculated


def is_double_hand_on_hip_simple(keypoints):
    """简化版双手叉腰检测"""
    if 4 not in keypoints or 7 not in keypoints or 2 not in keypoints or 5 not in keypoints:
        return False

    left_wrist = np.array(keypoints[7])
    right_wrist = np.array(keypoints[4])
    body_width = abs(keypoints[2][0] - keypoints[5][0])
    body_height = abs(keypoints[2][1] - keypoints[8][1]) if 8 in keypoints else 1

    wrist_dist = np.linalg.norm(left_wrist - right_wrist)
    y_diff = abs(left_wrist[1] - right_wrist[1])

    wrist_dist_threshold = max(body_width * 1.8, 200)
    y_diff_threshold = body_height * 0.3

    return wrist_dist < wrist_dist_threshold and y_diff < y_diff_threshold


def put_chinese_text(img, text, position, font_size=48, color=(0, 255, 255)):
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


def draw_pose(image, keypoints):
    """绘制姿态和坐标"""
    # 绘制骨架
    for start, end in SKELETON:
        if start in keypoints and end in keypoints:
            pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
            pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
            cv2.line(image, pt1, pt2, CONFIG["skeleton_color"], CONFIG["skeleton_thickness"])

    # 绘制关键点和坐标
    for kpt_id, (x, y) in keypoints.items():
        # 绘制关键点
        cv2.circle(image, (int(x), int(y)), CONFIG["keypoint_radius"], CONFIG["keypoint_color"], -1)

        # 准备坐标文本
        text = f"({int(x)},{int(y)})"
        text_pos = (int(x) - 50, int(y) + CONFIG["text_offset_y"])

        # 绘制带背景的文本
        image = draw_text_with_background(
            image, text, text_pos,
            CONFIG["text_color"],
            CONFIG["text_bg_color"]
        )

    # 突出显示手腕
    if 4 in keypoints:
        cv2.circle(image, (int(keypoints[4][0]), int(keypoints[4][1])), 20, (255, 0, 0), 4)
    if 7 in keypoints:
        cv2.circle(image, (int(keypoints[7][0]), int(keypoints[7][1])), 20, (0, 0, 255), 4)

    # 检测双手叉腰
    if is_double_hand_on_hip_simple(keypoints):
        image = put_chinese_text(image, "双手叉腰", (50, 50), font_size=360, color=(0, 255, 255))

    return image


# 主流程
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

# 保存结果
cv2.imwrite(output_path, image)
print(f"结果已保存到: {output_path}")

# 可选：显示结果
# cv2.imshow('Pose Estimation', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
