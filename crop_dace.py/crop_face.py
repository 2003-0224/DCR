# import os
#
# os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
# import cv2
# import numpy as np
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import normalize
# from tqdm import tqdm
# from insightface.app import FaceAnalysis
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import dlib
# import hopenet
# from collections import defaultdict
# import torchvision
#
# # 忽略警告
# import warnings
#
# warnings.filterwarnings("ignore", category=FutureWarning)
#
# # 检测 GPU 可用性并设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
#
# # 初始化 Hopenet 模型
# def load_hopenet_model(model_path="/data/yuyangchen/pre_model/MERC/NewWay/hopenet_robust_alpha1.pkl"):
#     model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=66)
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint, strict=False)
#     model = model.to(device)  # 移动模型到 GPU
#     model.eval()
#     return model
#
#
# # 初始化模型和工具
# try:
#     hopenet_model = load_hopenet_model()
# except Exception as e:
#     print(f"Failed to load Hopenet model: {e}")
#     exit(1)
#
# predictor = dlib.shape_predictor("/data/yuyangchen/pre_model/MERC/NewWay/shape_predictor_68_face_landmarks.dat")
# face_analyzer = FaceAnalysis(name='buffalo_l')
# face_analyzer.prepare(ctx_id=0)  # 使用 GPU (ctx_id=0 表示第一个 GPU)
# idx_tensor = torch.FloatTensor(list(range(66))).to(device)  # 移动 idx_tensor 到 GPU
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
#
# def predict_head_pose(model, image):
#     try:
#         img = transform(image).unsqueeze(0).to(device)  # 移动输入到 GPU
#         with torch.no_grad():
#             yaw, pitch, roll = model(img)
#             yaw = torch.sum(F.softmax(yaw, dim=1) * idx_tensor) * 3 - 99
#             pitch = torch.sum(F.softmax(pitch, dim=1) * idx_tensor) * 3 - 99
#             roll = torch.sum(F.softmax(roll, dim=1) * idx_tensor) * 3 - 99
#         return yaw.item(), pitch.item(), roll.item()
#     except Exception as e:
#         print(f"Error in head pose prediction: {e}")
#         return 0, 0, 0
#
#
# def align_face(frame, face_box):
#     x, y, w, h = face_box
#     x, y = max(0, x), max(0, y)
#     return frame[y:y + h, x:x + w]
#
#
# def is_frontal_face(image):
#     try:
#         yaw, pitch, roll = predict_head_pose(hopenet_model, image)
#         max_angle = max(abs(yaw), abs(pitch), abs(roll))
#         return max_angle < 50
#     except:
#         return False
#
#
# def calculate_lip_movement(aligned_face, face_box):
#     try:
#         rect = dlib.rectangle(left=0, top=0, right=face_box[2], bottom=face_box[3])
#         landmarks = predictor(aligned_face, rect)
#         lip_points = [landmarks.part(i) for i in range(48, 68)]
#         lip_height = abs(lip_points[3].y - lip_points[9].y)
#         return lip_height
#     except:
#         return 0
#
#
# def smooth_sequence(sequence, window_size=3):
#     if len(sequence) < window_size:
#         return sequence
#     return np.convolve(sequence, np.ones(window_size) / window_size, mode='valid')
#
#
# def detect_faces_with_retinaface(frame):
#     try:
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb_frame)
#         detected_faces = []
#         for face in faces:
#             x, y, w, h = face.bbox[0], face.bbox[1], face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]
#             if h / w < 1.7 and h > 100:
#                 detected_faces.append({
#                     'box': [int(x), int(y), int(w), int(h)],
#                     'encoding': face.normed_embedding
#                 })
#         return detected_faces
#     except Exception as e:
#         print(f"Error in face detection: {e}")
#         return []
#
#
# def merge_clusters_by_temporal_continuity(labels, frame_nums, encodings, merge_threshold=1.05):
#     new_labels = labels.copy()
#     unique_labels = sorted(set(labels) - {-1})
#     for i in range(len(unique_labels)):
#         for j in range(i + 1, len(unique_labels)):
#             label_i, label_j = unique_labels[i], unique_labels[j]
#             frames_i = [frame_nums[k] for k in range(len(labels)) if labels[k] == label_i]
#             frames_j = [frame_nums[k] for k in range(len(labels)) if labels[k] == label_j]
#             min_frame_diff = min([abs(fi - fj) for fi in frames_i for fj in frames_j], default=float('inf'))
#             if min_frame_diff <= 20:
#                 encoding_i = np.mean([encodings[k] for k in range(len(labels)) if labels[k] == label_i], axis=0)
#                 encoding_j = np.mean([encodings[k] for k in range(len(labels)) if labels[k] == label_j], axis=0)
#                 dist = np.linalg.norm(encoding_i - encoding_j)
#                 if dist < merge_threshold:
#                     new_labels[labels == label_j] = label_i
#                     print(f"Merged cluster {label_j} into {label_i}, distance={dist:.4f}")
#     unique_new_labels = sorted(set(new_labels) - {-1})
#     label_map = {old: new for new, old in enumerate(unique_new_labels)}
#     final_labels = new_labels.copy()
#     for i in range(len(final_labels)):
#         if final_labels[i] != -1:
#             final_labels[i] = label_map[final_labels[i]]
#     return final_labels
#
#
# def process_video(video_path, output_dir):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"无法打开视频文件: {video_path}")
#         return
#
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     processed_frames = (total_frames + 1) // 2
#     frame_count = 0
#     all_face_encodings = []
#     all_face_images = []
#     all_face_boxes = []
#     all_lip_movements = []
#     all_faces_data = []
#
#     with tqdm(total=processed_frames, desc=f"处理 {video_path}（第一次遍历）") as pbar:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_count += 1
#             if frame_count % 2 == 0:
#                 continue
#             faces = detect_faces_with_retinaface(frame)
#             for face in faces:
#                 x, y, w, h = face['box']
#                 aligned_face = align_face(frame, (x, y, w, h))
#                 is_frontal = is_frontal_face(aligned_face)
#                 all_faces_data.append({
#                     'aligned_face': aligned_face,
#                     'box': (x, y, w, h),
#                     'encoding': face['encoding'],
#                     'is_frontal': is_frontal,
#                     'frame_num': frame_count
#                 })
#             pbar.update(1)
#
#     cap.release()
#
#     if len(all_faces_data) == 0:
#         print(f"视频 {video_path} 未检测到人脸")
#         return
#
#     # 筛选正脸或保留所有面部数据
#     frontal_faces_count = sum(1 for face in all_faces_data if face['is_frontal'])
#     selected_faces = [face for face in all_faces_data if
#                       face['is_frontal']] if frontal_faces_count >= 8 else all_faces_data
#
#     for face in selected_faces:
#         all_face_encodings.append(face['encoding'])
#         all_face_images.append(face['aligned_face'])
#         all_face_boxes.append((face['frame_num'], *face['box']))
#         lip_movement = calculate_lip_movement(face['aligned_face'], (0, 0, face['box'][2], face['box'][3]))
#         all_lip_movements.append(lip_movement)
#
#     if len(all_face_encodings) == 0:
#         print(f"视频 {video_path} 未检测到足够人脸")
#         return
#
#     # 归一化特征并聚类
#     all_face_encodings = normalize(np.array(all_face_encodings), axis=1, norm='l2')
#     clustering = DBSCAN(eps=0.95, min_samples=2, metric="euclidean").fit(all_face_encodings)
#     labels = clustering.labels_
#     frame_nums = [face['frame_num'] for face in selected_faces]
#     labels = merge_clusters_by_temporal_continuity(labels, frame_nums, all_face_encodings)
#
#     # 说话者识别
#     unique_labels = set(labels)
#     label_lip_vars = {}
#     for label in unique_labels:
#         if label != -1:
#             lip_values = [all_lip_movements[i] for i, lbl in enumerate(labels) if lbl == label]
#             if len(lip_values) > 1:
#                 smoothed_lip_values = smooth_sequence(np.array(lip_values))
#                 lip_var = np.var(smoothed_lip_values) if len(smoothed_lip_values) > 1 else 0
#                 label_lip_vars[label] = lip_var
#             else:
#                 label_lip_vars[label] = 0
#     speaker_label = max(label_lip_vars.items(), key=lambda x: x[1])[0] if label_lip_vars else None
#
#     # 保存结果
#     for label in unique_labels:
#         if label == -1:
#             person_dir = os.path.join(output_dir, "unknown")
#         elif label == speaker_label:
#             person_dir = os.path.join(output_dir, "speaker")
#         else:
#             person_dir = os.path.join(output_dir, f"people_{label}")
#         os.makedirs(person_dir, exist_ok=True)
#
#         for i, lbl in enumerate(labels):
#             if lbl == label:
#                 frame_num = all_face_boxes[i][0]  # 帧号
#                 cropped_face = all_face_images[i]
#                 # 处理帧号重复（同一帧多个人脸）
#                 base_filename = f"frame{frame_num}.jpg"
#                 filename = base_filename
#                 suffix = 0
#                 while os.path.exists(os.path.join(person_dir, filename)):
#                     suffix += 1
#                     filename = f"frame{frame_num}_{suffix}.jpg"
#                 face_filename = os.path.join(person_dir, filename)
#                 cv2.imwrite(face_filename, cropped_face)
#
#
# def batch_process_videos(video_dir, output_base_dir):
#     # 获取视频文件列表，只保留以 'dia' 开头、以 .mp4 结尾且符合条件的文件
#     video_files = [f for f in os.listdir(video_dir)
#                    if os.path.isfile(os.path.join(video_dir, f))  # 确保是文件
#                    and not f.startswith('.')  # 排除隐藏文件
#                    and f.startswith('dia')  # 只保留以 'dia' 开头的文件
#                    and f.endswith('.mp4')]  # 只选择 .mp4 文件
#
#     # 调试：打印过滤后的文件列表
#     print("Video files to process:", video_files)
#
#     if not video_files:
#         print(f"No valid video files found in {video_dir}")
#         return
#
#     # 自定义排序函数，按 diaX_uttY 的数字顺序
#     def sort_key(filename):
#         parts = filename.replace('.mp4', '').split('_')
#         dia_num = int(parts[0].replace('dia', ''))
#         utt_num = int(parts[1].replace('utt', ''))
#         return (dia_num, utt_num)
#
#     video_files.sort(key=sort_key)
#     for video_file in video_files:
#         # 解析文件名：diaX_uttY.mp4
#         try:
#             dia_id, utt_id = map(lambda x: x.replace('dia', '').replace('utt', '').split('.')[0],
#                                  video_file.split('_')[:2])
#         except:
#             print(f"Invalid video filename format: {video_file}")
#             continue
#         # 创建输出目录 ./processed/X/Y/
#         output_dir = os.path.join(output_base_dir, dia_id, utt_id)
#         video_path = os.path.join(video_dir, video_file)
#         print(f"Processing video: {video_path}")
#         process_video(video_path, output_dir)
#
#
# if __name__ == "__main__":
#     # video_dir = "E:\\Data\\Mul-DED\\MELD\\MELD.Raw\\train_splits"  # 存放 diaX_uttY.mp4 的目录
#     video_dir = "/data1/public_datasets/MELD/MELD.Raw/test_splits"
#     # output_base_dir = "./processed"
#     output_base_dir = '/data/yuyangchen/data/MELD_face/test'
#     os.makedirs(output_base_dir, exist_ok=True)
#     batch_process_videos(video_dir, output_base_dir)
#     print("Processing completed.")


import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from tqdm import tqdm
from insightface.app import FaceAnalysis
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import dlib
import hopenet
from collections import defaultdict
import torchvision
import pandas as pd

# 忽略警告
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# 检测 GPU 可用性并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 初始化 Hopenet 模型
def load_hopenet_model(model_path="/data/home/chenqian/MERC-RL-MOSI/crop_dace.py/hopenet_robust_alpha1.pkl"):
    """加载 Hopenet 模型用于头部姿态估计。"""
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=66)
    # 强制映射到当前设备
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    return model


# 初始化模型和工具
try:
    hopenet_model = load_hopenet_model(
        model_path="/data/home/chenqian/MERC-RL-MOSI/crop_dace.py/hopenet_robust_alpha1.pkl")
except Exception as e:
    print(f"Failed to load Hopenet model: {e}")
    hopenet_model = None

dlib_predictor_path = "/data/home/chenqian/MERC-RL-MOSI/crop_dace.py/shape_predictor_68_face_landmarks.dat"
try:
    predictor = dlib.shape_predictor(dlib_predictor_path)
except Exception as e:
    print(f"Failed to load dlib predictor: {e}")
    predictor = None

face_analyzer = FaceAnalysis(name='buffalo_l')
face_analyzer.prepare(ctx_id=0 if device.type == 'cuda' else -1)
idx_tensor = torch.FloatTensor(list(range(66))).to(device)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_head_pose(model, image):
    """预测头部姿态 (Yaw, Pitch, Roll)。"""
    if model is None: return 0, 0, 0
    try:
        img = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            yaw, pitch, roll = model(img)
            yaw = torch.sum(F.softmax(yaw, dim=1) * idx_tensor) * 3 - 99
            pitch = torch.sum(F.softmax(pitch, dim=1) * idx_tensor) * 3 - 99
            roll = torch.sum(F.softmax(roll, dim=1) * idx_tensor) * 3 - 99
        return yaw.item(), pitch.item(), roll.item()
    except Exception as e:
        # print(f"Head pose error: {e}")
        return 0, 0, 0


def align_face(frame, face_box):
    """根据边界框裁剪人脸区域。"""
    x, y, w, h = face_box
    x, y = max(0, x), max(0, y)
    h_frame, w_frame = frame.shape[:2]
    y_end = min(y + h, h_frame)
    x_end = min(x + w, w_frame)
    return frame[y:y_end, x:x_end]


def is_frontal_face(image):
    """检查人脸是否大致为正面 (姿态角小于 50 度)。"""
    if hopenet_model is None: return True
    try:
        yaw, pitch, roll = predict_head_pose(hopenet_model, image)
        max_angle = max(abs(yaw), abs(pitch), abs(roll))
        return max_angle < 50
    except:
        return False


def calculate_lip_movement(aligned_face, face_box):
    """计算嘴唇的垂直运动 (未使用，但保留函数签名)。"""
    if predictor is None: return 0
    try:
        rect = dlib.rectangle(left=0, top=0, right=face_box[2], bottom=face_box[3])
        gray_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        landmarks = predictor(gray_face, rect)
        lip_height = abs(landmarks.part(62).y - landmarks.part(66).y)
        return lip_height
    except:
        return 0


def detect_faces_with_retinaface(frame):
    """使用 InsightFace/RetinaFace 检测人脸并获取 ArcFace 编码。"""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(rgb_frame)
        detected_faces = []
        for face in faces:
            x, y, xmax, ymax = face.bbox
            w, h = xmax - x, ymax - y
            # 筛选条件：长宽比和最小高度
            if h / w < 1.7 and h > 100:
                detected_faces.append({
                    'box': [int(x), int(y), int(w), int(h)],
                    'encoding': face.normed_embedding
                })
        return detected_faces
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []


def merge_clusters_by_temporal_continuity(labels, frame_nums, encodings, merge_threshold=1.05):
    """基于时间和特征相似性合并 DBSCAN 聚类结果。"""
    new_labels = labels.copy()
    unique_labels = sorted(set(labels) - {-1})

    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            label_i, label_j = unique_labels[i], unique_labels[j]
            frames_i = [frame_nums[k] for k in range(len(labels)) if labels[k] == label_i]
            frames_j = [frame_nums[k] for k in range(len(labels)) if labels[k] == label_j]

            # 检查时间上的接近性
            min_frame_diff = min([abs(fi - fj) for fi in frames_i for fj in frames_j], default=float('inf'))

            if min_frame_diff <= 20:
                # 检查特征距离
                encoding_i = np.mean([encodings[k] for k in range(len(labels)) if labels[k] == label_i], axis=0)
                encoding_j = np.mean([encodings[k] for k in range(len(labels)) if labels[k] == label_j], axis=0)
                dist = np.linalg.norm(encoding_i - encoding_j)

                if dist < merge_threshold:
                    new_labels[new_labels == label_j] = label_i

    # 重新映射标签
    unique_new_labels = sorted(set(new_labels) - {-1})
    label_map = {old: new for new, old in enumerate(unique_new_labels)}
    final_labels = new_labels.copy()
    for i in range(len(final_labels)):
        if final_labels[i] != -1:
            final_labels[i] = label_map[final_labels[i]]

    return final_labels


def process_video(video_path, output_dir):
    """
    处理单个视频，执行人脸提取、聚类，并保存裁剪后的人脸图像到 output_dir。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = (total_frames + 1) // 2
    frame_count = 0
    all_face_encodings = []
    all_face_images = []
    all_face_boxes = []
    all_faces_data = []

    # 第一次遍历：检测、裁剪、姿态和编码
    with tqdm(total=processed_frames, desc=f"处理 {os.path.basename(video_path)}（第一次遍历）") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count % 2 == 0: continue  # 只处理奇数帧

            faces = detect_faces_with_retinaface(frame)
            for face in faces:
                x, y, w, h = face['box']
                aligned_face = align_face(frame, (x, y, w, h))
                is_frontal = is_frontal_face(aligned_face)

                if aligned_face.size == 0: continue

                all_faces_data.append({
                    'aligned_face': aligned_face,
                    'box': (x, y, w, h),
                    'encoding': face['encoding'],
                    'is_frontal': is_frontal,
                    'frame_num': frame_count
                })
            pbar.update(1)

    cap.release()

    if len(all_faces_data) == 0:
        print(f"视频 {video_path} 未检测到人脸")
        return

    # 筛选正脸或保留所有面部数据
    frontal_faces_count = sum(1 for face in all_faces_data if face['is_frontal'])
    # 如果正脸帧数足够（>8），则只使用正脸进行聚类
    selected_faces = [face for face in all_faces_data if
                      face['is_frontal']] if frontal_faces_count >= 8 else all_faces_data

    if not selected_faces and all_faces_data: selected_faces = all_faces_data

    for face in selected_faces:
        all_face_encodings.append(face['encoding'])
        all_face_images.append(face['aligned_face'])
        all_face_boxes.append((face['frame_num'], *face['box']))

    if len(all_face_encodings) == 0:
        print(f"视频 {video_path} 未检测到足够人脸进行聚类")
        return

    # 聚类
    all_face_encodings_np = normalize(np.array(all_face_encodings), axis=1, norm='l2')
    # DBSCAN 聚类: eps=0.95 是一个经验值
    clustering = DBSCAN(eps=0.95, min_samples=2, metric="euclidean").fit(all_face_encodings_np)
    labels = clustering.labels_

    frame_nums = [face['frame_num'] for face in selected_faces]
    labels = merge_clusters_by_temporal_continuity(labels, frame_nums, all_face_encodings_np)

    # ------------------- 独白场景说话者简化逻辑 -------------------
    unique_labels = set(labels)
    # 统计最大的聚类作为“主要人物”（即说话者）
    label_counts = {label: np.sum(labels == label) for label in unique_labels if label != -1}

    if not label_counts:
        print(f"Warning: No main person cluster found (only outliers). Skipping save for {video_path}.")
        return

    main_person_label = max(label_counts, key=label_counts.get)

    # 保存结果
    for label in unique_labels:
        # 主要人物保存到 'speaker' 目录
        if label == main_person_label:
            person_dir = os.path.join(output_dir, "speaker")
        # 异常值保存到 'unknown' 目录
        elif label == -1:
            person_dir = os.path.join(output_dir, "unknown")
        # 忽略其他小聚类
        else:
            continue

        os.makedirs(person_dir, exist_ok=True)

        for i, lbl in enumerate(labels):
            if lbl == label:
                frame_num = all_face_boxes[i][0]
                cropped_face = all_face_images[i]

                # 处理帧号重复 (同一帧多个人脸)
                base_filename = f"frame{frame_num}.jpg"
                filename = base_filename
                suffix = 0
                while os.path.exists(os.path.join(person_dir, filename)):
                    suffix += 1
                    filename = f"frame{frame_num}_{suffix}.jpg"
                face_filename = os.path.join(person_dir, filename)
                if cropped_face.size > 0:
                    cv2.imwrite(face_filename, cropped_face)


# --------------------------------------------------------------------------------------


def batch_process_videos(csv_path, output_base_dir):
    """
    读取 CSV 文件，其中第一列是视频文件路径，第三列是情感分数。
    并将所有输出统一存放在 output_base_dir 下，不再区分 train/test。
    输出路径结构: output_base_dir/segment_id/video_id/
    """
    try:
        # 假设 CSV 没有 header，且第一列是路径，第三列是分数
        df = pd.read_csv(csv_path, header=None)
    except Exception as e:
        print(f"Error loading or parsing CSV: {e}")
        return

    # 1. 使用 output_base_dir 作为统一的输出根目录
    base_output = output_base_dir
    os.makedirs(base_output, exist_ok=True)
    print(f"\n--- Starting processing for CSV: {csv_path} (Output root: {base_output}) ---")

    # 遍历 CSV 的每一行
    for index, row in df.iterrows():
        video_path = str(row[0]).strip()  # 第一列：视频文件路径
        sentiment_score = row[2] if len(row) > 2 else 'N/A'

        if not os.path.isfile(video_path):
            print(f"Warning: Video file not found at {video_path}. Skipping.")
            continue

        # --- 路径解析和构建 ---

        # 1. 提取视频文件名 (e.g., '24.mp4')
        video_filename = os.path.basename(video_path)
        # 2. 提取视频ID (e.g., '24')
        video_id = os.path.splitext(video_filename)[0]

        # 3. 提取上层目录名作为 Segment ID (e.g., 'c7UH_rxdZv4')
        segment_id = os.path.basename(os.path.dirname(video_path))

        if not segment_id:
             print(f"Error: Could not extract segment ID from path: {video_path}. Skipping.")
             continue

        # 4. 构建输出目录结构：./base_output/segment_id/video_id/
        # 例如: /data/home/chenqian/CMU-MOSI/MOSI_face/c7UH_rxdZv4/24/
        output_segment_dir = os.path.join(base_output, segment_id)
        output_dir = os.path.join(output_segment_dir, video_id)
        os.makedirs(output_dir, exist_ok=True)

        # ----------------------

        # 在视频目录下保存情感分数，作为元数据
        try:
            with open(os.path.join(output_dir, "sentiment.txt"), "w") as f:
                f.write(str(sentiment_score))
        except Exception as e:
            print(f"Error saving sentiment score: {e}")

        print(f"  -> Processing video: {video_path} (Output: {output_dir})")

        # 调用视频处理函数
        process_video(video_path, output_dir)


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # 请替换为你实际的 CSV 文件路径
    mosi_train_csv_path = "/data/home/chenqian/MERC-RL-MOSI/crop_dace.py/all_train_data.csv"  # <-- 训练集 CSV 路径
    mosi_test_csv_path = "/data/home/chenqian/MERC-RL-MOSI/crop_dace.py/all_test_data.csv"  # <-- 测试集 CSV 路径

    # 请替换为你希望保存结果的输出基础目录
    output_base_dir = '/data/home/chenqian/CMU_MOSI/MOSI_face/'
    # ----------------------------------------------------------------

    os.makedirs(output_base_dir, exist_ok=True)

    # 1. 处理训练集 CSV
    # 所有输出将统一放置在 output_base_dir 下，不再区分 train/test 子目录
    batch_process_videos(mosi_train_csv_path, output_base_dir)

    # 2. 处理测试集 CSV
    # 所有输出将统一放置在 output_base_dir 下，不再区分 train/test 子目录
    batch_process_videos(mosi_test_csv_path, output_base_dir)

    print("\n--- All Processing Completed! ---")
    