import os
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm


def process_and_save_from_annotations(csv_path, input_base_path, output_base_path, split='train'):
    # 读取CSV注释文件
    df = pd.read_csv(csv_path)

    # 创建输出目录
    output_dir = os.path.join(output_base_path, split)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历CSV中的每一行
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        # 根据MELD的命名规则构造视频文件名
        # 构造视频文件名：diax_utty.mp4
        dialogue_id = row['Dialogue_ID']  # 直接使用数字，如 "0"
        utterance_id = row['Utterance_ID']  # 如 "0", "1", "2"
        file_name = f"dia{dialogue_id}_utt{utterance_id}.mp4"  # 如 "dia0_utt0.mp4"

        # 输入和输出路径
        input_file_path = os.path.join(input_base_path, file_name)
        output_file_path = os.path.join(output_dir, file_name.replace('.mp4', '.wav'))  # 输出为 .wav 文件

        # 跳过已存在的文件
        if os.path.exists(output_file_path):
            continue

        # 检查输入文件是否存在
        if not os.path.exists(input_file_path):
            print(f"Warning: {input_file_path} not found, skipping.")
            continue

        try:
            # 加载视频文件
            video = VideoFileClip(input_file_path)

            # # 根据StartTime和EndTime剪辑视频（可选）
            # start_time = parse_time(row['StartTime'])  # 需要自定义解析函数
            # end_time = parse_time(row['EndTime'])
            # video_clip = video.subclip(start_time, end_time)

            # 直接提取整个视频的音频（无需裁剪）
            audio = video.audio
            desired_sampling_rate = 16000
            resampled_audio = audio.set_fps(desired_sampling_rate)

            # 保存音频
            resampled_audio.write_audiofile(output_file_path, codec='pcm_s16le', verbose=False, logger=None)
        except Exception as e:
            print(f"Error processing {input_file_path}: {e}")


if __name__ == "__main__":
    # # 配置路径
    # csv_path = '/data/yuyangchen/data/MELD/train_sent_emo.csv'
    # input_base_path = '/data1/public_datasets/MELD/MELD.Raw/train_splits'
    # output_base_path = '/data/yuyangchen/data/MELD/train_A'

    # # 处理train分割
    # if os.path.exists(csv_path):
    #     process_and_save_from_annotations(csv_path, input_base_path, output_base_path, split='train')
    # else:
    #     print(f"CSV file {csv_path} not found, aborting.")

    # 配置路径
    csv_path = '/data/yuyangchen/data/MELD/test_sent_emo.csv'
    input_base_path = '/data1/public_datasets/MELD/MELD.Raw/test_splits'
    output_base_path = '/data/yuyangchen/data/MELD/test_A'

    # 处理test分割
    if os.path.exists(csv_path):
        process_and_save_from_annotations(csv_path, input_base_path, output_base_path, split='test')
    else:
        print(f"CSV file {csv_path} not found, aborting.")
