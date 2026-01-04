
import pandas as pd

# 1. 定义标签映射（根据图片中的映射关系）
emotion2id = {
    "neutral": 2,
    "anger": 3,
    "joy": 0,
    "sadness": 1,
    "fear": 6,
    "disgust": 5,
    "surprise": 4
}



# 2. 修改注释文件的函数
def clean_and_map_emotion_id(input_file, output_file):
    # 读取原始注释文件
    df = pd.read_csv(input_file)
    
    # 检查情感标签是否都在映射中
    unique_emotions = df["Emotion"].unique()
    for emotion in unique_emotions:
        if emotion not in emotion2id:
            raise ValueError(f"Emotion '{emotion}' not found in emotion2id mapping!")
    
    # 筛选需要的列
    keep_columns = ["Utterance", "Speaker", "Emotion", "Dialogue_ID", "Utterance_ID"]
    df = df[keep_columns]
    
    # 添加 emotion_id 列
    df["emotion_id"] = df["Emotion"].map(emotion2id)
    
    # 保存修改后的文件
    df.to_csv(output_file, index=False)
    print(f"Modified annotation file saved to {output_file}")
    
    # 打印类别分布
    print("\nEmotion ID distribution:")
    print(df["emotion_id"].value_counts().sort_index())

# 3. 主函数
def main():
    # # 读取原始标签文件
    # input_file = '/data/yuyangchen/data/MELD/train_sent_emo.csv'  # 替换为你的实际文件路径
    # output_file = '/data/yuyangchen/data/MELD/processed_train_T_emo.csv'  # 输出文件名,T表示已经处理好了T模态的数据
    # 输入和输出文件路径
    # 替换为实际路径
    train_input_file = "/data/yuyangchen/data/MELD/train_sent_emo.csv"
    dev_input_file = "meld_dev.csv"
    test_input_file = "/data/yuyangchen/data/MELD/test_sent_emo.csv"
    
    train_output_file = "/data/yuyangchen/data/MELD/processed_train_T_emo.csv"
    dev_output_file = "meld_dev_cleaned.csv"
    test_output_file = "/data/yuyangchen/data/MELD/processed_test_T_emo.csv"
    
    # 修改 train、dev 和 test 文件
    clean_and_map_emotion_id(train_input_file, train_output_file)
    # clean_and_map_emotion_id(dev_input_file, dev_output_file)
    clean_and_map_emotion_id(test_input_file, test_output_file)

if __name__ == "__main__":
    main()