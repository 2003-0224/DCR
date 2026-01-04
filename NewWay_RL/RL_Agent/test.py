import numpy as np

def load_sample_names(npz_path):
    """
    只读取 sample_names 字段，不做其它推测。
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "sample_names" not in data:
            print(f"[ERROR] {npz_path} 中不存在 sample_names 字段！")
            return None
        raw = data["sample_names"]

        # 统一成字符串
        processed = []
        for x in raw:
            if isinstance(x, bytes):
                processed.append(x.decode("utf-8").strip())
            else:
                processed.append(str(x).strip())

        return set(processed)

    except Exception as e:
        print(f"[ERROR] 无法读取 {npz_path}: {e}")
        return None


def check_alignment(text_path, audio_path, video_path):
    print("=========== checking sample_names alignment ===========")

    ids_T = load_sample_names(text_path)
    ids_A = load_sample_names(audio_path)
    ids_V = load_sample_names(video_path)

    if ids_T is None or ids_A is None or ids_V is None:
        print("读取失败，无法继续。")
        return

    print("\n数量统计：")
    print(f"Text : {len(ids_T)}")
    print(f"Audio: {len(ids_A)}")
    print(f"Video: {len(ids_V)}")

    # ==========================================
    # 三模态的交集 = 能对齐的最终样本
    # ==========================================
    common = ids_T & ids_A & ids_V
    print(f"\n可三模态对齐的样本数：{len(common)}")

    # ==========================================
    # 不对齐的 ID 输出
    # ==========================================
    print("\n--- Text 中有，但 Audio 没有 ---")
    print(list(ids_T - ids_A)[:30])
    if len(ids_T - ids_A) > 30:
        print("...")

    print("\n--- Text 中有，但 Video 没有 ---")
    print(list(ids_T - ids_V)[:30])
    if len(ids_T - ids_V) > 30:
        print("...")

    print("\n--- Audio 中有，但 Text 没有 ---")
    print(list(ids_A - ids_T)[:30])
    if len(ids_A - ids_T) > 30:
        print("...")

    print("\n--- Video 中有，但 Text 没有 ---")
    print(list(ids_V - ids_T)[:30])
    if len(ids_V - ids_T) > 30:
        print("...")

    print("\n=========== done ===========")
    return common


if __name__ == "__main__":
    text_npz = "/data/home/chenqian/models/text_model/test_text_features_general.npz"
    audio_npz = "/data/home/chenqian/models/audio_model/test_audio_features_general.npz"
    video_npz = "/data/home/chenqian/models/video_model/test_video_features_general.npz"

    check_alignment(text_npz, audio_npz, video_npz)
