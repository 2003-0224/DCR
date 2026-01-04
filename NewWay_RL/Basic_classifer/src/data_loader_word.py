"""
提取context，集合成样本，参与训练，并实现word level特征的提取：
"""
import torch
from torch import nn
import transformers
# import torchaudio
from transformers import RobertaTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
import string


class wordMELDDataset(Dataset):
    def __init__(self, data_path, tokenizer, context_len=5, max_seq_length=160, use_all_context=False,
                 add_speaker=False):
        """
        初始化 MELD 数据集类。
        
        参数：
        - data_path (str): 数据文件路径（例如 meld_train_cleaned.csv）。
        - tokenizer: RoBERTa 的 tokenizer（例如 RobertaTokenizer）。
        - context_len (int): 上下文长度（默认 5）,默认为历史对话
        - max_seq_length (int): 最大序列长度（默认 128）。 最大的有600多，详细见记录，上下文的最大记录，按照每个utterance_len为32左右来计算，全文的话就是512
        - use_all_context (bool): 是否使用整个对话的所有上下文（包括前置和后置，默认为 False）。
        - add_speaker (bool): 是否将 speaker 添加到 utterance 中（例如 "speaker: utterance"）。
        """
        # 读取数据
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.use_all_context = use_all_context
        self.add_speaker = add_speaker
        if context_len is None or use_all_context == True:
            self.max_seq_length = 512
        else:
            self.max_seq_length = max_seq_length

        # 按 Dialogue_ID 分组
        self.dialogues = self.data.groupby("Dialogue_ID")
        self.dialogue_ids = list(self.dialogues.groups.keys())

        if self.add_speaker and "Speaker" not in self.data.columns:
            raise ValueError("Data must contain a 'Speaker' column when add_speaker=True")

        # 打印数据集统计信息
        print(f"Loaded dataset from {data_path}")
        print(f"Total utterances: {len(self.data)}")
        print(f"Total dialogues: {len(self.dialogue_ids)}")
        print(f"Use full context: {use_all_context}")
        if not use_all_context:
            print(f"Context length: {context_len}")
        # print(f"Emotion ID distribution:\n{self.data['emotion_id'].value_counts().sort_index()}")
        if self.add_speaker:
            print("Adding speaker prefix to utterances")

    def __len__(self):
        """
        返回数据集的大小（总话语数）。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取第 idx 个样本。
        
        参数：
        - idx (int): 样本索引。
        
        返回：
        - dict: 包含以下键值对：
            - input_ids: RoBERTa 的输入 token IDs（形状：[max_seq_length]）。
            - attention_mask: 注意力掩码（形状：[max_seq_length]）。
            - label: 情感标签（emotion_id，标量）。
        """
        # 获取目标话语
        row = self.data.iloc[idx]
        dialogue_id = row["Dialogue_ID"]
        utterance_id = row["Utterance_ID"]
        target_utterance = row["Utterance"]
        emotion_id = row["emotion_id"]

        # 生成样本名
        sample_name = f"dia{dialogue_id}_utt{utterance_id}"

        # 添加 speaker 前缀
        if self.add_speaker:
            speaker = row["Speaker"]
            target_utterance = f"{speaker}: {target_utterance}"

        # 获取对话
        dialogue = self.dialogues.get_group(dialogue_id)

        # # 获取上下文（前 context_len 个话语）
        # start_idx = max(0, utterance_id - self.context_len)
        # context_utterances = dialogue[(dialogue["Utterance_ID"] >= start_idx) & 
        #                             (dialogue["Utterance_ID"] < utterance_id)]["Utterance"].tolist()

        # 获取上下文
        if self.use_all_context:
            all_utterances = dialogue.sort_values("Utterance_ID")[["Utterance", "Speaker"]].values
            target_idx = dialogue[dialogue["Utterance_ID"] == utterance_id].index[0]
            target_pos_in_list = dialogue.index.get_loc(target_idx)
            # 为所有 utterance 添加 speaker 前缀
            if self.add_speaker:
                all_utterances = [f"{speaker}: {utt}" for utt, speaker in all_utterances]
            else:
                all_utterances = [utt for utt, _ in all_utterances]
        else:
            context_utterances = dialogue[dialogue["Utterance_ID"] < utterance_id][["Utterance", "Speaker"]].values
            if self.context_len is not None and len(context_utterances) > self.context_len:
                context_utterances = context_utterances[-self.context_len:]
            # 为上下文和目标 utterance 添加 speaker 前缀
            if self.add_speaker:
                all_utterances = [f"{speaker}: {utt}" for utt, speaker in context_utterances] + [target_utterance]
            else:
                all_utterances = [utt for utt, _ in context_utterances] + [target_utterance]
            target_pos_in_list = len(context_utterances)

        # # 获取上下文（从对话开始到目标话语前）
        # context_utterances = dialogue[(dialogue["Utterance_ID"] < utterance_id)]["Utterance"].tolist()

        # # 如果设置了 max_context_len，限制上下文长度
        # if self.context_len is not None and len(context_utterances) > self.context_len:
        #     context_utterances = context_utterances[-self.context_len:]

        # 拼接上下文和目标话语
        # 格式：<s>utt_1</s></s>utt_2</s></s>utt_3</s>...
        input_text = "</s></s>".join(all_utterances)  # 在列表元素中间加入 </s></s>作为分隔符，开头和结尾的标记，tokenizer会自动添加（默认的）

        # 编码输入
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,  # 当标记长度大于最大值，是否进行截取
            return_tensors="pt"  # 选择返回的数据类型为pytorch张量
        )

        # 记录目标话语的起始和结束位置
        input_ids = encoding["input_ids"][0]
        sep_positions = [i for i, token_id in enumerate(input_ids) if
                         token_id == self.tokenizer.sep_token_id]  # self.tokenizer.sep_token_id 指的是分隔符</s>

        # 计算目标话语的起始和结束位置
        if target_pos_in_list == 0:  # 目标话语是第一个
            target_start_pos = 1  # 跳过 <s>
        else:
            # 目标话语从前一个 </s> 后开始
            if (target_pos_in_list * 2 - 1) >= len(sep_positions):  # 在测试集中，有一条数据的总长度大于512，需要进行特殊处理
                target_start_pos = sep_positions[-2]

            else:
                target_start_pos = sep_positions[target_pos_in_list * 2 - 1] + 1

        # 结束位置：目标话语后的第一个 </s> 或序列末尾
        if target_pos_in_list < len(all_utterances) - 1 and (target_pos_in_list + 1) * 2 <= len(sep_positions):
            target_end_pos = sep_positions[target_pos_in_list * 2]
        else:
            if (target_pos_in_list * 2 - 1) >= len(sep_positions):
                target_end_pos = sep_positions[-1]
            else:
                target_end_pos = sep_positions[-1] if input_ids[-1] in [self.tokenizer.pad_token_id,
                                                                        self.tokenizer.sep_token_id] else len(input_ids)

        # # 如果有上下文，目标起始位置是最后一个上下文的 </s> 后；否则从序列开头
        # target_start_pos = sep_positions[-2] + 1 if len(sep_positions) > 1 else 1  # 跳过 <s>
        # if input_ids[-1] == self.tokenizer.pad_token_id or input_ids[-1] == self.tokenizer.sep_token_id:
        #     target_end_pos = sep_positions[-1]
        # else:
        #     target_end_pos = len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(emotion_id, dtype=torch.long),
            "target_start_pos": target_start_pos,
            "target_end_pos": target_end_pos,
            "sample_name": sample_name
        }


# 测试代码
def test_dataset():
    # 加载 tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # 加载数据集
    train_dataset = wordMELDDataset(
        data_path="/data/yuyangchen/data/MELD/processed_test_T_emo.csv",
        tokenizer=tokenizer,
        context_len=None,
        max_seq_length=128,
        use_all_context=True,
        add_speaker=True
    )

    # 创建 DataLoader 以便测试
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 测试前几个样本
    for i, batch in enumerate(train_loader):
        if i >= 5:  # 只测试前 5 个样本
            break

        input_ids = batch["input_ids"][0]
        attention_mask = batch["attention_mask"][0]
        label = batch["label"][0]
        target_start_pos = batch["target_start_pos"][0].item()
        target_end_pos = batch["target_end_pos"][0].item()

        # 提取目标话语的词级 token 序列
        target_tokens = input_ids[target_start_pos:target_end_pos]
        decoded_text = tokenizer.decode(target_tokens, skip_special_tokens=False)

        # 解码整个输入以便对比
        full_decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)

        print(f"\nSample {i}:")
        print(f"Full decoded input text:\n{full_decoded_text}")
        print(f"Target start position: {target_start_pos}")
        print(f"Target end position: {target_end_pos}")
        print(f"Target utterance token IDs: {target_tokens.tolist()}")
        print(f"Target utterance decoded: {decoded_text}")
        print(f"Label: {label.item()}")


if __name__ == "__main__":
    test_dataset()
