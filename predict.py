import torch
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import jieba
import random

# 1. 加载预训练模型和分词器
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
model.eval()  # 设置为评估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 2. 读取训练集
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = self.load_data()

    def load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.readlines()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return inputs


# 3. Masked Language Modeling (MLM) 输入处理
def mask_tokens(inputs):
    """ 将文本中的某些token用[MASK]替代，模拟MLM任务 """
    # 随机mask掉部分token
    inputs_ids = inputs["input_ids"].squeeze(0)
    labels = inputs_ids.clone()
    probability_matrix = torch.full(labels.shape, 0.15)  # Mask掉15%的tokens
    special_tokens_mask = tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # 随机选择tokens来mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # 对非mask的token设为-100（表示不计算损失）

    inputs_ids[masked_indices] = tokenizer.mask_token_id
    inputs["input_ids"] = inputs_ids.unsqueeze(0)
    inputs["labels"] = labels.unsqueeze(0)

    return inputs


# 4. 创建训练数据加载器
train_dataset = TextDataset(file_path='train.txt', tokenizer=tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


# 5. 训练过程
def train(model, train_loader, epochs=3, learning_rate=5e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        loop = tqdm(train_loader, desc="Training", leave=True)
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = mask_tokens(batch)  # 对batch进行mask处理
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())


train(model, train_loader, epochs=8)
model.save_pretrained('./finetuned_bert')
tokenizer.save_pretrained('./finetuned_bert')

