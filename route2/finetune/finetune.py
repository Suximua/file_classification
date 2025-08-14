import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from utils.data_loader import load_data
from utils.metrics import calculate_metrics
import numpy as np


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def main():
    print("开始Finetune训练...")
    # 加载数据
    print("加载数据...")
    train_data, test_data = load_data('data/dataset.csv')

    # # 为了快速验证，只使用小部分数据
    # train_data = train_data.head(20)  # 只使用前20个样本
    # test_data = test_data.head(5)  # 只使用前5个样本

    print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

    # 准备标签映射
    class_names = ['简历', '合同', '小说', '发票', '数学', '法律',
                   '物理', '作文', '说明书', '方案', '英语', '论文']
    label2id = {label: i for i, label in enumerate(class_names)}
    id2label = {i: label for label, i in label2id.items()}

    # 加载模型和tokenizer
    model_path = 'BAAI/bge-m3'
    # 检查模型路径是否存在，如果不存在则使用模型名称从HuggingFace下载
    if os.path.exists(model_path):
        print(f"从本地路径加载模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(class_names),
            id2label=id2label,
            label2id=label2id
        )
    else:
        print(f"本地模型路径 {model_path} 不存在，将从HuggingFace下载模型")
        model_name = "BAAI/bge-m3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(class_names),
            id2label=id2label,
            label2id=label2id
        )
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"使用GPU进行训练: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("使用CPU进行训练")

    # 准备数据集
    print("准备训练数据集...")
    train_dataset = TextDataset(
        train_data['PlainText'].tolist(),
        [label2id[label] for label in train_data['Label']],
        tokenizer
    )

    print("准备测试数据集...")
    test_dataset = TextDataset(
        test_data['PlainText'].tolist(),
        [label2id[label] for label in test_data['Label']],
        tokenizer
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,  # 减小批次大小以节省内存
        per_device_eval_batch_size=8,   # 减小批次大小以节省内存
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=None,  # 禁用wandb等报告工具
        fp16=torch.cuda.is_available(),  # 如果有GPU则启用混合精度训练
        dataloader_num_workers=2,  # 减少worker数量以节省内存
        gradient_accumulation_steps=2,  # 梯度累积以模拟更大的批次
    )


    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # 训练模型
    print("开始训练模型...")
    trainer.train()

    # 保存模型
    print("保存模型...")
    model.save_pretrained('route2/finetune/saved_model')
    tokenizer.save_pretrained('route2/finetune/saved_model')
    print("模型保存完成")


if __name__ == "__main__":
    main()