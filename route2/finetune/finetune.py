import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from utils.data_loader import load_data
from utils.metrics import compute_metrics

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
    # 加载数据
    train_data, test_data = load_data('data/dataset.csv')
    
    # 准备标签映射
    class_names = ['简历', '合同', '小说', '发票', '数学', '法律', 
                  '物理', '作文', '说明书', '方案', '英语', '论文']
    label2id = {label: i for i, label in enumerate(class_names)}
    id2label = {i: label for label, i in label2id.items()}
    
    # 加载模型和tokenizer
    model_path = 'models/m3e-base'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id
    )
    
    # 准备数据集
    train_dataset = TextDataset(
        train_data['PlainText'].tolist(),
        [label2id[label] for label in train_data['Label']],
        tokenizer
    )
    
    test_dataset = TextDataset(
        test_data['PlainText'].tolist(),
        [label2id[label] for label in test_data['Label']],
        tokenizer
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 训练模型
    trainer.train()
    
    # 保存模型
    model.save_pretrained('route2/finetune/saved_model')
    tokenizer.save_pretrained('route2/finetune/saved_model')

if __name__ == "__main__":
    main()
