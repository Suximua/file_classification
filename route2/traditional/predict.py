import os
import numpy as np
import joblib
from transformers import AutoModel, AutoTokenizer
from utils.data_loader import load_data
from utils.metrics import calculate_metrics
import torch

def get_embeddings(texts, model_path):
    print("加载模型...")
    # 检查模型路径是否存在，如果不存在则使用模型名称从HuggingFace下载
    if os.path.exists(model_path):
        print(f"从本地路径加载模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
    else:
        print(f"本地模型路径 {model_path} 不存在，将从HuggingFace下载模型")
        model_name = "BAAI/bge-m3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    
    # 如果有GPU则使用GPU
    if torch.cuda.is_available():
        model = model.cuda()
        print("使用GPU计算嵌入向量")
    else:
        print("使用CPU计算嵌入向量")
    
    embeddings = []
    for i, text in enumerate(texts):
        if (i + 1) % 50 == 0 or i == len(texts) - 1:
            print(f"嵌入向量计算进度: {i+1}/{len(texts)}")
            
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # 如果有GPU则将数据移到GPU上
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(embedding.flatten())
    
    return np.vstack(embeddings)

def main():
    print("开始传统机器学习预测...")
    # 加载数据
    print("加载数据...")
    _, test_data = load_data('data/dataset.csv')
    print(f"测试集大小: {len(test_data)}")
    
    # 准备标签映射
    class_names = ['简历', '合同', '小说', '发票', '数学', '法律', 
                  '物理', '作文', '说明书', '方案', '英语', '论文']
    id2label = {i: label for i, label in enumerate(class_names)}
    
    # 加载模型
    model_path = 'BAAI/bge-m3'
    classifier = joblib.load('route2/traditional/best_model.pkl')
    
    # 获取嵌入向量
    print("生成测试集嵌入向量...")
    test_embeddings = get_embeddings(test_data['PlainText'].tolist(), model_path)
    
    # 预测
    print("开始预测...")
    predictions = [id2label[pred] for pred in classifier.predict(test_embeddings)]
    
    # 评估
    true_labels = test_data['Label'].tolist()
    results = calculate_metrics(true_labels, predictions, class_names)
    
    # 输出结果
    print("=== 传统机器学习结果 ===")
    print(f"精确率: {results['precision']}")
    print(f"召回率: {results['recall']}")
    print(f"F1值: {results['f1']}")
    print("\n混淆矩阵:")
    print(results['confusion_matrix'])
    
    # 错误分析
    errors = []
    for i, (true, pred) in enumerate(zip(true_labels, predictions)):
        if true != pred:
            errors.append((test_data.iloc[i]['LinkID'], true, pred))
    
    print("\n分类错误样本:")
    for link_id, true, pred in errors[:10]:
        print(f"文件ID: {link_id}, 真实标签: {true}, 预测标签: {pred}")

if __name__ == "__main__":
    main()
