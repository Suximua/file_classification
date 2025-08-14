import os
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer
from utils.data_loader import load_data
import joblib
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
        
        # 确保text是字符串类型
        if not isinstance(text, str):
            print(f"警告: 发现非字符串类型的文本 (索引 {i}): {type(text)}, 值: {text}")
            # 尝试转换为字符串，如果失败则使用空字符串
            text = str(text) if text is not None else ""
            
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
    print("开始传统机器学习训练...")
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
    
    # 获取嵌入向量
    model_path = 'BAAI/bge-m3'
    print("生成训练集嵌入向量...")
    train_embeddings = get_embeddings(train_data['PlainText'].tolist(), model_path)
    train_labels = [label2id[label] for label in train_data['Label']]
    
    print("生成测试集嵌入向量...")
    test_embeddings = get_embeddings(test_data['PlainText'].tolist(), model_path)
    test_labels = [label2id[label] for label in test_data['Label']]
    
    # 训练多个分类器
    classifiers = {
        'SVM': SVC(kernel='linear', probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'LogisticRegression': LogisticRegression(max_iter=1000)
    }
    
    best_model = None
    best_score = 0
    
    for name, clf in classifiers.items():
        print(f"训练 {name} 模型...")
        clf.fit(train_embeddings, train_labels)
        score = clf.score(test_embeddings, test_labels)
        print(f"{name} 准确率: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = clf
    
    # 保存最佳模型
    joblib.dump(best_model, 'route2/traditional/best_model.pkl')
    print(f"最佳模型已保存: {type(best_model).__name__}")

if __name__ == "__main__":
    main()

