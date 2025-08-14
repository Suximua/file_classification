import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer
from utils.data_loader import load_data
import joblib

def get_embeddings(texts, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding.flatten())
    
    return np.vstack(embeddings)

def main():
    # 加载数据
    train_data, test_data = load_data('data/dataset.csv')
    
    # 准备标签映射
    class_names = ['简历', '合同', '小说', '发票', '数学', '法律', 
                  '物理', '作文', '说明书', '方案', '英语', '论文']
    label2id = {label: i for i, label in enumerate(class_names)}
    
    # 获取嵌入向量
    model_path = 'models/m3e-base'
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

