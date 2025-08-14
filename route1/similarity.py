import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from utils.data_loader import load_data
from utils.metrics import compute_metrics


class SimilarityClassifier:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.class_centers = {}
        self.class_names = ['简历', '合同', '小说', '发票', '数学', '法律', '物理', '作文', '说明书', '方案', '英语', '论文']

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(1).detach().numpy()
    
    def compute_class_centers(self, train_data):
        for class_name in self.class_names:
            class_texts = train_data[train_data['label'] == class_name]['PlainText'].tolist()
            embeddings = np.vstack([self.get_embedding(text) for text in class_texts])
            self.class_centers[class_name] = np.mean(embeddings, axis=0)

    def predict(self, text):
        text_embedding = self.get_embedding(text)
        similarities = {
            class_name: cosine_similarity(text_embedding.reshape(1, -1))[0][0]
            for class_name, center in self.class_centers.items()
        }
        return max(similarities, key=similarities.get)
    
    def main():
        #数据加载
        train_data, test_data = load_data('data/dataset.csv')
        # 初始化分类器
        model_path = 'models/m3e-base'
        classifier = SimilarityClassifier(model_path)
        
        # 计算类别中心
        classifier.compute_class_centers(train_data)
        
        # 预测
        predictions = []
        for text in test_data['PlainText']:
            predictions.append(classifier.predict(text))
        
        # 评估
        true_labels = test_data['Label'].tolist()
        results = calculate_metrics(true_labels, predictions, classifier.class_names)
        
        # 输出结果
        print("=== 语义相似度匹配结果 ===")
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
        for link_id, true, pred in errors[:10]:  # 只显示前10个错误
            print(f"文件ID: {link_id}, 真实标签: {true}, 预测标签: {pred}")

    if __name__ == "__main__":
        main()