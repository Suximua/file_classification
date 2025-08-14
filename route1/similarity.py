import os
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from utils.data_loader import load_data
from utils.metrics import calculate_metrics

class SimilarityClassifier:
    def __init__(self, model_path):
        # 检查模型路径是否存在，如果不存在则使用模型名称从HuggingFace下载
        if os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
        else:
            print(f"本地模型路径 {model_path} 不存在，将从HuggingFace下载模型")
            model_name = "BAAI/bge-m3"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        
        # 如果有GPU则使用GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("使用GPU进行计算")
        else:
            print("使用CPU进行计算")
            
        self.class_centers = {}
        self.class_names = ['简历', '合同', '小说', '发票', '数学', '法律', '物理', '作文', '说明书', '方案', '英语', '论文']

    def get_embedding(self, text):
        # 确保text是字符串类型
        if not isinstance(text, str):
            text = str(text)
        # 如果文本为空，返回零向量
        if not text.strip():
            return np.zeros(self.model.config.hidden_size)
            
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        # 如果有GPU则将数据移到GPU上
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    def compute_class_centers(self, train_data):
        print("计算类别中心...")
        for i, class_name in enumerate(self.class_names):
            print(f"处理类别 {i+1}/{len(self.class_names)}: {class_name}")
            class_texts = train_data[train_data['Label'] == class_name]['PlainText'].tolist()
            # 处理类别中没有样本的情况
            if len(class_texts) == 0:
                print(f"警告: 类别 '{class_name}' 在训练数据中没有样本，跳过该类别")
                continue
            embeddings = np.vstack([self.get_embedding(text) for text in class_texts])
            self.class_centers[class_name] = np.mean(embeddings, axis=0)
        print("类别中心计算完成")

    def predict(self, text):
        # 如果没有任何类别中心（所有类别在训练集中都缺失），返回默认类别
        if not self.class_centers:
            return self.class_names[0]  # 默认返回第一个类别
            
        text_embedding = self.get_embedding(text)
        similarities = {}
        for class_name, center in self.class_centers.items():
            sim = cosine_similarity(text_embedding.reshape(1, -1), center.reshape(1, -1))[0][0]
            similarities[class_name] = sim
            
        # 如果没有计算出任何相似度（理论上不应该发生），返回默认类别
        if not similarities:
            return self.class_names[0]
            
        return max(similarities, key=similarities.get)
    
    @staticmethod
    def main():
        #数据加载
        print("加载数据...")
        train_data, test_data = load_data('data/dataset.csv')
        print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
        
        # 初始化分类器
        model_path = 'BAAI/bge-m3'
        classifier = SimilarityClassifier(model_path)
        
        # 计算类别中心
        classifier.compute_class_centers(train_data)
        
        # 检查是否有类别在训练集中完全缺失，并给出提示
        missing_classes = set(classifier.class_names) - set(classifier.class_centers.keys())
        if missing_classes:
            print(f"警告: 以下类别在训练集中没有样本: {', '.join(missing_classes)}")
        
        # 预测
        print("开始预测...")
        predictions = []
        test_texts = test_data['PlainText'].tolist()
        for i, text in enumerate(test_texts):
            if (i + 1) % 10 == 0 or i == len(test_texts) - 1:
                print(f"预测进度: {i+1}/{len(test_texts)}")
            predictions.append(classifier.predict(text))
        
        # 评估 - 只使用训练集中存在的类别
        true_labels = test_data['Label'].tolist()
        existing_class_names = list(classifier.class_centers.keys())
        results = calculate_metrics(true_labels, predictions, existing_class_names)
        
        # 输出结果
        print("=== 语义相似度匹配结果 ===")
        accuracy = sum(results['confusion_matrix'][i][i] for i in range(len(existing_class_names))) / len(test_data)
        print(f"总体准确率: {accuracy:.4f}")
        print("\n各分类指标:")
        print(f"{'类别':<8} {'精确率':<8} {'召回率':<8} {'F1值':<8}")
        print("-" * 40)
        for class_name in existing_class_names:
            precision = results['precision'].get(class_name, 0)
            recall = results['recall'].get(class_name, 0)
            f1 = results['f1'].get(class_name, 0)
            print(f"{class_name:<8} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f}")
        
        print("\n混淆矩阵:")
        # 打印表头
        header_label = "真实\\预测"
        print(f"{header_label:<8}", end="")
        for class_name in existing_class_names:
            print(f"{class_name:<6}", end="")
        print()
        
        # 打印矩阵内容
        for i, true_class in enumerate(existing_class_names):
            print(f"{true_class:<8}", end="")
            for j in range(len(existing_class_names)):
                print(f"{results['confusion_matrix'][i][j]:<6}", end="")
            print()
        
        # 错误分析
        errors = []
        for i, (true, pred) in enumerate(zip(true_labels, predictions)):
            if true != pred:
                errors.append((test_data.iloc[i]['LinkID'], true, pred))
        
        print(f"\n分类错误样本 (共{len(errors)}个错误):")
        print(f"{'文件ID':<15} {'真实标签':<8} {'预测标签':<8}")
        print("-" * 35)
        for link_id, true, pred in errors[:10]:  # 只显示前10个错误
            print(f"{link_id:<15} {true:<8} {pred:<8}")

if __name__ == "__main__":
    SimilarityClassifier.main()
