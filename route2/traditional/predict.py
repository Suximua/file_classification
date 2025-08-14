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
    print("开始传统机器学习预测...")
    # 加载数据
    print("加载数据...")
    _, test_data = load_data('data/dataset.csv')
    print(f"测试集大小: {len(test_data)}")
    
    # 准备标签映射
    class_names = ['简历', '合同', '小说', '发票', '数学', '法律',
                  '物理', '作文', '说明书', '方案', '英语', '论文']
    label2id = {label: i for i, label in enumerate(class_names)}
    id2label = {i: label for i, label in enumerate(class_names)}
    
    # 获取嵌入向量
    model_path = 'BAAI/bge-m3'
    print("生成测试集嵌入向量...")
    test_embeddings = get_embeddings(test_data['PlainText'].tolist(), model_path)
    test_labels = [label2id[label] for label in test_data['Label']]
    
    # 加载模型
    print("加载模型...")
    model = joblib.load('route2/traditional/best_model.pkl')
    
    # 预测
    print("预测中...")
    predictions = model.predict(test_embeddings)
    probabilities = model.predict_proba(test_embeddings) if hasattr(model, 'predict_proba') else None
    
    # 计算指标
    print("计算指标...")
    accuracy = np.mean(np.array(test_labels) == np.array(predictions))
    metrics = calculate_metrics(test_labels, predictions, class_names)
    
    # 打印结果
    print("\n=== 传统机器学习预测结果 ===")
    print(f"准确率: {accuracy:.4f}")
    print("详细指标:")
    print(f"{'类别':<8} {'精确率':<8} {'召回率':<8} {'F1值':<8}")
    print("-" * 40)
    
    for class_name in class_names:
        if class_name in metrics['precision']:
            precision = metrics['precision'][class_name]
            recall = metrics['recall'][class_name]
            f1 = metrics['f1'][class_name]
            print(f"{class_name:<8} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f}")
    
    # 保存结果
    results = []
    for i, (pred, true) in enumerate(zip(predictions, test_labels)):
        # 确保文本是字符串类型
        text = test_data.iloc[i]['PlainText']
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        result = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'true_label': id2label[true],
            'predicted_label': id2label[pred],
            'confidence': float(np.max(probabilities[i])) if probabilities is not None else None
        }
        results.append(result)
    
    import json
    with open('route2/traditional/prediction_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("预测结果已保存到 route2/traditional/prediction_results.json")

if __name__ == "__main__":
    main()
