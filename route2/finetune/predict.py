import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.data_loader import load_data
from utils.metrics import calculate_metrics

def main():
    # 加载数据
    _, test_data = load_data('data/dataset.csv')
    
    # 加载模型
    model_path = 'route2/finetune/saved_model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # 准备标签映射
    class_names = ['简历', '合同', '小说', '发票', '数学', '法律', 
                  '物理', '作文', '说明书', '方案', '英语', '论文']
    id2label = {i: label for i, label in enumerate(class_names)}
    
    # 预测
    predictions = []
    model.eval()
    with torch.no_grad():
        for text in test_data['PlainText']:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(id2label[pred_id])
    
    # 评估
    true_labels = test_data['Label'].tolist()
    results = calculate_metrics(true_labels, predictions, class_names)
    
    # 输出结果
    print("=== Finetune模型结果 ===")
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
