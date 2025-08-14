import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.data_loader import load_data
from utils.metrics import calculate_metrics

def main():
    print("开始Finetune预测...")
    # 加载数据
    print("加载数据...")
    _, test_data = load_data('data/dataset.csv')
    print(f"测试集大小: {len(test_data)}")

    # 加载模型
    model_path = 'route2/finetune/saved_model'
    # 检查微调模型是否存在，如果不存在则使用基础模型
    if os.path.exists(model_path):
        print(f"从本地路径加载微调模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        print(f"微调模型路径 {model_path} 不存在，将使用基础模型")
        base_model_path = 'BAAI/bge-m3'
        if os.path.exists(base_model_path):
            print(f"从本地路径加载基础模型: {base_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
        else:
            print(f"本地模型路径 {base_model_path} 不存在，将从HuggingFace下载模型")
            model_name = "BAAI/bge-m3"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # 如果有GPU则使用GPU
    if torch.cuda.is_available():
        model = model.cuda()
        print("使用GPU进行预测")
    else:
        print("使用CPU进行预测")

    # 准备标签映射
    class_names = ['简历', '合同', '小说', '发票', '数学', '法律',
                   '物理', '作文', '说明书', '方案', '英语', '论文']
    id2label = {i: label for i, label in enumerate(class_names)}
    label2id = {label: i for i, label in enumerate(class_names)}

    # 预测
    print("开始预测...")
    predictions = []
    test_texts = test_data['PlainText'].tolist()
    true_labels = test_data['Label'].tolist()
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            if (i + 1) % 10 == 0 or i == len(test_texts) - 1:
                print(f"预测进度: {i+1}/{len(test_texts)}")
            
            # 确保text是字符串类型
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
                
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # 如果有GPU则将数据移到GPU上
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(id2label[pred_id])

    # 评估
    results = calculate_metrics(true_labels, predictions, class_names)

    # 输出结果
    print("=== Finetune模型结果 ===")
    # 计算准确率
    correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
    accuracy = correct / len(true_labels)
    print(f"总体准确率: {accuracy:.4f}")
    print("\n各分类指标:")
    print(f"{'类别':<8} {'精确率':<8} {'召回率':<8} {'F1值':<8}")
    print("-" * 40)
    
    for class_name in class_names:
        if class_name in results['precision']:
            precision = results['precision'][class_name]
            recall = results['recall'][class_name]
            f1 = results['f1'][class_name]
            print(f"{class_name:<8} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f}")

    # 保存结果
    import json
    output_results = []
    for i, (true_label, pred_label) in enumerate(zip(true_labels, predictions)):
        # 确保文本是字符串类型
        text = test_texts[i]
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        output_results.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'true_label': true_label,
            'predicted_label': pred_label
        })
    
    with open('route2/finetune/prediction_results.json', 'w', encoding='utf-8') as f:
        json.dump(output_results, f, ensure_ascii=False, indent=2)
    print("\n预测结果已保存到 route2/finetune/prediction_results.json")

if __name__ == "__main__":
    main()
