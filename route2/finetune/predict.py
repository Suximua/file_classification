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

    # 预测
    print("开始预测...")
    predictions = []
    test_texts = test_data['PlainText'].tolist()
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            if (i + 1) % 10 == 0 or i == len(test_texts) - 1:
                print(f"预测进度: {i+1}/{len(test_texts)}")
                
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
    true_labels = test_data['Label'].tolist()
    results = calculate_metrics(true_labels, predictions, class_names)

    # 输出结果
    print("=== Finetune模型结果 ===")
    accuracy = sum(results['confusion_matrix'][i][i] for i in range(len(class_names))) / len(test_data)
    print(f"总体准确率: {accuracy:.4f}")
    print("\n各分类指标:")
    print(f"{'类别':<8} {'精确率':<8} {'召回率':<8} {'F1值':<8}")
    print("-" * 40)
    for class_name in class_names:
        precision = results['precision'].get(class_name, 0)
        recall = results['recall'].get(class_name, 0)
        f1 = results['f1'].get(class_name, 0)
        print(f"{class_name:<8} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f}")
    
    print("\n混淆矩阵:")
    # 打印表头
    header_label = "真实\\预测"
    print(f"{header_label:<8}", end="")
    for class_name in class_names:
        print(f"{class_name:<6}", end="")
    print()
    
    # 打印矩阵内容
    for i, true_class in enumerate(class_names):
        print(f"{true_class:<8}", end="")
        for j in range(len(class_names)):
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
    main()
