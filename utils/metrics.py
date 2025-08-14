from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np

def calculate_metrics(true_labels, predicted_labels, class_names):
    """
    计算分类指标
    
    Args:
        true_labels: 真实标签列表
        predicted_labels: 预测标签列表
        class_names: 类别名称列表
    """
    # 检查标签是字符串还是数字类型
    are_string_labels = isinstance(true_labels[0], str) if true_labels else False
    
    if are_string_labels:
        # 处理字符串标签（Finetune方法）
        # 获取所有唯一标签
        all_labels = list(set(true_labels + predicted_labels))
        
        # 计算指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, labels=all_labels, average=None, zero_division=0
        )
        
        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
        
        # 创建指标字典
        precision_dict = dict(zip(all_labels, precision))
        recall_dict = dict(zip(all_labels, recall))
        f1_dict = dict(zip(all_labels, f1))
        
        # 计算每个类别的TP, FP, FN, TN
        metrics_per_class = {}
        for i, label in enumerate(all_labels):
            tp = cm[i, i] if i < len(cm) and i < len(cm[0]) else 0
            fp = cm[:, i].sum() - tp if i < len(cm[0]) else 0
            fn = cm[i, :].sum() - tp if i < len(cm) else 0
            tn = cm.sum() - (tp + fp + fn)
            
            metrics_per_class[label] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn
            }
        
        return {
            'precision': precision_dict,
            'recall': recall_dict,
            'f1': f1_dict,
            'confusion_matrix': cm,
            'per_class': metrics_per_class
        }
    else:
        # 处理数字标签（Traditional方法）
        # 获取所有唯一标签
        all_labels = list(set(true_labels + predicted_labels))
        
        # 为每个标签创建一个映射，将标签索引映射到类别名称
        label_to_name = {i: class_names[i] for i in all_labels if i < len(class_names)}
        
        # 计算指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average=None, zero_division=0
        )
        
        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # 计算每个类别的TP, FP, FN, TN
        metrics_per_class = {}
        involved_labels = list(set(true_labels) | set(predicted_labels))
        
        for label in involved_labels:
            # 找到该标签在混淆矩阵中的索引
            if label in all_labels:
                idx = all_labels.index(label)
                tp = cm[idx, idx] if idx < len(cm) and idx < len(cm[0]) else 0
                fp = cm[:, idx].sum() - tp if idx < len(cm[0]) else 0
                fn = cm[idx, :].sum() - tp if idx < len(cm) else 0
                tn = cm.sum() - (tp + fp + fn)
            else:
                tp = fp = fn = tn = 0
                
            # 获取类别名称
            class_name = label_to_name.get(label, f"Class_{label}")
            
            # 确保索引有效
            prec = precision[all_labels.index(label)] if label in all_labels and all_labels.index(label) < len(precision) else 0
            rec = recall[all_labels.index(label)] if label in all_labels and all_labels.index(label) < len(recall) else 0
            f1_score = f1[all_labels.index(label)] if label in all_labels and all_labels.index(label) < len(f1) else 0
            
            metrics_per_class[class_name] = {
                'precision': prec,
                'recall': rec,
                'f1': f1_score,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn
            }
        
        # 创建按类别名称索引的指标字典
        precision_dict = {}
        recall_dict = {}
        f1_dict = {}
        
        for label in involved_labels:
            class_name = label_to_name.get(label, f"Class_{label}")
            if label in all_labels:
                idx = all_labels.index(label)
                prec = precision[idx] if idx < len(precision) else 0
                rec = recall[idx] if idx < len(recall) else 0
                f1_score = f1[idx] if idx < len(f1) else 0
            else:
                prec = rec = f1_score = 0
                
            precision_dict[class_name] = prec
            recall_dict[class_name] = rec
            f1_dict[class_name] = f1_score
        
        return {
            'precision': precision_dict,
            'recall': recall_dict,
            'f1': f1_dict,
            'confusion_matrix': cm,
            'per_class': metrics_per_class
        }
