from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def calculate_metrics(true_labels, predicted_labels, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=class_names, average=None
    )
    
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
    
    # 计算每个类别的TP, FP, FN, TN
    metrics_per_class = {}
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        metrics_per_class[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn
        }
    
    return {
        'precision': dict(zip(class_names, precision)),
        'recall': dict(zip(class_names, recall)),
        'f1': dict(zip(class_names, f1)),
        'confusion_matrix': cm,
        'per_class': metrics_per_class
    }
