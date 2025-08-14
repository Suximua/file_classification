import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path)
    
    # 获取所有类别（除了"法律"）
    class_names = ['简历', '合同', '小说', '发票', '数学', 
                  '物理', '作文', '说明书', '方案', '英语', '论文']
    
    # 筛选数据，排除"法律"类别
    filtered_data = data[data['Label'].isin(class_names)]
    
    # 确保每个类别在训练集和测试集中都有代表
    train_list = []
    test_list = []
    
    for label in class_names:
        # 获取该类别的所有数据
        class_data = filtered_data[filtered_data['Label'] == label]
        if len(class_data) > 0:
            # 如果该类别数据较少，确保至少有一个样本在测试集中
            if len(class_data) <= 5:
                train_data, test_data = train_test_split(class_data, test_size=1, random_state=42)
            else:
                train_data, test_data = train_test_split(class_data, test_size=0.2, random_state=42)
            train_list.append(train_data)
            test_list.append(test_data)
    
    # 合并所有类别的数据
    train = pd.concat(train_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)
    
    # 打乱数据顺序
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return train, test
