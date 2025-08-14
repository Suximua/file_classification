import os
import argparse

def download_model():
    """下载m3e-base模型到本地"""
    from transformers import AutoModel, AutoTokenizer
    
    model_name = "moka-ai/m3e-base"
    save_path = "models/m3e-base"
    
    if not os.path.exists(save_path):
        print(f"正在下载模型 {model_name} 到 {save_path}...")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("模型下载完成!")
    else:
        print("模型已存在，跳过下载")

def run_route1():
    """运行无需训练的方法"""
    print("\n=== 运行路线1: 语义相似度匹配 ===")
    from route1.similarity import main as similarity_main
    similarity_main()

def run_route2_finetune():
    """运行Finetune方法"""
    print("\n=== 运行路线2: Finetune方法 ===")
    from route2.finetune.finetune import main as finetune_main
    from route2.finetune.predict import main as predict_main
    
    print("开始训练模型...")
    finetune_main()
    print("开始预测和评估...")
    predict_main()

def run_route2_traditional():
    """运行传统机器学习方法"""
    print("\n=== 运行路线2: 传统机器学习 ===")
    from route2.traditional.train import main as train_main
    from route2.traditional.predict import main as predict_main
    
    print("开始训练模型...")
    train_main()
    print("开始预测和评估...")
    predict_main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="文本分类实验")
    parser.add_argument("--route", type=int, choices=[1, 2], help="选择运行路线 (1或2)")
    parser.add_argument("--method", type=str, choices=["finetune", "traditional"], 
                       help="路线2的方法选择 (finetune或traditional)")
    parser.add_argument("--download", action="store_true", help="下载模型")
    
    args = parser.parse_args()
    
    if args.download:
        download_model()
    
    if args.route == 1:
        run_route1()
    elif args.route == 2:
        if args.method == "finetune":
            run_route2_finetune()
        elif args.method == "traditional":
            run_route2_traditional()
        else:
            print("请指定路线2的方法: --method finetune 或 --method traditional")
    else:
        print("请指定运行路线: --route 1 或 --route 2")
