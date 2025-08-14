import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

# 下载模型
model_path = snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="BAAI/bge-m3",
    resume_download=True,
    force_download=True
)

print(f"模型已下载到: {model_path}")