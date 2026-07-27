import os
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    util
)
from peft import LoraConfig, TaskType
from pathlib import Path
from safetensors.torch import load_file, save_file
import shutil

# 通过设置HuggingFace的token来下载模型
DIR_PATH = Path(__file__).parent
print("Starting to load model...")
# 如果指定目录不存在模型 则下载模型文件
EXIST_LOCAL_MODEL = not (os.path.isdir(f"{DIR_PATH}/model") and len(os.listdir(f"{DIR_PATH}/model")) == 0)
print(f"EXIST_LOCAL_MODEL = {EXIST_LOCAL_MODEL}")
model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', cache_folder=f"{DIR_PATH}/model",
                            local_files_only=EXIST_LOCAL_MODEL)
print("Model loaded successfully.")

ADAPTER_PATH = f"{DIR_PATH.parent}/lora/qwen3-0.6b-lora"

if os.path.isdir(ADAPTER_PATH):
    print(f"已经创建adapter目录")
    model.load_adapter(ADAPTER_PATH)
    model.set_adapter("default")
else:
    print(f"尚未创建adapter目录")

def reset_weights_name(tensor_path):
    '''
    训练时用model.add_adapter() 底层会把模型包成PeftModel 权重key带有base_model.model.前缀
    保存adapter时这个前缀被保留
    当重新加载基础模型后再调用load_adapter() SentenceTransformer内部的Transformer模块结构不同导致key对不上 
    该方法用于修改模型训练后model.save保存的adapter模型权重的key中的base_model.model.前缀 
    使其adapter文件夹能够被load_adapter方法正确的加载到模型中 在模型的推理过程中使用adapter
    '''
    tensor_path += '/adapter_model.safetensors'
    weights = load_file(tensor_path)
    new_weights = {}
    for k, v in weights.items():
        new_key = k.replace("base_model.model.", "")
        new_weights[new_key] = v
    save_file(new_weights, tensor_path)

def get_embedding(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts)
    return embeddings.tolist()

def compare_similar(text1: str, text2: str):
    embed1 = model.encode([text1])
    embed2 = model.encode([text2])
    similarity = util.cos_sim(embed1, embed2)
    return similarity

def lora_fine_tune(lora_dataset: list):
    global model
    train_data = {"query": [], "chunk": [], "label": []}
    for data_id, query, chunk, label in lora_dataset:
        train_data['query'].append(query)
        train_data["chunk"].append(chunk)
        train_data["label"].append(label)
        cur_similar = compare_similar(query, chunk)
        print(f"id:{data_id} similar = {cur_similar} label: {label}")
    
    # 如果模型没有配置adapter 则给它配置对应的adapter
    try:
        active = model.active_adapters()
        print(f"active = {active}")
    except Exception as e:
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,                   
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model.add_adapter(peft_config)
        print(f"配置adapters完成 (Lora矩阵)")
    
    train_dataset = Dataset.from_dict(train_data)

    train_loss = losses.ContrastiveLoss(
        model=model,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,  # 常用
        margin=0.5,          # 负样本至少要拉开的距离，可调
    )

    args = SentenceTransformerTrainingArguments(
    save_strategy="no",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    learning_rate=2e-4,          
    warmup_ratio=0.1,
    bf16=True,
    fp16=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss
    )

    trainer.train()

    for data_id, query, chunk, label in lora_dataset:
        cur_similar = compare_similar(query, chunk)
        print(f"id:{data_id} similar = {cur_similar} label: {label}")

    model.save_pretrained(ADAPTER_PATH)
    reset_weights_name(ADAPTER_PATH)

def remove_adapter():
    '''删除掉adapter目录'''
    shutil.rmtree(ADAPTER_PATH)


if __name__ == "__main__":
    # test1()
    lora_fine_tune()