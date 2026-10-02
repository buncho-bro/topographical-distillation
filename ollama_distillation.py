import os
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
# LoRAを使用するためのライブラリ
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("Please install peft: pip install peft bitsandbytes")
    exit()

# --- 設定 ---
OLLAMA_API_URL = "http://localhost:11434/api/embed"
OLLAMA_TEACHER_MODEL = "llama3.1" # Ollamaで安定稼働するLlama 3.1 8Bモデル
STUDENT_MODEL_ID = "google/gemma-2-2b-it"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "models", "wikipedia_texts.pkl")
SAVE_DIR = os.path.join(BASE_DIR, "gemma_distilled_lora")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_teacher_embedding_from_ollama(text):
    """Ollama APIを叩いてTeacher(Llama3.1)の概念地形(ベクトル)を取得する"""
    payload = {
        "model": OLLAMA_TEACHER_MODEL,
        "input": text
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json()["embeddings"][0]
    except Exception as e:
        err_msg = response.text if 'response' in locals() else "No response"
        print(f"Ollama API Error: {e} | Detail: {err_msg}")
        return None

def run_hybrid_distillation():
    print("==================================================")
    print("Ollama Hybrid Topographical Distillation")
    print(f"Teacher : {OLLAMA_TEACHER_MODEL} (via Ollama)")
    print(f"Student : {STUDENT_MODEL_ID} (via Transformers + LoRA)")
    print("==================================================")
    
    # 1. データの準備
    print("\n[1] Loading texts...")
    with open(DATA_PATH, "rb") as f:
        texts = pickle.load(f)[:500] # 学習時間はかかるためまずは500件で実験
        
    # 2. Teacherの地形（ベクトル）をOllamaから事前抽出
    print(f"\n[2] Extracting Teacher Topography from Ollama ({OLLAMA_TEACHER_MODEL})...")
    teacher_vecs = []
    valid_texts = []
    for i, text in enumerate(texts):
        # APIの負荷を考慮して1件ずつ（必要なら並列化）
        vec = get_teacher_embedding_from_ollama(text)
        if vec:
            teacher_vecs.append(vec)
            valid_texts.append(text)
        if (i+1) % 50 == 0:
            print(f"  Extracted {i+1}/{len(texts)} concepts...")
            
    teacher_vecs = torch.tensor(teacher_vecs, dtype=torch.float32).to(device)
    teacher_vecs = F.normalize(teacher_vecs, p=2, dim=1)
    print(f"Successfully extracted {len(teacher_vecs)} concepts.")

    # 3. Student(2B)のロードとLoRAの適用
    print(f"\n[3] Loading Student Model ({STUDENT_MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit/8-bitロードを推奨（VRAM節約のため）。ここでは標準ロードでLoRAを適用
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # LoRA（Low-Rank Adaptation）の設定：脳のほんの一部だけを学習可能にする
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], # Attention層をターゲット
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(model, lora_config)
    student.print_trainable_parameters() # 学習パラメータが数%に圧縮されることを確認
    
    # 2Bの隠れ層の次元数を、Teacherの次元数に合わせるプロジェクター
    teacher_dim = teacher_vecs.shape[1]
    student_dim = student.config.hidden_size
    projector = nn.Linear(student_dim, teacher_dim).to(device)
    
    # 4. 蒸留学習（地形の同期）
    print("\n[4] Starting Distillation Training...")
    optimizer = optim.Adam(list(student.parameters()) + list(projector.parameters()), lr=1e-4)
    
    epochs = 3
    batch_size = 4 # VRAM制限を考慮して極小バッチ
    
    student.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        indices = np.arange(len(valid_texts))
        np.random.shuffle(indices)
        
        for i in range(0, len(valid_texts), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_texts = [valid_texts[idx] for idx in batch_idx]
            t_vecs = teacher_vecs[batch_idx]
            
            # Studentの推論
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            # 隠れ層の出力を得るために output_hidden_states=True を設定
            outputs = student(**inputs, output_hidden_states=True)
            
            # 最終層の平均プーリング
            s_hidden = outputs.hidden_states[-1].mean(dim=1)
            s_vecs = projector(s_hidden.to(torch.float32))
            s_vecs = F.normalize(s_vecs, p=2, dim=1)
            
            # MSE Loss (Topographical Distillation)
            loss = F.mse_loss(s_vecs, t_vecs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/(len(valid_texts)//batch_size):.4f}")
        
    print(f"Distillation finished in {time.time() - start_time:.1f} seconds.")
    
    # 5. 保存
    print("\n[5] Saving Distilled LoRA Weights...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    student.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    torch.save(projector.state_dict(), os.path.join(SAVE_DIR, "projector.pth"))
    
    print(f"Done! The distilled brain (LoRA) is saved at: {SAVE_DIR}")
    print("You can now load this LoRA adapter onto Gemma-2B to unleash the 27B-level concepts.")

if __name__ == "__main__":
    run_hybrid_distillation()
