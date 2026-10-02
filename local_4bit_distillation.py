import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# --- 設定 ---
TEACHER_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "models", "wikipedia_texts.pkl")
SAVE_DIR = os.path.join(BASE_DIR, "gemma_distilled_lora")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_4bit_distillation():
    print("==================================================")
    print("Fully Local 4-bit Topographical Distillation")
    print(f"Teacher : {TEACHER_MODEL_ID} (4-bit Quantized)")
    print(f"Student : {STUDENT_MODEL_ID} (4-bit + LoRA)")
    print("==================================================")
    
    # 4-bit 量子化の設定（極限までVRAMを節約）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 1. データの準備
    print("\n[1] Loading texts...")
    with open(DATA_PATH, "rb") as f:
        texts = pickle.load(f)[:100] # メモリと速度を考慮し、まずは100件の概念で高速にテスト
        
    # 2. Teacher (Llama 3.1 8B) のロードと地形抽出
    print(f"\n[2] Loading Teacher Model ({TEACHER_MODEL_ID})...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print("Extracting Topography from Llama 3.1...")
    teacher_model.eval()
    teacher_vecs = []
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = teacher_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            outputs = teacher_model(**inputs, output_hidden_states=True)
            # Llamaの最終隠れ層の平均を取得（これが概念地形の絶対座標）
            hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
            teacher_vecs.append(hidden.cpu())
            if (i+1) % 10 == 0:
                print(f"  Extracted {i+1}/{len(texts)} concepts...")
                
    teacher_vecs = torch.stack(teacher_vecs).to(device)
    teacher_vecs = F.normalize(teacher_vecs, p=2, dim=1)
    print("Teacher Topography extraction complete.")
    
    # メモリ解放（Teacherの役目は終わったため消去してVRAMを空ける）
    del teacher_model
    torch.cuda.empty_cache()

    # 3. Student (Gemma 2B) のロードとLoRA設定
    print(f"\n[3] Loading Student Model ({STUDENT_MODEL_ID})...")
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    student_tokenizer.pad_token = student_tokenizer.eos_token
    
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(student_model, lora_config)
    # VRAMを極限まで節約するための勾配チェックポイントの有効化
    student_model.gradient_checkpointing_enable()
    student.print_trainable_parameters()
    
    # プロジェクター（1.5Bの次元を7Bの次元に合わせる）
    teacher_dim = teacher_vecs.shape[1] 
    student_dim = student.config.hidden_size 
    projector = nn.Linear(student_dim, teacher_dim).to(device)
    
    # 4. 蒸留学習（地形の同期）
    print("\n[4] Starting Distillation Training...")
    optimizer = optim.Adam(list(student.parameters()) + list(projector.parameters()), lr=2e-4)
    epochs = 3
    batch_size = 1 # VRAMが溢れないように極小の1に設定
    
    student.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        indices = np.arange(len(texts))
        np.random.shuffle(indices)
        
        for step, i in enumerate(range(0, len(texts), batch_size)):
            batch_idx = indices[i:i+batch_size]
            batch_texts = [texts[idx] for idx in batch_idx]
            t_vecs = teacher_vecs[batch_idx]
            
            inputs = student_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = student(**inputs, output_hidden_states=True)
            
            s_hidden = outputs.hidden_states[-1].mean(dim=1)
            s_vecs = projector(s_hidden.to(torch.float32))
            s_vecs = F.normalize(s_vecs, p=2, dim=1)
            
            loss = F.mse_loss(s_vecs, t_vecs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 学習の進捗（フリーズしていないか）を画面に出す
            if (step + 1) % 10 == 0:
                print(f"  [Epoch {epoch+1}] Step {step+1}/{len(texts)//batch_size} | Current Loss: {loss.item():.4f}")
            
        print(f"=== Epoch {epoch+1}/{epochs} Completed | Average Loss: {total_loss/(len(texts)//batch_size):.4f} ===")
        
    print(f"Distillation finished in {time.time() - start_time:.1f} seconds.")
    
    # 5. 保存
    print("\n[5] Saving Distilled LoRA Weights...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    student.save_pretrained(SAVE_DIR)
    student_tokenizer.save_pretrained(SAVE_DIR)
    torch.save(projector.state_dict(), os.path.join(SAVE_DIR, "projector.pth"))
    print(f"Done! The distilled brain (LoRA) is saved at: {SAVE_DIR}")

if __name__ == "__main__":
    run_4bit_distillation()
