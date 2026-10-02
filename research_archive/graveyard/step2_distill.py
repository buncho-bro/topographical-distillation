import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# --- 設定 ---
STUDENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "target_texts.pkl")
INPUT_VEC_PATH = os.path.join(BASE_DIR, "extracted_teacher_topography.pt")
SAVE_DIR = os.path.join(BASE_DIR, "qwen_distilled_lora")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_target_token_indices(input_ids, target_ids):
    input_ids = input_ids.tolist()
    target_ids = target_ids.tolist()
    n = len(target_ids)
    for i in range(len(input_ids) - n + 1):
        if input_ids[i:i+n] == target_ids:
            return list(range(i, i+n))
    return None

def step2_distill_old_route():
    print("==================================================")
    print("Step 2: Absolute Coordinate Distillation (LoRA + Projector)")
    print("==================================================")
    
    if not os.path.exists(INPUT_VEC_PATH):
        print(f"Error: Could not find {INPUT_VEC_PATH}.")
        return

    with open(DATA_PATH, "rb") as f:
        dataset = pickle.load(f)
        
    print("Loading pre-computed Teacher Topography from disk...")
    teacher_vecs = torch.load(INPUT_VEC_PATH).to(device)
    teacher_dim = teacher_vecs.shape[1] # 4096
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"Loading Student Model ({STUDENT_MODEL_ID}) in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    # 💡 1. Student本体をLoRAで学習可能にする
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(model, lora_config)
    student_dim = student.config.hidden_size # 1536
    
    # 💡 2. プロジェクター（翻訳機）も同時に学習させる
    projector = nn.Linear(student_dim, teacher_dim).to(device)
    # dtypeを揃える
    projector = projector.to(torch.bfloat16)
    
    print("\nStarting Training (Both LoRA and Projector)...")
    # LoRAのパラメータとプロジェクターのパラメータの両方をOptimizerに渡す
    optimizer = optim.Adam(list(student.parameters()) + list(projector.parameters()), lr=2e-4)
    epochs = 3
    batch_size = 8
    
    student.train()
    projector.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        
        for step, i in enumerate(range(0, len(dataset), batch_size)):
            batch_idx = indices[i:i+batch_size]
            batch_data = [dataset[idx] for idx in batch_idx]
            t_vecs_batch = teacher_vecs[batch_idx].to(device).to(torch.bfloat16)
            
            batch_texts = [d["text"] for d in batch_data]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = student(**inputs, output_hidden_states=True)
            
            s_vecs_list = []
            for b_idx in range(len(batch_data)):
                target_str = batch_data[b_idx]["target"]
                target_tokens = tokenizer(target_str, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
                
                input_ids = inputs.input_ids[b_idx]
                target_indices = find_target_token_indices(input_ids, target_tokens)
                
                if target_indices is None:
                    vec = outputs.hidden_states[-1][b_idx].mean(dim=0)
                else:
                    vec = outputs.hidden_states[-1][b_idx, target_indices, :].mean(dim=0)
                
                s_vecs_list.append(vec)
                
            s_hidden_batch = torch.stack(s_vecs_list).to(torch.bfloat16)
            
            # 💡 Studentの出力をプロジェクターに通してTeacherの次元に合わせる
            projected_batch = projector(s_hidden_batch)
            projected_batch = F.normalize(projected_batch, p=2, dim=1)
            
            # 💡 絶対座標でのMSE Loss（「ズル」が発生した元凶）
            loss = F.mse_loss(projected_batch, t_vecs_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (step + 1) % 10 == 0:
                print(f"  [Epoch {epoch+1}] Step {step+1}/{len(dataset)//batch_size} | Absolute Loss: {loss.item():.4f}")
            
        print(f"=== Epoch {epoch+1}/{epochs} Completed | Average Loss: {total_loss/(len(dataset)//batch_size):.4f} ===")
        
    print(f"Training finished in {time.time() - start_time:.1f} seconds.")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    student.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    
    # 💡 プロジェクターも保存する
    torch.save(projector.state_dict(), os.path.join(SAVE_DIR, "projector.pth"))
    
    print(f"Done! Distilled brain and projector saved at: {SAVE_DIR}")

if __name__ == "__main__":
    step2_distill_old_route()
