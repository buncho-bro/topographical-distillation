import os
import time
import torch
import torch.optim as optim
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# --- 設定 ---
STUDENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "models", "wikipedia_texts.pkl")
SAVE_DIR = os.path.join(BASE_DIR, "qwen_text_distilled_lora")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step3_baseline_distill():
    print("==================================================")
    print("Step 3: Standard Text Distillation (Baseline)")
    print("==================================================")
    
    with open(DATA_PATH, "rb") as f:
        texts = pickle.load(f)[:100]
        
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
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(model, lora_config)
    student.print_trainable_parameters()
    
    print("\nStarting Standard Text-based Training...")
    # 一般的なテキスト蒸留（CausalLMの言語モデリング）なのでProjectorは不要
    optimizer = optim.Adam(student.parameters(), lr=2e-4)
    epochs = 3
    batch_size = 1 
    
    student.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        indices = np.arange(len(texts))
        np.random.shuffle(indices)
        
        for step, i in enumerate(range(0, len(texts), batch_size)):
            batch_idx = indices[i:i+batch_size]
            batch_texts = [texts[idx] for idx in batch_idx]
            
            # Causal LMとして学習（labelsを渡すことで内部でCrossEntropyLossが計算される）
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            inputs["labels"] = inputs["input_ids"].clone()
            
            outputs = student(**inputs)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (step + 1) % 10 == 0:
                print(f"  [Epoch {epoch+1}] Step {step+1}/{len(texts)//batch_size} | Text Loss: {loss.item():.4f}")
            
        print(f"=== Epoch {epoch+1}/{epochs} Completed | Average Loss: {total_loss/(len(texts)//batch_size):.4f} ===")
        
    print(f"Standard Text Distillation finished in {time.time() - start_time:.1f} seconds.")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    student.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Done! Standard distilled brain saved at: {SAVE_DIR}")

if __name__ == "__main__":
    step3_baseline_distill()
