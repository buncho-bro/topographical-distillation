import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- 設定 ---
STUDENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "target_texts.pkl")
INPUT_VEC_PATH = os.path.join(BASE_DIR, "extracted_teacher_topography.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ベンチマークテストセット ---
analogy_tests = [
    ("王様", "男", "女", "女王"),
    ("東京", "日本", "フランス", "パリ"),
    ("医者", "病院", "学校", "教師"),
    ("車", "道路", "線路", "電車"),
    ("昼", "太陽", "夜", "月"),
]

hierarchy_tests = [
    ("動物", "犬", "机"),
    ("科学", "物理学", "りんご"),
    ("感情", "怒り", "石"),
    ("コンピュータ", "ソフトウェア", "自然"),
    ("宇宙", "銀河", "政治"),
]

def find_target_token_indices(input_ids, target_ids):
    input_ids = input_ids.tolist()
    target_ids = target_ids.tolist()
    n = len(target_ids)
    for i in range(len(input_ids) - n + 1):
        if input_ids[i:i+n] == target_ids:
            return list(range(i, i+n))
    return None

def get_word_embedding_with_translator(model, tokenizer, translator, word):
    """【真の評価】Studentの推論出力を、翻訳機（Translator）を通してTeacherの地形で解釈する"""
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Studentの素の出力 (1536次元)
    hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
    
    # 💡 翻訳機を通す！ (4096次元のTeacher空間へ)
    translated_hidden = translator(hidden.to(translator.weight.dtype))
    
    return F.normalize(translated_hidden.to(torch.float32), p=2, dim=0)

def run_benchmark(model, tokenizer, translator, model_name):
    print(f"\n--- Testing {model_name} ---")
    model.eval()
    translator.eval()
    
    analogy_scores = []
    for a, b, c, d in analogy_tests:
        va, vb, vc, vd = [get_word_embedding_with_translator(model, tokenizer, translator, w) for w in (a, b, c, d)]
        v_pred = F.normalize(va - vb + vc, p=2, dim=0)
        score = F.cosine_similarity(v_pred.unsqueeze(0), vd.unsqueeze(0)).item()
        analogy_scores.append(score)
    avg_analogy = np.mean(analogy_scores)
    
    hierarchy_scores = []
    for base, target, outlier in hierarchy_tests:
        v_base, v_tgt, v_out = [get_word_embedding_with_translator(model, tokenizer, translator, w) for w in (base, target, outlier)]
        sim_tgt = F.cosine_similarity(v_base.unsqueeze(0), v_tgt.unsqueeze(0)).item()
        sim_out = F.cosine_similarity(v_base.unsqueeze(0), v_out.unsqueeze(0)).item()
        score = sim_tgt - sim_out
        hierarchy_scores.append(score)
    avg_hierarchy = np.mean(hierarchy_scores)
    
    return avg_analogy, avg_hierarchy

def step6_translator_injection():
    print("==================================================")
    print("Step 6: Translator Injection (Frozen Student)")
    print("==================================================")
    
    if not os.path.exists(INPUT_VEC_PATH):
        print(f"Error: Could not find {INPUT_VEC_PATH}.")
        return

    with open(DATA_PATH, "rb") as f:
        dataset = pickle.load(f)
        
    print("Loading pre-computed Teacher Topography from disk...")
    teacher_vecs = torch.load(INPUT_VEC_PATH).to(device)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"Loading Student Model ({STUDENT_MODEL_ID}) in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    # 💡 Student本体は完全にFrozen（凍結・無学習）
    student = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    student.eval()
    
    teacher_dim = teacher_vecs.shape[1] # 4096
    student_dim = student.config.hidden_size # 1536
    
    # 💡 翻訳機（超小型プロジェクター）の準備。これ"だけ"を学習させる。
    translator = nn.Linear(student_dim, teacher_dim).to(device)
    translator = translator.to(torch.bfloat16) # 計算型合わせ
    
    print("\nTraining Translator (Student and Teacher Topographies are FROZEN)...")
    optimizer = optim.Adam(translator.parameters(), lr=1e-3)
    epochs = 3
    batch_size = 16
    
    start_time = time.time()
    for epoch in range(epochs):
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        total_loss = 0
        
        for step, i in enumerate(range(0, len(dataset), batch_size)):
            batch_idx = indices[i:i+batch_size]
            batch_data = [dataset[idx] for idx in batch_idx]
            t_vecs_batch = teacher_vecs[batch_idx].to(device).to(torch.bfloat16)
            
            batch_texts = [d["text"] for d in batch_data]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            
            # Studentは計算のみ（勾配不要）
            with torch.no_grad():
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
            
            # 💡 Studentの出力を翻訳機に通し、Teacherの座標（絶対座標）と合わせる
            translated_batch = translator(s_hidden_batch)
            translated_batch = F.normalize(translated_batch, p=2, dim=1)
            
            # 翻訳機の誤差（文字化け度合い）
            loss = F.mse_loss(translated_batch, t_vecs_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"=== Epoch {epoch+1}/{epochs} | Avg Translator Loss: {total_loss/(len(dataset)//batch_size):.5f} ===")
        
    print(f"Translation Dictionary Trained in {time.time() - start_time:.1f} seconds.")
    
    # 運命のベンチマークテスト
    res_analogy, res_hierarchy = run_benchmark(student, tokenizer, translator, "Translated Chimera (Frozen 1.5B + Teacher Topology)")
    
    print("\n==================================================")
    print("              FINAL BENCHMARK RESULTS               ")
    print("==================================================")
    print(f"{'Model':<40} | {'Analogy':<15} | {'Hierarchy':<15}")
    print("-" * 75)
    print(f"{'Teacher_7B (Genius)':<40} | {'0.6644':<15} | {'0.2148':<15}")
    print(f"{'Base_Student_1.5B (Raw)':<40} | {'0.7715':<15} | {'0.0937':<15}")
    print(f"{'Translated Chimera (Ours)':<40} | {res_analogy:<15.4f} | {res_hierarchy:<15.4f}")
    print("==================================================")

if __name__ == "__main__":
    step6_translator_injection()
