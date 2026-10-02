import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- 設定（同一次元でのキメラ検証） ---
# ベースモデルをTeacher（階層理解担当）、InstructモデルをStudent（類推担当）とする
TEACHER_MODEL_ID = "Qwen/Qwen2.5-1.5B"
STUDENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "target_texts.pkl")
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
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
    
    if translator is not None:
        hidden = translator(hidden.to(translator.weight.dtype))
        
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

def run_benchmark(model, tokenizer, model_name, translator=None):
    model.eval()
    if translator:
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

def step7_same_dim_chimera():
    print("==================================================")
    print("Step 7: Same-Dimension Translated Chimera Test")
    print("==================================================")
    
    if not os.path.exists(DATA_PATH):
        print("Error: Please run step0_generate_target_data.py first.")
        return

    with open(DATA_PATH, "rb") as f:
        dataset = pickle.load(f)
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    results = {}
    
    # ---------------------------------------------------------
    # 1. Teacher (1.5B Base) からの抽出とベンチマーク
    # ---------------------------------------------------------
    print(f"\n[Loading Teacher: {TEACHER_MODEL_ID}]")
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    teacher.eval()
    
    res_a, res_h = run_benchmark(teacher, teacher_tok, "Teacher")
    results["Teacher_1.5B_Base"] = (res_a, res_h)
    
    print("Extracting Teacher's Topography...")
    teacher_vecs = []
    with torch.no_grad():
        for data in dataset:
            input_tokens = teacher_tok(data["text"], return_tensors="pt", add_special_tokens=True).input_ids[0].to(device)
            target_tokens = teacher_tok(data["target"], return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
            indices = find_target_token_indices(input_tokens, target_tokens)
            
            inputs = {"input_ids": input_tokens.unsqueeze(0)}
            outputs = teacher(**inputs, output_hidden_states=True)
            
            if indices is None:
                vec = outputs.hidden_states[-1][0].mean(dim=0)
            else:
                vec = outputs.hidden_states[-1][0, indices, :].mean(dim=0)
            teacher_vecs.append(vec.cpu())
            
    teacher_vecs = torch.stack(teacher_vecs)
    teacher_vecs = F.normalize(teacher_vecs, p=2, dim=1)
    
    del teacher
    torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # 2. Student (1.5B Instruct) の準備とベンチマーク
    # ---------------------------------------------------------
    print(f"\n[Loading Student: {STUDENT_MODEL_ID}]")
    student_tok = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    student = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    student.eval()
    
    res_a, res_h = run_benchmark(student, student_tok, "Student")
    results["Student_1.5B_Instruct"] = (res_a, res_h)
    
    # ---------------------------------------------------------
    # 3. 翻訳機 (1536 -> 1536) の学習 (Translated Chimera)
    # ---------------------------------------------------------
    hidden_dim = student.config.hidden_size # 1536
    translator = nn.Linear(hidden_dim, hidden_dim).to(device).to(torch.bfloat16)
    
    print("\n[Training Translator (1536 -> 1536) ...]")
    optimizer = optim.Adam(translator.parameters(), lr=1e-3)
    epochs = 3
    batch_size = 16
    
    for epoch in range(epochs):
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        total_loss = 0
        for step, i in enumerate(range(0, len(dataset), batch_size)):
            batch_idx = indices[i:i+batch_size]
            batch_data = [dataset[idx] for idx in batch_idx]
            t_vecs_batch = teacher_vecs[batch_idx].to(device).to(torch.bfloat16)
            
            batch_texts = [d["text"] for d in batch_data]
            inputs = student_tok(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            
            with torch.no_grad():
                outputs = student(**inputs, output_hidden_states=True)
            
            s_vecs_list = []
            for b_idx in range(len(batch_data)):
                target_str = batch_data[b_idx]["target"]
                target_tokens = student_tok(target_str, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
                input_ids = inputs.input_ids[b_idx]
                target_indices = find_target_token_indices(input_ids, target_tokens)
                
                if target_indices is None:
                    vec = outputs.hidden_states[-1][b_idx].mean(dim=0)
                else:
                    vec = outputs.hidden_states[-1][b_idx, target_indices, :].mean(dim=0)
                s_vecs_list.append(vec)
                
            s_hidden_batch = torch.stack(s_vecs_list).to(torch.bfloat16)
            
            translated_batch = translator(s_hidden_batch)
            translated_batch = F.normalize(translated_batch, p=2, dim=1)
            loss = F.mse_loss(translated_batch, t_vecs_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  -> Epoch {epoch+1} Loss: {total_loss/(len(dataset)//batch_size):.5f}")
        
    # ---------------------------------------------------------
    # 4. Translated Chimera のベンチマーク
    # ---------------------------------------------------------
    print("\n[Evaluating Same-Dimension Translated Chimera]")
    res_a, res_h = run_benchmark(student, student_tok, "Chimera", translator)
    results["Translated_Chimera_(Same_Dim)"] = (res_a, res_h)
    
    # ---------------------------------------------------------
    # 最終結果
    # ---------------------------------------------------------
    print("\n==================================================")
    print("       SAME-DIMENSION CHIMERA TEST RESULTS          ")
    print("==================================================")
    print(f"{'Model':<35} | {'Analogy':<15} | {'Hierarchy':<15}")
    print("-" * 70)
    for name, (a, h) in results.items():
        print(f"{name:<35} | {a:<15.4f} | {h:<15.4f}")
    print("==================================================")

if __name__ == "__main__":
    step7_same_dim_chimera()
