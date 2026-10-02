import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- 設定 ---
TEACHER_MODEL_ID = "Qwen/Qwen2.5-7B"
STUDENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ARCHIVE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ARCHIVE_DIR)
TOPO_LORA_DIR = os.path.join(PROJECT_ROOT, "qwen_distilled_lora")
TEXT_LORA_DIR = os.path.join(PROJECT_ROOT, "qwen_text_distilled_lora")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 高度な推論ベンチマークセット ---
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

def get_word_embedding(model, tokenizer, word, projector=None):
    """単語の概念ベクトルを取得（プロジェクターがあれば通す）"""
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
    
    if projector is not None:
        hidden = projector(hidden.to(projector.weight.dtype))
        
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

def run_tests(model, tokenizer, model_name, projector=None):
    print(f"\n--- Testing {model_name} ---")
    model.eval()
    if projector is not None:
        projector.eval()
        print("  *(Evaluating with Translator/Projector)*")
    
    analogy_scores = []
    print("1. Analogy (類推) Test:")
    for a, b, c, d in analogy_tests:
        va = get_word_embedding(model, tokenizer, a, projector)
        vb = get_word_embedding(model, tokenizer, b, projector)
        vc = get_word_embedding(model, tokenizer, c, projector)
        vd = get_word_embedding(model, tokenizer, d, projector)
        
        v_pred = F.normalize(va - vb + vc, p=2, dim=0)
        score = F.cosine_similarity(v_pred.unsqueeze(0), vd.unsqueeze(0)).item()
        analogy_scores.append(score)
    
    avg_analogy = np.mean(analogy_scores)
    print(f"  -> Score: {avg_analogy:.4f}")
    
    hierarchy_scores = []
    print("2. Hierarchy (概念階層) Test:")
    for base, target, outlier in hierarchy_tests:
        v_base = get_word_embedding(model, tokenizer, base, projector)
        v_tgt = get_word_embedding(model, tokenizer, target, projector)
        v_out = get_word_embedding(model, tokenizer, outlier, projector)
        
        sim_tgt = F.cosine_similarity(v_base.unsqueeze(0), v_tgt.unsqueeze(0)).item()
        sim_out = F.cosine_similarity(v_base.unsqueeze(0), v_out.unsqueeze(0)).item()
        score = sim_tgt - sim_out
        hierarchy_scores.append(score)
        
    avg_hierarchy = np.mean(hierarchy_scores)
    print(f"  -> Score: {avg_hierarchy:.4f}")
    
    return avg_analogy, avg_hierarchy

def step4_advanced_benchmark():
    print("==================================================")
    print("Step 4: Advanced Topographical Reasoning Benchmark (Bug Fix)")
    print("==================================================")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    results = {}
    
    # 1. Teacher (7B)
    print("\n[Loading Teacher (7B)]")
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    results["Teacher_7B"] = run_tests(teacher, teacher_tok, "Teacher (7B Genius)")
    del teacher
    torch.cuda.empty_cache()

    # 2. Base Student (1.5B)
    print("\n[Loading Base Student (1.5B)]")
    student_tok = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    student = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    results["Base_Student_1.5B"] = run_tests(student, student_tok, "Base Student (1.5B Pre-Trained)")
    
    # 3. Text Distilled
    print("\n[Loading Text-Distilled Student (1.5B)]")
    if os.path.exists(TEXT_LORA_DIR):
        text_student = PeftModel.from_pretrained(student, TEXT_LORA_DIR)
        results["Text_Distilled_1.5B"] = run_tests(text_student, student_tok, "Text Distilled Student (1.5B Standard)")
        text_student.unload()
        torch.cuda.empty_cache()
    else:
        print("  -> Skipped: qwen_text_distilled_lora not found.")

    # 4. Topo Distilled (with Projector if available)
    print("\n[Loading Topo-Distilled Student (1.5B)]")
    if os.path.exists(TOPO_LORA_DIR):
        topo_student = PeftModel.from_pretrained(student, TOPO_LORA_DIR)
        
        # 💡 ここで保存されたプロジェクター（翻訳機）をロードする
        projector_path = os.path.join(TOPO_LORA_DIR, "projector.pth")
        loaded_projector = None
        if os.path.exists(projector_path):
            print("  -> Found projector.pth! Loading Translator...")
            # 💡 保存された重みファイルから直接次元数（3584等）を読み取ってモデルを構築する
            state_dict = torch.load(projector_path, map_location=device, weights_only=True)
            out_features, in_features = state_dict['weight'].shape
            
            loaded_projector = nn.Linear(in_features, out_features).to(device)
            loaded_projector.load_state_dict(state_dict)
            loaded_projector = loaded_projector.to(torch.bfloat16)
        
        results["Topo_Distilled_1.5B"] = run_tests(topo_student, student_tok, "Topo Distilled Student (1.5B Topographical)", loaded_projector)
    else:
        print("  -> Skipped: qwen_distilled_lora not found.")
        
    print("\n==================================================")
    print("              BENCHMARK RESULTS SUMMARY             ")
    print("==================================================")
    print(f"{'Model':<30} | {'Analogy (類推)':<15} | {'Hierarchy (階層)':<15}")
    print("-" * 65)
    for model_name, (analogy, hierarchy) in results.items():
        print(f"{model_name:<30} | {analogy:<15.4f} | {hierarchy:<15.4f}")
    print("==================================================")

if __name__ == "__main__":
    step4_advanced_benchmark()
