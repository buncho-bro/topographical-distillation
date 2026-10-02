import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- 設定 ---
STUDENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_LORA_DIR = os.path.join(BASE_DIR, "qwen_text_distilled_lora")
ISOLATED_LORA_DIR = os.path.join(BASE_DIR, "qwen_isolated_distilled_lora")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

analogy_tests = [("王様", "男", "女", "女王"), ("東京", "日本", "フランス", "パリ"), ("医者", "病院", "学校", "教師"), ("車", "道路", "線路", "電車"), ("昼", "太陽", "夜", "月")]
hierarchy_tests = [("動物", "犬", "机"), ("科学", "物理学", "りんご"), ("感情", "怒り", "石"), ("コンピュータ", "ソフトウェア", "自然"), ("宇宙", "銀河", "政治")]

def get_word_embedding(model, tokenizer, word):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

def run_benchmark(model, tokenizer, model_name):
    print(f"\n--- Testing {model_name} ---")
    model.eval()
    
    analogy_scores = []
    for a, b, c, d in analogy_tests:
        va, vb, vc, vd = [get_word_embedding(model, tokenizer, w) for w in (a, b, c, d)]
        v_pred = F.normalize(va - vb + vc, p=2, dim=0)
        score = F.cosine_similarity(v_pred.unsqueeze(0), vd.unsqueeze(0)).item()
        analogy_scores.append(score)
    avg_analogy = np.mean(analogy_scores)
    
    hierarchy_scores = []
    for base, target, outlier in hierarchy_tests:
        v_base, v_tgt, v_out = [get_word_embedding(model, tokenizer, w) for w in (base, target, outlier)]
        sim_tgt = F.cosine_similarity(v_base.unsqueeze(0), v_tgt.unsqueeze(0)).item()
        sim_out = F.cosine_similarity(v_base.unsqueeze(0), v_out.unsqueeze(0)).item()
        score = sim_tgt - sim_out
        hierarchy_scores.append(score)
    avg_hierarchy = np.mean(hierarchy_scores)
    
    return avg_analogy, avg_hierarchy

def step10_hybrid_chimera():
    print("==================================================")
    print("Step 10: Hybrid Chimera (LoRA Merging)")
    print("==================================================")
    
    if not os.path.exists(TEXT_LORA_DIR) or not os.path.exists(ISOLATED_LORA_DIR):
        print("Error: Required LoRA directories not found.")
        print(f"TEXT_LORA_DIR: {os.path.exists(TEXT_LORA_DIR)}")
        print(f"ISOLATED_LORA_DIR: {os.path.exists(ISOLATED_LORA_DIR)}")
        return

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"\n[Loading Base Model: {STUDENT_MODEL_ID}]")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    print("\n[Loading and Merging LoRAs]")
    # 1. テキスト蒸留LoRA（Analogy特化）をロード
    print(" - Loading Text Distilled LoRA (Analogy Part)...")
    student = PeftModel.from_pretrained(base_model, TEXT_LORA_DIR, adapter_name="text_lora")
    
    # 2. 純粋概念蒸留LoRA（Hierarchy特化）をロード
    print(" - Loading Isolated Concept LoRA (Hierarchy Part)...")
    student.load_adapter(ISOLATED_LORA_DIR, adapter_name="isolated_lora")
    
    # 3. 2つのLoRAを 50% : 50% で合体させる（キメラ生成）
    print(" - Fusing Adapters (50% Analogy + 50% Hierarchy)...")
    student.add_weighted_adapter(
        adapters=["text_lora", "isolated_lora"],
        weights=[0.5, 0.5],
        adapter_name="hybrid_chimera",
        combination_type="cat"
    )
    student.set_adapter("hybrid_chimera")
    
    # ---------------------------------------------------------
    # 運命のベンチマーク
    # ---------------------------------------------------------
    res_a, res_h = run_benchmark(student, tokenizer, "Hybrid Chimera (Text 50% + Concept 50%)")
    
    print("\n==================================================")
    print("         FINAL HYBRID CHIMERA RESULTS               ")
    print("==================================================")
    print(f"{'Model':<45} | {'Analogy':<15} | {'Hierarchy':<15}")
    print("-" * 80)
    print(f"{'Teacher_7B (Genius)':<45} | {'0.6644':<15} | {'0.2148':<15}")
    print(f"{'Text Distilled LoRA (Analogy Specialist)':<45} | {'0.8228':<15} | {'0.0771':<15}")
    print(f"{'Isolated Concept LoRA (Hierarchy Specialist)':<45} | {'0.6433':<15} | {'0.2323':<15}")
    print(f"{'Hybrid Chimera (Ours)':<45} | {res_a:<15.4f} | {res_h:<15.4f}")
    print("==================================================")

if __name__ == "__main__":
    step10_hybrid_chimera()
