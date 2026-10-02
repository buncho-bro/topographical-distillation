import os
import torch
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import shutil
import json

# --- 設定 ---
STUDENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEXT_LORA_DIR = os.path.join(BASE_DIR, "qwen_text_distilled_lora")
ISOLATED_LORA_DIR = os.path.join(BASE_DIR, "qwen_isolated_distilled_lora")

# 手術後の半身LoRAの保存先
HALF_TEXT_DIR = os.path.join(BASE_DIR, "qwen_half_text_lora")
HALF_ISOLATED_DIR = os.path.join(BASE_DIR, "qwen_half_isolated_lora")

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

def surgical_amputation(src_dir, dest_dir, layers_to_zero):
    """指定した層のLoRAパラメータを完全にゼロに切断（無効化）する外科手術"""
    if not os.path.exists(src_dir):
        return False
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # config.json をコピー
    shutil.copy(os.path.join(src_dir, "adapter_config.json"), os.path.join(dest_dir, "adapter_config.json"))
    
    # safetensors の読み込み
    weights_path = os.path.join(src_dir, "adapter_model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(src_dir, "adapter_model.bin") # 古い形式のフォールバック
        state_dict = torch.load(weights_path)
    else:
        state_dict = load_file(weights_path)
    
    new_state_dict = {}
    cut_count = 0
    keep_count = 0
    
    for key, tensor in state_dict.items():
        # キーから層番号を抽出 (例: base_model.model.model.layers.5.self_attn...)
        layer_num = -1
        for part in key.split('.'):
            if part.isdigit():
                layer_num = int(part)
                break
                
        if layer_num in layers_to_zero:
            # この層は切断する（ゼロ埋め）
            new_state_dict[key] = torch.zeros_like(tensor)
            cut_count += 1
        else:
            # この層は生かす
            new_state_dict[key] = tensor
            keep_count += 1
            
    # 新しい体を保存
    save_file(new_state_dict, os.path.join(dest_dir, "adapter_model.safetensors"))
    print(f"  -> Surgery completed for {os.path.basename(dest_dir)}: Cut {cut_count} tensors, Kept {keep_count} tensors.")
    return True

def step11_frankenstein_surgery():
    print("==================================================")
    print("Step 11: Frankenstein Surgery (Layer-wise LoRA Composing)")
    print("==================================================")
    
    # ---------------------------------------------------------
    # 1. 外科手術フェーズ
    # ---------------------------------------------------------
    print("\n[Phase 1: Surgical Amputation of LoRAs]")
    
    # Text LoRA の上半身（14〜27）を残し、下半身（0〜13）をゼロ切断
    layers_0_to_13 = list(range(0, 14))
    print(" - Operating on Text Distilled LoRA (Keeping Upper Body 14-27)...")
    success_text = surgical_amputation(TEXT_LORA_DIR, HALF_TEXT_DIR, layers_0_to_13)
    
    # Isolated LoRA の下半身（0〜13）を残し、上半身（14〜27）をゼロ切断
    layers_14_to_27 = list(range(14, 28))
    print(" - Operating on Isolated Concept LoRA (Keeping Lower Body 0-13)...")
    success_iso = surgical_amputation(ISOLATED_LORA_DIR, HALF_ISOLATED_DIR, layers_14_to_27)
    
    if not (success_text and success_iso):
        print("Error: Could not find original LoRA directories for surgery.")
        return

    # ---------------------------------------------------------
    # 2. キメラの縫合（マージ）フェーズ
    # ---------------------------------------------------------
    print("\n[Phase 2: Fusing the Frankenstein Chimera]")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    print(" - Sewing Upper Body (Text LoRA) onto the base...")
    student = PeftModel.from_pretrained(base_model, HALF_TEXT_DIR, adapter_name="upper_body")
    
    print(" - Sewing Lower Body (Isolated Concept LoRA) onto the base...")
    student.load_adapter(HALF_ISOLATED_DIR, adapter_name="lower_body")
    
    print(" - Applying High-Voltage (Merging at 100% : 100%)...")
    # 不要な部分はゼロになっているため、1.0 : 1.0 の全力で結合して完全体に！
    # cat を使うことで r=32 と r=128 でも安全に結合可能
    student.add_weighted_adapter(
        adapters=["upper_body", "lower_body"],
        weights=[1.0, 1.0],
        adapter_name="frankenstein",
        combination_type="cat"
    )
    student.set_adapter("frankenstein")
    
    # ---------------------------------------------------------
    # 3. 運命の起動実験（ベンチマーク）
    # ---------------------------------------------------------
    print("\n[Phase 3: Awakening the Monster]")
    res_a, res_h = run_benchmark(student, tokenizer, "Frankenstein Chimera (Layer-Split)")
    
    print("\n==================================================")
    print("           FRANKENSTEIN CHIMERA RESULTS             ")
    print("==================================================")
    print(f"{'Model':<45} | {'Analogy':<15} | {'Hierarchy':<15}")
    print("-" * 80)
    print(f"{'Teacher_7B (Genius)':<45} | {'0.6644':<15} | {'0.2148':<15}")
    print(f"{'Text LoRA (Upper Body / Output)':<45} | {'0.8228':<15} | {'0.0771':<15}")
    print(f"{'Isolated LoRA (Lower Body / Concept)':<45} | {'0.6433':<15} | {'0.2323':<15}")
    print(f"{'Simple Hybrid (50:50 Fade)':<45} | {'0.7443':<15} | {'0.1462':<15}")
    print(f"{'Frankenstein (Layer-Split Surgery)':<45} | {res_a:<15.4f} | {res_h:<15.4f}")
    print("==================================================")

if __name__ == "__main__":
    step11_frankenstein_surgery()
