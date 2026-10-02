import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- 設定 ---
TEACHER_MODEL_ID = "Qwen/Qwen2.5-7B"
STUDENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ベンチマークテストセット（Step 4と同じ） ---
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

def get_word_embedding(model, tokenizer, word):
    """単語ベクトルを取得（すげ替えた辞書を通過した『Logits空間』での分布）"""
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # 単語の最後のトークンが出力する15万次元の確率分布（Logits）を「単語の意味ベクトル」として使う
    logits = outputs.logits[0, -1, :] 
    return F.normalize(logits.to(torch.float32), p=2, dim=0)

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

def mathematical_topology_injection():
    print("==================================================")
    print("Step 5: Zero-shot Mathematical Topology Injection")
    print("==================================================")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print("\n1. Loading Teacher's Dictionary (lm_head) in 4-bit...")
    # ※ 本来はパラメータだけをロードすればVRAMを節約できますが、今回はHuggingFaceの構造上モデルごとロードします
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    # dequantize (4-bit -> float32) して数学的計算のための純粋な重み行列を取り出す
    print("   Extracting and dequantizing Teacher's topology parameters...")
    # bitsandbytesのLinear4bitから重みを復元（※近似的な抽出）
    # 注: 厳密にはFP16のモデルをロードした方が正確ですが、VRAM節約のため4bitから無理やり抽出します
    teacher_weight = teacher.lm_head.weight.data.clone().to(torch.float32)
    del teacher
    torch.cuda.empty_cache()
    
    print("\n2. Loading Student Model (1.5B)...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    student = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    student_weight = student.lm_head.weight.data.clone().to(torch.float32)
    student_dim = student_weight.shape[1] # 1536
    
    print("\n3. Performing Mathematical Topology Compression (SVD)...")
    print(f"   Compressing Teacher's shape from 4096 to {student_dim} dimensions...")
    # 特異値分解 (SVD) を用いて、Teacherの4096次元の地形を、重要な情報（階層）を残したまま1536次元に圧縮
    U, S, V = torch.svd_lowrank(teacher_weight, q=student_dim)
    compressed_teacher_weight = torch.matmul(U, torch.diag(S)) # [Vocab, 1536]
    
    print("\n4. Performing Orthogonal Procrustes Alignment (Spatial Rotation)...")
    print("   Rotating the compressed Teacher topology to perfectly match the Student's axis...")
    
    # 7Bと1.5Bで語彙サイズが微妙に違うため、共通する語彙数に切り詰める
    min_vocab = min(compressed_teacher_weight.shape[0], student_weight.shape[0])
    T_sub = compressed_teacher_weight[:min_vocab, :]
    S_sub = student_weight[:min_vocab, :]
    
    # プロクラステス解析：共通語彙を用いて、空間の軸が最もピッタリ重なる「回転行列 R」を計算
    M = torch.matmul(T_sub.T, S_sub)
    U_p, _, V_p = torch.svd(M)
    R = torch.matmul(U_p, V_p.T) # 回転行列 [1536, 1536]
    
    # Teacherの地形を回転させ、Studentの語彙サイズに合わせてカットする
    aligned_teacher_weight = torch.matmul(compressed_teacher_weight[:student_weight.shape[0], :], R)
    
    print("\n5. Injecting the Mathematical Topology into the Student...")
    # 学習（LoRA等）を一切行わず、Studentの出力辞書を、この生成した完璧な地形に物理的にすげ替える
    # (※ bitsandbytesの4bit層の重みは直接上書きできないため、出力層を標準のLinear層に置換します)
    new_lm_head = nn.Linear(student_dim, student.config.vocab_size, bias=False).to(device)
    # 脳本体の計算型（BFloat16）に合わせる
    new_lm_head.weight.data = aligned_teacher_weight.to(torch.bfloat16)
    student.lm_head = new_lm_head
    
    print("\nInjection Complete! Evaluating the Chimera Model (No Training Done)...")
    
    # 評価実行
    results = {}
    
    # ベースの評価（すげ替え前と同等の結果になるか確認するため、本来は比較用にもう一つベースが必要ですが、
    # 以前のデータ Analogy:0.7715, Hierarchy:0.0937 と比較します）
    res_analogy, res_hierarchy = run_benchmark(student, tokenizer, "Zero-shot Injected Chimera (1.5B + 7B Topology)")
    
    print("\n==================================================")
    print("              ZERO-SHOT BENCHMARK RESULTS           ")
    print("==================================================")
    print(f"{'Model':<40} | {'Analogy':<15} | {'Hierarchy':<15}")
    print("-" * 75)
    print(f"{'Base_Student_1.5B (Previous)':<40} | {'0.7715':<15} | {'0.0937':<15}")
    print(f"{'Zero-shot Injected Chimera':<40} | {res_analogy:<15.4f} | {res_hierarchy:<15.4f}")
    print("==================================================")
    print("If successful, Analogy remains high (Student's reasoning) and Hierarchy improves (Teacher's knowledge)!")

if __name__ == "__main__":
    mathematical_topology_injection()
