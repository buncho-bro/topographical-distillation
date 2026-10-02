import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- 設定 ---
TEACHER_MODEL_ID = "Qwen/Qwen2.5-7B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_word_embedding_isolated(model, tokenizer, word):
    """単語単体を入力（過去のStep4と同じ方法）"""
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

def get_word_embedding_contextual(model, tokenizer, word):
    """文脈に埋め込んで、その位置からピンポイント抽出（今のStep1の抽出方法）"""
    # 意図的に自然な文脈を作る
    context_text = f"一般的に言って、{word}は非常に重要な概念であると考えられる。"
    
    input_tokens = tokenizer(context_text, return_tensors="pt", add_special_tokens=True).input_ids[0].to(device)
    target_tokens = tokenizer(word, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    
    indices = find_target_token_indices(input_tokens, target_tokens)
    
    inputs = {"input_ids": input_tokens.unsqueeze(0)}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    if indices is None:
        # トークナイズ都合で見つからない場合はフォールバック
        hidden = outputs.hidden_states[-1][0].mean(dim=0)
    else:
        # 💡文脈の中の、ターゲット単語の位置だけを抽出！
        hidden = outputs.hidden_states[-1][0, indices, :].mean(dim=0)
        
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

def run_benchmark(model, tokenizer, extraction_mode):
    print(f"--- Running Benchmark ({extraction_mode}) ---")
    
    get_embed_fn = get_word_embedding_isolated if extraction_mode == "Isolated" else get_word_embedding_contextual
    
    analogy_scores = []
    for a, b, c, d in analogy_tests:
        va, vb, vc, vd = [get_embed_fn(model, tokenizer, w) for w in (a, b, c, d)]
        v_pred = F.normalize(va - vb + vc, p=2, dim=0)
        score = F.cosine_similarity(v_pred.unsqueeze(0), vd.unsqueeze(0)).item()
        analogy_scores.append(score)
    avg_analogy = np.mean(analogy_scores)
    
    hierarchy_scores = []
    for base, target, outlier in hierarchy_tests:
        v_base, v_tgt, v_out = [get_embed_fn(model, tokenizer, w) for w in (base, target, outlier)]
        sim_tgt = F.cosine_similarity(v_base.unsqueeze(0), v_tgt.unsqueeze(0)).item()
        sim_out = F.cosine_similarity(v_base.unsqueeze(0), v_out.unsqueeze(0)).item()
        score = sim_tgt - sim_out
        hierarchy_scores.append(score)
    avg_hierarchy = np.mean(hierarchy_scores)
    
    return avg_analogy, avg_hierarchy

def step8_verify_teacher_extraction():
    print("==================================================")
    print("Step 8: Teacher Extraction Quality Verification")
    print("==================================================")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"\n[Loading Teacher Model: {TEACHER_MODEL_ID}]")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    model.eval()
    
    # 1. 単語単体での抽出（過去のベンチマークと同じ）
    res_a_iso, res_h_iso = run_benchmark(model, tokenizer, "Isolated")
    
    # 2. 文脈からのピンポイント抽出（今のデータ生成と同じ）
    res_a_ctx, res_h_ctx = run_benchmark(model, tokenizer, "Contextual")
    
    print("\n==================================================")
    print("         TEACHER EXTRACTION QUALITY TEST            ")
    print("==================================================")
    print(f"{'Extraction Method':<30} | {'Analogy':<15} | {'Hierarchy':<15}")
    print("-" * 65)
    print(f"{'1. Isolated (Single Word)':<30} | {res_a_iso:<15.4f} | {res_h_iso:<15.4f}")
    print(f"{'2. Contextual (Current Step1)':<30} | {res_a_ctx:<15.4f} | {res_h_ctx:<15.4f}")
    print("==================================================")
    print("If Contextual Hierarchy is significantly lower than Isolated,")
    print("our data generation (Step 1) is failing to capture the true topography.")

if __name__ == "__main__":
    step8_verify_teacher_extraction()
