import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# --- 設定 ---
TEACHER_MODEL_ID = "Qwen/Qwen2.5-7B"
STUDENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCHIVE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ARCHIVE_DIR)

# --- ベンチマークテストセット（これらは学習データから厳密に除外） ---
analogy_tests = [("王様", "男", "女", "女王"), ("東京", "日本", "フランス", "パリ"), ("医者", "病院", "学校", "教師"), ("車", "道路", "線路", "電車"), ("昼", "太陽", "夜", "月")]
hierarchy_tests = [("動物", "犬", "机"), ("科学", "物理学", "りんご"), ("感情", "怒り", "石"), ("コンピュータ", "ソフトウェア", "自然"), ("宇宙", "銀河", "政治")]

# --- 純度100%の概念アンカー（ベンチマーク単語を除外した約300語） ---
concept_words = [
    "時間", "空間", "存在", "無", "真理", "論理", "哲学", "倫理", "道徳", "正義", "自由", "平等", "平和", "戦争", "歴史", "未来", "過去", "永遠",
    "数学", "化学", "生物", "幾何学", "代数", "確率", "統計", "量子", "素粒子", "光", "重力", "磁力", "電気", "エネルギー", "質量", "速度",
    "経済", "市場", "資本", "通貨", "労働", "産業", "貿易", "企業", "社会", "文化", "法律", "国家", "民主主義", "独裁", "革命", "権利",
    "芸術", "音楽", "絵画", "彫刻", "文学", "詩", "演劇", "映画", "美", "表現", "創造", "伝統", "宗教", "信仰", "神", "霊", "魂",
    "植物", "樹木", "花", "草", "森", "海", "山", "川", "砂漠", "気候", "天気", "雨", "雪", "風", "雷", "地球", "惑星", "星",
    "機械", "道具", "エンジン", "ロボット", "回路", "通信", "金属", "鉄", "金", "銀", "銅", "水", "火", "土", "空気", "酸素", "水素",
    "喜び", "悲しみ", "恐怖", "驚き", "愛", "憎しみ", "希望", "絶望", "孤独", "幸福", "苦痛", "快楽", "記憶", "思考", "意識", "無意識",
    "鳥", "魚", "昆虫", "哺乳類", "爬虫類", "細胞", "臓器", "心臓", "脳", "血液", "病気", "健康", "治療", "薬", "毒", "栄養",
    "色", "赤", "青", "緑", "黄", "白", "黒", "形", "丸", "四角", "三角", "線", "点", "面", "立体", "数", "無限", "ゼロ",
    "親", "子", "家族", "友人", "敵", "集団", "個人", "言葉", "文字", "文法", "意味", "会話", "情報", "知識", "知恵", "データ",
    "都市", "村", "建築", "家", "橋", "塔", "服", "食料", "料理", "武器", "防具", "乗り物", "船", "飛行機", "自転車", "歩行",
    "春", "夏", "秋", "冬", "朝", "夕", "今日", "明日", "昨日", "世紀", "年代", "年齢", "若者", "老人", "子供", "大人", "男性", "女性",
    "思考", "行動", "変化", "維持", "破壊", "構築", "分析", "統合", "評価", "判断", "選択", "決定", "成功", "失敗", "挑戦", "逃避",
    "物質", "精神", "理論", "実践", "原因", "結果", "目的", "手段", "部分", "全体", "偶然", "必然", "絶対", "相対", "普遍", "特殊"
]

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

def step9_isolated_concept_distillation():
    print("==================================================")
    print("Step 9: Isolated Concept Distillation (The Final Truth)")
    print("==================================================")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # ---------------------------------------------------------
    # 1. Teacherからの純度100%地形の抽出
    # ---------------------------------------------------------
    print(f"\n[Loading Teacher Model: {TEACHER_MODEL_ID}]")
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    teacher.eval()
    
    print(f"Extracting 100% Pure Concept Topography ({len(concept_words)} words)...")
    teacher_vecs = []
    for word in concept_words:
        vec = get_word_embedding(teacher, teacher_tok, word)
        teacher_vecs.append(vec.cpu())
        
    teacher_vecs = torch.stack(teacher_vecs)
    
    del teacher
    torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # 2. Studentのロードとペアワイズ蒸留
    # ---------------------------------------------------------
    print(f"\n[Loading Student Model: {STUDENT_MODEL_ID}]")
    student_tok = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    
    # プロジェクターなしで脳本体(LoRA)の空間構造を直接書き換える
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 全アテンション層を学習
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(model, lora_config)
    
    print("\n[Starting Pairwise Topography Distillation]")
    print("Mapping pure conceptual distances without any projectors...")
    
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    epochs = 15 # 単語だけなので高速。エポックを回してじっくり彫り込む
    batch_size = 16
    
    student.train()
    for epoch in range(epochs):
        indices = np.arange(len(concept_words))
        np.random.shuffle(indices)
        total_loss = 0
        
        for step, i in enumerate(range(0, len(concept_words), batch_size)):
            batch_idx = indices[i:i+batch_size]
            batch_words = [concept_words[idx] for idx in batch_idx]
            t_vecs_batch = teacher_vecs[batch_idx].to(device)
            
            # 純粋な単語の入力
            inputs = student_tok(batch_words, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = student(**inputs, output_hidden_states=True)
            
            # 各単語のベクトル（最後のトークンを抽出）
            s_vecs_list = []
            for b_idx in range(len(batch_words)):
                # 単語のみの入力なので、平均化してベクトルとする
                # inputのマスク部分を除外して平均化
                mask = inputs.attention_mask[b_idx].unsqueeze(-1)
                hidden = outputs.hidden_states[-1][b_idx] * mask
                vec = hidden.sum(dim=0) / mask.sum()
                s_vecs_list.append(vec)
                
            s_vecs_batch = torch.stack(s_vecs_list)
            s_vecs_batch = F.normalize(s_vecs_batch.to(torch.float32), p=2, dim=1)
            t_vecs_batch = t_vecs_batch.to(torch.float32)
            
            # ペアワイズ行列の計算（プロジェクターのズルなし！）
            S_sim = torch.matmul(s_vecs_batch, s_vecs_batch.T)
            T_sim = torch.matmul(t_vecs_batch, t_vecs_batch.T)
            
            loss = F.mse_loss(S_sim, T_sim)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  -> Epoch {epoch+1}/{epochs} | Topography Loss: {total_loss/(len(concept_words)//batch_size):.5f}")

    # ---------------------------------------------------------
    # 3. 運命のベンチマーク
    # ---------------------------------------------------------
    print("\n==================================================")
    print("      FINAL BENCHMARK: THE ISOLATED TOPOGRAPHY      ")
    print("==================================================")
    res_a, res_h = run_benchmark(student, student_tok, "Isolated Concept Distilled Student")
    
    print("\n[Final Results Summary]")
    print(f"{'Model':<40} | {'Analogy':<15} | {'Hierarchy':<15}")
    print("-" * 75)
    print(f"{'Teacher_7B (Genius)':<40} | {'0.6644':<15} | {'0.2148':<15}")
    print(f"{'Base_Student_1.5B (Raw)':<40} | {'0.7715':<15} | {'0.0937':<15}")
    print(f"{'Isolated Concept Distilled (Ours)':<40} | {res_a:<15.4f} | {res_h:<15.4f}")
    print("==================================================")
    
    # 💡ハイブリッド用にLoRAパーツを保存！
    save_path = os.path.join(PROJECT_ROOT, "qwen_isolated_distilled_lora")
    student.save_pretrained(save_path)
    print(f"\n[Saved] Isolated Concept LoRA saved to: {save_path}")

if __name__ == "__main__":
    step9_isolated_concept_distillation()
