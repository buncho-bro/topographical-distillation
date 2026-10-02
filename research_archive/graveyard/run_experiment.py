import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import pickle

# --- 設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "models", "wikipedia_texts.pkl")
SAVE_MODEL_DIR = os.path.join(BASE_DIR, "distilled_student_model")
RESULT_PATH = os.path.join(BASE_DIR, "experiment_results.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ベンチマーク用データ（概念の距離感テスト） ---
# 人間の直感的な類似度（0.0〜1.0）
benchmark_pairs = [
    ("犬", "猫", 0.8),
    ("自動車", "トラック", 0.9),
    ("宇宙", "銀河", 0.85),
    ("コンピュータ", "ソフトウェア", 0.7),
    ("りんご", "みかん", 0.75),
    ("犬", "自動車", 0.1),
    ("りんご", "宇宙", 0.05),
    ("政治", "経済", 0.6),
    ("日本", "アメリカ", 0.4),
    ("音楽", "美術", 0.7)
]

def evaluate_model(model, tokenizer, is_sentence_transformer=False):
    """モデルの概念理解度（スピアマン相関係数）を計測"""
    model.eval()
    similarities = []
    human_scores = []
    
    with torch.no_grad():
        for word1, word2, score in benchmark_pairs:
            if is_sentence_transformer:
                vec1 = model.encode([word1])[0]
                vec2 = model.encode([word2])[0]
            else:
                # Student(rinna)のエンコード（最終層の平均プーリング）
                inputs1 = tokenizer(word1, return_tensors="pt").to(device)
                inputs2 = tokenizer(word2, return_tensors="pt").to(device)
                out1 = model(**inputs1).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                out2 = model(**inputs2).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                vec1, vec2 = out1, out2
                
            # コサイン類似度
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append(sim)
            human_scores.append(score)
            
    # 人間の感覚との相関（高いほど概念を正しく理解している）
    correlation, _ = spearmanr(similarities, human_scores)
    return correlation

def run_experiment():
    print("==================================================")
    print("Topographical Distillation Experiment")
    print("==================================================")
    
    results = {}
    
    # 1. モデルのロード
    print("[1] Loading Models...")
    print("Teacher: paraphrase-multilingual-MiniLM-L12-v2 (Genius Brain)")
    teacher = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    print("Student: rinna/japanese-gpt2-xsmall (Minimum AI, ~40M params)")
    student_name = "rinna/japanese-gpt2-xsmall"
    tokenizer = AutoTokenizer.from_pretrained(student_name)
    tokenizer.pad_token = tokenizer.eos_token
    student = AutoModel.from_pretrained(student_name).to(device)
    
    # Studentの出力をTeacherの次元(384)に合わせるためのProjection Layer
    # ※ GPT2-xsmallの隠れ層は通常1024次元（または768）-> 今回はrinna xsmallなので768次元
    hidden_size = student.config.hidden_size
    projector = nn.Linear(hidden_size, 384).to(device)
    
    # 2. 蒸留前のベンチマークテスト
    print("\n[2] Running Benchmarks (Pre-Distillation)...")
    teacher_score = evaluate_model(teacher, None, is_sentence_transformer=True)
    student_pre_score = evaluate_model(student, tokenizer)
    
    print(f"Teacher Score (Internet standard) : {teacher_score:.4f}")
    print(f"Student Pre-Distillation Score    : {student_pre_score:.4f}")
    
    results["Teacher_Score"] = teacher_score
    results["Student_Pre_Score"] = student_pre_score
    
    # 3. データの準備
    print("\n[3] Preparing Training Data...")
    with open(DATA_PATH, "rb") as f:
        texts = pickle.load(f)[:1000] # 高速化のため1000件で蒸留
        
    print("Pre-computing Teacher Concepts (Topography)...")
    # 教師モデルの概念ベクトル（絶対座標）
    teacher_vecs = torch.tensor(teacher.encode(texts, show_progress_bar=True), dtype=torch.float32).to(device)
    teacher_vecs = F.normalize(teacher_vecs, p=2, dim=1)
    
    # 4. 概念蒸留の実行
    print("\n[4] Starting Conceptual Distillation...")
    optimizer = optim.Adam(list(student.parameters()) + list(projector.parameters()), lr=5e-5)
    epochs = 5
    batch_size = 16 # VRAM消費を抑えるため小さめ
    
    student.train()
    projector.train()
    
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        indices = np.arange(len(texts))
        np.random.shuffle(indices)
        
        for i in range(0, len(texts), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_texts = [texts[idx] for idx in batch_idx]
            
            # Teacher vectors
            t_vecs = teacher_vecs[batch_idx]
            
            # Student forward
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            outputs = student(**inputs)
            
            # 最終層の平均（Sentence Embeddingの代わり）
            s_hidden = outputs.last_hidden_state.mean(dim=1)
            s_vecs = projector(s_hidden)
            s_vecs = F.normalize(s_vecs, p=2, dim=1)
            
            # MSE Loss (Topographical alignment)
            loss = F.mse_loss(s_vecs, t_vecs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Distillation Loss: {total_loss/(len(texts)//batch_size):.4f}")
        
    print(f"Distillation completed in {time.time() - start_time:.1f} seconds.")
    
    # 5. 蒸留後のベンチマークテスト
    print("\n[5] Running Benchmarks (Post-Distillation)...")
    student.eval()
    projector.eval()
    
    # 蒸留後モデル用の評価関数（プロジェクションを通す）
    def evaluate_distilled():
        similarities = []
        human_scores = []
        with torch.no_grad():
            for word1, word2, score in benchmark_pairs:
                i1 = tokenizer(word1, return_tensors="pt").to(device)
                i2 = tokenizer(word2, return_tensors="pt").to(device)
                h1 = student(**i1).last_hidden_state.mean(dim=1)
                h2 = student(**i2).last_hidden_state.mean(dim=1)
                v1 = projector(h1).squeeze().cpu().numpy()
                v2 = projector(h2).squeeze().cpu().numpy()
                
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                similarities.append(sim)
                human_scores.append(score)
        corr, _ = spearmanr(similarities, human_scores)
        return corr
        
    student_post_score = evaluate_distilled()
    print(f"Student Post-Distillation Score   : {student_post_score:.4f}")
    results["Student_Post_Score"] = student_post_score
    
    # 6. 成果物の保存
    print("\n[6] Saving Distilled Model & Results...")
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    student.save_pretrained(SAVE_MODEL_DIR)
    tokenizer.save_pretrained(SAVE_MODEL_DIR)
    torch.save(projector.state_dict(), os.path.join(SAVE_MODEL_DIR, "projector.pth"))
    
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Artifacts saved to: {BASE_DIR}")
    print("Experiment Finished Successfully.")

if __name__ == "__main__":
    run_experiment()
