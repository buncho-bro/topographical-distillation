import gradio as gr
import os
import gc
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import LoraConfig, get_peft_model, PeftModel
from threading import Thread

# --- グローバル状態（チャット・ベンチマーク用） ---
current_model = None
current_tokenizer = None
current_model_name = "None"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 純度100%の概念アンカー（約150語） ---
CONCEPT_WORDS = [
    "時間", "空間", "存在", "無", "真理", "論理", "哲学", "倫理", "道徳", "正義", "自由", "平等", "平和", "戦争", "歴史", "未来", "過去", "永遠",
    "数学", "化学", "生物", "幾何学", "代数", "確率", "統計", "量子", "素粒子", "光", "重力", "磁力", "電気", "エネルギー", "質量", "速度",
    "経済", "市場", "資本", "通貨", "労働", "産業", "貿易", "企業", "社会", "文化", "法律", "国家", "民主主義", "独裁", "革命", "権利",
    "芸術", "音楽", "絵画", "彫刻", "文学", "詩", "演劇", "映画", "美", "表現", "創造", "伝統", "宗教", "信仰", "神", "霊", "魂",
    "植物", "樹木", "花", "草", "森", "海", "山", "川", "砂漠", "気候", "天気", "雨", "雪", "風", "雷", "地球", "惑星", "星",
    "機械", "道具", "エンジン", "ロボット", "回路", "通信", "金属", "鉄", "金", "銀", "銅", "水", "火", "土", "空気", "酸素", "水素",
    "喜び", "悲しみ", "恐怖", "驚き", "愛", "憎しみ", "希望", "絶望", "孤独", "幸福", "苦痛", "快楽", "記憶", "思考", "意識", "無意識"
]

analogy_tests = [("王様", "男", "女", "女王"), ("東京", "日本", "フランス", "パリ"), ("医者", "病院", "学校", "教師"), ("車", "道路", "線路", "電車"), ("昼", "太陽", "夜", "月")]
hierarchy_tests = [("動物", "犬", "机"), ("科学", "物理学", "りんご"), ("感情", "怒り", "石"), ("コンピュータ", "ソフトウェア", "自然"), ("宇宙", "銀河", "政治")]

def free_vram():
    global current_model, current_tokenizer, current_model_name
    if current_model is not None:
        del current_model
        current_model = None
    if current_tokenizer is not None:
        del current_tokenizer
        current_tokenizer = None
    current_model_name = "None"
    gc.collect()
    torch.cuda.empty_cache()

def get_word_embedding(model, tokenizer, word):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

# ==========================================
# 1. 蒸留ロジック
# ==========================================
def distill_process(teacher_id, student_id, save_dir, r_value, epochs_val):
    log_text = "=== Starting Isolated Concept Distillation ===\n"
    yield log_text
    
    try:
        free_vram()
        os.makedirs(save_dir, exist_ok=True)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        log_text += f"\n[1/3] Loading Teacher ({teacher_id}) to VRAM...\n"
        yield log_text
        teacher_tok = AutoTokenizer.from_pretrained(teacher_id)
        teacher = AutoModelForCausalLM.from_pretrained(teacher_id, quantization_config=bnb_config, device_map={"": "cuda:0"})
        teacher.eval()
        
        log_text += f"Extracting 100% Pure Concept Topography ({len(CONCEPT_WORDS)} words)...\n"
        yield log_text
        teacher_vecs = []
        for word in CONCEPT_WORDS:
            vec = get_word_embedding(teacher, teacher_tok, word)
            teacher_vecs.append(vec.cpu())
        teacher_vecs = torch.stack(teacher_vecs)
        
        log_text += "Extraction Complete! Unloading Teacher from VRAM to save memory...\n"
        yield log_text
        del teacher
        del teacher_tok
        gc.collect()
        torch.cuda.empty_cache()
        
        log_text += f"\n[2/3] Loading Student ({student_id}) and setting up LoRA...\n"
        yield log_text
        student_tok = AutoTokenizer.from_pretrained(student_id)
        student_tok.pad_token = student_tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(student_id, quantization_config=bnb_config, device_map={"": "cuda:0"})
        
        lora_config = LoraConfig(
            r=int(r_value),
            lora_alpha=int(r_value)*2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        student = get_peft_model(model, lora_config)
        
        log_text += "\n[3/3] Starting Pairwise Topography Training...\n"
        yield log_text
        optimizer = optim.Adam(student.parameters(), lr=1e-4)
        batch_size = 16
        epochs = int(epochs_val)
        
        student.train()
        for epoch in range(epochs):
            indices = np.arange(len(CONCEPT_WORDS))
            np.random.shuffle(indices)
            total_loss = 0
            
            for step, i in enumerate(range(0, len(CONCEPT_WORDS), batch_size)):
                batch_idx = indices[i:i+batch_size]
                batch_words = [CONCEPT_WORDS[idx] for idx in batch_idx]
                t_vecs_batch = teacher_vecs[batch_idx].to(device)
                
                inputs = student_tok(batch_words, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = student(**inputs, output_hidden_states=True)
                
                s_vecs_list = []
                for b_idx in range(len(batch_words)):
                    mask = inputs.attention_mask[b_idx].unsqueeze(-1)
                    hidden = outputs.hidden_states[-1][b_idx] * mask
                    vec = hidden.sum(dim=0) / mask.sum()
                    s_vecs_list.append(vec)
                    
                s_vecs_batch = torch.stack(s_vecs_list)
                s_vecs_batch = F.normalize(s_vecs_batch.to(torch.float32), p=2, dim=1)
                t_vecs_batch = t_vecs_batch.to(torch.float32)
                
                S_sim = torch.matmul(s_vecs_batch, s_vecs_batch.T)
                T_sim = torch.matmul(t_vecs_batch, t_vecs_batch.T)
                
                loss = F.mse_loss(S_sim, T_sim)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            log_text += f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/(len(CONCEPT_WORDS)//batch_size):.5f}\n"
            yield log_text
            
        log_text += f"\n=== Saving LoRA to: {save_dir} ===\n"
        yield log_text
        student.save_pretrained(save_dir)
        
        log_text += "Distillation Complete! You can now load the model in the Chat/Benchmark tab.\n"
        yield log_text
        
        free_vram()
        
    except Exception as e:
        log_text += f"\nERROR: {str(e)}\n"
        yield log_text

# ==========================================
# 2. モデルロード（チャット/ベンチマーク用）
# ==========================================
def load_model_for_chat(base_id, lora_dir, use_lora):
    global current_model, current_tokenizer, current_model_name
    free_vram()
    
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        current_tokenizer = AutoTokenizer.from_pretrained(base_id)
        current_model = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb_config, device_map={"": "cuda:0"})
        
        if use_lora and os.path.exists(lora_dir):
            current_model = PeftModel.from_pretrained(current_model, lora_dir)
            current_model_name = "Distilled (LoRA) Student"
        else:
            current_model_name = "Base Student"
            
        return f"Successfully loaded: {current_model_name}"
    except Exception as e:
        return f"Error loading model: {str(e)}"

# ==========================================
# 3. チャットロジック
# ==========================================
def chat_fn(message, history):
    global current_model, current_tokenizer
    if current_model is None:
        yield "Please load a model first in the options above."
        return
        
    messages = []
    for user_msg, ai_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": ai_msg})
    messages.append({"role": "user", "content": message})
    
    text = current_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = current_tokenizer(text, return_tensors="pt").to(device)
    
    streamer = TextIteratorStreamer(current_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
    )
    
    thread = Thread(target=current_model.generate, kwargs=generation_kwargs)
    thread.start()
    
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text

# ==========================================
# 4. ベンチマークロジック
# ==========================================
def run_benchmark_ui():
    global current_model, current_tokenizer, current_model_name
    if current_model is None:
        return "Please load a model first."
        
    log = f"Running Benchmark for: {current_model_name}...\n\n"
    
    analogy_scores = []
    for a, b, c, d in analogy_tests:
        va, vb, vc, vd = [get_word_embedding(current_model, current_tokenizer, w) for w in (a, b, c, d)]
        v_pred = F.normalize(va - vb + vc, p=2, dim=0)
        score = F.cosine_similarity(v_pred.unsqueeze(0), vd.unsqueeze(0)).item()
        analogy_scores.append(score)
    avg_analogy = np.mean(analogy_scores)
    
    hierarchy_scores = []
    for base, target, outlier in hierarchy_tests:
        v_base, v_tgt, v_out = [get_word_embedding(current_model, current_tokenizer, w) for w in (base, target, outlier)]
        sim_tgt = F.cosine_similarity(v_base.unsqueeze(0), v_tgt.unsqueeze(0)).item()
        sim_out = F.cosine_similarity(v_base.unsqueeze(0), v_out.unsqueeze(0)).item()
        score = sim_tgt - sim_out
        hierarchy_scores.append(score)
    avg_hierarchy = np.mean(hierarchy_scores)
    
    log += f"Analogy (推論力): {avg_analogy:.4f}\n"
    log += f"Hierarchy (階層理解): {avg_hierarchy:.4f}\n"
    return log


# ==========================================
# Gradio UI 構築
# ==========================================

TEACHER_CHOICES = [
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "meta-llama/Meta-Llama-3.1-8B",
    "google/gemma-2-9b",
    "mistralai/Mistral-7B-v0.3"
]

STUDENT_CHOICES = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it"
]

with gr.Blocks(title="Concept Forge", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚗️ Concept Forge (Isolated Concept Distillation)")
    gr.Markdown("VRAMを節約しつつ、巨大モデルから小規模モデルへ「純粋な概念地形（哲学）」だけを抽出・移植するOSSツールです。")
    
    with gr.Tabs():
        # --- TAB 1: 蒸留 ---
        with gr.TabItem("⚙️ Distillation"):
            with gr.Row():
                with gr.Column():
                    t_id = gr.Dropdown(choices=TEACHER_CHOICES, value="Qwen/Qwen2.5-7B", label="Teacher Model (リストから選択 or 手動入力)", allow_custom_value=True)
                    s_id = gr.Dropdown(choices=STUDENT_CHOICES, value="Qwen/Qwen2.5-1.5B-Instruct", label="Student Model (リストから選択 or 手動入力)", allow_custom_value=True)
                    save_d = gr.Textbox(label="Drive/Save Path for LoRA", value="./concept_lora_output")
                    r_val = gr.Slider(minimum=8, maximum=256, step=8, value=128, label="LoRA Rank (r)")
                    ep_val = gr.Slider(minimum=1, maximum=50, step=1, value=15, label="Epochs")
                    start_btn = gr.Button("🚀 Start Distillation", variant="primary")
                with gr.Column():
                    output_log = gr.Textbox(label="Process Log", lines=20, max_lines=20, interactive=False)
                    
            start_btn.click(distill_process, inputs=[t_id, s_id, save_d, r_val, ep_val], outputs=[output_log])
            
        # --- TAB 2: チャット & テスト ---
        with gr.TabItem("💬 Chat & Benchmark"):
            with gr.Row():
                base_txt = gr.Dropdown(choices=STUDENT_CHOICES, value="Qwen/Qwen2.5-1.5B-Instruct", label="Base Model", allow_custom_value=True, scale=2)
                lora_txt = gr.Textbox(label="LoRA Path (Optional)", value="./concept_lora_output", scale=2)
                use_lora_chk = gr.Checkbox(label="Apply LoRA", value=True)
                load_btn = gr.Button("Load Model", variant="secondary")
            
            load_status = gr.Textbox(label="Status", interactive=False)
            load_btn.click(load_model_for_chat, inputs=[base_txt, lora_txt, use_lora_chk], outputs=[load_status])
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.ChatInterface(fn=chat_fn)
                with gr.Column(scale=1):
                    bench_btn = gr.Button("Run Topography Benchmark", variant="primary")
                    bench_out = gr.Textbox(label="Benchmark Results", lines=10)
                    bench_btn.click(run_benchmark_ui, inputs=[], outputs=[bench_out])

if __name__ == "__main__":
    # ローカルでブラウザを自動で開く
    demo.launch(inbrowser=True)
