import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "distilled_student_model")

# 実行環境の判定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_chat():
    print("==================================================")
    print("Loading Distilled Minimum AI (rinna-gpt2-xsmall)")
    print("==================================================")
    
    start_load = time.time()
    
    # 蒸留されたモデルとトークナイザをロード（文章生成用にAutoModelForCausalLMを使う）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    # 保存したディレクトリから生成用モデルとしてロード
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)
    
    print(f"Model loaded in {time.time() - start_load:.2f} seconds.")
    print(f"Device: {device.type.upper()}")
    print("Ready to chat. Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("\n[You]: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
                
            # プロンプトの作成（ベースモデルなので簡単な文脈を与える）
            prompt = f"質問: {user_input}\n回答:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # --- 速度計測開始 ---
            start_gen = time.time()
            
            # 文章生成（極小モデルなので一瞬で終わるはず）
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,       # 最大50トークン生成
                    do_sample=True,          # サンプリングを有効化
                    temperature=0.7,         # 少し創造的に
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # --- 速度計測終了 ---
            end_gen = time.time()
            gen_time = end_gen - start_gen
            
            # デコード
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 入力プロンプト部分を削って回答だけを抽出
            answer = generated_text[len(prompt):].strip()
            
            # 回答と速度の表示
            print(f"[Distilled AI]: {answer}")
            print(f"  --> (Generation Time: {gen_time:.3f} seconds)")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_chat()
