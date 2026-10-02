import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- 設定 ---
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_DIR = os.path.join(BASE_DIR, "qwen_distilled_lora")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_qwen_chat():
    print("==================================================")
    print("Loading Distilled Chimera AI")
    print(f"Base Brain : {BASE_MODEL_ID} (1.5B)")
    print(f"LoRA Brain : Teacher's Topography (7B-level)")
    print("==================================================")
    
    start_load = time.time()
    
    # 4-bitでベースモデルをロード（VRAM節約＆高速化）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # ベースモデルに「蒸留された概念地形（LoRA）」をガシャッと合体させる
    print("Fusing Teacher's Topography into the Base Brain...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()
    
    print(f"Model successfully loaded in {time.time() - start_load:.2f} seconds.")
    print("Ready to chat! Type 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("\n[You]: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
            
            # Qwen-Instruct用のチャットテンプレートを適用
            messages = [
                {"role": "system", "content": "あなたは優秀なアシスタントです。"},
                {"role": "user", "content": user_input}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(device)
            
            # --- 速度計測開始 ---
            start_gen = time.time()
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            # 入力プロンプト部分を切り落として回答だけを取得
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # --- 速度計測終了 ---
            end_gen = time.time()
            gen_time = end_gen - start_gen
            
            answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"\n[Distilled AI]:\n{answer}")
            print(f"  --> (Generation Time: {gen_time:.3f} seconds)")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_qwen_chat()
