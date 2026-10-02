import os
import torch
import torch.nn.functional as F
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- 設定 ---
TEACHER_MODEL_ID = "Qwen/Qwen2.5-7B"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "target_texts.pkl")
OUTPUT_VEC_PATH = os.path.join(BASE_DIR, "extracted_teacher_topography.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_target_token_indices(input_ids, target_ids):
    """入力トークン列の中から、ターゲットのトークン列が一致する最初のインデックス範囲を返す"""
    input_ids = input_ids.tolist()
    target_ids = target_ids.tolist()
    n = len(target_ids)
    for i in range(len(input_ids) - n + 1):
        if input_ids[i:i+n] == target_ids:
            return list(range(i, i+n))
    return None

def step1_extract():
    print("==================================================")
    print("Step 1: Extracting Topography (Targeted Token)")
    print("==================================================")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    with open(DATA_PATH, "rb") as f:
        dataset = pickle.load(f)
        
    print(f"Loading Teacher Model ({TEACHER_MODEL_ID}) in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_ID, quantization_config=bnb_config, device_map={"": "cuda:0"})
    model.eval()
    
    teacher_vecs = []
    valid_dataset = []
    
    print("Extracting vectors for targeted tokens...")
    with torch.no_grad():
        for i, data in enumerate(dataset):
            text = data["text"]
            target = data["target"]
            
            # 入力文とターゲット単語をそれぞれトークナイズ
            # add_special_tokens=False を指定して、純粋な単語のIDを取得
            input_tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
            target_tokens = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids[0]
            
            # ターゲットがどこにあるか探す
            indices = find_target_token_indices(input_tokens, target_tokens)
            
            if indices is None:
                # トークナイズの都合で一致しない場合はスキップ
                continue
                
            inputs = {"input_ids": input_tokens.unsqueeze(0).to(device)}
            outputs = model(**inputs, output_hidden_states=True)
            
            # 💡【重要】ターゲット単語のトークン位置だけのベクトルを取り出し、平均化する
            target_hidden = outputs.hidden_states[-1][0, indices, :]
            mean_hidden = target_hidden.mean(dim=0)
            
            teacher_vecs.append(mean_hidden.cpu())
            valid_dataset.append(data)
            
            if len(valid_dataset) % 10 == 0:
                print(f"  Extracted {len(valid_dataset)} valid concepts...")
                
    teacher_vecs = torch.stack(teacher_vecs)
    teacher_vecs = F.normalize(teacher_vecs, p=2, dim=1)
    
    # 有効だったデータだけを保存し直す（Step2でインデックスがズレないようにするため）
    with open(DATA_PATH, "wb") as f:
        pickle.dump(valid_dataset, f)
    
    torch.save(teacher_vecs, OUTPUT_VEC_PATH)
    print(f"Teacher Topography ({len(teacher_vecs)} tokens) successfully saved to {OUTPUT_VEC_PATH}!")
    print("VRAM will now be completely freed. Please run step2_distill.py next.")

if __name__ == "__main__":
    step1_extract()
