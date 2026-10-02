import os
import pickle
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "advanced_texts.pkl")

# ベンチマークと被らない単語リスト（事実用）
wiki_subjects = ["量子力学", "ルネサンス", "光合成", "マグマ", "バイオテクノロジー", "古代エジプト", "ブラックホール", "細胞分裂", "産業革命", "深海魚", "超伝導", "地殻変動", "DNA", "暗号資産", "オーロラ"]
wiki_predicates = ["について研究が進んでいる。", "は複雑なメカニズムを持つ。", "の歴史は非常に古い。", "は現代科学の重要なテーマである。", "の構造は未だ完全には解明されていない。", "は多くの専門家によって議論されている。"]

# ベンチマークと被らない単語リスト（論理・階層用）
categories_high = ["無機物", "芸術", "生態系", "経済", "言語", "幾何学", "思想", "素粒子", "仮想空間", "エネルギー"]
categories_mid = ["金属", "絵画", "森林", "通貨", "文法", "多角形", "哲学", "クォーク", "ネットワーク", "電力"]
categories_low = ["鉄", "油絵", "針葉樹", "紙幣", "動詞", "三角形", "倫理学", "アップクォーク", "プロトコル", "ボルト"]

logic_templates = [
    # 階層構造
    "{low}は{mid}の具体例であり、さらにそれは{high}という巨大な概念に内包される。",
    "{high}の抽象度は{mid}よりも高く、最も具体的なインスタンスとして{low}が存在する。",
    "もし対象が{low}であれば、それは自動的に{mid}の性質を満たすが、すべての{mid}が{low}であるとは限らない。",
    # 独立・排他構造
    "{high}の文脈において、{mid}と他の要素は排他的な関係にある。",
    "{low}の存在は{high}の枠組みの中で定義されるが、それ自体は独立した機能を持つ。",
    # 論理推論
    "前提Aが{low}を含み、前提Bが{mid}を示すならば、結論は必然的に{high}の領域に属する。",
    "ある要素が{mid}に属さない場合、それが{low}である可能性は論理的に否定される。"
]

def generate_data():
    print("Generating Advanced Dataset (Wiki 50% + Logic/Hierarchy 50%)...")
    texts = []
    
    # 1. Wiki風の事実データ (500件)
    for _ in range(500):
        subj = random.choice(wiki_subjects)
        pred = random.choice(wiki_predicates)
        # バリエーションを増やすために装飾語をつける
        adj = random.choice(["一般的に", "歴史的に見ても", "最新の論文によれば", "専門家の間では", "驚くべきことに"])
        text = f"{adj}、{subj}{pred}"
        texts.append(text)
        
    # 2. 論理・階層パズルデータ (500件)
    for _ in range(500):
        h = random.choice(categories_high)
        m = random.choice(categories_mid)
        l = random.choice(categories_low)
        template = random.choice(logic_templates)
        text = template.format(high=h, mid=m, low=l)
        texts.append(text)
        
    # シャッフルして保存
    random.shuffle(texts)
    
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(texts, f)
        
    print(f"Generated {len(texts)} texts successfully!")
    print("Samples:")
    for t in texts[:5]:
        print(f" - {t}")
    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_data()
