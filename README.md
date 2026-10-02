# Concept Forge

Concept Forge は、大規模言語モデルの意味空間にある「概念の地形」を、より小さなモデルへ LoRA で蒸留するための Gradio アプリ兼研究プロジェクトです。

## すぐに使う

### Windows（推奨）

1. Python 3.10 以降と CUDA 対応環境を用意します。
2. `start_concept_forge.bat` をダブルクリックします。
3. 初回は仮想環境の作成と依存パッケージの導入後、Web UI が起動します。

### 手動起動

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python concept_forge_webui.py
```

4-bit 量子化には、CUDA と環境に対応した `bitsandbytes` が必要です。Hugging Face のモデルは初回実行時にダウンロードされるため、各モデルの利用条件にも従ってください。

## 研究の結論

単語を文中から取り出すのではなく、単独で教師モデルへ入力して概念表現を抽出し、概念間のペアワイズ・コサイン類似度を小さなモデルへ蒸留する方法を検証しました。

| モデル / 条件 | Analogy | Hierarchy |
| --- | ---: | ---: |
| Qwen2.5-7B Teacher（単語単独） | 0.6644 | 0.2148 |
| Qwen2.5-1.5B-Instruct Base | 0.7715 | 0.0937 |
| Concept-distilled Student | 0.6563 | 0.2280 |

小規模な独自ベンチマークでは、蒸留後の Student の階層スコアが Teacher と同等以上になりました。ただし、これは探索的な結果であり、一般的な推論能力や他タスクへの改善を保証するものではありません。詳しい条件と数値は [evidence_report.md](evidence_report.md) を参照してください。

## プロジェクト構成

```text
concept-forge/
├─ concept_forge_webui.py
├─ requirements.txt
├─ start_concept_forge.bat
├─ README.md
├─ evidence_report.md
└─ research_archive/
   ├─ generate_evidence.py
   ├─ step4_advanced_benchmark.py
   ├─ step9_isolated_concept_distillation.py
   ├─ step10_hybrid_chimera.py
   ├─ step11_frankenstein_surgery.py
   └─ graveyard/
```

## 各ファイルの役割

### コアアプリ

- `concept_forge_webui.py`: 蒸留、比較、チャットを操作するメインの Gradio アプリ
- `requirements.txt`: コアアプリに必要な依存パッケージ
- `start_concept_forge.bat`: Windows用ワンクリック起動ファイル

### ドキュメント

- `README.md`: セットアップ、研究の概要、プロジェクト案内
- `evidence_report.md`: 公開用の検証結果と評価指標

### Research Archive

- `generate_evidence.py`: Concept Forge の LoRA を評価し、証明レポートを生成
- `step4_advanced_benchmark.py`: 高度な類推・階層ベンチマーク
- `step9_isolated_concept_distillation.py`: 単語単独抽出による純粋蒸留のCUI版
- `step10_hybrid_chimera.py`: テキストLoRAと概念LoRAのマージ実験
- `step11_frankenstein_surgery.py`: レイヤー単位のLoRA合成実験
- `graveyard/`: 失敗・旧方式を含む実験過程の保存場所

## 公開対象

GitHubにはコードと評価レポートのみを収録しています。生成されたモデル重み、LoRAアダプター、抽出テンソル、キャッシュデータは容量が大きいため含めていません。
