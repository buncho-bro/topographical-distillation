# Concept Forge

Concept Forge は、大規模言語モデルの意味空間にある「概念の地形」を、より小さなモデルへ LoRA で蒸留するための Gradio アプリ兼研究プロジェクトです。

このリポジトリはツール本体だけでなく、LLM Topography を探究した過程も保存しています。最終的なコアツール、主要な研究手法、従来方式が失敗した理由を確認できる研究アーカイブの3層で構成されています。

## すぐに使う

### Windows（推奨）

1. Python 3.10 以降と CUDA 対応環境を用意します。
2. `start_concept_forge.bat` をダブルクリックします。
3. 初回は仮想環境の作成と依存パッケージの導入後、Web UI が起動します。

### 手動起動

```bash
python -m venv .venv
./.venv/Scripts/Activate.ps1
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

### 🚀 コアアプリ

- `concept_forge_webui.py`: VRAM効率を意識したメインのGradioアプリ。概念地形の抽出、蒸留、比較、チャットを操作
- `requirements.txt`: コアアプリに必要な依存パッケージ
- `start_concept_forge.bat`: Windows用ワンクリック起動ファイル

### 📖 ドキュメント

- `README.md`: セットアップ、研究の概要、プロジェクト案内
- `evidence_report.md`: 公開用の検証結果と評価指標

### 🔬 Research Archive

- `generate_evidence.py`: Concept Forge のLoRAを評価し、証明レポートを生成
- `step4_advanced_benchmark.py`: Pairwise Cosine Similarityを用いた類推・階層ベンチマーク
- `step9_isolated_concept_distillation.py`: 非線形プロジェクターを使わず、単語単独抽出で行う純粋蒸留のCUI版
- `step10_hybrid_chimera.py`: Text LoRAとConcept LoRAを組み合わせるハイブリッド実験
- `step11_frankenstein_surgery.py`: `.safetensors` をレイヤー単位で合成するLoRA外科手術実験

### 💀 Graveyard of Failed Ideas

`research_archive/graveyard/` は本番利用向けではありません。失敗・旧方式を、なぜ従来のマッピングが機能しなかったかを追跡できる研究記録として残しています。

- `step2_distill.py`: 線形プロジェクターがアンカー点を過学習する問題を検証
- `step5_zero_shot_injection.py` / `step6_translator_injection.py`: 異なる次元間のSVD・Translator方式を検証
- `step7_same_dim_chimera.py`: MSEによる空間平滑化とHierarchy低下を検証
- `step8_verify_teacher_extraction.py`: 単語単独抽出と文脈内抽出を比較し、文脈化によるHierarchy平坦化を検証

## 公開対象

GitHubにはコードと評価レポートのみを収録しています。生成されたモデル重み、LoRAアダプター、抽出テンソル、キャッシュデータは容量が大きいため含めていません。
