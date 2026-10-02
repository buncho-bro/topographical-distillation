# Topographical Distillation

巨大な言語モデルが持つ意味空間の「地形」を、より小さな言語モデルへ移すための実験プロジェクトです。文章生成を模倣する通常の蒸留に加え、隠れ表現や概念間のコサイン類似度を直接合わせる方法を検証しています。

## 主な結果

初期実験では `paraphrase-multilingual-MiniLM-L12-v2` を教師、`rinna/japanese-gpt2-xsmall` を生徒として、1,000件のテキストで学習しました。独自の意味類似度ベンチマークにおける Spearman 相関は次のように変化しました。

| モデル | スコア |
| --- | ---: |
| Teacher | 0.7173 |
| Student（蒸留前） | 0.2796 |
| Student（蒸留後） | 0.4985 |

追加の Qwen 実験では、文脈内ではなく単語を単独で入力して概念表現を抽出し、ペアごとのコサイン類似度行列を LoRA で合わせる方法を検証しました。

| モデル / 条件 | Analogy | Hierarchy |
| --- | ---: | ---: |
| Qwen2.5-7B Teacher（単語単独） | 0.6644 | 0.2148 |
| Qwen2.5-1.5B-Instruct Base | 0.7715 | 0.0937 |
| Concept-distilled Student | 0.6563 | 0.2280 |

これらは小規模な独自ベンチマーク上の探索的結果です。一般的な推論能力や他のタスクへの改善を保証するものではありません。再現性と外部ベンチマークでの検証が今後の課題です。

## ファイル構成

- `run_experiment.py`: 初期の地形蒸留実験
- `step0_generate_target_data.py` ～ `step12_generate_evidence.py`: Qwenを使った段階的な実験
- `concept_forge_webui.py`: 蒸留と比較を行うGradio UI
- `experiment_report.md`: 初期実験の詳細レポート
- `evidence_report.md`: 単語単独抽出とLoRA蒸留の結果
- `experiment_results.json`: 初期実験の機械可読な結果

## セットアップ

Python 3.10以降と、CUDA対応GPUを推奨します。

```bash
python -m venv .venv
pip install -r requirements.txt
```

4-bit量子化を使うスクリプトは、CUDA環境と対応する `bitsandbytes` が必要です。Hugging Face上のモデルは初回実行時にダウンロードされます。モデルごとの利用条件にも従ってください。

## 実行例

初期実験:

```bash
python run_experiment.py
```

Qwenによるターゲットデータ生成、教師表現の抽出、蒸留:

```bash
python step0_generate_target_data.py
python step1_extract.py
python step2_distill.py
```

単語単独での概念蒸留と評価:

```bash
python step9_isolated_concept_distillation.py
python step12_generate_evidence.py
```

Web UI:

```bash
python concept_forge_webui.py
```

一部のスクリプトはローカルのデータファイルや、それ以前のステップで生成した成果物を前提とします。パスとモデルIDは各スクリプト冒頭の設定で変更できます。

## 公開対象について

このリポジトリには再現用コードと評価結果を収録しています。生成されたモデル重み、LoRAアダプター、抽出テンソル、キャッシュデータは容量が大きいため含めていません。

## レポート

- [初期実験レポート](experiment_report.md)
- [追加検証レポート](evidence_report.md)

