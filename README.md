📁 Repository Structure & The Journey to the Truth

This repository is not just a tool; it is a chronological archive of our research into Large Language Model Topography. We have divided the scripts into the Final Tools, the Research Architectures, and the Graveyard of Failed Ideas (which serve as mathematical proofs of why traditional methods fail).

🌟 Core Tools (The Final Solution)

These are the production-ready tools derived from our final conclusions.

concept_forge_webui.py The Ultimate Topography Distiller (Web UI). The culmination of our research. A VRAM-efficient, Gradio-based application that allows anyone to extract pure conceptual topography from massive Teacher models (e.g., 32B/70B) and distill it into tiny Student models (e.g., 0.5B/1.5B) using Isolated Concept Distillation.
step12_generate_evidence.py The Evidence Builder. An automated script that generates the evidence_report.md. It proves mathematically that Contextual Embeddings flatten hierarchy and that Isolated Concept Distillation successfully clones the Teacher's depth.
🔬 The Research Architectures (Experiments & Surgery)
step4_advanced_benchmark.py The Topography Evaluator. Our custom evaluation metric that calculates the Analogy (Linear Reasoning / Vector Arithmetic) and Hierarchy (Non-linear Concept Depth) scores using Pairwise Cosine Similarity.
step8_verify_teacher_extraction.py The Trap of Contextualization. A verification script that compares vectors extracted from single words vs. words embedded in sentences. It proved our greatest discovery: Context flattens hierarchy due to Attention mechanism noise.
step9_isolated_concept_distillation.py Isolated Concept Distillation (Raw Script). The foundational script that performs Pairwise Distillation using 100% pure isolated words without any non-linear projectors.
step10_hybrid_chimera.py Mixture of LoRAs (MoE-LoRA). An experimental script that merges a Text-Distilled LoRA (Analogy Specialist) and an Isolated-Concept LoRA (Hierarchy Specialist) into a single Hybrid AI.
step11_frankenstein_surgery.py Layer-wise LoRA Surgery. A highly advanced script that mathematically dissects .safetensors files to apply Concept LoRAs only to the lower (thinking) layers and Text LoRAs to the upper (output) layers.
💀 The Graveyard of Failed Ideas (Proofs of Failure)

Do not use these for production. They remain here as empirical evidence of why traditional mapping fails.

step2_distill.py The Projector Overfitting Trap. Attempts to match absolute coordinates using an nn.Linear projector. Result: The projector achieves a Loss of 0.0000 by "memorizing" anchor points, completely destroying the topology for unknown words.
step5_zero_shot_injection.py & step6_translator_injection.py The Dimensional Catastrophe. Attempts to map a 1536-dim Student space to a 4096-dim Teacher space using SVD Procrustes alignment and frozen Translators. Result: Severe "mojibake" (catastrophic forgetting) of semantic placement.
step7_same_dim_chimera.py The Linear Smoothing Trap. Attempts to map spaces of the same dimension (1536 -> 1536) using MSE Loss. Result: The MSE forces the space to "iron out", resulting in artificially high Analogy but completely flattened Hierarchy.
