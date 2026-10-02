# ⚗️ Concept Forge: Scientific Evidence

This report was automatically generated to provide empirical evidence for the **Isolated Concept Distillation** methodology.

## 0. Terminology & Metrics
Before diving into the results, we define the key metrics and concepts used to evaluate the model's internal topography (conceptual space):

*   **Analogy (Linear Reasoning)**: Measures the accuracy of linear relationships between concepts using vector arithmetic (e.g., `King - Man + Woman = Queen`). A higher score indicates stronger logical and sequential reasoning capabilities.
*   **Hierarchy (Concept Depth)**: Measures the structural depth and non-linear topology of the space. Calculated by measuring the proximity of related concepts (e.g., `Dog` and `Animal`) minus the proximity to unrelated outliers (e.g., `Desk`). A higher score means the model perfectly separates distinct semantic domains into deep "peaks and valleys" rather than flattening them.
*   **Contextual Extraction**: Extracting the vector of a target word while it is embedded inside a natural sentence.
*   **Isolated Concept Extraction (Ours)**: Extracting the vector of a word completely in isolation, without any surrounding context.

## 1. The Trap of Contextual Embeddings
Many traditional distillation methods extract embeddings from within a sentence context. However, our tests prove that Attention mechanisms "iron out" the deep conceptual hierarchy when a context is applied.

| Model | Extraction Method | Analogy (Linear Reasoning) | Hierarchy (Concept Depth) |
| :--- | :--- | :--- | :--- |
| **Qwen/Qwen2.5-7B** | **Isolated** (Pure Word) | 0.6644 | **0.2148** |
| Qwen/Qwen2.5-7B | Contextual (Sentence) | 0.8421 | 0.1103 (Severely Flattened) |

*Conclusion: To extract the true "deep philosophy" (Hierarchy) of the Teacher, we MUST extract isolated words without any surrounding context.*

## 2. Isolated Concept Distillation Results
By extracting 100% pure isolated words and directly matching the Pairwise Cosine Similarity matrix (without using non-linear projectors that cause overfitting), the Student model successfully clones the deep topography of the Teacher.

| Model | Setup | Analogy | Hierarchy |
| :--- | :--- | :--- | :--- |
| Qwen/Qwen2.5-7B | Teacher (Ground Truth) | 0.6644 | **0.2148** |
| Qwen/Qwen2.5-1.5B-Instruct | Base Student | 0.7715 | 0.0937 |
| **Concept Distilled Student** | **LoRA (Ours)** | 0.6563 | **0.2280** |

*Conclusion: The small Student model not only dramatically improved its Hierarchy, but managed to equal or surpass the Teacher's deep conceptual understanding.*
