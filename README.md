# Podcast_Summarisation
# Automated Podcast Content Summarization Using Parameter-Efficient Fine-Tuning and Speech Recognition

## Abstract
This project explores the integration of speech recognition and parameter-efficient fine-tuning (PEFT) techniques to automatically summarize lengthy podcast episodes. By converting audio content into text using the Whisper model and subsequently applying LoRA-based fine-tuning to pre-trained transformer models (such as BART and T5), we aim to produce concise, coherent, and informative summaries. This approach balances computational efficiency with high-quality results, ultimately improving accessibility and enabling quick content filtering across large audio repositories.

## Overview
Podcasts are rich sources of information, but their long-form nature makes identifying relevant sections challenging. The need to quickly find key ideas in hours of spoken audio is critical for journalists, researchers, students, and anyone managing vast audio libraries. Our solution begins with Whisper for accurate transcription, followed by fine-tuning large language models using LoRA—a technique that updates only a small portion of model parameters. This parameter-efficient approach reduces training overhead while preserving model quality, making it more accessible for a range of use cases and computational environments.

We benchmark our approach on a legislative document summarization dataset (Billsum) as a stand-in for complex, long-form content. While initially designed for textual data, Billsum’s complexity offers insights into how these techniques scale and generalize, suggesting that our workflow could handle domain adaptation for different types of content, including other spoken materials or specialized domains.

## What is the Problem?
The problem lies in automatically extracting key insights from large volumes of spoken content. Manually summarizing podcasts is time-consuming, error-prone, and infeasible at scale. Users need a quick way to assess whether a given episode meets their interests or requirements, and traditional search tools and metadata often fall short. A robust automated summarizer can streamline content discovery, save time, and improve user engagement with audio platforms.

## Why is this Problem Interesting?
As audio content proliferates—spanning news, educational lectures, entertainment, and research interviews—the ability to rapidly extract key points and summaries is invaluable. This capability is relevant not only to casual listeners but also to journalists, content curators, market researchers, educators, and accessibility advocates. By improving the efficiency of information retrieval and comprehension, this approach contributes to more inclusive access to knowledge and can assist in real-time decision-making scenarios.

## Proposed Approach
1. **Transcription with Whisper**: Convert audio segments into accurate transcripts. Whisper’s reliability in handling diverse accents and variable audio quality sets a strong foundation.
2. **Summarization via LoRA-Fine-Tuned Models**: Apply LoRA-based PEFT to BART and T5 summarization models. This reduces the percentage of trainable parameters (~2-4%) while maintaining strong ROUGE scores. Parameter-efficient training allows experimentation with multiple architectures without excessive computation.

## Rationale Behind the Approach
Traditional full fine-tuning of large language models can be resource-intensive, both computationally and financially. LoRA provides a cost-effective alternative, enabling rapid adaptation of large pre-trained models to new tasks without retraining all parameters. While text summarization is well-studied, extending these techniques to spoken content (via transcripts) is less explored. By repurposing text-based datasets like Billsum, we can assess performance in summarizing complex, structured content similar in length and detail to podcast transcripts.

Comparisons with standard extractive or rule-based summarization methods underscore the benefits of generative, context-aware models. Earlier research often focused on news articles or short documents, whereas our approach targets longer, more varied transcripts—an area where advanced models can excel.

## Key Components & Results
**Key Components**:  
- **Whisper** for transcription  
- **BART (Base/Large) and T5 (Medium/Small) Models** fine-tuned with LoRA  
- **Evaluation Metrics**: ROUGE scores, evaluation and training loss

**Performance Highlights**:
- BART-Large (LoRA) achieves the best ROUGE-1 (~0.5653), indicating high-quality summaries.
- T5 and BART-Base variants show moderate performance but at lower computational costs.
- Summaries remain coherent and contextually relevant, benefiting from Whisper’s quality transcripts and the rich latent knowledge in pre-trained models.

**Limitations**:
- Whisper accuracy may degrade with poor audio quality or highly domain-specific jargon.
- Although parameter-efficient, some models still require significant GPU memory for large batches or longer sequences.
- Domain adaptation can further improve results; current tests on Billsum simulate complexity but may not fully capture nuances of certain podcast genres.

## Experiment Setup
**Dataset**:  
- **Billsum**: A Hugging Face dataset of U.S. legislative bills with expert-written summaries. While not podcast-specific, it simulates long, complex textual input to test the summarization capabilities of our approach.

**Implementation Details**:  
- **Hyperparameters**: Batch sizes adjusted for GPU memory (e.g., batch size=8), warm-up steps (~500), and weight decay (0.01).
- **Training Environment**: Experiments conducted on GPU-enabled cloud instances with limited memory, guiding choice of gradient accumulation steps (e.g., 16) to fit sequences into memory efficiently.
- **Model Architectures**:  
  - **BART Variants**: Encoder-decoder transformers known for strong summarization capabilities.
  - **T5 Variants**: Text-to-text framework enabling flexible summary generation.
  - **LoRA Adaptation**: Injecting low-rank matrices into certain transformer layers reduces full-model retraining and expedites experimentation.

## Experiment Results
**Main Results**:  
| Model               | ROUGE-1  | ROUGE-2  | ROUGE-L  | ROUGE-Lsum | Eval Loss | Train Loss |
|---------------------|----------|----------|----------|------------|-----------|------------|
| **BART-Large (LoRA)**  | 0.5653   | 0.3477   | 0.4333   | 0.4641     | 1.4295    | 1.4806     |
| **BART-Base (LoRA)**   | 0.2369   | 0.1916   | 0.2302   | 0.2310     | 1.7250    | 1.9093     |
| **T5-Medium (LoRA)**   | 0.2460   | 0.2021   | 0.2388   | 0.2389     | 1.2384    | 1.4548     |
| **T5-Small (LoRA)**    | 0.2400   | 0.1904   | 0.2315   | 0.2315     | 1.6472    | 1.8503     |

**Supplementary Results**:  
Adjusting learning rates, batch sizes, and epochs revealed that performance scales with training time and careful parameter tuning. Resource constraints (e.g., GPU memory) required smaller batch sizes and gradient accumulation. Slight improvements were noted with longer training and increased epochs, though diminishing returns highlight the importance of balanced hyperparameter choices.

## Discussion
Our results show that parameter-efficient fine-tuning can achieve competitive ROUGE scores while reducing computational demands. BART-Large emerges as the top contender, but smaller models like T5 or BART-Base may suffice in less resource-rich scenarios or where slightly lower accuracy is acceptable.

Compared to baseline approaches found in literature, our LoRA method is more cost-effective and flexible. While the Billsum dataset approximates the complexity of podcast transcripts, future work could incorporate real-world podcast transcripts or domain-specific data. Enhanced domain adaptation, improved data preprocessing, and more sophisticated prompt engineering could further improve performance. Memory considerations remain a practical constraint, emphasizing the need for careful model and batch size selection.

## Conclusion
By integrating Whisper-based transcription with LoRA-fine-tuned summarization models, we have developed a system capable of producing concise and coherent summaries of lengthy, complex content. Our approach demonstrates that parameter-efficient fine-tuning can strike a balance between accuracy and resource constraints, paving the way for scalable solutions to handle ever-growing audio libraries. Continued refinement in hyperparameter tuning, dataset selection, and domain adaptation can further enhance these results, ultimately making podcast summarization a more accessible and powerful tool.

## References
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Billsum Dataset](https://huggingface.co/datasets/billsum)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al.)](https://arxiv.org/abs/2106.09685)
- [Weights & Biases](https://wandb.ai)
