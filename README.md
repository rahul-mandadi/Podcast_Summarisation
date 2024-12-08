# Podcast_Summarisation
# Automated Podcast Content Summarization Using Parameter-Efficient Fine-Tuning and Speech Recognition

## Abstract
This project explores a scalable and efficient pipeline for automatically transcribing and summarizing podcast content. Leveraging OpenAI’s Whisper model for speech-to-text conversion, we apply Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA to BART and T5 summarization models. Our goal is to achieve high-quality, concise summaries with reduced computational overhead. This approach aims to enhance accessibility, enable efficient content filtering, and assist users in quickly identifying relevant podcast episodes from large audio collections.

## Overview
Long-form audio content, such as podcasts, poses challenges for users who need to quickly glean key insights or determine relevance. Manual transcription and summarization are impractical at scale. This project proposes an automated solution that integrates accurate transcription and advanced summarization models. Starting with Whisper for transcription, we fine-tune pre-trained transformer-based summarizers (BART and T5 variants) using LoRA, a technique that updates a small subset of parameters. This method maintains model quality while reducing training resources and time.

The following sections detail the problem, motivation, and societal impact before describing our chosen approaches, the rationale for these methods, and any related work. We then present a comprehensive experiment setup, including datasets, model architectures, training parameters, and computational environments. After discussing our main and supplementary results, we offer a critical discussion of performance trade-offs and potential improvements. The conclusion summarizes our achievements and provides references to relevant literature and tools.

## What is the Problem?
Manual summarization of podcast episodes is time-consuming and does not scale with the rapidly increasing volume of audio content. Listeners need a quick way to assess whether a podcast episode is relevant. Our problem is thus to automatically convert audio into text and then distill that text into a concise summary. This allows users to efficiently filter content, save time, and discover new material more easily.

## Why is this Problem Interesting?
As the popularity of podcasts soars, rapidly digesting and cataloging this content becomes essential. Automated summarization can significantly impact media monitoring, education, journalism, corporate communications, and accessibility for individuals with hearing impairments. By providing summaries, we help users navigate large collections of audio content, ultimately contributing to better information retrieval and democratized access to knowledge.

## Proposed Approach
Our approach integrates two main steps:
1. **Transcription with Whisper**: We first leverage the Whisper model to generate accurate transcripts from podcast audio. Whisper’s robustness ensures high-quality textual input for subsequent summarization.
2. **Summarization with LoRA-Fine-Tuned Models**: We fine-tune BART (Base and Large) and T5 (Medium and Small) models using LoRA, which updates a fraction of parameters. This reduces computational overhead and training complexity, making it feasible to experiment with multiple models and configurations efficiently.

## Rationale Behind the Approach
Traditional full fine-tuning of large language models is expensive and time-consuming. LoRA-based PEFT methods allow us to adapt powerful models at a fraction of the cost. While summarization of written text has been widely studied, applying these methods directly to automatically generated transcripts is less common. By combining speech recognition and PEFT-based summarization, we can deliver a more sustainable and scalable solution.

Compared to previous approaches:
- We rely on a state-of-the-art transcription model (Whisper).
- We adopt a parameter-efficient fine-tuning strategy (LoRA) to handle large models without excessive computational demands.
- We apply our approach to domain-specific text (e.g., legislative bills with Billsum) to assess performance and versatility.

## Key Components & Results
**Key Components:**
- **Transcription**: Whisper for robust speech-to-text conversion.
- **Summarization Models**: BART-Base, BART-Large, T5-Medium, and T5-Small, all fine-tuned using LoRA.
- **Metrics**: ROUGE scores to quantify the quality of generated summaries; evaluation and training loss metrics to gauge model convergence and stability.

**Limitations:**
- Whisper’s performance may degrade with poor audio quality or unusual accents.
- Certain podcast topics may require domain-specific fine-tuning.
- Some configurations yield shorter summaries that may omit essential details, indicating room for further fine-tuning and optimization.

## Experiment Setup
**Dataset**:  
- **Billsum**: A dataset of U.S. legislative bills used as a proxy for long-form text. Although not originally for podcasts, it helps simulate summarization of lengthy content segments. It contains thousands of samples, each with source text and target summaries, enabling quantitative evaluation with ROUGE metrics.

**Implementation Details**:
- **Models**: BART-Base, BART-Large, T5-Medium, T5-Small with LoRA-based fine-tuning.
- **Parameters**: Typically around 2-4% of parameters updated via LoRA, significantly reducing training overhead.
- **Environment**: Experiments run on GPU-enabled cloud instances. We use Hugging Face Transformers, PyTorch, and Weights & Biases (W&B) for logging and experiment tracking.

**Model Architecture**:
- **BART**: An encoder-decoder transformer pretrained on massive corpora, suitable for generative tasks like summarization.
- **T5**: A text-to-text transformer that uniformly treats every NLP task.  
- LoRA fine-tuning injects additional low-rank parameter matrices into the model, allowing adaptation without large-scale retraining.

## Experiment Results
### Main Results
Below is a comparison of the main results for different models. ROUGE-1 and ROUGE-2 measure the overlap of unigrams and bigrams with the reference summaries, respectively, while ROUGE-L and ROUGE-Lsum consider longest common subsequences, offering a more nuanced measure of summary quality.

| Model               | ROUGE-1  | ROUGE-2  | ROUGE-L  | ROUGE-Lsum | Eval Loss | Train Loss |
|---------------------|----------|----------|----------|------------|-----------|------------|
| **BART-Large (LoRA)**  | 0.5653   | 0.3477   | 0.4333   | 0.4641     | 1.4295    | 1.4806     |
| **BART-Base (LoRA)**   | 0.2369   | 0.1916   | 0.2302   | 0.2310     | 1.7250    | 1.9093     |
| **T5-Medium (LoRA)**   | 0.2460   | 0.2021   | 0.2388   | 0.2389     | 1.2384    | 1.4548     |
| **T5-Small (LoRA)**    | 0.2400   | 0.1904   | 0.2315   | 0.2315     | 1.6472    | 1.8503     |

### Supplementary Results
We experimented with different learning rates, batch sizes, and the proportion of parameters trained under LoRA. Fine-tuning a small percentage of parameters consistently reduced training costs while preserving model performance. Varying summary length (gen_len) and adjusting early stopping criteria also influenced the final ROUGE scores.

## Discussion
BART-Large fine-tuned with LoRA stands out as the top performer, but it comes with higher baseline resource requirements. Smaller models like T5-Small and BART-Base, while less accurate, may still be suitable for deployment on resource-limited devices. The trade-off between performance and efficiency is central to this project’s approach.

In comparison to state-of-the-art full fine-tuning approaches, our LoRA-based method yields strong results at a fraction of the computational cost. If certain scores are not optimal, possible reasons include limited domain adaptation or the complexity of transcribed spoken language. Further efforts may involve domain-specific pre-training, improved audio quality, and advanced prompt engineering.

## Conclusion
We successfully integrated Whisper-based transcription with LoRA-fine-tuned summarization models to produce concise and coherent summaries of complex, long-form content. By striking a balance between computational efficiency and summarization quality, we have taken a step towards making large-scale podcast analysis accessible and user-friendly. Future work may explore more domain-adapted datasets, other speech modalities, and further parameter-efficient optimization techniques.

## References
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Billsum Dataset](https://huggingface.co/datasets/billsum)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al.)](https://arxiv.org/abs/2106.09685)
- [Weights & Biases](https://wandb.ai)
