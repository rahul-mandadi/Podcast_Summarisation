# Podcast_Summarisation
# Automated Podcast Content Summarization Using Parameter-Efficient Fine-Tuning and Speech Recognition
By Rahul Reddy Mandadi and Ashwin Khairnar

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

### Handling Long Inputs

Many of the models we use have a maximum input token length (typically 512 tokens), which can pose challenges when dealing with lengthy podcasts or transcripts that exceed this limit. To address this, we can employ several strategies to ensure our summarization process remains accurate and efficient:

1. **Chunking and Batching**:  
   We split lengthy transcripts into smaller, manageable segments (e.g., 512-token chunks). Each chunk is summarized independently, and these partial summaries are then combined or further summarized to produce a coherent final summary. This approach ensures that we stay within the model’s token constraints.

2. **Sliding Window Approach**:  
   By using a sliding window, we maintain a certain overlap between consecutive chunks. This overlap preserves contextual continuity, ensuring that information from one segment isn’t lost when moving on to the next. The final summary can integrate insights from each windowed segment, producing a more comprehensive result.

3. **Extractive Summaries as Preprocessing**:  
   Before applying our generative summarization models, we may first generate a quick extractive summary using a simpler model or keyword extraction technique. This extractive summary serves as a shorter, distilled input, which the PEFT-fine-tuned model can then transform into a higher-quality abstractive summary. This two-step process allows the model to work within token limits while still capturing essential content.

By employing these techniques, we can ensure that even the longest podcast transcripts can be efficiently processed. As a result, users receive concise, coherent, and contextually rich summaries, regardless of the original audio’s length.

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

## Model Repository Links

To ensure reproducibility and facilitate easy access, we have hosted our fine-tuned models on Hugging Face. Below are the direct links to each model used in this project:

| Model                 | Description                                           | Hugging Face Repository                                                                                           |
|-----------------------|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **BART-Large (LoRA)** | BART-Large model fine-tuned with LoRA for summarization | [ashwin0211/bart-large-cnn-repo](https://huggingface.co/ashwin0211/bart-large-cnn-repo) |
| **BART-Base (LoRA)**  | BART-Base model fine-tuned with LoRA for summarization  | [RahulMandadi/lora_Bart_base_cnn](https://huggingface.co/RahulMandadi/lora_Bart_base_cnn)   |
| **T5-Medium (LoRA)**  | T5-Medium model fine-tuned with LoRA for summarization  | [ashwin0211/lora_t5_medium](https://huggingface.co/ashwin0211/lora_t5_medium)               |
| **T5-Small (LoRA)**   | T5-Small model fine-tuned with LoRA for summarization   | [RahulMandadi/lora_T5_small](https://huggingface.co/RahulMandadi/lora_T5_small)             |

### How to Use the Models

1. **Installation**:
   Ensure you have the Hugging Face Transformers library installed. You can install it using pip:

    ```bash
    pip install transformers
    ```

2. **Loading a Model and Tokenizer**:
   Here's an example of how to load the BART-Large (LoRA) model and its tokenizer:

    ```python
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "ashwin0211/bart-large-cnn-repo"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    ```

3. **Generating Summaries**:
   Use the loaded model to generate summaries from transcripts:

    ```python
    def generate_summary(text, model, tokenizer, max_length=150, min_length=40, do_sample=False):
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    transcript = "Your podcast transcript goes here..."
    summary = generate_summary(transcript, model, tokenizer)
    print(summary)
    ```

## References

- [Hugging Face Transformers Documentation: Summarization Task](https://huggingface.co/docs/transformers/main/en/tasks/summarization)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Billsum Dataset](https://huggingface.co/datasets/billsum)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al.)](https://arxiv.org/abs/2106.09685)
- [Weights & Biases](https://wandb.ai)
- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
