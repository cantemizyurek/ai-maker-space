---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:156
- loss:MatryoshkaLoss
- loss:MultipleNegativesRankingLoss
base_model: Snowflake/snowflake-arctic-embed-l
widget:
- source_sentence: What was the typical token context length for most models last
    year, and which model was a notable exception?
  sentences:
  - 'Here’s the rest of the transcript. It’s bland and generic, but my phone can pitch
    bland and generic Christmas movies to Netflix now!

    LLM prices crashed, thanks to competition and increased efficiency

    The past twelve months have seen a dramatic collapse in the cost of running a
    prompt through the top tier hosted LLMs.

    In December 2023 (here’s the Internet Archive for the OpenAI pricing page) OpenAI
    were charging $30/million input tokens for GPT-4, $10/mTok for the then-new GPT-4
    Turbo and $1/mTok for GPT-3.5 Turbo.'
  - 'Gemini 1.5 Pro also illustrated one of the key themes of 2024: increased context
    lengths. Last year most models accepted 4,096 or 8,192 tokens, with the notable
    exception of Claude 2.1 which accepted 200,000. Today every serious provider has
    a 100,000+ token model, and Google’s Gemini series accepts up to 2 million.'
  - 'Qwen2.5-Coder-32B is an LLM that can code well that runs on my Mac talks about
    Qwen2.5-Coder-32B in November—an Apache 2.0 licensed model!


    I can now run a GPT-4 class model on my laptop talks about running Meta’s Llama
    3.3 70B (released in December)'
- source_sentence: What does the training cost of DeepSeek v3 suggest about the future
    of training costs for AI models?
  sentences:
  - 'A lot of people are excited about AI agents—an infuriatingly vague term that
    seems to be converging on “AI systems that can go away and act on your behalf”.
    We’ve been talking about them all year, but I’ve seen few if any examples of them
    running in production, despite lots of exciting prototypes.

    I think this is because of gullibility.

    Can we solve this? Honestly, I’m beginning to suspect that you can’t fully solve
    gullibility without achieving AGI. So it may be quite a while before those agent
    dreams can really start to come true!

    Code may be the best application

    Over the course of the year, it’s become increasingly clear that writing code
    is one of the things LLMs are most capable of.'
  - 'I think this means that, as individual users, we don’t need to feel any guilt
    at all for the energy consumed by the vast majority of our prompts. The impact
    is likely neglible compared to driving a car down the street or maybe even watching
    a video on YouTube.

    Likewise, training. DeepSeek v3 training for less than $6m is a fantastic sign
    that training costs can and should continue to drop.

    For less efficient models I find it useful to compare their energy usage to commercial
    flights. The largest Llama 3 model cost about the same as a single digit number
    of fully loaded passenger flights from New York to London. That’s certainly not
    nothing, but once trained that model can be used by millions of people at no extra
    training cost.'
  - '“Agents” still haven’t really happened yet

    I find the term “agents” extremely frustrating. It lacks a single, clear and widely
    understood meaning... but the people who use the term never seem to acknowledge
    that.

    If you tell me that you are building “agents”, you’ve conveyed almost no information
    to me at all. Without reading your mind I have no way of telling which of the
    dozens of possible definitions you are talking about.'
- source_sentence: How does providing a lot of example code to an LLM help in solving
    coding problems?
  sentences:
  - 'So training an LLM still isn’t something a hobbyist can afford, but it’s no longer
    the sole domain of the super-rich. I like to compare the difficulty of training
    an LLM to that of building a suspension bridge—not trivial, but hundreds of countries
    around the world have figured out how to do it. (Correction: Wikipedia’s Suspension
    bridges by country category lists 44 countries).

    You can run LLMs on your own devices

    In January of this year, I thought it would be years before I could run a useful
    LLM on my own computer. GPT-3 and 3.5 were pretty much the only games in town,
    and I thought that even if the model weights were available it would take a $10,000+
    server to run them.'
  - 'Longer inputs dramatically increase the scope of problems that can be solved
    with an LLM: you can now throw in an entire book and ask questions about its contents,
    but more importantly you can feed in a lot of example code to help the model correctly
    solve a coding problem. LLM use-cases that involve long inputs are far more interesting
    to me than short prompts that rely purely on the information already baked into
    the model weights. Many of my tools were built using this pattern.'
  - 'Today $30/mTok gets you OpenAI’s most expensive model, o1. GPT-4o is $2.50 (12x
    cheaper than GPT-4) and GPT-4o mini is $0.15/mTok—200x cheaper than GPT-4, nearly
    7x cheaper than GPT-3.5 and massively more capable than that model.

    Other model providers charge even less. Anthropic’s Claude 3 Haiku (from March,
    but still their cheapest model) is $0.25/mTok. Google’s Gemini 1.5 Flash is $0.075/mTok
    and their Gemini 1.5 Flash 8B is $0.0375/mTok—that’s 27x cheaper than GPT-3.5
    Turbo last year.

    I’ve been tracking these pricing changes under my llm-pricing tag.'
- source_sentence: Which model mentioned in the context is the least expensive per
    million tokens, and how does its price compare to GPT-35 Turbo from last year?
  sentences:
  - 'Nothing yet from Anthropic or Meta but I would be very surprised if they don’t
    have their own inference-scaling models in the works. Meta published a relevant
    paper Training Large Language Models to Reason in a Continuous Latent Space in
    December.

    Was the best currently available LLM trained in China for less than $6m?

    Not quite, but almost! It does make for a great attention-grabbing headline.

    The big news to end the year was the release of DeepSeek v3—dropped on Hugging
    Face on Christmas Day without so much as a README file, then followed by documentation
    and a paper the day after that.'
  - 'The boring yet crucial secret behind good system prompts is test-driven development.
    You don’t write down a system prompt and find ways to test it. You write down
    tests and find a system prompt that passes them.


    It’s become abundantly clear over the course of 2024 that writing good automated
    evals for LLM-powered systems is the skill that’s most needed to build useful
    applications on top of these models. If you have a strong eval suite you can adopt
    new models faster, iterate better and build more reliable and useful product features
    than your competition.

    Vercel’s Malte Ubl:'
  - 'Today $30/mTok gets you OpenAI’s most expensive model, o1. GPT-4o is $2.50 (12x
    cheaper than GPT-4) and GPT-4o mini is $0.15/mTok—200x cheaper than GPT-4, nearly
    7x cheaper than GPT-3.5 and massively more capable than that model.

    Other model providers charge even less. Anthropic’s Claude 3 Haiku (from March,
    but still their cheapest model) is $0.25/mTok. Google’s Gemini 1.5 Flash is $0.075/mTok
    and their Gemini 1.5 Flash 8B is $0.0375/mTok—that’s 27x cheaper than GPT-3.5
    Turbo last year.

    I’ve been tracking these pricing changes under my llm-pricing tag.'
- source_sentence: How does the context differentiate between training an LLM from
    scratch and fine-tuning an existing model for hobbyists?
  sentences:
  - 'I run a bunch of them on my laptop. I run Mistral 7B (a surprisingly great model)
    on my iPhone. You can install several different apps to get your own, local, completely
    private LLM. My own LLM project provides a CLI tool for running an array of different
    models via plugins.

    You can even run them entirely in your browser using WebAssembly and the latest
    Chrome!

    Hobbyists can build their own fine-tuned models

    I said earlier that building an LLM was still out of reach of hobbyists. That
    may be true for training from scratch, but fine-tuning one of those models is
    another matter entirely.'
  - 'In 2024, almost every significant model vendor released multi-modal models. We
    saw the Claude 3 series from Anthropic in March, Gemini 1.5 Pro in April (images,
    audio and video), then September brought Qwen2-VL and Mistral’s Pixtral 12B and
    Meta’s Llama 3.2 11B and 90B vision models. We got audio input and output from
    OpenAI in October, then November saw SmolVLM from Hugging Face and December saw
    image and video models from Amazon Nova.

    In October I upgraded my LLM CLI tool to support multi-modal models via attachments.
    It now has plugins for a whole collection of different vision models.'
  - 'Except... you can run generated code to see if it’s correct. And with patterns
    like ChatGPT Code Interpreter the LLM can execute the code itself, process the
    error message, then rewrite it and keep trying until it works!

    So hallucination is a much lesser problem for code generation than for anything
    else. If only we had the equivalent of Code Interpreter for fact-checking natural
    language!

    How should we feel about this as software engineers?

    On the one hand, this feels like a threat: who needs a programmer if ChatGPT can
    write code for you?'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on Snowflake/snowflake-arctic-embed-l
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: cosine_accuracy@1
      value: 1.0
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 1.0
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 1.0
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 1.0
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 1.0
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.3333333333333333
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.20000000000000004
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.10000000000000002
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 1.0
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 1.0
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 1.0
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 1.0
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 1.0
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 1.0
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 1.0
      name: Cosine Map@100
---

# SentenceTransformer based on Snowflake/snowflake-arctic-embed-l

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Snowflake/snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l). It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Snowflake/snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l) <!-- at revision d8fb21ca8d905d2832ee8b96c894d3298964346b -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 1024 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'How does the context differentiate between training an LLM from scratch and fine-tuning an existing model for hobbyists?',
    'I run a bunch of them on my laptop. I run Mistral 7B (a surprisingly great model) on my iPhone. You can install several different apps to get your own, local, completely private LLM. My own LLM project provides a CLI tool for running an array of different models via plugins.\nYou can even run them entirely in your browser using WebAssembly and the latest Chrome!\nHobbyists can build their own fine-tuned models\nI said earlier that building an LLM was still out of reach of hobbyists. That may be true for training from scratch, but fine-tuning one of those models is another matter entirely.',
    'Except... you can run generated code to see if it’s correct. And with patterns like ChatGPT Code Interpreter the LLM can execute the code itself, process the error message, then rewrite it and keep trying until it works!\nSo hallucination is a much lesser problem for code generation than for anything else. If only we had the equivalent of Code Interpreter for fact-checking natural language!\nHow should we feel about this as software engineers?\nOn the one hand, this feels like a threat: who needs a programmer if ChatGPT can write code for you?',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value   |
|:--------------------|:--------|
| cosine_accuracy@1   | 1.0     |
| cosine_accuracy@3   | 1.0     |
| cosine_accuracy@5   | 1.0     |
| cosine_accuracy@10  | 1.0     |
| cosine_precision@1  | 1.0     |
| cosine_precision@3  | 0.3333  |
| cosine_precision@5  | 0.2     |
| cosine_precision@10 | 0.1     |
| cosine_recall@1     | 1.0     |
| cosine_recall@3     | 1.0     |
| cosine_recall@5     | 1.0     |
| cosine_recall@10    | 1.0     |
| **cosine_ndcg@10**  | **1.0** |
| cosine_mrr@10       | 1.0     |
| cosine_map@100      | 1.0     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 156 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 156 samples:
  |         | sentence_0                                                                        | sentence_1                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                               |
  | details | <ul><li>min: 14 tokens</li><li>mean: 23.1 tokens</li><li>max: 38 tokens</li></ul> | <ul><li>min: 43 tokens</li><li>mean: 135.28 tokens</li><li>max: 214 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                             | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
  |:-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>When did Google release their gemini-20-flash-thinking-exp model?</code>                                         | <code>OpenAI are not the only game in town here. Google released their first entrant in the category, gemini-2.0-flash-thinking-exp, on December 19th.<br>Alibaba’s Qwen team released their QwQ model on November 28th—under an Apache 2.0 license, and that one I could run on my own machine. They followed that up with a vision reasoning model called QvQ on December 24th, which I also ran locally.<br>DeepSeek made their DeepSeek-R1-Lite-Preview model available to try out through their chat interface on November 20th.<br>To understand more about inference scaling I recommend Is AI progress slowing down? by Arvind Narayanan and Sayash Kapoor.</code>                                                                                                |
  | <code>What is the name of the vision reasoning model released by Alibaba’s Qwen team, and when was it released?</code> | <code>OpenAI are not the only game in town here. Google released their first entrant in the category, gemini-2.0-flash-thinking-exp, on December 19th.<br>Alibaba’s Qwen team released their QwQ model on November 28th—under an Apache 2.0 license, and that one I could run on my own machine. They followed that up with a vision reasoning model called QvQ on December 24th, which I also ran locally.<br>DeepSeek made their DeepSeek-R1-Lite-Preview model available to try out through their chat interface on November 20th.<br>To understand more about inference scaling I recommend Is AI progress slowing down? by Arvind Narayanan and Sayash Kapoor.</code>                                                                                                |
  | <code>What is the main function of Claude Artifacts as described in the context?</code>                                | <code>We already knew LLMs were spookily good at writing code. If you prompt them right, it turns out they can build you a full interactive application using HTML, CSS and JavaScript (and tools like React if you wire up some extra supporting build mechanisms)—often in a single prompt.<br>Anthropic kicked this idea into high gear when they released Claude Artifacts, a groundbreaking new feature that was initially slightly lost in the noise due to being described half way through their announcement of the incredible Claude 3.5 Sonnet.<br>With Artifacts, Claude can write you an on-demand interactive application and then let you use it directly inside the Claude interface.<br>Here’s my Extract URLs app, entirely generated by Claude:</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          768,
          512,
          256,
          128,
          64
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 10
- `per_device_eval_batch_size`: 10
- `num_train_epochs`: 10
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 10
- `per_device_eval_batch_size`: 10
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch | Step | cosine_ndcg@10 |
|:-----:|:----:|:--------------:|
| 1.0   | 16   | 0.9846         |
| 2.0   | 32   | 1.0            |


### Framework Versions
- Python: 3.13.2
- Sentence Transformers: 4.1.0
- Transformers: 4.51.3
- PyTorch: 2.7.0
- Accelerate: 1.6.0
- Datasets: 3.5.1
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MatryoshkaLoss
```bibtex
@misc{kusupati2024matryoshka,
    title={Matryoshka Representation Learning},
    author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    year={2024},
    eprint={2205.13147},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->