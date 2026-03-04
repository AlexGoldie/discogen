DESCRIPTION
The MUSE (Machine Unlearning of Sensitive Entities) News Benchmark is designed to evaluate machine unlearning capabilities in large language models (LLMs) with respect to copyrighted news content. This task specifically focuses on news articles from various reputable sources covering diverse topics, providing a realistic testbed for assessing whether models can selectively forget specific content while maintaining general knowledge and capabilities. The goal is to train the model to "unlearn" specific copyrighted news articles (forget set) while preserving knowledge of other news content (retain set).

DATASET COMPOSITION
Training data from the MUSE-News dataset:

- Forget corpus: 889 news articles to be unlearned
- Retain corpus (retain1): 1,777 news articles to maintain model capabilities
- Format: Raw text passages for pretraining-style unlearning

EVALUATION DATASET COMPOSITION
Evaluation data from the MUSE-News dataset:

- Knowledge memorization (knowmem): 100 QA pairs from forget set (forget_qa split), 100 QA pairs from retain set (retain_qa split)
- Verbatim memorization (verbmem): Text completion tasks on forget set to test exact memorization

TASK OBJECTIVE
The primary goal is to implement an unlearning algorithm that satisfies:

1. Content Forgetting: Remove the model's ability to reproduce or recall specific details from the forget set
2. Knowledge Retention: Maintain strong performance on the retain set

EVALUATION METRICS
The task uses 5 evaluation metrics:

1. forget_knowmem_ROUGE: ROUGE-L F1 score on 100 forget QA pairs from forget_qa split (lower is better, indicates reduced knowledge recall)
2. retain_knowmem_ROUGE: ROUGE-L F1 score on 100 retain QA pairs from retain_qa split (higher is better, indicates preserved knowledge)
3. forget_verbmem_ROUGE: ROUGE-L F1 score on text completions from forget split testing verbatim memorization (lower is better, indicates reduced memorization)
4. privleak: Privacy leakage metric using min_k% probability on forget split (lower is better, indicates successful unlearning)
5. extraction_strength: Exact extraction capability measured on forget split verbatim memorization tasks (lower is better, indicates reduced memorization)

The goal is to optimize for all of them, from which a final score will be computed.
