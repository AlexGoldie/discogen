DESCRIPTION
The TOFU (Task of Fictitious Unlearning) dataset is a benchmark designed to evaluate machine unlearning capabilities in large language models (LLMs). It consists of question-answer pairs derived from autobiographies of 200 entirely fictitious authors. The synthetic nature ensures the data is distinct from any pretraining corpus, providing a controlled environment for testing selective forgetting. The goal is to train the model to "unlearn" specific authors (forget set) while preserving its knowledge of other authors (retain set) and general world knowledge.

DATASET COMPOSITION
The TOFU dataset contains:

- 200 Fictitious Authors: Each with a complete fictional biography and associated QA pairs
- Question-Answer Pairs: Covering biographical details, writing styles, and life events of each author (20 QA pairs per author)
- Total Size: 4,000 QA pairs across all authors
- Format: Conversational QA format compatible with instruction-tuned models

TASK CONFIGURATION
This task uses the following splits from the TOFU dataset:

- forget10: 10% of the dataset (20 authors) - information to be unlearned
- retain90: 90% of the dataset (180 authors) - information to be preserved
- holdout10: 10% holdout set for evaluation purposes

TASK OBJECTIVE
The primary goal is to implement an unlearning algorithm that satisfies:

1. Selective Forgetting: Remove the model's ability to recall information about the 20 authors in the forget set
2. Knowledge Retention: Maintain strong performance on the 180 authors in the retain set

EVALUATION METRICS
The task uses 6 evaluation metrics:

1. forget_quality: KS test comparing truth ratio distributions (correct vs perturbed answers) on forget set (higher is better)
2. forget_Q_A_Prob: Probability assigned to correct answers on forget set (lower is better)
3. forget_Q_A_ROUGE: ROUGE-L recall between generated and correct answers on forget set (lower is better)
4. model_utility: Harmonic mean aggregate across 9 sub-metrics measuring retention on retain set, real authors, and world facts (higher is better)
   - Retain set (180 authors): probability, ROUGE-L, truth ratio
   - Real authors: normalized probability, ROUGE-L, truth ratio
   - World facts: normalized probability, ROUGE-L, truth ratio
5. privleak: Membership inference attack vulnerability using min_k metric (lower is better)
6. extraction_strength: Measures model's ability to memorize/extract information from forget set (lower is better)

The goal is to optimize for all of them, from which a final score will be computed.
