DESCRIPTION
The WMDP (Weapons of Mass Destruction Proxy) Cyber Benchmark is designed to evaluate and mitigate hazardous cybersecurity knowledge in large language models (LLMs). It consists of 1,987 multiple-choice questions covering sensitive cybersecurity topics including reconnaissance, vulnerability discovery, exploitation techniques, and cyber tactics. The goal is to unlearn hazardous cybersecurity knowledge (forget corpus) while preserving general language capabilities (retain corpus).

DATASET COMPOSITION
The WMDP Cyber dataset contains:

- Forget corpus: 1,000 text documents containing hazardous cybersecurity information
- Retain corpus: 4,473 text documents of general content for maintaining model utility
- Format: Pretraining-style text data and multiple-choice evaluation questions

TASK CONFIGURATION
This task uses the following components:

- cyber-forget-corpus: Text data containing hazardous cybersecurity knowledge to be unlearned
- cyber-retain-corpus: General text data to maintain model capabilities

TASK OBJECTIVE
The primary goal is to implement an unlearning algorithm that satisfies:

1. Hazardous Knowledge Removal: Reduce the model's ability to answer questions about sensitive cybersecurity topics
2. General Capability Preservation: Maintain strong performance on general language tasks

EVALUATION METRICS
The task uses 2 evaluation metrics:

1. wmdp_cyber/acc: Accuracy on the 1,987-question WMDP Cyber multiple-choice benchmark (lower is better)
2. mmlu_stem/acc: Accuracy on the STEM subsection of Massive Multitask Language Understanding (MMLU) benchmark containing 3,153 questions, tested via multiple-choice questions from lm_eval harness (higher is better)

The goal is to optimize for both of them, from which a final score will be computed.
