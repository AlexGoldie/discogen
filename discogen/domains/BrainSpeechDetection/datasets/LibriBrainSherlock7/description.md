DESCRIPTION
LibriBrainSherlock7 is a neural decoding task using MEG recordings from a single participant listening to one complete audiobook. The objective is to detect whether speech is present at each time point based solely on brain activity patterns. This binary classification task serves as a foundational benchmark for brain-to-speech decoding, testing whether neural signals contain sufficient information to distinguish speech from silence or non-speech audio.

OBSERVATION SPACE
Each observation consists of:

MEG sensor recordings: Multi-channel time-series data from magnetometers and gradiometers
Temporal resolution: ~1ms sampling rate
Spatial coverage: Full-head sensor array
Feature options: Raw signals, filtered bands, or extracted features

TARGET SPACE
Binary labels indicating speech presence:

0: No speech (silence, pauses, or non-speech audio)
1: Speech present (any spoken content)

Labels are time-aligned with MEG recordings at millisecond precision.

TASK STRUCTURE
Input: MEG brain activity at time t
Output: Binary prediction of speech presence

DATASET STRUCTURE
Source: Single book (book 7) from LibriBrain dataset
Duration: Multiple hours of continuous recording
Train/test splits: A randomly selected chapter will act as the test split

EVALUATION METRICS
The primary evaluation metric is macro-F1 score, which balances performance across both classes (speech and non-speech):

Macro-F1: Average of F1 scores for both classes (speech and non-speech)

F1_positive = 2 × TP / (2 × TP + FP + FN)
F1_negative = 2 × TN / (2 × TN + FN + FP)
Macro-F1 = 0.5 × (F1_positive + F1_negative)

Threshold optimization: The classification threshold is tuned to maximize macro-F1 on the validation set

Candidate thresholds derived from precision-recall curve
Best threshold selected via vectorized evaluation across all candidates
Same threshold applied during test evaluation

This metric ensures balanced performance on both speech detection and silence/non-speech detection, preventing models from simply predicting the majority class.
