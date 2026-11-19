# Evaluation metrics and ablation studies in related work

COSC 497 Senior Design Project I
Project: LLM Hallucination Detection

This note answers the supervisor's request to check which evaluation metrics and ablation studies the related works use, and what this project should add as a result. It covers the papers already listed in `paper-matrix.md`.

## 1. Metrics used in related work

| Work | Task setting | Metrics reported |
|---|---|---|
| SciHal25 overview (Li et al., 2025) | Claim vs. scientific reference, 3 coarse and 8 fine labels | Weighted F1 (primary, used for ranking) |
| Schopf et al. (2025) | SciHal25 system | Weighted F1 |
| Galimzianova et al. (2025) | SciHal25 system | Weighted F1 on the leaderboard; accuracy, macro F1, weighted F1 and per-class precision, recall and F1 on a 360-sample development set |
| SummaC (Laban et al., 2022) | Summary vs. source document, binary | Balanced accuracy (headline: 74.4% for SummaCConv) |
| TRUE (Honovich et al., 2022) | 11 grounded-text datasets, binary | Example-level evaluation rather than system-level correlation; AUC-ROC (to verify in the paper) |
| AlignScore (Zha et al., 2023) | Context vs. claim alignment | AUC-ROC on SummaC and TRUE; Spearman correlation on other datasets |
| SelfCheckGPT (Manakul et al., 2023) | No reference, sampled responses | AUC-PR at sentence level; Pearson and Spearman correlation with human judgements at passage level |
| RAGTruth (Niu et al., 2024) | RAG responses with annotated spans | Precision, recall and F1 at response level; character-level precision, recall and F1 at span level |
| MiniCheck (Tang et al., 2024) | Claim vs. grounding document | Balanced accuracy on LLM-AggreFact (to verify in the paper); fixed mid-range threshold instead of per-dataset tuning; cost comparison |

### Observations

- The metric depends on the label format. Multi-class tasks (SciHal25) use weighted F1. Binary grounded-consistency benchmarks (SummaC, TRUE, LLM-AggreFact) prefer balanced accuracy or AUC-ROC, because the classes are usually imbalanced and plain accuracy is misleading.
- Threshold-free metrics (AUC-ROC, AUC-PR) are used when the detector outputs a score. They separate the quality of the score from the choice of threshold.
- When a single threshold is needed, papers either tune it on validation data (SummaC) or fix it in advance (MiniCheck). The policy is always stated.
- Only RAGTruth evaluates below the response level (spans). Our dataset has no span labels, so this does not apply.

## 2. Ablations and comparisons in related work

| Work | What was varied | Finding |
|---|---|---|
| Schopf et al. (2025) | Ensemble vs. single model | Ensemble 0.60 vs. best single model 0.57 weighted F1 |
| | Prompting vs. fine-tuning | Best zero-shot LLM 0.49; every fine-tuned encoder 0.50 or higher |
| | Claim decomposition in the prompt | Lower score: 0.44 vs. 0.49 |
| | Training data batches | DeBERTa-v3 NLI: batch 3 alone 0.57, batch 2 alone 0.50, both 0.56 |
| Galimzianova et al. (2025) | Encoder | DeBERTa-v3-large-NLI 0.58, ModernBERT-large 0.57, SciBERT-NLI 0.35 |
| | Model size | ModernBERT large vs. base: 0.57 vs. 0.56 |
| | Direct fine-tuning vs. feature pipeline | 0.58 vs. 0.51 |
| | Fine-tuned encoder vs. few-shot LLM | 0.58 vs. 0.45 |
| SummaC | Granularity of the NLI input | Sentence-level pairs work better than whole documents, because NLI training data is sentence-level |
| AlignScore | Classification head, splitting method, upstream training task | Chunk-level splitting performs best |
| SelfCheckGPT | Number of samples, external knowledge, choice of LLM | Reported as separate analyses |
| RAGTruth | Hallucination type | Span-level recall is reported per type |
| MiniCheck | Removing each kind of synthetic training data; claim decomposition; decontextualization | The two kinds of synthetic data complement each other |

### Observations

- Neither SciHal25 system paper isolates the effect of NLI pretraining. Both attribute their best result to it, but neither trains the same encoder with and without an NLI checkpoint.
- The SciHal25 overview reports no system-level error analysis and no per-class results for systems.
- The most common ablation axes are: encoder family, model size, training data, input format or granularity, and single model vs. ensemble.

## 3. Comparison with this project

Already in `experimental-protocol.md`:

- weighted F1 (primary), macro F1, per-class precision, recall and F1, accuracy, confusion matrix;
- mean and standard deviation over six grouped folds;
- input-condition ablations: context + response, prompt + context + response, prompt + response, response only;
- encoder comparison: BERT, RoBERTa, DeBERTa-v3;
- zero-shot pretrained NLI vs. fine-tuned models.

Missing compared with the literature:

| Gap | Source | Cost to add |
|---|---|---|
| Balanced accuracy | SummaC, MiniCheck | Computed from the saved out-of-fold predictions |
| AUC-ROC and AUC-PR | TRUE, AlignScore, SelfCheckGPT | Needs the predicted probability for each example, not only the label |
| Stated threshold policy | SummaC, MiniCheck | One sentence in the protocol |
| Results per hallucination type | RAGTruth | The `hallucination_type` field is already kept for error analysis |
| NLI pretraining ablation | Not done in the SciHal25 papers | One extra run: DeBERTa-v3 with and without the NLI checkpoint |
| Binary vs. three-way NLI output | AlignScore | Compare the current combined non-entailment mapping with a binary head |
| Ensemble vs. single model | Schopf et al. | Majority vote over the three fine-tuned models, no retraining |
| Model size | Galimzianova et al. | Base vs. large, if GPU time allows |

## 4. Proposed additions

In priority order:

1. Add balanced accuracy, AUC-ROC and AUC-PR to the results table for every model.
2. State the threshold policy (fixed 0.5 unless stated otherwise).
3. Add a per-hallucination-type recall table for the best model.
4. Run the NLI pretraining ablation on DeBERTa-v3.
5. Report the majority-vote ensemble.
6. Present the existing input-condition and zero-shot vs. fine-tuned results explicitly as ablations in the report.

Items 1 to 3 and 5 need no new training. Item 4 needs one additional cross-validation run.

## 5. Limits of this review

- The SciHal25 papers were read in full. The entries for SummaC, TRUE, AlignScore, SelfCheckGPT, RAGTruth and MiniCheck are based on abstracts and summaries, and the two cells marked "to verify" must be checked against the papers.
- Carla and Uban (2025) is not covered yet.
- All numbers in sections 1 and 2 are the authors' reported results on their own datasets and are not comparable with our results.

## References

- Li et al. (2025), SciHal25 overview: https://aclanthology.org/2025.sdp-1.29/
- Schopf et al. (2025): https://aclanthology.org/2025.sdp-1.33/
- Galimzianova et al. (2025): https://aclanthology.org/2025.sdp-1.34/
- Laban et al. (2022), SummaC: https://doi.org/10.1162/tacl_a_00453
- Honovich et al. (2022), TRUE: https://aclanthology.org/2022.naacl-main.287/
- Zha et al. (2023), AlignScore: https://aclanthology.org/2023.acl-long.634/
- Manakul et al. (2023), SelfCheckGPT: https://aclanthology.org/2023.emnlp-main.557/
- Niu et al. (2024), RAGTruth: https://aclanthology.org/2024.acl-long.585/
- Tang et al. (2024), MiniCheck: https://arxiv.org/abs/2404.10774
