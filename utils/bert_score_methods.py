# Before using script, run...
# pip install evaluate
# from evaluate import load
from bert_score import score
import torch


def compute_bert_score(references, predictions):
    # Generate prediction_list and reference_list
    reference_list = []  # list of strings (each string is a reference)
    prediction_list = []  # list of strings (each string is a prediction)

    for i in range(len(references)):
        reference = references[i]
        top_k_pred = predictions[i]
        k_val = len(top_k_pred)

        prediction_list += top_k_pred
        reference_list += [reference for _ in range(k_val)]

    # Load BERTScore evaluation metric
    # bertscore = load("bertscore")

    # results = bertscore.compute(predictions=prediction_list, references=reference_list, lang="en")
    P, R, F1 = score(prediction_list, reference_list, lang='en', verbose=True)
    results = {"meanF1": F1.mean().tolist(), "precision": P.tolist(), "recall": R.tolist(), "f1": F1.tolist()}
    return results
