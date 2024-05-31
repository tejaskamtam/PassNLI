# Before using script, run...
# pip install evaluate
from evaluate import load


# Pass in a list of lists of tuples, where each tuple is of the form ([top_k_pred], reference)
def compute_bert_score(instance_list):
    # Generate prediction_list and reference_list
    prediction_list = []  # list of strings (each string is a prediction)
    reference_list = []  # list of strings (each string is a reference)

    for instance in instance_list:
        (top_k_pred, reference) = instance
        k_val = len(top_k_pred)

        prediction_list += top_k_pred
        prediction_list += [reference for _ in range(k_val)]

    # Load BLEU evaluation metric
    bertscore = evaluate.load("bertscore")

    results = bleu.compute(
        predictions=prediction_list, references=reference_list, lang="en"
    )
    print(results)
    return results
