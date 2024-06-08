# Before using script, run...
# pip install evaluate
from evaluate import load


# Break a string into a list of characters
def custom_tokenizer(input_string):
    token_list = [char for char in input_string]
    return token_list


def compute_bleu_score(references, predictions, use_custom_tokenizer=False):
    # Generate prediction_list and reference_list
    reference_list = []  # list of lists of strings
    prediction_list = []  # list of strings (each string is a prediction)

    for i in range(len(references)):
        reference = references[i]
        top_k_pred = predictions[i]
        k_val = len(top_k_pred)

        prediction_list += top_k_pred
        reference_list += [[reference] for _ in range(k_val)]

    # Load BLEU evaluation metric
    bleu = load("bleu")

    # Default tokenizer is minimalistic. Custom tokenizer breaks string into chars
    results = None
    if use_custom_tokenizer is True:
        results = bleu.compute(predictions=prediction_list, references=reference_list, tokenizer=custom_tokenizer)
    else:
        results = bleu.compute(predictions=prediction_list, references=reference_list)
    return results
