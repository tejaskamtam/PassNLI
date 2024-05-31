# Before using script, run...
# pip install evaluate
from evaluate import load


# Break a string into a list of characters
def custom_tokenizer(input_string):
    token_list = [char for char in input_string]
    return token_list


# Pass in a list of lists of tuples, where each tuple is of the form ([top_k_pred], reference)
def compute_bleu_score(instance_list, use_custom_tokenizer=False):
    # Generate prediction_list and reference_list
    prediction_list = []  # list of strings (each string is a prediction)
    reference_list = (
        []
    )  # list of lists of strings (each list of strings is a list of references)

    for instance in instance_list:
        (top_k_pred, reference) = instance
        k_val = len(top_k_pred)

        prediction_list += top_k_pred
        reference_list += [[reference] for _ in range(k_val)]

    # Load BLEU evaluation metric
    bleu = evaluate.load("bleu")

    # Default tokenizer is minimalistic. Custom tokenizer breaks string into chars
    results = None
    if use_custom_tokenizer is True:
        results = bleu.compute(
            predictions=prediction_list,
            references=reference_list,
            tokenizer=custom_tokenizer,
        )
    else:
        results = bleu.compute(predictions=prediction_list, references=reference_list)
    print(results)
    return results
