from results import *
from Levenshtein import distance
from statistics import mean
import sys

def compute_mean_lev(references, predictions, use_mins=True):
    lev_agg = []
    for i in range(len(references)):
        lev_scores = [distance(references[i], pred) for pred in predictions[i]]
        if use_mins is True:
            lev_agg.append(min(lev_scores))
        else:
            lev_agg += lev_scores
    return str(mean(lev_agg)) + "\n"


if __name__ == "__main__":
    sys.stdout.write("GPT4o\n")
    sys.stdout.write(compute_mean_lev(labels, gpt4o_0shot))
    sys.stdout.write(compute_mean_lev(labels, gpt4o_0shot_cot))
    sys.stdout.write(compute_mean_lev(labels, gpt4o_5shot))
    sys.stdout.write(compute_mean_lev(labels, gpt4o_5shot_cot))
    sys.stdout.write(compute_mean_lev(labels, gpt4o_10shot))
    sys.stdout.write(compute_mean_lev(labels, gpt4o_10shot_cot))

    sys.stdout.write("\nGPT4 Turbo\n")
    sys.stdout.write(compute_mean_lev(labels, gpt4turbo_0shot))
    sys.stdout.write(compute_mean_lev(labels, gpt4turbo_0shot_cot))
    sys.stdout.write(compute_mean_lev(labels, gpt4turbo_5shot))
    sys.stdout.write(compute_mean_lev(labels, gpt4turbo_5shot_cot))
    sys.stdout.write(compute_mean_lev(labels, gpt4turbo_10shot))
    sys.stdout.write(compute_mean_lev(labels, gpt4turbo_10shot_cot))

    sys.stdout.write("\nGPT3 Turbo\n")
    sys.stdout.write(compute_mean_lev(labels, gpt3turbo_0shot))
    # sys.stdout.write(compute_mean_lev(labels, gpt3turbo_0shot_cot))
    sys.stdout.write(compute_mean_lev(labels, gpt3turbo_5shot))
    sys.stdout.write(compute_mean_lev(labels, gpt3turbo_5shot_cot))
    sys.stdout.write(compute_mean_lev(labels, gpt3turbo_10shot))
    sys.stdout.write(compute_mean_lev(labels, gpt3turbo_10shot_cot))