from bleu_score_methods import compute_bleu_score
from bert_score_methods import compute_bert_score
from results import *
import sys

if __name__ == "__main__":
    # Compute BLEU scores (using custom tokenizer)
    bleu_score_gpt4o_0shot = compute_bleu_score(labels, gpt4o_0shot, True)
    bleu_score_gpt4o_0shot_cot = compute_bleu_score(labels, gpt4o_0shot_cot, True)
    bleu_score_gpt4o_5shot = compute_bleu_score(labels, gpt4o_5shot, True)
    bleu_score_gpt4o_5shot_cot = compute_bleu_score(labels, gpt4o_5shot_cot, True)
    bleu_score_gpt4o_10shot = compute_bleu_score(labels, gpt4o_10shot, True)
    bleu_score_gpt4o_10shot_cot = compute_bleu_score(labels, gpt4o_10shot_cot, True)

    bleu_score_gpt4turbo_0shot = compute_bleu_score(labels, gpt4turbo_0shot, True)
    bleu_score_gpt4turbo_0shot_cot = compute_bleu_score(labels, gpt4turbo_0shot_cot, True)
    bleu_score_gpt4turbo_5shot = compute_bleu_score(labels, gpt4turbo_5shot, True)
    bleu_score_gpt4turbo_5shot_cot = compute_bleu_score(labels, gpt4turbo_5shot_cot, True)
    bleu_score_gpt4turbo_10shot = compute_bleu_score(labels, gpt4turbo_10shot, True)
    bleu_score_gpt4turbo_10shot_cot = compute_bleu_score(labels, gpt4turbo_10shot_cot, True)

    bleu_score_gpt3turbo_0shot = compute_bleu_score(labels, gpt3turbo_0shot, True)
    bleu_score_gpt3turbo_0shot_cot = compute_bleu_score(labels, gpt3turbo_0shot_cot, True)
    bleu_score_gpt3turbo_5shot = compute_bleu_score(labels, gpt3turbo_5shot, True)
    bleu_score_gpt3turbo_5shot_cot = compute_bleu_score(labels, gpt3turbo_5shot_cot, True)
    bleu_score_gpt3turbo_10shot = compute_bleu_score(labels, gpt3turbo_10shot, True)
    bleu_score_gpt3turbo_10shot_cot = compute_bleu_score(labels, gpt3turbo_10shot_cot, True)

    # Output BLEU Scores to stdout
    sys.stdout.write("bleu_score_gpt4o_0shot = " + str(bleu_score_gpt4o_0shot) + "\n")
    sys.stdout.write("bleu_score_gpt4o_0shot_cot = " + str(bleu_score_gpt4o_0shot_cot) + "\n")
    sys.stdout.write("bleu_score_gpt4o_5shot = " + str(bleu_score_gpt4o_5shot) + "\n")
    sys.stdout.write("bleu_score_gpt4o_5shot_cot = " + str(bleu_score_gpt4o_5shot_cot) + "\n")
    sys.stdout.write("bleu_score_gpt4o_10shot = " + str(bleu_score_gpt4o_10shot) + "\n")
    sys.stdout.write("bleu_score_gpt4o_10shot_cot = " + str(bleu_score_gpt4o_10shot_cot) + "\n")

    sys.stdout.write("bleu_score_gpt4turbo_0shot = " + str(bleu_score_gpt4turbo_0shot) + "\n")
    sys.stdout.write("bleu_score_gpt4turbo_0shot_cot = " + str(bleu_score_gpt4turbo_0shot_cot) + "\n")
    sys.stdout.write("bleu_score_gpt4turbo_5shot = " + str(bleu_score_gpt4turbo_5shot) + "\n")
    sys.stdout.write("bleu_score_gpt4turbo_5shot_cot = " + str(bleu_score_gpt4turbo_5shot_cot) + "\n")
    sys.stdout.write("bleu_score_gpt4turbo_10shot = " + str(bleu_score_gpt4turbo_10shot) + "\n")
    sys.stdout.write("bleu_score_gpt4turbo_10shot_cot = " + str(bleu_score_gpt4turbo_10shot_cot) + "\n")

    sys.stdout.write("bleu_score_gpt3turbo_0shot = " + str(bleu_score_gpt3turbo_0shot) + "\n")
    sys.stdout.write("bleu_score_gpt3turbo_0shot_cot = " + str(bleu_score_gpt3turbo_0shot_cot) + "\n")
    sys.stdout.write("bleu_score_gpt3turbo_5shot = " + str(bleu_score_gpt3turbo_5shot) + "\n")
    sys.stdout.write("bleu_score_gpt3turbo_5shot_cot = " + str(bleu_score_gpt3turbo_5shot_cot) + "\n")
    sys.stdout.write("bleu_score_gpt3turbo_10shot = " + str(bleu_score_gpt3turbo_10shot) + "\n")
    sys.stdout.write("bleu_score_gpt3turbo_10shot_cot = " + str(bleu_score_gpt3turbo_10shot_cot) + "\n")

    # Compute BERT scores
    bert_score_gpt4o_0shot = compute_bert_score(labels, gpt4o_0shot)
    bert_score_gpt4o_0shot_cot = compute_bert_score(labels, gpt4o_0shot_cot)
    bert_score_gpt4o_5shot = compute_bert_score(labels, gpt4o_5shot)
    bert_score_gpt4o_5shot_cot = compute_bert_score(labels, gpt4o_5shot_cot)
    bert_score_gpt4o_10shot = compute_bert_score(labels, gpt4o_10shot)
    bert_score_gpt4o_10shot_cot = compute_bert_score(labels, gpt4o_10shot_cot)

    bert_score_gpt4turbo_0shot = compute_bert_score(labels, gpt4turbo_0shot)
    bert_score_gpt4turbo_0shot_cot = compute_bert_score(labels, gpt4turbo_0shot_cot)
    bert_score_gpt4turbo_5shot = compute_bert_score(labels, gpt4turbo_5shot)
    bert_score_gpt4turbo_5shot_cot = compute_bert_score(labels, gpt4turbo_5shot_cot)
    bert_score_gpt4turbo_10shot = compute_bert_score(labels, gpt4turbo_10shot)
    bert_score_gpt4turbo_10shot_cot = compute_bert_score(labels, gpt4turbo_10shot_cot)

    bert_score_gpt3turbo_0shot = compute_bert_score(labels, gpt3turbo_0shot)
    bert_score_gpt3turbo_0shot_cot = compute_bert_score(labels, gpt3turbo_0shot_cot)
    bert_score_gpt3turbo_5shot = compute_bert_score(labels, gpt3turbo_5shot)
    bert_score_gpt3turbo_5shot_cot = compute_bert_score(labels, gpt3turbo_5shot_cot)
    bert_score_gpt3turbo_10shot = compute_bert_score(labels, gpt3turbo_10shot)
    bert_score_gpt3turbo_10shot_cot = compute_bert_score(labels, gpt3turbo_10shot_cot)

    # Output BERT Scores to stdout
    sys.stdout.write("bert_score_gpt4o_0shot = " + str(bert_score_gpt4o_0shot) + "\n")
    sys.stdout.write("bert_score_gpt4o_0shot_cot = " + str(bert_score_gpt4o_0shot_cot) + "\n")
    sys.stdout.write("bert_score_gpt4o_5shot = " + str(bert_score_gpt4o_5shot) + "\n")
    sys.stdout.write("bert_score_gpt4o_5shot_cot = " + str(bert_score_gpt4o_5shot_cot) + "\n")
    sys.stdout.write("bert_score_gpt4o_10shot = " + str(bert_score_gpt4o_10shot) + "\n")
    sys.stdout.write("bert_score_gpt4o_10shot_cot = " + str(bert_score_gpt4o_10shot_cot) + "\n")

    sys.stdout.write("bert_score_gpt4turbo_0shot = " + str(bert_score_gpt4turbo_0shot) + "\n")
    sys.stdout.write("bert_score_gpt4turbo_0shot_cot = " + str(bert_score_gpt4turbo_0shot_cot) + "\n")
    sys.stdout.write("bert_score_gpt4turbo_5shot = " + str(bert_score_gpt4turbo_5shot) + "\n")
    sys.stdout.write("bert_score_gpt4turbo_5shot_cot = " + str(bert_score_gpt4turbo_5shot_cot) + "\n")
    sys.stdout.write("bert_score_gpt4turbo_10shot = " + str(bert_score_gpt4turbo_10shot) + "\n")
    sys.stdout.write("bert_score_gpt4turbo_10shot_cot = " + str(bert_score_gpt4turbo_10shot_cot) + "\n")

    sys.stdout.write("bert_score_gpt3turbo_0shot = " + str(bert_score_gpt3turbo_0shot) + "\n")
    sys.stdout.write("bert_score_gpt3turbo_0shot_cot = " + str(bert_score_gpt3turbo_0shot_cot) + "\n")
    sys.stdout.write("bert_score_gpt3turbo_5shot = " + str(bert_score_gpt3turbo_5shot) + "\n")
    sys.stdout.write("bert_score_gpt3turbo_5shot_cot = " + str(bert_score_gpt3turbo_5shot_cot) + "\n")
    sys.stdout.write("bert_score_gpt3turbo_10shot = " + str(bert_score_gpt3turbo_10shot) + "\n")
    sys.stdout.write("bert_score_gpt3turbo_10shot_cot = " + str(bert_score_gpt3turbo_10shot_cot) + "\n")