'''
Note: Evaluation script modified from code in the vec2text github's Python files and README.

Sources:
1. general structure and get_gtr_embeddings function from:
        https://github.com/jxmorris12/vec2text/tree/master?tab=readme-ov-file#similarly-you-can-invert-gtr-base-embeddings-with-the-following-example
2. bleu, tf1, and exact; and cosine similarity code modified from _text_comparison_metrics, and
        eval_generation_metrics in vec2text/trainers/base.py:
        https://github.com/jxmorris12/vec2text/blob/master/vec2text/trainers/base.py
 
Original Author: John X. Morris
Editor: Emily Weiss
'''

import math
import vec2text
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from datasets import load_dataset
import argparse
import json
import evaluate # for evaluation (bleu and tf1...)
from tqdm import tqdm
import numpy
import datasets
import sys
import os

import nltk
nltk.download('punkt')

MAX_TEXTS = 20000

bleu_score = evaluate.load("sacrebleu")

'''Exact function from Source 1.'''
def get_gtr_embeddings(text_list,
                       encoder: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       max_length=32) -> torch.Tensor:

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=max_length,
                       truncation=True,
                       padding="max_length",).to("cuda")

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

    return embeddings

"""Based on _text_comparison_metrics function from link in Source 2."""
def get_tf1(preds, truth):
    f1s = []
    for i in range(len(preds)):
        true_words = nltk.tokenize.word_tokenize(truth[i])
        pred_words = nltk.tokenize.word_tokenize(preds[i])

        true_words_set = set(true_words)
        pred_words_set = set(pred_words)

        TP = len(true_words_set & pred_words_set)
        FP = len(true_words_set) - len(true_words_set & pred_words_set)
        FN = len(pred_words_set) - len(true_words_set & pred_words_set)

        P = (TP) / (TP + FP + 1e-20)
        R = (TP) / (TP + FN + 1e-20)

        try:
            f1 = (2 * P * R) / (P + R + 1e-20)
        except ZeroDivisionError:
            f1 = 0.0
        f1s.append(f1)

    return sum(f1s)/len(f1s)


'''Based on _text_comparison_metrics function from link in Source 2.'''
def get_exact(preds, truth):
    return sum(numpy.array(truth)  == numpy.array(preds))/len(numpy.array(truth)  == numpy.array(preds))


'''Based on eval_generation_metrics function from link in Source 2.'''
def get_cos(preds_embeddings, truth_embeddings):
    return torch.nn.CosineSimilarity(dim=1)(preds_embeddings, truth_embeddings).mean().item()

def get_random_texts():
    with open("random_texts.txt", "r") as f:
        texts = f.readlines()
    return texts

'''https://huggingface.co/datasets/tabilab/biosses'''
def get_biosses():
    ds = load_dataset("tabilab/biosses")
    return ds['train']['sentence1'][:MAX_TEXTS]

'''https://huggingface.co/datasets/mteb/medrxiv-clustering-p2p from MTEB benchmark'''
def get_medrxiv():
    ds = load_dataset("mteb/medrxiv-clustering-p2p")
    return ds['test']['sentences'][:MAX_TEXTS]

'''https://huggingface.co/datasets/mteb/biorxiv-clustering-p2p from MTEB benchmark'''
def get_biorxiv():
    ds = load_dataset("mteb/biorxiv-clustering-p2p")
    return ds['test']['sentences'][:MAX_TEXTS]

def main(args):

    # Initialize the models
    if args.model == "gtr-base":
        encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
        corrector = vec2text.load_pretrained_corrector(args.model)
    elif args.model == "hallisky/gtr-nq-32-corrector-5epoch":
        encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

        inversion_model = vec2text.models.InversionModel.from_pretrained("hallisky/gtr-nq-32-base-5epoch", cache_dir="/gscratch/xlab/hallisky/cache/")
        corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained("hallisky/gtr-nq-32-corrector-5epoch", cache_dir="/gscratch/xlab/hallisky/cache/")
        corrector = vec2text.load_corrector(inversion_model, corrector_model)
    else:
        sys.exit("Invalid model provided")

    # get nq dev split
    if args.dataset == "random_texts":
        truth = get_random_texts()
    elif args.dataset == "biosses":
        truth = get_biosses()
    elif args.dataset == "medrxiv":
        truth = get_medrxiv()
    elif args.dataset == "biorxiv":
        truth = get_biorxiv()
    else:
        sys.exit("Unsupported dataset.")

    results = []
    results_embeddings = []
    truth_embeddings = []

    # batch the embeddings and inversion
    for i in tqdm(range(0, len(truth), args.batch_size)):
        current_batch = truth[i: i + args.batch_size]

        embeddings = get_gtr_embeddings(current_batch, encoder, tokenizer, 32)

        current_results = vec2text.invert_embeddings(
            embeddings=embeddings.cuda(),
            corrector=corrector,
            num_steps=args.num_steps
        )

        current_results_embeddings = get_gtr_embeddings(current_results, encoder, tokenizer, 32)

        results.extend(current_results)  # Extend results with current batch
        results_embeddings.extend(current_results_embeddings)  # Append embeddings directly
        truth_embeddings.extend(embeddings)  # Append embeddings directly

    # get metrics
    bleu_result = bleu_score.compute(references=truth, predictions=results)
    print("Bleu: ", bleu_result)

    tf1_result = get_tf1(preds=results, truth=truth)
    print("TF1: ", tf1_result)

    exact_result = get_exact(preds=results, truth=truth)
    print("Exact: ", exact_result)

    cos_result = get_cos(preds_embeddings=torch.stack(results_embeddings), truth_embeddings=torch.stack(truth_embeddings))
    print("Cos Sim: ", cos_result)

    metrics = {"num_steps": args.num_steps,
           "bleu_result": bleu_result,
           "tf1_result": tf1_result,
           "exact_result": exact_result,
           "cosine_similarity": cos_result,
           }

    # Save to file
    if args.model == "gtr-base":
        with open(f"gtr_pt_{args.num_steps}_{args.dataset}_results.json", "w") as f:
            json.dump(metrics, f, indent=4)
    else:
        with open(f"gtr_ours_{args.num_steps}_{args.dataset}_results.json", "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument('--batch_size', type=int, default=384, help="eval batch size")
    parser.add_argument('--num_steps', type=int, default=0, help="num of corrector steps, 0-20")
    parser.add_argument('--model', type=str, default="gtr-base", help="Options: [hallisky/gtr-nq-32-corrector-5epoch, gtr-base]")
    parser.add_argument('--dataset', type=str, default="random_texts", help="Options: random_texts, biosses, medrxiv, biorxiv")

    main(parser.parse_args())

    """
    python evaluate_ood.py --model gtr-base --dataset random_texts --num_steps 0 --batch_size 384
    python evaluate_ood.py --model gtr-base --dataset random_texts --num_steps 20 --batch_size 384

    python evaluate_ood.py --model gtr-base --dataset biosses --num_steps 0 --batch_size 384
    python evaluate_ood.py --model gtr-base --dataset biosses --num_steps 20 --batch_size 384

    python evaluate_ood.py --model gtr-base --dataset medrxiv --num_steps 0 --batch_size 384
    python evaluate_ood.py --model gtr-base --dataset medrxiv --num_steps 20 --batch_size 384

    python evaluate_ood.py --model gtr-base --dataset biorxiv --num_steps 0 --batch_size 384
    python evaluate_ood.py --model gtr-base --dataset biorxiv --num_steps 20 --batch_size 384
    """