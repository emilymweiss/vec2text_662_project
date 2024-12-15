'''
Note: Evaluation script modified from code in the vec2text github's Python files and README.

Sources:
1. general structure and get_gtr_embeddings function from:
        https://github.com/jxmorris12/vec2text/tree/master?tab=readme-ov-file#similarly-you-can-invert-gtr-base-embeddings-with-the-following-example
2. bleu, tf1, and exact; and cosine similarity code modified from _text_comparison_metrics, and
        eval_generation_metrics in vec2text/trainers/base.py:
        https://github.com/jxmorris12/vec2text/blob/master/vec2text/trainers/base.py

Original Author: John X. Morris
Editors: Emily Weiss, Skyler Hallinan
'''

os.environ["HF_HOME"] = "/gscratch/xlab/hallisky/cache/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/gscratch/xlab/hallisky/cache/"
os.environ["HF_DATASETS_CACHE"] = "/gscratch/xlab/hallisky/cache/"
os.environ["TRANSFORMERS_CACHE"] = "/gscratch/xlab/hallisky/cache/"

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

import nltk
nltk.download('punkt')

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


def get_nq_eval_text():
    ds = load_dataset("jxm/nq_corpus_dpr")
    return ds['dev']['text']


def main(args):

    # Initialize the models
    if args.model == "hallisky/gtr-nq-32-corrector-5epoch":
        encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

        inversion_model = vec2text.models.InversionModel.from_pretrained("hallisky/gtr-nq-32-base-5epoch", cache_dir="/gscratch/xlab/hallisky/cache/")
        corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained("hallisky/gtr-nq-32-corrector-5epoch", cache_dir="/gscratch/xlab/hallisky/cache/")
    else:
        pass

    corrector = vec2text.load_corrector(inversion_model, corrector_model)

    # get nq dev split
    truth = get_nq_eval_text()

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
    with open(f"gtr_nq_{args.num_steps}_results.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument('--batch_size', type=int, default=384, help="eval batch size")
    # parser.add_argument('--output_dir', type=str, default="ood_results")
    # parser.add_argument('--max_length', type=int, default=32, help="max length of embedding")
    parser.add_argument('--num_steps', type=int, default=0, help="num of corrector steps, 0-20")
    parser.add_argument('--model', type=str, default="hallisky/gtr-nq-32-corrector-5epoch", help="Options: [hallisky/gtr-nq-32-corrector-5epoch, hallisky/gtr-nq-32-hypothesizer-5epoch]")

    main(parser.parse_args())

    """
    python3 -m vec2text.evaluate_ours_gtr --num_steps 0 --batch_size 384
    python3 -m vec2text.evaluate_ours_gtr --num_steps 20 --batch_size 384
    """
