'''
Note: Evaluation script modified from code in the vec2text github Python files and README.

Sources:
1. general structure from:
        https://github.com/jxmorris12/vec2text/tree/master?tab=readme-ov-file#similarly-you-can-invert-gtr-base-embeddings-with-the-following-example
2. bleu, tf1, and exact; and cosine similarity code modified from _text_comparison_metrics, and
        eval_generation_metrics in vec2text/trainers/base.py:
        https://github.com/jxmorris12/vec2text/blob/master/vec2text/trainers/base.py
3. get_embeddings_openai function modified from Morris et al.'s' example:
        https://colab.research.google.com/drive/14RQFRF2It2Kb8gG3_YDhP_6qE0780L8h?usp=sharing
4. Embeddings docs: https://platform.openai.com/docs/guides/embeddings
5. get_msmarco_eval_text based on load_msmarco_corpus and dataset_from_args functions from:
        https://github.com/jxmorris12/vec2text/blob/master/vec2text/data_helpers.py


Original Author: John X. Morris
Editors: Leonardo Blas Urrutia, Emily Weiss, Skyler Hallinan
'''

from openai import OpenAI
import os

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
# nltk.download('punkt')

bleu_score = evaluate.load("sacrebleu")

os.environ["OPENAI_API_KEY"] = "PLACEHOLDER"
client = OpenAI(api_key = "PLACEHOLDER")


''' Modified from Morris et al.'s example in Source 3: 
https://colab.research.google.com/drive/14RQFRF2It2Kb8gG3_YDhP_6qE0780L8h?usp=sharing '''
def get_embeddings_openai(text_list, model="text-embedding-ada-002") -> torch.Tensor:
    batches = math.ceil(len(text_list) / 128)
    outputs = []
    for batch in range(batches):
        text_list_batch = text_list[batch * 128 : (batch + 1) * 128]

        # for text in text_list_batch:
        response = client.embeddings.create(
            input=text_list_batch,
            model=model,
            encoding_format="float",  # override default base64 encoding...
        )

        outputs.extend([e.embedding for e in response.data])
    return torch.tensor(outputs)


"""Based on _text_comparison_metrics function from link in Source 2."""
def get_tf1(preds, truth):
    precision = 0
    recall = 0

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


''' Based on load_msmarco_corpus and dataset_from_args functions from link in Source 5.'''
def get_msmarco_eval_text():

    msmarco_dataset = datasets.load_dataset("Tevatron/msmarco-passage-corpus")

    # get 1% split for validation -- 88,419 rows
    msmarco_corpus = msmarco_dataset["train"].train_test_split(test_size=0.01)

    return msmarco_corpus["test"]["text"]


def main(args):

    # initialize the models
    if args.model == "hallisky/ada-msmarco-128-corrector-5epoch":
        encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
        inversion_model = vec2text.models.InversionModel.from_pretrained("hallisky/ada-msmarco-128-base-5epoch",
                                                                         cache_dir="/gscratch/xlab/hallisky/cache/")
        corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained("hallisky/ada-msmarco-128-corrector-5epoch",
                                                                                cache_dir="/gscratch/xlab/hallisky/cache/")
        
        corrector = vec2text.load_corrector(inversion_model, corrector_model)

    else:
        pass


    # get 1% MSMARCO split
    truth = get_msmarco_eval_text()

    results = []
    results_embeddings = []
    truth_embeddings = []

    # batch the embeddings and inversion
    for i in tqdm(range(0, len(truth), args.batch_size)):
        current_batch = truth[i: i + args.batch_size]

        embeddings = get_embeddings_openai(current_batch)

        current_results = vec2text.invert_embeddings(
            embeddings=embeddings.cuda(),
            corrector=corrector,
            num_steps=args.num_steps
        )

        current_results_embeddings = get_embeddings_openai(current_results)

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
    with open(f"openai_msmarco_{args.num_steps}_ours_results.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument('--batch_size', type=int, default=384, help="eval batch size")
    # parser.add_argument('--output_dir', type=str, default="ood_results")
    # parser.add_argument('--max_length', type=int, default=32, help="max length of embedding")
    parser.add_argument('--num_steps', type=int, default=20, help="num of corrector steps, 0-20")
    parser.add_argument('--model', type=str, default="hallisky/ada-msmarco-128-corrector-5epoch", help="Options: [hallisky/ada-msmarco-128-corrector-5epoch]")

    main(parser.parse_args())

    """
    # python3 -m vec2text.evaluate_ours_openai --num_steps 0 --batch_size 384;
    python3 -m vec2text.evaluate_ours_openai --num_steps 20 --batch_size 64

    """