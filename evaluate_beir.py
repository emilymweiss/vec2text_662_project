'''
Note: Evaluation script modified from code in the vec2text github Python files and README.

Sources:
1. general structure and get_gtr_embeddings function from: 
        https://github.com/jxmorris12/vec2text/tree/master?tab=readme-ov-file#similarly-you-can-invert-gtr-base-embeddings-with-the-following-example
2. bleu and tf1 code modified from _text_comparison_metrics in vec2text/trainers/base.py: 
        https://github.com/jxmorris12/vec2text/blob/master/vec2text/trainers/base.py
        
Original Author: John X. Morris 
Editors: Emily Weiss, Skyler Hallinan
'''
import os
# TODO Emily: Delete if you ever run this
os.environ["HF_HOME"] = "/gscratch/xlab/hallisky/cache/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/gscratch/xlab/hallisky/cache/"
os.environ["HF_DATASETS_CACHE"] = "/gscratch/xlab/hallisky/cache/"
os.environ["TRANSFORMERS_CACHE"] = "/gscratch/xlab/hallisky/cache/"

import vec2text
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from datasets import load_dataset
import argparse
import json
import evaluate # for evaluation (bleu and tf1...)
from tqdm import tqdm
import sys

import nltk
nltk.download('punkt_tab')

# os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
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
    # precision = 0
    # recall = 0

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

        # precision += P
        # recall += R

    return sum(f1s)/len(f1s)

def main(args):
    
    corpus_key = "corpus"
    # Different loading for different dataset
    if args.eval_dataset == "quora":
        ds = load_dataset("BeIR/quora", "corpus", cache_dir="/gscratch/xlab/hallisky/cache/")
    elif args.eval_dataset == "msmarco":
        ds = load_dataset("BeIR/msmarco", "corpus", cache_dir="/gscratch/xlab/hallisky/cache/")
    elif args.eval_dataset == "climate-fever":
        ds = load_dataset("BeIR/climate-fever", "corpus", cache_dir="/gscratch/xlab/hallisky/cache/")
    elif args.eval_dataset == "fever":
        ds = load_dataset("BeIR/fever", "corpus", cache_dir="/gscratch/xlab/hallisky/cache/")
    elif args.eval_dataset == "dbpedia-entity":
        ds = load_dataset("BeIR/dbpedia-entity", "corpus", cache_dir="/gscratch/xlab/hallisky/cache/")
    elif args.eval_dataset == "nq":
        ds = load_dataset("BeIR/nq", "corpus", cache_dir="/gscratch/xlab/hallisky/cache/")
    elif args.eval_dataset == "hotpotqa":
        ds = load_dataset("BeIR/hotpotqa", "corpus", cache_dir="/gscratch/xlab/hallisky/cache/")
    elif args.eval_dataset == "fiqa":
        # TODO: Skyler, this dataset is only 57.6K rows long. Will that be a problem?
        ds = load_dataset("BeIR/fiqa", "corpus", cache_dir="/gscratch/xlab/hallisky/cache/")
    elif args.eval_dataset == "webis-touche2020":
        ds = load_dataset("BeIR/webis-touche2020", "corpus", cache_dir="/gscratch/xlab/hallisky/cache/")
    elif args.eval_dataset == "cqadupstack-generated-queries":
        # TODO: this comes with a warning on hugging face 
        ds = load_dataset("BeIR/cqadupstack-generated-queries")
        corpus_key="train"
    elif args.eval_dataset == "arguana":
        # this is also small, 8.67k rows 
        ds = load_dataset("BeIR/arguana", "queries")
        corpus_key = "queries"
    elif args.eval_dataset == "scidocs":
        ds = load_dataset("BeIR/scidocs", "corpus", cache_dir="/gscratch/xlab/hallisky/cache/")
    elif args.eval_dataset == "trec-covid":
        # this is also small, 171k rows
        ds = load_dataset("BeIR/trec-covid", "corpus", cache_dir="/gscratch/xlab/hallisky/cache/")
    elif args.eval_dataset == "scifact":
        # 5.18k rows 
        ds = load_dataset("BeIR/scifact", "corpus", cache_dir="/gscratch/xlab/hallisky/cache/")
    elif args.eval_dataset == "nfcorpus":
        # 3.63k rows 
        ds = load_dataset("BeIR/nfcorpus", "corpus", cache_dir="/gscratch/xlab/hallisky/cache/")
    else: 
        print("Valid dataset not passed in")
        import sys; sys.exit()

    print("Cache settings:")
    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("HUGGINGFACE_HUB_CACHE:", os.environ.get("HUGGINGFACE_HUB_CACHE"))
    print("HF_DATASETS_CACHE:", os.environ.get("HF_DATASETS_CACHE"))
    print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))

    NUM_EVAL = 100000
    small_text = ds[corpus_key]['text'][:NUM_EVAL]

    # Initialize the models
    encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

    if args.model == "gtr-base":
        corrector = vec2text.load_pretrained_corrector("gtr-base")
    elif args.model == "hallisky/gtr-nq-32-corrector-5epoch":
        inversion_model = vec2text.models.InversionModel.from_pretrained("hallisky/gtr-nq-32-base-5epoch", cache_dir="/gscratch/xlab/hallisky/cache/")
        corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained("hallisky/gtr-nq-32-corrector-5epoch", cache_dir="/gscratch/xlab/hallisky/cache/")
        corrector = vec2text.load_corrector(inversion_model, corrector_model)
    ## Vec2Text API does not suport only loading the inverter
    # elif args.model == "hallisky/gtr-nq-32-base-5epoch":
    #     corrector = vec2text.load_pretrained_corrector("hallisky/gtr-nq-32-base-5epoch")
    else:
        sys.exit("Invalid model provided")


    results = []

    # batch_size = 2048 # uses about 68K VRAM
    # batch_size = 256 # uses about 11k of VRAM

    for i in tqdm(range(0, len(small_text), args.batch_size)):
        current_batch = small_text[i: i + args.batch_size]

        embeddings = get_gtr_embeddings(current_batch, encoder, tokenizer, args.max_length)

        current_results = vec2text.invert_embeddings(
            embeddings=embeddings.cuda(),
            corrector=corrector,
            num_steps=args.num_steps,
        )

        results = results + current_results

    assert len(results) == len(small_text), "len of references must match len of predictions"

    bleu_result = bleu_score.compute(references=small_text, predictions=results)
    print(bleu_result)

    tf1_result = get_tf1(preds=results, truth=small_text)
    print(tf1_result)

    results = {"data_name": args.eval_dataset,
               "bleu_result": bleu_result,
               "tf1_result": tf1_result
               }
 
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to file 
    with open(os.path.join(args.output_dir, f"{args.eval_dataset}_{args.num_steps}_{args.model.split('/')[-1]}_results.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument('--batch_size', type=int, default=150, help="eval batch size")
    parser.add_argument('--eval_dataset', type=str, choices=["quora", "msmarco", "climate-fever", "fever", "dbpedia-entity", "nq", "hotpotqa", "fiqa", "webis-touche2020", "cqadupstack-generated-queries", "arguana", "scidocs", "trec-covid", "scifact", "nfcorpus"], help="eval dataset to use")
    parser.add_argument('--output_dir', type=str, default="ood_results")
    parser.add_argument('--max_length', type=int, default=32, help="max length of embedding") 
    parser.add_argument('--num_steps', type=int, default=0, help="max number of steps") 
    parser.add_argument('--model', type=str, default="hallisky/gtr-nq-32-corrector-5epoch", help="Other options: [hallisky/gtr-nq-32-corrector-5epoch, hallisky/gtr-nq-32-hypothesizer-5epoch]") 



    main(parser.parse_args())

    """
    Run this code

    # For nq, use max length of 32

# For quora
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset quora \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results;

    # For msmarco
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset msmarco \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results;

    # For climate-fever
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset climate-fever \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results

    # For fever
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset fever \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results;

    # For dbpedia-entity
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset dbpedia-entity \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results;

    # For nq
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset nq \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results

    # For hotpotqa
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset hotpotqa \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results;

    # For fiqa
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset fiqa \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results;

    # For webis-touche2020
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset webis-touche2020 \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results;

    # For cqadupstack-generated-queries
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset cqadupstack-generated-queries \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results;

    # For arguana
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset arguana \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results;

    # For scidocs
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset scidocs \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results;

    # For trec-covid
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset trec-covid \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results;

    # For scifact
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset scifact \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results;

    # For nfcorpus
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset nfcorpus \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 20 \
        --output_dir ood_results

   # 0 step
   # For quora
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset quora \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results;

    # For msmarco
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset msmarco \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \ 
        --output_dir ood_results;

    # For climate-fever
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset climate-fever \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results

    # For fever
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset fever \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results;

    # For dbpedia-entity
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset dbpedia-entity \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results;

    # For nq
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset nq \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results

    # For hotpotqa
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset hotpotqa \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results;

    # For fiqa
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset fiqa \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results;

    # For webis-touche2020
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset webis-touche2020 \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results;

    # For cqadupstack-generated-queries
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset cqadupstack-generated-queries \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results;

    # For arguana
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset arguana \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results;

    # For scidocs
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset scidocs \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results;

    # For trec-covid
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset trec-covid \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results;

    # For scifact
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset scifact \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results;

    # For nfcorpus
    python3 -m vec2text.evaluate_beir \
        --batch_size 190 \
        --eval_dataset nfcorpus \
        --max_length 32 \
        --model hallisky/gtr-nq-32-corrector-5epoch \
        --num_steps 0 \
        --output_dir ood_results

  # NOT WORKING
   # For quora
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset quora \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results;

    # For msmarco
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset msmarco \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \ 
        --output_dir ood_results;

    # For climate-fever
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset climate-fever \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results

    # For fever
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset fever \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results;

    # For dbpedia-entity
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset dbpedia-entity \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results;

    # For nq
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset nq \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results

    # For hotpotqa
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset hotpotqa \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results;

    # For fiqa
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset fiqa \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results;

    # For webis-touche2020
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset webis-touche2020 \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results;

    # For cqadupstack-generated-queries
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset cqadupstack-generated-queries \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results;

    # For arguana
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset arguana \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results;

    # For scidocs
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset scidocs \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results;

    # For trec-covid
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset trec-covid \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results;

    # For scifact
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset scifact \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results;

    # For nfcorpus
    python3 -m vec2text.evaluate_beir \
        --batch_size 150 \
        --eval_dataset nfcorpus \
        --max_length 32 \
        --model hallisky/gtr-nq-32-base-5epoch \
        --output_dir ood_results

    """
