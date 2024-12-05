# Vec2Text CSCI662 Fall 2024 Final Project

Morris et al.'s original repository can be found here: https://github.com/jxmorris12/vec2text.

This repository contains scripts used to evaluate the pretrained Vec2Text models publicly available on HuggingFace, and the models we trained for our reproduction.
Scripts of the format `evalute_[pretrained/ours]_[gtr/openai/beir].py` contain the appropriate calls at the bottom of the file. 

## Dependencies
Dependencies required to run our evaluation scripts can be found in `requirements.txt`.

## Getting the data
We and Morris et al. use the following version of the Natural Questions corpus to train and evaluate all GTR models: https://huggingface.co/datasets/jxm/nq_corpus_dpr

We and Morris et al. use the following version of the MSMARCO corpus to train and evaluate all OpenAI models: https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus

We use 15 datasets from the BEIR benchmark for additional evaluation of the GTR models:

1. quora - https://huggingface.co/datasets/BeIR/quora
2. msmarco - https://huggingface.co/datasets/BeIR/msmarco
3. climate-fever - https://huggingface.co/datasets/BeIR/climate-fever
4. fever - https://huggingface.co/datasets/BeIR/fever
5. dbpedia-entity - https://huggingface.co/datasets/BeIR/dbpedia-entity
6. nq - https://huggingface.co/datasets/BeIR/nq
7. hotpotqa - https://huggingface.co/datasets/BeIR/hotpotqa
8. fiqa - https://huggingface.co/datasets/BeIR/fiqa
9. webis-touche2020 - https://huggingface.co/datasets/BeIR/webis-touche2020
10. cqadupstack - https://huggingface.co/datasets/BeIR/cqadupstack-generated-queries
11. arguana - https://huggingface.co/datasets/BeIR/arguana
12. scidocs - https://huggingface.co/datasets/BeIR/scidocs
13. trec-covid - https://huggingface.co/datasets/BeIR/trec-covid
14. scifact - https://huggingface.co/datasets/BeIR/scifact
15. nfcorpus - https://huggingface.co/datasets/BeIR/nfcorpus

## Pretrained models
Morris et al. make the following pretrained models publicly available on HuggingFace:

1. GTR, Natural Questions, 32, Base - https://huggingface.co/jxm/gtr__nq__32
2. GTR, Natural Questions, 32, Corrector - https://huggingface.co/jxm/gtr__nq__32__correct
3. OpenAI, MSMARCO, 128, Base - https://huggingface.co/jxm/vec2text__openai_ada002__msmarco__msl128__hypothesizer
4. OpenAI, MSMARCO, 128, Corrector - https://huggingface.co/jxm/vec2text__openai_ada002__msmarco__msl128__corrector

We make the following models from our reproduction attempts publicly available on HuggingFace:

1. GTR, Natural Questions, 32, Base - https://huggingface.co/hallisky/gtr-nq-32-base-5epoch
2. GTR, Natural Questions, 32, Corrector - https://huggingface.co/hallisky/gtr-nq-32-corrector-5epoch
3. OpenAI, MSMARCO, 128, Base - https://huggingface.co/hallisky/ada-msmarco-128-base-5epoch
4. OpenAI, MSMARCO, 128, Corrector - **TODO ADD**


We make the following models from our ablations publicly available on HuggingFace:

- **TODO**

## Training the models with the [vec2text](https://github.com/jxmorris12/vec2text/tree/master) codebase

From the base [vec2text](https://github.com/jxmorris12/vec2text/tree/master) directory, we run the following commands to produce our models:

1. GTR, Natural Questions, 32, Base
```
python3 -m vec2text.run --per_device_train_batch_size 1420 --per_device_eval_batch_size 1420 --max_seq_length 32 --model_name_or_path t5-base --dataset_name nq --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs 5 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment inversion --lr_scheduler_type constant_with_warmup --exp_group_name oct-gtr --learning_rate 0.001 --output_dir ./saves/gtr-nq-full-5epoch-1420 --save_steps 2000
```

2. GTR, Natural Questions, 32, Corrector
```
python3 -m vec2text.run --per_device_train_batch_size 1420 --per_device_eval_batch_size 1420 --max_seq_length 32 --model_name_or_path t5-base --dataset_name nq --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs 5 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment corrector --lr_scheduler_type constant_with_warmup --exp_group_name oct-gtr --learning_rate 0.001 --output_dir ./saves/gtr-corrector-nq-full-5epoch-1420 --save_steps 2000 --corrector_model_alias gtr_nq__msl32__5epoch
```

3. OpenAI, MSMARCO, 128, Base
```
python3 -m vec2text.run --per_device_train_batch_size 350 --per_device_eval_batch_size 350 --max_seq_length 128 --model_name_or_path t5-base --dataset_name msmarco --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True  --num_train_epochs 5 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment inversion --lr_scheduler_type constant_with_warmup --embedder_model_api text-embedding-ada-002  --exp_group_name jun3-openai-4gpu-ddp-3 --learning_rate 0.001 --output_dir ./saves/msmarco-350-ada-5epoch --save_steps 2000 --use_less_data 1000000 
```

5. OpenAI, MSMARCO, 128, Corrector
```
python3 -m vec2text.run --per_device_train_batch_size 350 --per_device_eval_batch_size 350 --max_seq_length 32 --model_name_or_path t5-base --dataset_name nq --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs 5 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment corrector --lr_scheduler_type constant_with_warmup --exp_group_name oct-gtr --learning_rate 0.001 --output_dir ./saves/ada-corrector-msmarco-full-5epoch-350 --save_steps 2000 --corrector_model_alias ada_msmarco_msl128_5epoch  --use_less_data 1000000 
```


## Running the evaluation scripts 

As mentioned earlier in the README, scripts of the format `evalute_[pretrained/ours]_[gtr/openai/beir].py` contain the appropriate calls at the bottom of their respective files. 
These calls produced the results shown in the the [Evaluation results](https://github.com/emilymweiss/vec2text_662_project/edit/main/README.md#evaluation-results) section. 


### A note about evaluating our 0 step model results 
We discovered that Morris et al.'s [run.py](https://github.com/jxmorris12/vec2text/blob/master/vec2text/run.py) in the Vec2Text Github evaluates the hypothesizer, 0 step models once they have been trained.
Thus, we did not explicitly run our evaluation script on our base models, and instead chose to report the results of the built-in evaluation in [Evaluation results](https://github.com/emilymweiss/vec2text_662_project/edit/main/README.md#evaluation-results). 

## Evaluation results 

**TODO: ADD TABLES FROM PAPER FOR CORE RESULTS**


# Citation to Original Paper
```
@misc{morris2023text,
      title={Text Embeddings Reveal (Almost) As Much As Text},
      author={John X. Morris and Volodymyr Kuleshov and Vitaly Shmatikov and Alexander M. Rush},
      year={2023},
      eprint={2310.06816},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

John Morris, Volodymyr Kuleshov, Vitaly Shmatikov, and Alexander Rush. 2023. [Text Embeddings Reveal (Almost) As Much As Text](https://aclanthology.org/2023.emnlp-main.765). In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12448–12460, Singapore. Association for Computational Linguistics.
