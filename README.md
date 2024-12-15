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
4. OpenAI, MSMARCO, 128, Corrector - https://huggingface.co/hallisky/ada-msmarco-128-corrector-5epoch

## Training the models with the [vec2text](https://github.com/jxmorris12/vec2text/tree/master) codebase

From the base [vec2text](https://github.com/jxmorris12/vec2text/tree/master) directory, we run the following commands to produce our models:

1. GTR, Natural Questions, 32, Base
```
python3 -m vec2text.run --per_device_train_batch_size 1420 --per_device_eval_batch_size 1420 --max_seq_length 32 --model_name_or_path t5-base --dataset_name nq --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs 5 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment inversion --lr_scheduler_type constant_with_warmup --exp_group_name oct-gtr --learning_rate 0.001 --output_dir ./saves/gtr-nq-full-5epoch-1420 --save_steps 2000
```

2. GTR, Natural Questions, 32, Corrector
```
python3 -m vec2text.run --per_device_train_batch_size 1420 --per_device_eval_batch_size 1420 --max_seq_length 32 --model_name_or_path t5-base --dataset_name nq --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs 3.3 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment corrector --lr_scheduler_type constant_with_warmup --exp_group_name oct-gtr --learning_rate 0.001 --output_dir ./saves/gtr-corrector-nq-full-5epoch-1420 --save_steps 2000 --corrector_model_alias gtr_nq__msl32__5epoch
```

3. OpenAI, MSMARCO, 128, Base
```
python3 -m vec2text.run --per_device_train_batch_size 350 --per_device_eval_batch_size 350 --max_seq_length 128 --model_name_or_path t5-base --dataset_name msmarco --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True  --num_train_epochs 5 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment inversion --lr_scheduler_type constant_with_warmup --embedder_model_api text-embedding-ada-002  --exp_group_name jun3-openai-4gpu-ddp-3 --learning_rate 0.001 --output_dir ./saves/msmarco-350-ada-5epoch --save_steps 2000 --use_less_data 1000000 
```

4. OpenAI, MSMARCO, 128, Corrector
```
python3 -m vec2text.run --per_device_train_batch_size 350 --per_device_eval_batch_size 350 --max_seq_length 128 --model_name_or_path t5-base --dataset_name msmarco --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs 5 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment corrector --lr_scheduler_type constant_with_warmup --exp_group_name oct-gtr --learning_rate 0.001 --output_dir ./saves/ada-corrector-msmarco-full-5epoch-350 --save_steps 2000 --corrector_model_alias ada_msmarco_msl128_5epoch  --use_less_data 1000000 
```


## Running the evaluation scripts 

As mentioned earlier in the README, the scripts `evaluate_ours_gtr.py`, `evaluate_beir.py`, and `evaluate_pretrained_openai.py` contain the commands to reproduce our results at the end of the files. 
These calls produced the results shown in the the [Evaluation results](https://github.com/emilymweiss/vec2text_662_project/edit/main/README.md#evaluation-results) section.
We copy the calls here for convenience:
1. Evaluate our GTR, Natural Questions, 32, Corrector with 20 corrective steps:
```
python evaluate_ours_gtr.py --num_steps 20 --batch_size 384
```

2. Evaluate our OpenAI, MSMARCO, 128, Corrector with 20 corrective steps:
```
python evaluate_ours_openai.py --num_steps 20 --batch_size 64
```

3. Evaluate the authors' pre-trained OpenAI, MSMARCO, 128, Base and Corrector models with 20 corrective steps:
```
python evaluate_pretrained_openai.py --num_steps 0 --batch_size 384
python evaluate_pretrained_openai.py --num_steps 20 --batch_size 64
```

4. Evaluate the GTR/NQ models on select datasets from the BEIR benchmark. (These commands are only for our 20-step model. To change the number of corrective steps, change --num_steps to 0. To change the model to the pre-trained model, change --model to gtr-base.)
```
python evaluate_beir.py --batch_size 190 --eval_dataset quora --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset msmarco --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset climate-fever --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset fever --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset dbpedia-entity --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset nq --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset hotpotqa --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset fiqa --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset webis-touche2020 --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset cqadupstack-generated-queries --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset arguana --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset scidocs --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset trec-covid --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset scifact --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
python evaluate_beir.py --batch_size 190 --eval_dataset nfcorpus --max_length 32 --model hallisky/gtr-nq-32-corrector-5epoch --num_steps 20 --output_dir ood_results
```


5. Finally, to evaluate the authors' pre-trained GTR, Natural Questions, 32, Base and Corrector models, we ran the following script from the authors' README: [Evaluate the models from the papers](https://github.com/jxmorris12/vec2text/tree/master#evaluate-the-models-from-the-papers).
Note that we replaced `"jxm/gtr__nq__32__correct"` in the following line with `"jxm/gtr__nq__32"`...
```python
experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
     "jxm/gtr__nq__32__correct"
)
```
... and changed `num_gen_recursive_steps` to 0 to evaluate their pre-trained base model. 



### A note about evaluating 0 step models
We discovered that Morris et al.'s training script, [run.py](https://github.com/jxmorris12/vec2text/blob/master/vec2text/run.py) in the Vec2Text Github, evaluates 0-step models at the end of training.
Thus, we did not explicitly run our evaluation scripts for the in-domain datasets on our 0-step models. 
Instead, we chose to report the results of the built-in evaluation for those models, as seen in [Evaluation results](https://github.com/emilymweiss/vec2text_662_project/edit/main/README.md#evaluation-results). 

## Running our ablations 
### Training the GTE/NQ models:
1. GTE/NQ Base
```
python run.py --per_device_train_batch_size 1420 --per_device_eval_batch_size 1420 --max_seq_length 32 --model_name_or_path t5-base --dataset_name nq --embedder_model_name gte_base --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs 5 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment inversion --lr_scheduler_type constant_with_warmup --exp_group_name oct-gtr --learning_rate 0.001 --output_dir ./saves/gte-nq-full-5epoch-1420 --save_steps 2000 
```

2. GTE/NQ Corrector
```
python run.py --per_device_train_batch_size 300 --per_device_eval_batch_size 300 --max_seq_length 32 --model_name_or_path t5-base --dataset_name nq --embedder_model_name gte_base --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs 5 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment corrector --lr_scheduler_type constant_with_warmup --exp_group_name oct-gtr --learning_rate 0.001 --output_dir ./saves/gte-corrector-nq-full-5epoch-300 --save_steps 2000 --corrector_model_alias gte_nq__msl32__5epoch
```

### Evaluating pre-trained GTR/NQ on additional data
Note: If no `random_texts.txt` is available, run `python create_random_texts.py`. 

We use the following commands to evaluate the authors' pre-trained GTR/NQ models on additional OOD data:
```
python evaluate_ood.py --model gtr-base --dataset random_texts --num_steps 0 --batch_size 384
python evaluate_ood.py --model gtr-base --dataset random_texts --num_steps 20 --batch_size 384

python evaluate_ood.py --model gtr-base --dataset biosses --num_steps 0 --batch_size 384
python evaluate_ood.py --model gtr-base --dataset biosses --num_steps 20 --batch_size 384

python evaluate_ood.py --model gtr-base --dataset medrxiv --num_steps 0 --batch_size 384
python evaluate_ood.py --model gtr-base --dataset medrxiv --num_steps 20 --batch_size 384

python evaluate_ood.py --model gtr-base --dataset biorxiv --num_steps 0 --batch_size 384
python evaluate_ood.py --model gtr-base --dataset biorxiv --num_steps 20 --batch_size 384
```

The commands are replicated inside `evaluate_ood.py` in the `ood-ablation` folder. 

## Evaluation results 

Our core reproduction results are shown in the tables below. 
These results were produced using the steps dicsussed in [Running the evaluation scripts](https://github.com/emilymweiss/vec2text_662_project/edit/main/README.md#running-the-evaluation-scripts).

<img width="770" alt="Table1" src="https://github.com/user-attachments/assets/6779451a-5c79-4d80-a070-5cd0daa1362e" />


<img width="751" alt="Table2" src="https://github.com/user-attachments/assets/d6556496-a720-48f8-88fe-cc7bca2f54f5" />



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
