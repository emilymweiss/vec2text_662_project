# Vec2Text CSCI662 Fall 2024 Final Project

The authors' original repository can be found here: https://github.com/jxmorris12/vec2text.

This repository contains scripts used to evaluate the pretrained Vec2Text models publicly available on HuggingFace, and the models we trained for our reproduction.
Scripts of the format `evalute_[pretrained/ours]_[gtr/openai/beir].py` contain the appropriate calls at the bottom of the file. 

## Dependencies
Dependencies required to run our evaluation scripts can be found in `requirements.txt`.

## Getting the data
We and the original authors use the following version of the Natural Questions corpus to train and evaluate all GTR models: https://huggingface.co/datasets/jxm/nq_corpus_dpr

We and the original authors use the following version of the MSMARCO corpus to train and evaluate all OpenAI models: https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus

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

## Running the evaluation scripts 

## Evaluation results 

### A note about our 0 step model results 
