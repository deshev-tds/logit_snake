# Long-form Hallucination Detection with Self-elicitation

ACL Findings 2025

## Environment

python=3.10.0
```bash
pip install -r requirements.txt
```

Environment has been tested on Ubuntu machines with Nvidia A800 and RTX 4090 GPUs with CUDA 12.4. Please feel free to contact us if you have issues.

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
```

## Data


## Script

`$MODEL` refers to LLM path or name (e.g. `GPT4`, `/root/llama3-8b`)

Note: see `utils/llm.py` for more supported models or adjust code for your APIs.

Running baselines (e.g. IO) with cached intermediate results.
```bash
python methods/calibration/IO/main.py \
--device_map auto \
--model_path $MODEL \
--verbose \
--info sentence \
--language en \
--data_path ./data/dataset/MedHallu_en \
--cls_cache ./cache/MedHallu_en-sentence/cls \
--context_cache ./cache/MedHallu_en-sentence/ctx
```

Running our self-elicit with cached intermediate results.
```bash
python methods/selfelicit/main.py \
--device_map auto \
--model_path $MODEL \
--knowledge global \
--verbose \
--info sentence \
--language en \
--data_path ./data/dataset/MedHallu_en \
--cls_cache ./cache/MedHallu_en-sentence/cls \
--context_cache ./cache/MedHallu_en-sentence/ctx \
--knowledge_cache ./cache/MedHallu_en-sentence/global \
--judge self_kg \
--postfix my_experiment
```

Note:
`cls_cache`: cache folder for classification of sentences as check-worthy
`context_cache`: de-contextualization for sentences
`knowledge_cache`: extract statements from sentences
