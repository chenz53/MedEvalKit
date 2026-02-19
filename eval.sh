#!/bin/bash
# export HF_ENDPOINT=https://hf-mirror.com
# MMMU-Medical-test,MMMU-Medical-val,PMC_VQA,MedQA_USMLE,MedMCQA,PubMedQA,OmniMedVQA,Medbullets_op4,Medbullets_op5,MedXpertQA-Text,MedXpertQA-MM,SuperGPQA,HealthBench,IU_XRAY,CheXpert_Plus,MIMIC_CXR,CMB,CMExam,CMMLU,MedQA_MCMLE,VQA_RAD,SLAKE,PATH_VQA,MedFrameQA,Radrestruct
EVAL_DATASETS="OmniMedVQA" 
DATASETS_PATH="hf"

# List of models and their HuggingFace paths to test
declare -A MODELS
MODELS=(
    # ["Qwen3-VL"]="Qwen/Qwen3-VL-4B-Instruct"
    ["MedGemma"]="google/MedGemma-1.5-4b-it"
    # ["MedGemma"]="/home/user/checkpoints/medgemma-1.5-4b-it-gradient-merged"
    # Add more models here if needed, e.g. ["OtherModel"]="other/model-path"
)

#vllm setting
CUDA_VISIBLE_DEVICES="0"
TENSOR_PARALLEL_SIZE="1"
USE_VLLM="False"

#Eval setting
SEED=42
REASONING="True"
TEST_TIMES=1

# Eval LLM setting
MAX_NEW_TOKENS=2048
MAX_IMAGE_NUM=6
TEMPERATURE=0.7
TOP_P=0.95
REPETITION_PENALTY=1

# LLM judge setting
USE_LLM_JUDGE="False"
GPT_MODEL="gpt-4.1-2025-04-14"
JUDGE_MODEL_TYPE="openai"  # openai or gemini or deepseek or claude
API_KEY=""
BASE_URL=""

for MODEL_NAME in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL_NAME]}"
    # Extract the model name from the HuggingFace path (last part after '/')
    ACTUAL_MODEL_NAME="$(basename "$MODEL_PATH")"
    OUTPUT_PATH="eval_results/${ACTUAL_MODEL_NAME}/${EVAL_DATASETS}"
    mkdir -p "$OUTPUT_PATH" # Ensure output directory exists

    echo "Running evaluation for $MODEL_NAME from $MODEL_PATH..."

    python eval.py \
        --eval_datasets "$EVAL_DATASETS" \
        --datasets_path "$DATASETS_PATH" \
        --output_path "$OUTPUT_PATH" \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --seed $SEED \
        --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --use_vllm "$USE_VLLM" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --max_image_num "$MAX_IMAGE_NUM" \
        --temperature "$TEMPERATURE"  \
        --top_p "$TOP_P" \
        --repetition_penalty "$REPETITION_PENALTY" \
        --reasoning "$REASONING" \
        --use_llm_judge "$USE_LLM_JUDGE" \
        --judge_model_type "$JUDGE_MODEL_TYPE" \
        --judge_model "$GPT_MODEL" \
        --api_key "$API_KEY" \
        --base_url "$BASE_URL" \
        --test_times "$TEST_TIMES"
done
