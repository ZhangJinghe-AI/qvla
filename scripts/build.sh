#!/usr/bin/env bash

# --model pi05 \
# --checkpoint /data/share/pi05_libero_finetuned_v044 \

# --model groot_n17 \
# --checkpoint /data/share/GR00T-N1.7-LIBERO/libero_goal \
# --embodiment-tag LIBERO_PANDA \
# --processor-model-name-or-path /data/share/Cosmos-Reason2-2B \

# --llm-act-scale-mode dynamic \
# --llm-act-scale-granularity per_block \
# --llm-group-size 16 \

# --llm-act-outlier-selective-channels \
# --llm-act-outlier-global \
# --llm-act-outlier-std-k 3 \
# --llm-act-outlier-fit-tokens image_lang_pad \

# --llm-smooth-alpha 0.9 \
# --dit-smooth-step-pmean-p 4 \

# # LLM: W4A4
# export HF_ENDPOINT=https://hf-mirror.com
# CUDA_VISIBLE_DEVICES=5 uv run python scripts/build_pack.py build \
#     --model groot_n17 \
#     --checkpoint /data/share/GR00T-N1.7-LIBERO/libero_goal \
#     --embodiment-tag LIBERO_PANDA \
#     --processor-model-name-or-path /data/share/Cosmos-Reason2-2B \
#     --device cuda \
#     --params-dtype bfloat16 \
#     --build-seed 0 \
#     --dit-include-regex '' \
#     --llm-weight-format nvfp \
#     --llm-act-format nvfp \
#     --llm-weight-bits 4 \
#     --llm-act-bits 4 \
#     --dit-weight-bits 16 \
#     --dit-act-bits 16 \
#     --num-samples 30 \
#     --noise-ensemble-k 1 \
#     --calibration-source file \
#     --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
#     --calibration-noise-mode per_sample \
#     --llm-act-scale-mode dynamic \
#     --llm-act-scale-granularity per_block \
#     --llm-group-size 16 \
#     --llm-quant gptq \
#     --llm-gptq-block-size 128 \
#     --llm-pipeline smooth \
#     --llm-smooth-alpha 1.0 \
#     --output /data/share/qvla-packs/gr00tN17_libero_goal-chunksize16-seed0-W4A4_llm_nvfpxnvfp-ns30-k1-pipeline_smooth-llm_smooth_alpha10-act_dynamic_perblock_16-test1.pt
    
 
 
# # DiT: W4A4
# export HF_ENDPOINT=https://hf-mirror.com
# CUDA_VISIBLE_DEVICES=6 uv run python scripts/build_pack.py build \
#     --model pi05 \
#     --checkpoint /data/share/pi05_libero_finetuned_v044 \
#     --device cuda \
#     --params-dtype bfloat16 \
#     --build-seed 0 \
#     --llm-include-regex '' \
#     --dit-weight-format nvfp \
#     --dit-act-format nvfp \
#     --llm-weight-bits 16 \
#     --llm-act-bits 16 \
#     --dit-weight-bits 4 \
#     --dit-act-bits 4 \
#     --num-samples 30 \
#     --noise-ensemble-k 4 \
#     --calibration-source file \
#     --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
#     --calibration-noise-mode per_sample \
#     --dit-act-scale-mode dynamic \
#     --dit-act-scale-granularity per_block \
#     --dit-group-size 16 \
#     --dit-quant gptq \
#     --dit-gptq-block-size 128 \
#     --dit-pipeline none \
#     --output /data/share/qvla-packs/pi05-chunksize50-seed0-W4A4_dit_nvfpxnvfp-ns30-k4-pipeline_none-act_dynamic_perblock_16-test1.pt
    


# LLM+DiT: W4A4
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=5 uv run python scripts/build_pack.py build \
    --model groot_n17 \
    --checkpoint /data/share/GR00T-N1.7-LIBERO/libero_goal \
    --embodiment-tag LIBERO_PANDA \
    --processor-model-name-or-path /data/share/Cosmos-Reason2-2B \
    --device cuda \
    --params-dtype bfloat16 \
    --build-seed 0 \
    --llm-weight-format nvfp \
    --llm-act-format nvfp \
    --dit-weight-format nvfp \
    --dit-act-format nvfp \
    --llm-weight-bits 4 \
    --llm-act-bits 4 \
    --dit-weight-bits 4 \
    --dit-act-bits 4 \
    --num-samples 30 \
    --noise-ensemble-k 4 \
    --calibration-source file \
    --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
    --calibration-noise-mode per_sample \
    --llm-act-scale-mode dynamic \
    --llm-act-scale-granularity per_block \
    --llm-group-size 16 \
    --dit-act-scale-mode dynamic \
    --dit-act-scale-granularity per_block \
    --dit-group-size 16 \
    --llm-quant gptq \
    --llm-gptq-block-size 128 \
    --dit-quant gptq \
    --dit-gptq-block-size 128 \
    --llm-pipeline clip \
    --llm-act-outlier-std-k 3 \
    --llm-act-outlier-fit-tokens image_lang_pad \
    --llm-act-outlier-selective-channels \
    --dit-pipeline clip \
    --dit-act-outlier-std-k 3 \
    --dit-act-outlier-std-k-down 0.0 \
    --dit-act-outlier-std-k-up 0.25 \
    --dit-act-outlier-fit-tokens skip_first \
    --dit-act-outlier-selective-channels \
    --output /data/share/qvla-packs/gr00tN17_libero_goal-chunksize16-seed0-W4A4_llm_dit_nvfpxnvfp-ns30-k4-llm_pipeline_clip-clip_mean_3std_image_langpad_mask_selective-dit_pipeline_clip-clip_mean_3std_00_025_skipfirst_selective-llm_dit_act_dynamic_perblock_16-test1.pt
       