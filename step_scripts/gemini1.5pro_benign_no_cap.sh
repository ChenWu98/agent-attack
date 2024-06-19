python scripts/eval_step.py \
  --result_dir step_test_gemini_som \
  --provider google \
  --model gemini-1.5-pro-preview-0409 \
  --mode completion \
  --max_obs_length 15360 \
  --no_cap \
  --temperature 0 \
  --action_set_tag som  --observation_type image_som
