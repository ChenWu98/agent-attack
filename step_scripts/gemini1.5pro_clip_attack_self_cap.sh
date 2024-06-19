python scripts/eval_step_attack.py \
  --attack clip_attack \
  --result_dir step_test_gemini_som \
  --provider google \
  --model gemini-1.5-pro-preview-0409 \
  --mode completion \
  --max_obs_length 15360 \
  --temperature 0 \
  --action_set_tag som  --observation_type image_som

# Get the results
python scripts/report_metric.py \
  --result_file step_test_gemini_som/target_correct_clip_attack_cap.json
