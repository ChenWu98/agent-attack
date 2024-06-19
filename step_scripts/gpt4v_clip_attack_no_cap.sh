python scripts/eval_step_attack.py \
  --attack clip_attack \
  --no_cap \
  --result_dir step_test_gpt4v_som \
  --model gpt-4-vision-preview \
  --provider openai \
  --temperature 0 \
  --action_set_tag som  --observation_type image_som

# Get the results
python scripts/report_metric.py \
  --result_file step_test_gpt4v_som_no_cap/target_correct_clip_attack_no_cap.json
