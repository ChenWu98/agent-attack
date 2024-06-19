python scripts/eval_step_attack.py \
  --attack bim_caption \
  --result_dir step_test_gpt4o_som \
  --model gpt-4o-2024-05-13 \
  --provider openai \
  --temperature 0 \
  --action_set_tag som  --observation_type image_som

# Get the results
python scripts/report_metric.py \
  --result_file step_test_gpt4o_som/target_correct_bim_caption_cap.json \
  --supports_goal_misdirection
