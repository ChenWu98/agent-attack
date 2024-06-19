python scripts/eval_step_attack.py \
  --attack bim_caption \
  --result_dir step_test_claude_som \
  --provider anthropic \
  --model claude-3-opus-20240229 \
  --temperature 0 \
  --action_set_tag som  --observation_type image_som

# Get the results
python scripts/report_metric.py \
  --result_file step_test_claude_som/target_correct_bim_caption_cap.json \
  --supports_goal_misdirection
