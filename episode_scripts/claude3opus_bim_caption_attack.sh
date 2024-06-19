# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline_attack.py \
  --attack bim_caption \
  --result_dir pipeline_test_claude_som \
  --provider anthropic \
  --model claude-3-opus-20240229 \
  --temperature 0 \
  --action_set_tag som  --observation_type image_som

# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline_attack.py \
  --attack bim_caption \
  --pass_id 1 \
  --result_dir pipeline_test_claude_som \
  --provider anthropic \
  --model claude-3-opus-20240229 \
  --temperature 0 \
  --action_set_tag som  --observation_type image_som

# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline_attack.py \
  --attack bim_caption \
  --pass_id 2 \
  --result_dir pipeline_test_claude_som \
  --provider anthropic \
  --model claude-3-opus-20240229 \
  --temperature 0 \
  --action_set_tag som  --observation_type image_som

# Get the results
python scripts/report_metric.py \
  --result_file pipeline_test_claude_som/target_correct_bim_caption_cap.json \
  --supports_goal_misdirection
