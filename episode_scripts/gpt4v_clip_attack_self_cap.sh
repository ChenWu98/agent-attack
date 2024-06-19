# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline_attack.py \
  --attack clip_attack \
  --result_dir pipeline_test_gpt4v_som \
  --model gpt-4-vision-preview \
  --captioning_model gpt-4-vision-preview \
  --provider openai \
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
  --attack clip_attack \
  --pass_id 1 \
  --result_dir pipeline_test_gpt4v_som \
  --model gpt-4-vision-preview \
  --captioning_model gpt-4-vision-preview \
  --provider openai \
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
  --attack clip_attack \
  --pass_id 2 \
  --result_dir pipeline_test_gpt4v_som \
  --model gpt-4-vision-preview \
  --captioning_model gpt-4-vision-preview \
  --provider openai \
  --temperature 0 \
  --action_set_tag som  --observation_type image_som

# Get the results
python scripts/report_metric.py \
  --result_file pipeline_test_gpt4v_som/target_correct_clip_attack_cap.json
