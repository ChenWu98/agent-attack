# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline.py \
  --result_dir pipeline_test_claude_som \
  --provider anthropic \
  --model claude-3-opus-20240229 \
  --captioning_model claude-3-opus-20240229 \
  --temperature 0 \
  --action_set_tag som  --observation_type image_som
