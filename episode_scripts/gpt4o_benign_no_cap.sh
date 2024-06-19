# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline.py \
  --result_dir pipeline_test_gpt4o_som \
  --model gpt-4o-2024-05-13 \
  --provider openai \
  --temperature 0 \
  --no_cap \
  --action_set_tag som  --observation_type image_som
