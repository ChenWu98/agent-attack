# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline.py \
  --result_dir pipeline_test_gemini_som \
  --provider google \
  --model gemini-1.5-pro-preview-0409 \
  --mode completion \
  --max_obs_length 15360 \
  --temperature 0 \
  --action_set_tag som  --observation_type image_som
