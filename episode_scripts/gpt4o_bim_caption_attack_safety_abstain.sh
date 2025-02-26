# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline_attack.py \
  --attack bim_caption \
  --result_dir pipeline_test_gpt4o_som_safety_abstain \
  --model gpt-4o-2024-05-13 \
  --provider openai \
  --temperature 0 \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s_safety_abstain.json \
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
  --result_dir pipeline_test_gpt4o_som_safety_abstain \
  --model gpt-4o-2024-05-13 \
  --provider openai \
  --temperature 0 \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s_safety_abstain.json \
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
  --result_dir pipeline_test_gpt4o_som_safety_abstain \
  --model gpt-4o-2024-05-13 \
  --provider openai \
  --temperature 0 \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s_safety_abstain.json \
  --action_set_tag som  --observation_type image_som
