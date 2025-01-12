# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline_attack_tree_search_attack.py \
  --attack bim_caption \
  --model gpt-4o-2024-05-13 \
  --agent_type "search"   --max_depth 0  --branching_factor 3  --vf_budget 20   \
  --result_dir pipeline_test_gpt4o_som_tree_search_attack \
  --provider openai \
  --temperature 0 \
  --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
  --action_set_tag som  --observation_type image_som \
  --top_p 0.95   --temperature 1.0  --max_steps 3

# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline_attack_tree_search_attack.py \
  --attack bim_caption \
  --model gpt-4o-2024-05-13 \
  --agent_type "search"   --max_depth 0  --branching_factor 3  --vf_budget 20   \
  --result_dir pipeline_test_gpt4o_som_tree_search_attack \
  --provider openai \
  --temperature 0 \
  --pass_id 1 \
  --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
  --action_set_tag som  --observation_type image_som \
  --top_p 0.95   --temperature 1.0  --max_steps 3

# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline_attack_tree_search_attack.py \
  --attack bim_caption \
  --model gpt-4o-2024-05-13 \
  --agent_type "search"   --max_depth 0  --branching_factor 3  --vf_budget 20   \
  --result_dir pipeline_test_gpt4o_som_tree_search_attack \
  --provider openai \
  --temperature 0 \
  --pass_id 2 \
  --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
  --action_set_tag som  --observation_type image_som \
  --top_p 0.95   --temperature 1.0  --max_steps 3
