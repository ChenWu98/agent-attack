# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline_attack_eval_refine_attack_only.py \
  --attack bim_caption \
  --result_dir pipeline_test_gpt4o_som_evaluator_attack_only \
  --model gpt-4o-2024-05-13 \
  --provider openai \
  --temperature 0 \
  --action_set_tag som \
  --observation_type image_som \
  --max_num_attempts 2 \
  --reflexion_evaluator model


# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline_attack_eval_refine_attack_only.py \
  --attack bim_caption \
  --result_dir pipeline_test_gpt4o_som_evaluator_attack_only \
  --model gpt-4o-2024-05-13 \
  --provider openai \
  --temperature 0 \
  --pass_id 1 \
  --action_set_tag som \
  --observation_type image_som \
  --max_num_attempts 2 \
  --reflexion_evaluator model

# Restart the docker each time before running the evaluation
cd ../visualwebarena/
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
bash scripts/reset_classifieds.sh
bash prepare.sh
cd ../agent-attack/

python scripts/eval_pipeline_attack_eval_refine_attack_only.py \
  --attack bim_caption \
  --result_dir pipeline_test_gpt4o_som_evaluator_attack_only \
  --model gpt-4o-2024-05-13 \
  --provider openai \
  --temperature 0 \
  --pass_id 2 \
  --action_set_tag som \
  --observation_type image_som \
  --max_num_attempts 2 \
  --reflexion_evaluator model
