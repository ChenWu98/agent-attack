import argparse
import json

parser = argparse.ArgumentParser(description="Report the metric for the attack")
parser.add_argument(
    "--result_file",
    type=str,
    required=True,
    help="The result file to report",
)
parser.add_argument(
    "--supports_goal_misdirection",
    action="store_true",
    help="Whether the attack supports goal misdirection",
)
args = parser.parse_args()

illusioning_file = "exp_data/illusioning_idx.json"

illusioning_idx = json.load(open(illusioning_file))
result_data = json.load(open(args.result_file))

illusioning_correct = []
illusioning_count = 0
if args.supports_goal_misdirection:
    goal_misdirection_correct = []
    goal_misdirection_count = 0

for example_id, correct in result_data.items():
    if not example_id.endswith("_cap"):
        example_id = example_id[:-2]
        assert example_id.endswith("_cap")
    if example_id in illusioning_idx:
        illusioning_correct.append(correct)
        illusioning_count += 1
    else:
        if args.supports_goal_misdirection:
            goal_misdirection_correct.append(correct)
            goal_misdirection_count += 1

print("Illusioning success rate:", sum(illusioning_correct) / illusioning_count)
if args.supports_goal_misdirection:
    print("Goal misdirection success rate:", sum(goal_misdirection_correct) / goal_misdirection_count)
