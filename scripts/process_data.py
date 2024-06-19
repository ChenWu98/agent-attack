import argparse
import os

from browser_env.env_config import *

FILES = ["data.json", "config.json", "obs_text.txt"]

REDDIT_PREV = "http://127.0.0.1:9999"
SHOPPING_PREV = "http://127.0.0.1:7770"
WIKIPEDIA_PREV = "http://127.0.0.1:8888"
HOMEPAGE_PREV = "http://127.0.0.1:4399"
CLASSIFIEDS_PREV = "http://127.0.0.1:9980"


parser = argparse.ArgumentParser(description="Process the data")
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The directory of the data",
)
args = parser.parse_args()

adv_dirs = os.listdir(os.path.join(args.data_dir, "agent_adv"))
adv_dirs = [d for d in adv_dirs if d != ".DS_Store"]
adv_dirs = [os.path.join(args.data_dir, "agent_adv", d) for d in adv_dirs]

clean_dirs = os.listdir(os.path.join(args.data_dir, "agent_clean"))
clean_dirs = [d for d in clean_dirs if d != ".DS_Store"]
clean_dirs = [os.path.join(args.data_dir, "agent_clean", d) for d in clean_dirs]

for directory in adv_dirs + clean_dirs:
    for file in FILES:
        if not os.path.exists(os.path.join(directory, file)):
            continue
        file_str = open(os.path.join(directory, file)).read()

        if REDDIT and (REDDIT != REDDIT_PREV):
            file_str = file_str.replace(REDDIT_PREV, REDDIT)
        if SHOPPING and (SHOPPING != SHOPPING_PREV):
            file_str = file_str.replace(SHOPPING_PREV, SHOPPING)
        if WIKIPEDIA and (WIKIPEDIA != WIKIPEDIA_PREV):
            file_str = file_str.replace(WIKIPEDIA_PREV, WIKIPEDIA)
        if HOMEPAGE and (HOMEPAGE != HOMEPAGE_PREV):
            file_str = file_str.replace(HOMEPAGE_PREV, HOMEPAGE)
        if CLASSIFIEDS and (CLASSIFIEDS != CLASSIFIEDS_PREV):
            file_str = file_str.replace(CLASSIFIEDS_PREV, CLASSIFIEDS)

        with open(os.path.join(directory, file), "w") as f:
            f.write(file_str)
            print(f"Processed {file} in {directory}")


print("Data processed successfully")
