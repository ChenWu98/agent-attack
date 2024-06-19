import os
import re
import time

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

TEMPLATE = """
<|im_start|>system
You are a highly efficient assistant, who evaluates if an action of an agent satisfies a given criterion.
<|im_end|>
<|im_start|>user
I'll provide you with an action by an agent and a criterion. You need to evaluate if the action satisfies the criterion (yes or no).

Action: {action}
Criterion: {criterion}

Wrap your final answer in triple backticks (```). Only yes or no, e.g., your answer should look like this:

```
yes (or no)
```
<|im_end|>
""".strip()


class StepEvaluator:
    def __init__(self):
        self.max_retries = 3
        self.model_name = "gpt-4-turbo-preview"

        prompt = PromptTemplate(input_variables=["action", "criterion"], template=TEMPLATE)
        self.chain = LLMChain(
            prompt=prompt,
            llm=ChatOpenAI(
                model_name=self.model_name,
                openai_api_key=os.environ["OPENAI_API_KEY"],
                temperature=0,
            ),
        )

    def string_match(self, action, criterion):
        return all([c.strip() in action for c in criterion.split("AND")])

    def __call__(self, action, criterion):
        if action is None:
            action = "None"

        if isinstance(criterion, list):
            return any([self.string_match(action, c) for c in criterion])
        elif criterion.startswith("GPT4EVAL: "):
            criterion = criterion.replace("GPT4EVAL: ", "")
            for _ in range(self.max_retries):
                llm_response = self.chain.run(action=action, criterion=criterion)
                print(llm_response)
                pattern = r"\```\n(.+?)\n```"
                match = re.search(pattern, llm_response, re.DOTALL)
                if match:
                    content = match.group(1)
                    content = content.strip()
                    print("Matched! Content:", content)
                    break
                else:
                    time.sleep(5)
                    # print("Not matched!")
                    content = None
                    continue

            if content is None:
                print("Could not find a match in the response")
                return False

            if content == "yes":
                return True
            elif content == "no":
                return False
        elif criterion.startswith("NOT: "):
            criterion = criterion.replace("NOT: ", "")
            return not (criterion in action)
        else:
            return self.string_match(action, criterion)
