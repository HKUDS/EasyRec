import os
import json
import time
import openai
import numpy as np

openai.api_key = ""

def get_gpt_response_w_system(instruction, prompt):
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0125', # You can also utilize gpt-4o-mini, which is more cost-effective.
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]
    )
    response = completion.choices[0].message.content
    # print(response)
    return response

# Here is an example code to generate the user/item profiles
# In real scenarios, we first generate the item profiles and then the user profiles.

item_prompt = {}
with open('../data/arts/prompts/item_prompt.json', 'r') as f:
    for _line in f.readlines():
        _data = json.loads(_line)
        item_prompt[_data["item_id"]] = _data["prompt"]

user_prompt = {}
with open('../data/arts/prompts/user_prompt.json', 'r') as f:
    for _line in f.readlines():
        _data = json.loads(_line)
        user_prompt[_data["user_id"]] = _data["prompt"]

item_system_prompt = ""
with open('instruction/item_system_prompt.txt', 'r') as f:
    for line in f.readlines():
        item_system_prompt += line

user_system_prompt = ""
with open('instruction/user_system_prompt.txt', 'r') as f:
    for line in f.readlines():
        user_system_prompt += line

pick_item_id = np.random.choice(len(item_prompt))
pick_user_id = np.random.choice(len(user_prompt))

class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

print(Colors.GREEN + f"Generating Profile for Item {pick_item_id}" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(item_system_prompt)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The Input Prompt is:\n" + Colors.END)
print(item_prompt[pick_item_id])
print("---------------------------------------------------\n")
response = get_gpt_response_w_system(item_system_prompt, item_prompt[pick_item_id])
print(Colors.GREEN + "Generated Results:\n" + Colors.END)
print(response)

print('=' * 20)
time.sleep(1)


print(Colors.GREEN + f"Generating Profile for User {pick_user_id}" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(user_system_prompt)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The Input Prompt is:\n" + Colors.END)
print(user_prompt[pick_user_id])
print("---------------------------------------------------\n")
response = get_gpt_response_w_system(user_system_prompt, user_prompt[pick_user_id])
print(Colors.GREEN + "Generated Results:\n" + Colors.END)
print(response)