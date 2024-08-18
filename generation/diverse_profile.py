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

# Here is an example code to diversify the user/item profiles

item_profile = {}
with open('../data/arts/item_profile.json', 'r') as f:
    for _line in f.readlines():
        _data = json.loads(_line)
        item_profile[_data["item_id"]] = _data["profile"]

user_profile = {}
with open('../data/arts/user_profile.json', 'r') as f:
    for _line in f.readlines():
        _data = json.loads(_line)
        user_profile[_data["user_id"]] = _data["profile"]

item_system_prompt = ""
with open('instruction/item_system_prompt_diverse.txt', 'r') as f:
    for line in f.readlines():
        item_system_prompt += line

user_system_prompt = ""
with open('instruction/user_system_prompt_diverse.txt', 'r') as f:
    for line in f.readlines():
        user_system_prompt += line

pick_item_id = np.random.choice(len(item_profile))
pick_user_id = np.random.choice(len(user_profile))

class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

print(Colors.GREEN + f"Diversify Profile for Item {pick_item_id}" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(item_system_prompt)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The Input Prompt is:\n" + Colors.END)
print(item_profile[pick_item_id])
print("---------------------------------------------------\n")
response = get_gpt_response_w_system(item_system_prompt, item_profile[pick_item_id])
print(Colors.GREEN + "Generated Results:\n" + Colors.END)
print(response)

print('=' * 20)
time.sleep(1)


print(Colors.GREEN + f"Diversify Profile for User {pick_user_id}" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(user_system_prompt)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The Input Prompt is:\n" + Colors.END)
print(user_profile[pick_user_id])
print("---------------------------------------------------\n")
response = get_gpt_response_w_system(user_system_prompt, user_profile[pick_user_id])
print(Colors.GREEN + "Generated Results:\n" + Colors.END)
print(response)