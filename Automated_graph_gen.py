import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display
from openai import AzureOpenAI
import seaborn as sns
import json
import re
from lida import Manager, TextGenerationConfig,llm


api_key = "8aa00b22342c40f5882e116427e55dd7"
endpoint = "https://feat1.openai.azure.com/"
DEPLOYMENT_NAME = "heybuddy"
API_VERSION = "2024-02-01"

API_KEY = "4e08d219725c475a8623f1618397fc2e"
ENDPOINT = "https://lida.openai.azure.com/openai/deployments/college/chat/completions?api-version=2024-02-15-preview"

file = "C:/Users/Kannan T/Downloads/supermarket_sales - Sheet1 2.csv"
if file:
    if file.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.endswith('.xlsx'):
        data = pd.read_excel(file)

chat_client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_deployment="heybuddy",
    api_key=api_key,
    api_version="2024-02-01"
)

columns_and_types = pd.DataFrame({'Column': data.columns, 'Data Type': data.dtypes})

columns_types_str = ", ".join(f"{row['Column']} ({row['Data Type']})" for _, row in columns_and_types.iterrows())

text_gen = llm(provider="openai", api_type="azure", azure_endpoint = ENDPOINT, api_key = API_KEY ,api_version = "2023-07-01-preview")

lida = Manager(text_gen=text_gen)

textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)

summary = lida.summarize(data, summary_method="default", textgen_config=textgen_config)


def goal_gen(summary,columns_types_str):
    prompt = f"""generate 4 goals based on the given data to plot graphs.
        summary of the data: {summary}
        Data Columns and Types: {columns_types_str} 
        Note : return only the goals in the form of a list."""

    response = chat_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.1,
        model="feature")

    response_content = response.choices[0].message.content
    goals = [line.split(". ", 1)[-1] for line in response_content.split("\n")]
    return goals

def graph_gen(goal):
    print("Goal: ",goal)
    prompt = f"""
    You are a graph plot generator. Analyze the data and the user goal to give a Python code to generate the graph.
    You can use matplotlib and seaborn libraries for visualization. Make the graph easy to understand and more attractive.
    Data Columns and Types: {columns_types_str}
    goal : {goal}
    Note : Don't give any example data by yourself. Data is already given and address the dataframe as 'data' always.
    """

    response = chat_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.1,
        model="feature")

    response_content = response.choices[0].message.content
    # print(response_content)

    code_blocks = re.findall(r"```python\n(.*?)\n```", response_content, re.DOTALL)

    if code_blocks:
        code_blocks = "\n".join(code_blocks)
        # print(type(code_blocks))
        # print(code_blocks)
    else:
        print('None')

    exec(code_blocks)

goals = goal_gen(summary,columns_types_str)
for goal in goals:
    graph_gen(goal)
while True:
    print("Do you want to generate more goals? (yes/no): ", end="", flush=True)
    cont = input()
    if cont.lower() == "yes":
        goals = goal_gen(summary, columns_types_str)
        for goal in goals:
            graph_gen(goal)
    elif cont.lower() == "no":
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")