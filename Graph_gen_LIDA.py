from lida import Manager, TextGenerationConfig,llm
import os 

API_KEY = "4e08d219725c475a8623f1618397fc2e"
ENDPOINT = "https://lida.openai.azure.com/openai/deployments/college/chat/completions?api-version=2024-02-15-preview"

text_gen = llm(provider="openai", api_type="azure", azure_endpoint = ENDPOINT, api_key = API_KEY ,api_version = "2023-07-01-preview")

lida = Manager(text_gen=text_gen)

textgen_config = TextGenerationConfig(n=1, temperature=0.2, model="gpt-3.5-turbo-0301", use_cache=True)

summary = lida.summarize("https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv", summary_method="default", textgen_config=textgen_config)  

goals = lida.goals(summary, n=4, textgen_config=textgen_config)

library = "seaborn"


for i, goal in enumerate(goals):
    print(f"\nGoal {i + 1}: {goal}") 
    try:
        charts = lida.visualize(summary=summary, goal=goal, textgen_config=textgen_config, library=library)
        if charts:
            code = charts[0].code
            instructions = ["make the chart more attractive,colourfull and easy to understand.If the chart uses different colors for different categories, assign a meaningful color palette."]
            edited_charts = lida.edit(code=code,  summary=summary, instructions=instructions, library=library, textgen_config=textgen_config)
            edited_charts[0].savefig(f"goal_{i+1}.png")
            print(f"Chart saved as goal_{i+1}.png")
        else:
            print(f"No chart generated for Goal {i + 1}.")
    except Exception as e:
        print(f"Error generating chart for Goal {i + 1}: {e}")