from lida import Manager, TextGenerationConfig,llm
import os 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display


API_KEY = "4e08d219725c475a8623f1618397fc2e"
ENDPOINT = "https://lida.openai.azure.com/openai/deployments/college/chat/completions?api-version=2024-02-15-preview"

text_gen = llm(provider="openai", api_type="azure", azure_endpoint = ENDPOINT, api_key = API_KEY ,api_version = "2023-07-01-preview")

lida = Manager(text_gen=text_gen)

textgen_config = TextGenerationConfig(n=1, temperature=0.2, model="gpt-3.5-turbo-0301", use_cache=True)

summary = lida.summarize("C:/Users/Kannan T/Documents/DATA SET/ITC.csv", summary_method="default", textgen_config=textgen_config)  

goals = lida.goals(summary, n=4, textgen_config = textgen_config)

library = "seaborn"

def generate_user_plot(user_goal):
        charts = lida.visualize(summary=summary, goal=user_goal, textgen_config=textgen_config, library=library)
        if charts:
                code = charts[0].code
                instructions = ["make the chart more attractive,colourfull and easy to understand.If the chart uses different colors for different categories, assign a meaningful color palette."]
                edited_charts = lida.edit(code=code,  summary=summary, instructions=instructions, library=library, textgen_config=textgen_config)
                edited_charts[0].savefig(f"user_goal.png")
                plt.figure(figsize=(10, 6))
                plt.imshow(plt.imread(f"user_goal.png"))
                plt.axis('off')
                plt.show()
                print(f"Chart saved as user_goal.png")


def plot_goals(goals):
    for i, goal in enumerate(goals):
        goal_str = str(goal)
        question = goal_str.split(",")[0]
        print(f"\nGoal {i + 1}: {question})") 
        try:
            charts = lida.visualize(summary=summary, goal=goal, textgen_config=textgen_config, library=library)
            if charts:
                code = charts[0].code
                instructions = ["make the chart more attractive,colourfull and easy to understand.If the chart uses different colors for different categories, assign a meaningful color palette."]
                edited_charts = lida.edit(code=code,  summary=summary, instructions=instructions, library=library, textgen_config=textgen_config)
                edited_charts[0].savefig(f"goal_{i+1}.png")
                plt.figure(figsize=(10, 6))
                plt.imshow(plt.imread(f"goal_{i+1}.png"))
                plt.axis('off')
                plt.show()
                print(f"Chart saved as goal_{i+1}.png")
            else:
                print(f"No chart generated for Goal {i + 1}.")
        except Exception as e:
            print(f"Error generating chart for Goal {i + 1}: {e}")


def main():

    while True:
        user_goal = input("Give a goal/prompt to generate a graph or type 'no' to exit: ")
        if user_goal.lower() == "no":
            print("Exiting program. Goodbye!")
            break
        elif user_goal.strip():
            print("Generating graph for the given goal...")
            generate_user_plot(user_goal)
        else:
            print("Invalid input. Please provide a valid goal or type 'no' to exit.")
            continue

    global goals
    global summary
    all_goals = goals
    n = 4
    plot_goals(goals)
    while True:
        print("Do you want to generate more goals? (yes/no): ", end="", flush=True)
        cont = input()
        if cont.lower() == "yes":
            n += 4
            new_goals = lida.goals(summary, n=n, textgen_config = textgen_config)
            new_goals = [goal for goal in new_goals if str(goal) not in [str(g) for g in all_goals]]
            if not new_goals:
                print("No new goals generated. Please try again.")
                continue
            all_goals.extend(new_goals)
            last_4_goals = new_goals[-4:]
            plot_goals(last_4_goals)
        elif cont.lower() == "no":
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")


if __name__ == "__main__":
    main()