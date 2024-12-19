from langchain.prompts import HumanMessage
#from langchain.schema import HumanMessage
from langchain.chat_models import AzureChatOpenAI
import pandas as pd
import matplotlib.pyplot as plt
import json

API_KEY = "4e08d219725c475a8623f1618397fc2e"
ENDPOINT = "https://lida.openai.azure.com/openai/deployments/college/chat/completions?api-version=2024-02-15-preview"
DEPLOYMENT_NAME = "college"

llm = AzureChatOpenAI(
    openai_api_base=ENDPOINT,
    openai_api_key=API_KEY,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_version="2024-02-15-preview"
)

def read_csv_file(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df, df.columns.tolist()
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

def get_plot_suggestion(columns, user_goal):
    try:
        llm_prompt = (
            f"Given the columns {columns} from a dataset and the user goal '{user_goal}', "
            """recommend the most relevant columns for visualization and suggest an appropriate plot type.
            Your output should only be column for X axis, Y axis and the plot type. Don't return anything else.
            Output format is (x axis column, y axis column, plot type). 
            If the plot type is histogram, output format should be (column, plot type)."""
        )

        human_message = HumanMessage(content=llm_prompt)

        llm_response = llm([human_message])

        # Extract and clean the response content
        response_content = llm_response.content.strip()
        print("Raw LLM Response:", response_content)

        # Parse the response manually
        if response_content.startswith("(") and response_content.endswith(")"):
            response_content = response_content[1:-1]  # Remove parentheses
            parts = [part.strip().strip("'\"") for part in response_content.split(",")]
            if len(parts) == 3:  # For plots with x, y, and type
                return parts[0], parts[1], parts[2]
            elif len(parts) == 2:  # For histograms with column and type
                return parts[0], parts[1]
            else:
                raise ValueError("Unexpected response format.")
        else:
            raise ValueError("Response is not in the expected tuple format.")
    except Exception as e:
        raise RuntimeError(f"Error in get_plot_suggestion: {e}")

def generate_plot(df, plot_details):
    try:
        if len(plot_details) == 3:  
            x_col = plot_details[0]
            y_col = plot_details[1]
            plot_type = plot_details[2].lower()
            plt.figure(figsize=(10, 6))
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(True)
            if "line" in plot_type:
                plt.plot(df[x_col], df[y_col], marker='o')
            elif "bar" in plot_type:
                plt.bar(df[x_col], df[y_col], color='skyblue')
            elif "scatter" in plot_type:
                plt.scatter(df[x_col], df[y_col], alpha=0.7, color='purple')
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
        elif len(plot_details) == 2: 
            column = plot_details[0]
            plot_type = plot_details[1].lower()
            plt.figure(figsize=(10, 6))
            plt.xlabel(column)
            plt.grid(True)
            if "histogram" in plot_type:
                plt.hist(df[column], bins=20, color='green', alpha=0.7)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
        else:
            raise ValueError("Invalid plot details format.")
        plt.show()
    except Exception as e:
        raise RuntimeError(f"Error in generate_plot: {e}")

def analyze_goal(state):
    """Main analysis function."""
    try:
        csv_path = state.get("csv_path")
        user_goal = state.get("user_goal")
        if not csv_path or not user_goal:
            raise ValueError("State must include 'csv_path' and 'user_goal'.")

        df, columns = read_csv_file(csv_path)
        plot_suggestion = get_plot_suggestion(columns, user_goal)
        print("LLM Plot Suggestion:", plot_suggestion)
        generate_plot(df, plot_suggestion)
    except Exception as e:
        raise RuntimeError(f"Error in analyze_goal: {e}")

def main():
    """Main function to run the workflow."""
    state = {
        "csv_path": "C:/Users/Kannan T/Documents/DATA SET/IRIS.csv",
        "user_goal": "I want to see the occurance of each type of species in the dataset."
    }

    try:
        analyze_goal(state)
    except Exception as e:
        print(f"Workflow failed: {e}")

if __name__ == "__main__":
    main()