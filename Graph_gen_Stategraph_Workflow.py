from typing import List, Optional, TypedDict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.chat_models import AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

API_KEY = "4e08d219725c475a8623f1618397fc2e"
ENDPOINT = "https://lida.openai.azure.com/openai/deployments/college/chat/completions?api-version=2024-02-15-preview"
DEPLOYMENT_NAME = "college"

class GenerativeUIState(TypedDict, total=False):
    input: HumanMessage
    suggested_plot: Optional[str]
    """Suggested plot type based on goal and data."""
    plot_path: Optional[str]
    """Path to the saved plot image."""
    result: Optional[str]
    """Plain text response."""

def suggest_plot_type(state: GenerativeUIState, config: RunnableConfig) -> GenerativeUIState:
    """
    LLM analyzes the user's goal and suggests the most appropriate plot type and columns.
    """
    initial_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that suggests appropriate plots for a user-provided CSV file based on their goals."),
            ("human", "Given the columns {columns} from a dataset and the user goal '{goal}', "
            """recommend the most relevant columns for visualization and suggest an appropriate plot type.
            Your output should only be column for X axis, Y axis and the plot type. Don't return anything else.
            Output format is (x axis column, y axis column, plot type). 
            If the plot type is histogram, output format should be (column, plot type)."""
            )
        ]
    )

    model = AzureChatOpenAI(
        openai_api_base = ENDPOINT,
        openai_api_key = API_KEY,
        deployment_name = DEPLOYMENT_NAME,
        openai_api_version = "2024-02-15-preview"
    )

    csv_file = state["input"]["csv_file"]
    goal = state["input"]["goal"]

    # Read a preview of the CSV file to pass to the LLM
    df = pd.read_csv(csv_file)
    columns = ", ".join(df.columns)

    # Invoke the LLM with goal and column details
    chain = initial_prompt | model
    result = chain.invoke({"goal": goal, "columns": columns}, config)

    if not isinstance(result, AIMessage):
        raise ValueError("Invalid result from model. Expected AIMessage.")

    return {"suggested_plot": str(result.content)}

def parse_csv_and_generate_plot(csv_file: str, plot_details: str) -> str:
    """
    Generate a plot based on the suggested plot type and save it to a file.
    """
    try:
        df = pd.read_csv(csv_file)
        plot_details = plot_details.strip("() ").replace("'", "").split(",")

        if len(plot_details) == 3:  # For plots requiring X and Y axes
            x_col = plot_details[0].strip()
            y_col = plot_details[1].strip()
            plot_type = plot_details[2].strip().lower()
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
            elif "box plot" in plot_type:
                sns.boxplot(x = df[x_col], y = df[y_col]) 
                #plt.title(f"Box Plot of {column}")
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")

        elif len(plot_details) == 2:  # For single-column plots like histograms
            column = plot_details[0].strip()
            plot_type = plot_details[1].strip().lower()
            plt.figure(figsize=(10, 6))
            plt.xlabel(column)
            plt.grid(True)

            if "histogram" in plot_type:
                plt.hist(df[column], bins=20, color='green', alpha=0.7)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")

        else:
            raise ValueError("Invalid plot details format.")

        # Save the plot as a file
        output_path = "output_plot.png"
        plt.savefig(output_path)
        plt.close()
        return output_path

    except Exception as e:
        raise RuntimeError(f"Error in generate_plot: {e}")

def generate_plot(state: GenerativeUIState) -> GenerativeUIState:
    """
    Generate the suggested plot and save it to a file.
    """
    suggested_plot = state["suggested_plot"]
    csv_file = state["input"]["csv_file"]

    try:
        plot_path = parse_csv_and_generate_plot(csv_file, suggested_plot)
        return {"plot_path": plot_path}
    except Exception as e:
        return {"result": f"Error generating plot: {str(e)}"}

def check_next_step(state: GenerativeUIState) -> str:
    """
    Decide whether to generate a plot or finish.
    """
    if "suggested_plot" in state:
        return "generate_plot"
    elif "plot_path" in state or "result" in state:
        return END
    else:
        raise ValueError("Invalid state flow.")

def create_graph() -> CompiledGraph:
    """
    Create the workflow graph.
    """
    workflow = StateGraph(GenerativeUIState)

    # Define nodes
    workflow.add_node("suggest_plot_type", suggest_plot_type)
    workflow.add_node("generate_plot", generate_plot)

    # Conditional edges
    workflow.add_conditional_edges("suggest_plot_type", check_next_step)

    # Set start and finish points
    workflow.set_entry_point("suggest_plot_type")
    workflow.set_finish_point("generate_plot")

    return workflow.compile()

# Prepare user input
csv_file_path = "C:/Users/Kannan T/Documents/DATA SET/Titanic-Dataset.csv" 
user_goal = "I want to see the relationship between sex and pclass"

# Create the graph
graph = create_graph()

# Define the input state
input_state = {"input": {"csv_file": csv_file_path, "goal": user_goal}}

# Run the graph
result = graph.invoke(input_state)

# Display the generated plot (if available)
if "plot_path" in result:
    from IPython.display import Image, display
    display(Image(filename=result["plot_path"]))
else:
    print(result.get("result", "No plot generated."))
