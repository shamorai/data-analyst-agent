## df - pandas
## agent - [llm, tools, prompt]
## ui -
import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool #it is  the decorator to convert any python function into a tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st



_df: pd.DataFrame = None

def set_df(df:pd.DataFrame):
    global _df
    _df = df

if "messages" not in st.session_state:
    st.session_state.messages = []    

@tool
def get_data_info():
    """Get columns, data types, shape, and first 5 rows of the uploaded CSV file"""
    if _df is None:
        return "No CSV loaded"

    return f"""
        Shape: {_df.shape},
        Columns & Types: {_df.dtypes.to_string()},
        First 5 Rows: {_df.head().to_string()}


     """

@tool
def filter_tool(condition: str):
    """
    Filter rows using a pandas query condition.
    Example: 'Age > 30 and Department == 'Engineering''
    """

    if  _df is None:
        return "No CSV loaded"

    try:
        result = _df.query(condition)
        return f"Found{len(result)} rows: {result.to_string()}"
    except Exception as e:
        return f"Error: {e}"



@tool
def analyze_data(expression: str):
    """
    Run a pandas expression on the dataframe (referred to as '_df').
    Examples:
       df['Salary'].mean()
       df.groupby('Department')['Salary'].mean()
       df['city'].value_counts()
    """

    if  _df is None:
            return "No CSV loaded"


    try:
        df = _df
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
    


all_tools = [get_data_info, filter_tool, analyze_data]
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-20b")


system_prompt = """
You are a data analyst assistant. You help users explore and analyze CSV data.
Use the tools to answer questions:
   -get_data_info -> see columns, types and sample rows
   -filter_data -> filter rows by condition
   -analyze_data -> run pandas expressions for stats and aggregations
Always use get_data_info first if you are unsure about the column names.   
"""

MEMORY = InMemorySaver()

agent = create_agent(
    model=llm, tools=all_tools, system_prompt=system_prompt, checkpointer=MEMORY
)


def run_agent(query:str):
    res = agent.invoke({"messages":[{"role":"user","content": query}]}, {"configurable": {"thread_id": "1"}},)

    answer = res["messages"][-1].content
    return answer

### lets build our UI

st.header("Data Analysis AI Agent")
file = st.file_uploader(label="Select your CSV File", type="csv")
if file:
    df = pd.read_csv(file)
    set_df(df)
    st.dataframe(df.head())
    # st.sucess(f"Loaded {df.shape[0]} rows.")
    st.success(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")

for msg in st.session_state.messages:
    role = msg["role"]
    content =msg["content"]    
    st.chat_message(role).markdown(content)

    
    # Question Input Field
query = st.chat_input("Ask a question about your dataset:")
if query:
     st.chat_message("user").markdown(query)
     st.session_state.messages.append({"role":"user","content":"query"})
     res = run_agent(query)
     st.chat_message("ai").markdown(res)
     st.session_state.messages.append({"role":"ai","content":"res"})
