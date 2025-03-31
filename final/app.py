import streamlit as st
import pandas as pd
import time
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

# 🎨 **Streamlit UI Configuration**
st.set_page_config(page_title="LangChain SQL Chat", page_icon="🦜", layout="wide")
st.title("Chat with SQL DB")

# 🔑 **API Key Input**
st.sidebar.title("🔧 Settings")
api_key = st.sidebar.text_input("🗝️ GRoq API Key", type="password")
model_choice = st.sidebar.selectbox("🤖 Choose LLM Model", ["gemma2-9b-it","llama3-8b-8192", "deepseek-r1-distill-qwen-32b"])
if not api_key:
    st.warning("Please enter your GRoq API Key.")
    st.stop()

# 📂 **Database Selection**
dbfilepath = (Path(__file__).parent / "csv_to_sql/reviews.db").absolute()

if not dbfilepath.exists():
    st.error(f"❌ Database file not found: {dbfilepath}")
    st.stop()

# 🛠️ **Database Connection**
@st.cache_resource(ttl="2h")
def configure_db():
    creator = lambda: sqlite3.connect(str(dbfilepath))
    return SQLDatabase(create_engine("sqlite:///", creator=creator))

db = configure_db()

# 🤖 **LLM Model Initialization**
llm = ChatGroq(groq_api_key=api_key, model_name=model_choice, streaming=True)

# 🏗️ **SQL Agent & Toolkit**
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# # 📊 **Show Database Schema**
# st.sidebar.subheader("🗄️ Database Information")
# with st.sidebar.expander("📌 View Table Names", expanded=False):
#     tables = db.run("SELECT name FROM sqlite_master WHERE type='table';")
#     st.write(tables)

# 🏷️ **Show Column Names for the 'reviews' Table**
st.sidebar.subheader("📌 Columns in 'reviews' Table")
columns = [
    "authorName", "googleMapsPlaceId", "placeAddress", "placeName", "placeUrl",
    "provider", "reviewDate", "reviewRating", "reviewText", "reviewTitle",
    "reviewUrl", "sentiment"
]
st.sidebar.write("\n".join([f"🔹 `{col}`" for col in columns]))

# 📜 **Chat History**
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 📝 **User Input**
user_query = st.chat_input("🔍 Ask something about the database...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # ⏳ Track Query Execution Time
    start_time = time.time()
    
    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        execution_time = time.time() - start_time  # ⏱️ End Time

        # 📊 **Display Query Execution Time**
        st.caption(f"⏳ Query executed in {execution_time:.2f} seconds")
        
        # 📋 **Check if response is tabular**
        try:
            df = pd.DataFrame(response)
            st.dataframe(df.style.set_properties(**{"background-color": "#f4f4f4"}))
        except:
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# 🎯 **Query History & Retry**
st.sidebar.subheader("📜 Query History")
if len(st.session_state.messages) > 1:
    for i, msg in enumerate(st.session_state.messages[::-1]):  
        if msg["role"] == "user":
            if st.sidebar.button(f"🔄 {msg['content'][:30]}...", key=f"retry_{i}"):
                st.session_state["messages"].append({"role": "user", "content": msg["content"]})
                st.rerun()
