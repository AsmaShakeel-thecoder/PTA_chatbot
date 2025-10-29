import streamlit as st
import pandas as pd
from datetime import timedelta
from openai import OpenAI
from pandasql import sqldf
import os
import re

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "saved_questions" not in st.session_state:
    st.session_state.saved_questions = []

# --- OpenAI Client Setup ---
api_key = st.secrets["openai"]["api_key_pta"]
if not api_key:
    raise ValueError("OPENAI_API_KEY_PTA environment variable not set.")
client = OpenAI(api_key=api_key)

# --- Data Loading ---
@st.cache_data
def load_data(file_path):
    try:
        file_id = "1hQZl1-KTC74893N8lp--qIla6cvxH5sN"
        url = f"https://drive.google.com/uc?id={file_id}"
        df = pd.read_csv(url)
        df['Start'] = pd.to_timedelta(df['Start'], errors='coerce')
        df['End'] = pd.to_timedelta(df['End'], errors='coerce')
        if 'Duration (HH:MM:SS)' in df.columns:
            df['Duration'] = pd.to_timedelta(df['Duration (HH:MM:SS)'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' not found.")
        return pd.DataFrame()

file_path = "Combined_Idling_Report.csv"
df_main = load_data(file_path)

if df_main.empty:
    st.stop()

# --- Sidebar Filters ---
st.sidebar.title("Data Filters")
selected_filters = {}
filtered_df = df_main.copy()

for col in df_main.columns:
    if pd.api.types.is_datetime64_any_dtype(df_main[col]) or pd.api.types.is_timedelta64_dtype(df_main[col]):
        continue
    unique_values = sorted(map(str, filtered_df[col].unique().tolist()))
    selected_values = st.sidebar.multiselect(f"Select {col}", unique_values)
    selected_filters[col] = selected_values

for col, values in selected_filters.items():
    if values:
        filtered_df = filtered_df[filtered_df[col].astype(str).isin(values)]

# --- Main Dashboard ---
st.title("Idling Data Dashboard")

if any(selected_filters.values()):
    st.subheader("Active Filters")
    for col, values in selected_filters.items():
        if values:
            st.markdown(f"- **{col}**: {', '.join(map(str, values))}")
else:
    st.info("No filters selected. Displaying all vehicles.")

st.subheader("Key Metrics")
col1, col2 = st.columns(2)
with col1:
    total_duration = filtered_df["Duration"].sum()
    st.metric("Total Idling Time", str(total_duration).split('.')[0])
with col2:
    if not filtered_df.empty:
        avg_duration = filtered_df["Duration"].mean()
        st.metric("Average Duration", str(avg_duration).split('.')[0])
    else:
        st.metric("Average Duration", "N/A")

st.subheader("Filtered Data")
st.dataframe(filtered_df.head(200))

csv = filtered_df.to_csv(index=False)
st.download_button("Download Full Filtered Data as CSV", csv, "filtered_data.csv", "text/csv")

# --- Ask About the Data ---
st.subheader("💬 Ask about the data")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask me anything about the idling data:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    filter_description = ""
    if any(selected_filters.values()):
        filter_description = "The user has applied the following filters: "
        for col, values in selected_filters.items():
            if values:
                filter_description += f"{col} in ({', '.join([f'\'{v}\'' for v in values])}), "
        filter_description = filter_description.strip(', ') + ". "

    with st.spinner("Generating SQL query..."):
        sql_prompt = f"""
        You are a SQL query generator. Your task is to write a single, valid SQL query to answer a question about a pandas DataFrame named 'filtered_df'.
        The DataFrame has the following columns, data types, and a sample of the data.

        DataFrame Head:
        {filtered_df.head(2).to_string()}

        Data types: {dict(df_main.dtypes)}.

        IMPORTANT: Your response MUST be only the SQL query, with no additional text or explanation. Use standard SQL syntax.
        """
        messages_for_sql_llm = [{"role": "system", "content": sql_prompt}]
        messages_for_sql_llm.extend(st.session_state.messages)

        sql_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages_for_sql_llm,
            temperature=0.0
        )

        query = sql_response.choices[0].message.content.strip()
        st.write("Generated SQL Query:", query)

    # --- SQL Execution and Formatting ---
    try:
        with st.spinner("Executing query and getting results..."):
            query_results = sqldf(query, locals())
            st.write("Generated SQL Query Results:", query_results)

            # 🔧 FIX: Convert nanosecond columns to human-readable durations
            def convert_nanoseconds_to_best_unit(ns):
                if pd.isna(ns):
                    return "N/A"
                if ns >= 8.64e13:
                    return f"{ns / 8.64e13:.2f} days"
                elif ns >= 3.6e12:
                    return f"{ns / 3.6e12:.2f} hours"
                elif ns >= 6e10:
                    return f"{ns / 6e10:.2f} minutes"
                else:
                    return f"{ns / 1e9:.2f} seconds"

            for col in query_results.columns:
                if pd.api.types.is_numeric_dtype(query_results[col]):
                    if any(kw in col.lower() for kw in ['duration', 'total', 'sum', 'avg', 'min', 'max']):
                        query_results[col] = query_results[col].apply(convert_nanoseconds_to_best_unit)

            # --- Get Natural Language Answer ---
            natural_language_prompt = f"""
            You are a helpful data assistant. A user asked a question, and you have the results of a SQL query.
            Your task is to provide a short, concise, natural language answer based on the query results.

            User's original question: {user_input}
            Query results (DataFrame):
            {query_results.to_string(index=False)}

            Answer:
            """
            messages_for_nl_llm = [{"role": "system", "content": natural_language_prompt}]
            messages_for_nl_llm.extend(st.session_state.messages[-3:])

            natural_language_response = client.chat.completions.create(
                model="gpt-4",
                messages=messages_for_nl_llm,
                temperature=0.4
            )

            final_answer = natural_language_response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

            with st.chat_message("assistant"):
                st.markdown(final_answer)

            if st.button("💾 Save this question", key=f"save_button_{len(st.session_state.saved_questions)}"):
                st.session_state.saved_questions.append(user_input)
                st.success("Question saved!")
                st.rerun()

    except Exception as e:
        st.error(f"An error occurred while executing the query: {e}")
        st.write("Please try rephrasing your question.")

# --- Saved Questions ---
st.subheader("💾 Saved Questions")
if st.session_state.saved_questions:
    st.info("You have saved the following questions:")
    for i, q in enumerate(st.session_state.saved_questions):
        st.markdown(f"- **{i+1}:** {q}")
    if st.button("Clear All Saved Questions"):
        st.session_state.saved_questions = []
        st.rerun()
else:
    st.info("To save an answer, click the 'Save' button after getting a useful answer.")

# --- Visualizations ---
st.subheader("Visualizations")

if not filtered_df.empty:
    st.markdown("#### Total Idling Time by Vehicle")
    vehicle_idling = filtered_df.groupby("Vehicle")["Duration"].sum().dt.total_seconds() / 3600
    chart_data = pd.DataFrame(vehicle_idling).reset_index()
    chart_data.columns = ['Vehicle', 'Total Idling Hours']
    st.bar_chart(chart_data, x='Vehicle', y='Total Idling Hours')
else:
    st.info("No data to display charts.")



