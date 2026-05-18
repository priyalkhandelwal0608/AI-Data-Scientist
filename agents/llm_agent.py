from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

def dataset_summary(df):

    prompt = f"""
    You are an expert data scientist.

    Analyze this dataset.

    Columns:
    {df.columns.tolist()}

    Dataset Shape:
    {df.shape}

    First 5 Rows:
    {df.head().to_string()}

    Provide:
    1. Dataset overview
    2. Possible ML tasks
    3. Data quality issues
    4. Recommendations
    """

    response = llm.invoke(prompt)

    return response


def ask_dataset_question(df, question):

    prompt = f"""
    Dataset Columns:
    {df.columns.tolist()}

    Sample Data:
    {df.head().to_string()}

    User Question:
    {question}
    """

    response = llm.invoke(prompt)

    return response