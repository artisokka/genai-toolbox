import os
import re
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from io import StringIO

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

# Load FAISS index
def load_faiss_index(index_path, embedding_model):
    # Load the FAISS index using the same embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Create the RAG system using FAISS and OpenAI API
def create_rag_system(index_path, embedding_model='sentence-transformers/all-MiniLM-L6-v2', model_name="llama3.2"):

    faiss_index_path = os.path.join(index_path, "index.faiss")
    if not os.path.exists(faiss_index_path):
        return None

    # Load the FAISS index
    vector_store = load_faiss_index(index_path, embedding_model)

    # Initialize LangChain OpenAI LLM
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_api_base=os.environ.get("OPENAI_BASE_URL")
    )

    # Create a more detailed prompt template
    prompt_template = """
    You are an expert medical data scientist assistant. You have access to the following context extracted from scientific papers and documents.

    Your task is to:

    1. Analyze the context to identify important clinical variables, measurements, and data types relevant to the patient population or study.
    2. Design a synthetic tabular dataset schema capturing these key features.
    3. Generate **realistic synthetic patient data** — creating new, plausible values that reflect the characteristics described but **never copying or reproducing any actual patient data or text** from the documents.
    4. Provide narrative patient reports based on the synthetic data for each synthetic patient in the table using column named 'patient_report'.
    5. Explain any assumptions or reasoning behind your data generation choices.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    - Do **not** copy or reuse any real patient data, names, or exact values found in the documents.
    - Create entirely new synthetic data that is consistent with the clinical context and realistic distributions.
    - Clearly highlight any assumptions or estimations you make.
    - Format your response with:
        - A table schema with variable names, types, and descriptions.
        - Synthetic patient data rows.
        - Patient narrative reports that are consistent with the synthetic data.
    - If the context is insufficient to generate data, explicitly state that.

    """

    # Create a template for formatting the input for the model
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    # Create a RetrievalQA chain that combines the vector store with the model
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

# Function to run the RAG system with a user question
def get_answer(question, qa_chain):
    answer = qa_chain.run(question)
    return answer

# Function to extract a table from the output text
def extract_table(output_text):
    # Find all markdown-style tables (rows with pipes)
    table_blocks = re.findall(r"(\|.+?\|(?:\n\|.+?\|)+)", output_text, re.DOTALL)
    
    if not table_blocks:
        return None  # No tables found

    # Choose the second table (the first is the schema)
    if len(table_blocks) < 2:
        return None
    
    table = table_blocks[1]

    # Remove separator row (---|---|...) if present
    lines = table.strip().split("\n")
    if len(lines) >= 2 and re.match(r"\|\s*-+", lines[1]):
        lines.pop(1)  # remove the alignment row

    cleaned_table = "\n".join(lines)

    # Read with pandas
    try:
        df = pd.read_csv(StringIO(cleaned_table), sep="|")
        df = df.dropna(axis=1, how="all")  # Remove any empty columns from padding
        df.columns = [col.strip() for col in df.columns]
        return df
    except Exception as e:
        print(f"Failed to parse table: {e}")
        return None

if __name__ == "__main__":
    # Path to the FAISS index directory
    index_path = "DataIndex"

    # Initialize the RAG system
    rag_system = create_rag_system(index_path)

    # Get user input and generate the answer
    while True:
        user_question = input("Ask your question (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            print("Exiting the RAG system.")
            break
        answer = get_answer(user_question, rag_system)
        print(f"Answer: {answer}")