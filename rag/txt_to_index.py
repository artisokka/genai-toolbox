import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Function to read all text files and prepare them for vector embedding
def load_text_files(text_folder):
    texts = []
    print(f"Loading text files from: {text_folder}")
    text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]
    print(f"Found {len(text_files)} text files")
    
    for file_name in text_files:
        file_path = os.path.join(text_folder, file_name)
        print(f"Reading file: {file_name}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                texts.append(file.read())
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
    
    print(f"Successfully loaded {len(texts)} text files")
    return texts

# Create FAISS index from text files
def create_faiss_index(text_folder, index_path, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
    # Convert paths to absolute Windows paths
    text_folder = str(Path(text_folder).resolve())
    index_path = str(Path(index_path).resolve())
    
    print(f"Using absolute paths:")
    print(f"Text folder: {text_folder}")
    print(f"Index path: {index_path}")
    
    # Check if paths exist
    print(f"\nChecking paths:")
    print(f"Text folder exists: {os.path.exists(text_folder)}")
    print(f"Index folder exists: {os.path.exists(index_path)}")
    
    texts = load_text_files(text_folder)
    if not texts:
        print("No text files found to index!")
        return
        
    print(f"\nCreating embeddings for {len(texts)} documents...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.from_texts(texts, embeddings)

    # Create the index directory if it doesn't exist
    os.makedirs(index_path, exist_ok=True)
    print(f"Ensured index directory exists at: {index_path}")
    
    # Save the FAISS index to disk
    print(f"\nSaving index...")
    try:
        # First try to remove existing files
        faiss_path = os.path.join(index_path, "index.faiss")
        pkl_path = os.path.join(index_path, "index.pkl")
        if os.path.exists(faiss_path):
            os.remove(faiss_path)
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
        print("Cleared existing index files")
        
        # Save new index
        vector_store.save_local(index_path)
        print(f"FAISS index saved successfully to {index_path}")
        
        # Verify files were created
        index_files = os.listdir(index_path)
        print(f"\nFiles in index directory:")
        for file in index_files:
            file_path = os.path.join(index_path, file)
            size = os.path.getsize(file_path)
            print(f"- {file} ({size} bytes)")
            
    except Exception as e:
        print(f"Error saving index: {e}")
        raise

if __name__ == "__main__":
    # Get the script's directory
    script_dir = Path(__file__).resolve().parent
    
    # Use exact paths we can see in the directory
    text_folder = script_dir / "DataTxt"
    index_path = script_dir / "DataIndex"
    
    print(f"Starting indexing process...")
    print(f"Script directory: {script_dir}")
    print(f"Text folder path: {text_folder}")
    print(f"Index path: {index_path}")
    
    create_faiss_index(text_folder, index_path)