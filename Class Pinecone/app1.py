import os
import time
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------------
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedding_dim = model.get_sentence_embedding_dimension()
print("Embedding dimension:", embedding_dim)
# saving embedding dimension to a file
import numpy as np
np.save("embedding_dim.npy", embedding_dim)


# ---------------------------------------------------------
# Initialize Pinecone 
#PineCone API Key:-pcsk_4gj7NP_KpYr5fcRNwSiFiTRHtUWaMAui5obMoh9Hxtnoivgh5h3KiYfCdowsXywjvrwTUG
# ---------------------------------------------------------
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access API key
openai_key = os.getenv("pinecone_api_key")
pc=Pinecone(api_key=openai_key)

index_name = "resume-search-index"

# Delete index if exists
if pc.has_index(index_name):
    pc.delete_index(index_name)

# Create index
pc.create_index(
    name=index_name,
    dimension=embedding_dim,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_EAST_1
    )
)

index = pc.Index(host=pc.describe_index(index_name).host)
print("Index created successfully")



pdf_folder_path = r"Education/2-999"

def extract_text_from_pdfs(folder_path):
    documents = {}
    doc_id = 1

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            reader = PdfReader(file_path)

            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "

            if text.strip():
                documents[f"doc_{doc_id}"] = text
                doc_id += 1

    return documents

start_time = time.time()
documents = extract_text_from_pdfs(pdf_folder_path)
end_time = time.time()

print(f"Text extraction completed in {end_time - start_time:.2f} seconds")
print(f"Total documents loaded: {len(documents)}")



# Generate Embeddings
# ---------------------------------------------------------
doc_ids = list(documents.keys())
doc_texts = list(documents.values())

embeddings = model.encode(
    doc_texts,
    batch_size=50,
    show_progress_bar=True
).tolist()

vectors = list(zip(doc_ids, embeddings))


# Upsert into Pinecone
# ---------------------------------------------------------
index.upsert(vectors=vectors)



# Semantic Queries
# ---------------------------------------------------------
query_text1 = "data engineering resume, azure data factory, azure databricks"
query_text2 = "data science machine learning langchain gen ai agentic ai"

query_embedding = model.encode(query_text1).tolist()

results = index.query(
    vector=query_embedding,
    top_k=5,
    include_values=False
)

print("\nSearch Results:")
print(results)


import time

# ---------------------------------------------------------
# Queries to compare (Before vs After)
# ---------------------------------------------------------
queries = {
    "Sentence 1": "data engineering resume, azure data factory, azure databricks",
    "Sentence 2": "data science machine learning langchain gen ai agentic ai",
    "Sentence 3": "data science machine learning langchain gen ai agentic ai"
}

results_table = []



# ---------------------------------------------------------
# Run Performance Benchmark
# ---------------------------------------------------------
for label, query_text in queries.items():
    print(f"\nRunning {label} query...")

    # Embedding timing
    embed_start = time.time()
    query_embedding = model.encode(query_text).tolist()
    embed_end = time.time()

    # Pinecone query timing
    search_start = time.time()
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_values=False
    )
    search_end = time.time()

    embedding_time = embed_end - embed_start
    pinecone_time = search_end - search_start
    total_time = search_end - embed_start

    results_table.append({
        "Query Type": label,
        "Embedding Time (s)": round(embedding_time, 4),
        "Pinecone Query Time (s)": round(pinecone_time, 4),
        "Total Time (s)": round(total_time, 4)
    })

# ---------------------------------------------------------
# Print Comparison Table
# ---------------------------------------------------------
print("\n================ PERFORMANCE COMPARISON ================\n")
print(f"{'Query':<10} | {'Embedding (s)':<14} | {'Pinecone (s)':<14} | {'Total (s)':<10}")
print("-" * 60)

for row in results_table:
    print(
        f"{row['Query Type']:<10} | "
        f"{row['Embedding Time (s)']:<14} | "
        f"{row['Pinecone Query Time (s)']:<14} | "
        f"{row['Total Time (s)']:<10}"
    )

print("\n========================================================\n")
