import os
import time
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_dim = model.get_sentence_embedding_dimension()
print("Embedding dimension:", embedding_dim)
# saving embedding dimension to a file
import numpy as np
np.save("embedding_dim.npy", embedding_dim)


# ---------------------------------------------------------
# Initialize Pinecone 
#PineCone API Key:-pcsk_4gj7NP_KpYr5fcRNwSiFiTRHtUWaMAui5obMoh9Hxtnoivgh5h3KiYfCdowsXywjvrwTUG
# ---------------------------------------------------------
api_key = "pcsk_4gj7NP_KpYr5fcRNwSiFiTRHtUWaMAui5obMoh9Hxtnoivgh5h3KiYfCdowsXywjvrwTUG"   # ⚠️ move to env variable in real projects
pc = Pinecone(api_key=api_key)

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

# ---------------------------------------------------------
# PDF Text Extraction
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Generate Embeddings
# ---------------------------------------------------------
doc_ids = list(documents.keys())
doc_texts = list(documents.values())

embeddings = model.encode(
    doc_texts,
    batch_size=16,
    show_progress_bar=True
).tolist()

vectors = list(zip(doc_ids, embeddings))

# ---------------------------------------------------------
# Upsert into Pinecone
# ---------------------------------------------------------
index.upsert(vectors=vectors) # function is used to store document embeddings in a vector database. It either inserts new vectors or updates existing ones based on the document ID, enabling semantic search and retrieval in RAG systems." 

# ---------------------------------------------------------
# Wait Until Indexing Completes -It waits until the vector database finishes indexing all uploaded vectors.
# Vector DBs (like Pinecone) index in the background, so if you search immediately after upsert, you might get incomplete results.
# ---------------------------------------------------------

def wait_until_indexing_complete(idx, expected_count, check_interval=5):
    while True:
        stats = idx.describe_index_stats()
        current_count = stats.total_vector_count
        print(f"Indexed: {current_count}/{expected_count}")
        if current_count >= expected_count:
            break
        time.sleep(check_interval)

wait_until_indexing_complete(index, len(documents))

# ---------------------------------------------------------
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


#==========================================================
#==========================================================
# import os
# import time
# from pypdf import PdfReader
# from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric
# from sentence_transformers import SentenceTransformer
# import numpy as np

# # ---------------------------------------------------------
# # Load Embedding Model
# # ---------------------------------------------------------
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# embedding_dim = model.get_sentence_embedding_dimension()
# print("Embedding dimension:", embedding_dim)

# np.save("embedding_dim.npy", embedding_dim)

# # ---------------------------------------------------------
# # Initialize Pinecone
# # ---------------------------------------------------------
# api_key = "pcsk_4gj7NP_KpYr5fcRNwSiFiTRHtUWaMAui5obMoh9Hxtnoivgh5h3KiYfCdowsXywjvrwTUG"
# pc = Pinecone(api_key=api_key)

# index_name = "resume-search-index"

# # Delete existing index once
# if pc.has_index(index_name):
#     pc.delete_index(index_name)

# # Create index once
# pc.create_index(
#     name=index_name,
#     dimension=embedding_dim,
#     metric=Metric.COSINE,
#     spec=ServerlessSpec(
#         cloud=CloudProvider.AWS,
#         region=AwsRegion.US_EAST_1
#     )
# )

# index = pc.Index(host=pc.describe_index(index_name).host)
# print("Index created successfully")

# # ---------------------------------------------------------
# # PDF Extraction Function (same)
# # ---------------------------------------------------------
# def extract_text_from_pdfs(folder_path):
#     documents = {}
#     doc_id = 1

#     for file_name in os.listdir(folder_path):
#         if file_name.lower().endswith(".pdf"):
#             file_path = os.path.join(folder_path, file_name)
#             reader = PdfReader(file_path)

#             text = ""
#             for page in reader.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + " "

#             if text.strip():
#                 documents[f"doc_{doc_id}"] = text
#                 doc_id += 1

#     return documents

# # ---------------------------------------------------------
# # Main Parent Folder (contains 5 subfolders)
# # ---------------------------------------------------------
# base_path = r"Education"

# # Example:
# # Education/
# #   ├── folder1
# #   ├── folder2
# #   ├── folder3
# #   ├── folder4
# #   └── folder5

# folders = sorted(os.listdir(base_path))

# # ---------------------------------------------------------
# # Semantic Queries
# # ---------------------------------------------------------
# query_text1 = "data engineering resume, azure data factory, azure databricks"
# query_text2 = "data science machine learning langchain gen ai agentic ai"

# # ---------------------------------------------------------
# # Loop through each folder
# # ---------------------------------------------------------
# for i, folder in enumerate(folders, start=1):

#     folder_path = os.path.join(base_path, folder)

#     if not os.path.isdir(folder_path):
#         continue

#     print(f"\n==============================")
#     print(f"Processing Folder {i}: {folder}")
#     print(f"==============================")

#     start_time = time.time()

#     documents = extract_text_from_pdfs(folder_path)

#     print(f"Documents loaded: {len(documents)}")

#     doc_ids = list(documents.keys())
#     doc_texts = list(documents.values())

#     # Generate embeddings
#     embeddings = model.encode(
#         doc_texts,
#         batch_size=16,
#         show_progress_bar=True
#     ).tolist()

#     vectors = list(zip(doc_ids, embeddings))

#     # Upsert
#     index.upsert(vectors=vectors)

#     # Wait until indexing
#     while True:
#         stats = index.describe_index_stats()
#         count = stats.total_vector_count
#         print(f"Indexed so far: {count}")
#         if count >= len(vectors):
#             break
#         time.sleep(3)

#     # -------------------------------------------------
#     # Run Query for this folder
#     # -------------------------------------------------
#     query_embedding = model.encode(query_text1).tolist()

#     results = index.query(
#         vector=query_embedding,
#         top_k=5,
#         include_values=False
#     )

#     end_time = time.time()

#     print(f"\n⏱ Time for Folder {i}: {end_time - start_time:.2f} seconds")

#     print(f"\n🔍 Results for Folder {i}:")
#     print(results)

#     print("\n------------------------------------")

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
