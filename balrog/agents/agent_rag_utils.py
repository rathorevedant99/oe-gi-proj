import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

def parse_mediawiki_xml(xml_path):
    """Parses MediaWiki XML and extracts full text per page."""
    ns = {"mw": "http://www.mediawiki.org/xml/export-0.10/"}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    pages = root.findall(".//mw:page", namespaces=ns)
    extracted = []

    for page in pages:
        title = page.find("mw:title", namespaces=ns).text
        revision = page.find(".//mw:revision/mw:text", namespaces=ns)

        if revision is not None and revision.text:
            text_content = revision.text.strip()
            extracted.append((title, text_content))

    # print(f"Extracted {len(extracted)} pages.")
    return extracted


def build_faiss_index(data, model, faiss_index_path=None, storage_path=None):
    """Encodes pages with embeddings and stores in FAISS HNSW index along with full text."""
    titles, texts = zip(*data)  # Extract titles and content
    
    # Convert text into embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity

    # Use HNSW for scalable nearest-neighbor search
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors in HNSW graph
    index.hnsw.efConstruction = 128  # Better recall
    index.add(embeddings)

    # Store full content alongside titles
    doc_store = [{"title": t, "content": c} for t, c in zip(titles, texts)]

    # Save FAISS index and document store
    faiss.write_index(index, faiss_index_path)
    with open(storage_path, "wb") as f:
        pickle.dump(doc_store, f)

    print(f"FAISS index and document store saved.")
    return index, doc_store


def load_faiss_index(faiss_index_path, storage_path):
    """Loads the FAISS index and document store if they exist."""
    if os.path.exists(faiss_index_path) and os.path.exists(storage_path):
        index = faiss.read_index(faiss_index_path)
        with open(storage_path, "rb") as f:
            doc_store = pickle.load(f)  # Load stored titles and texts
        print("Loaded FAISS index and document store from disk.")
        return index, doc_store
    else:
        print("No saved index found. Build it first.")
        return None, None
    

def search_faiss(index, doc_store, query, model, top_k=5):
    """Search FAISS index for similar documents and return titles + content."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)  # Normalize query

    distances, indices = index.search(query_embedding, top_k)

    # Retrieve the titles and content for the top-k results
    results = []
    for idx in indices[0]:
        doc = doc_store[idx]  # Retrieve from stored dictionary
        results.append((doc["title"], doc["content"]))

    return results