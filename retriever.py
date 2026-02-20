import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "vector_store/index.faiss"
META_PATH = "vector_store/metadata.pkl"

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)


def retrieve(query, top_k=5, category=None, year=None):

    expanded_query = f"""
    Find sections discussing recent advances, innovations,
    improvements, or efficiency gains related to:
    {query}
    """

    query_embedding = model.encode([expanded_query])
    distances, indices = index.search(np.array(query_embedding), top_k * 5)

    candidates = []

    for i in indices[0]:
        item = metadata[i]

        # Category filter
        if category:
            if item["category"].lower() != category.lower():
                continue

        # Year filter
        if year:
            if str(year) not in str(item["year"]):
                continue

        candidates.append(item)

    # ---- Section Priority Scoring ----
    section_priority = {
        "abstract": 3,
        "conclusion": 3,
        "introduction": 2,
        "results": 2,
        "experiments": 2,
        "method": 1,
        "methods": 1
    }

    def section_score(section_name):
        section_name = section_name.lower()
        for key in section_priority:
            if key in section_name:
                return section_priority[key]
        return 0

    # Sort candidates by section importance (descending)
    candidates = sorted(
    zip(candidates, distances[0]),
    key=lambda x: (section_score(x[0].get("section", "")), -x[1]),
    reverse=True
    )
    candidates = [x[0] for x in candidates]

    # ---- Diversity Selection AFTER sorting ----
    selected = []
    seen_papers = set()

    for item in candidates:
        if item["paper"] not in seen_papers:
            selected.append(item)
            seen_papers.add(item["paper"])

        if len(selected) >= top_k:
            break

    # Fallback if too strict
    if len(selected) < top_k:
        for item in candidates:
            if item not in selected:
                selected.append(item)
            if len(selected) >= top_k:
                break

    return selected


