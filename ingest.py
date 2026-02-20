import os
import faiss
import pickle
import numpy as np
import fitz
import json
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

BASE_PATH = "data/papers"
INDEX_PATH = "vector_store/index.faiss"
META_PATH = "vector_store/metadata.pkl"

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# -------------------------------
# Remove reference section
# -------------------------------
def remove_references(text):
    patterns = [
        r"\nreferences\n",
        r"\nreference\n",
        r"\nbibliography\n"
    ]

    lower_text = text.lower()

    for pattern in patterns:
        match = re.search(pattern, lower_text)
        if match:
            return text[:match.start()]

    return text


# -------------------------------
# Extract text from PDF
# -------------------------------
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# -------------------------------
# Split into sections
# -------------------------------
def split_into_sections(text):
    """
    Split paper text into sections based on common headings.
    """
    section_patterns = [
        r"\nabstract\n",
        r"\nintroduction\n",
        r"\nrelated work\n",
        r"\nmethod\n",
        r"\nmethods\n",
        r"\nexperiments\n",
        r"\nresults\n",
        r"\nconclusion\n"
    ]

    sections = {}
    lower_text = text.lower()

    matches = []

    for pattern in section_patterns:
        for match in re.finditer(pattern, lower_text):
            matches.append((match.start(), match.group().strip()))

    matches = sorted(matches, key=lambda x: x[0])

    for i in range(len(matches)):
        start_pos = matches[i][0]
        section_name = matches[i][1].strip()
        end_pos = matches[i + 1][0] if i + 1 < len(matches) else len(text)

        section_text = text[start_pos:end_pos]
        sections[section_name] = section_text

    return sections


# -------------------------------
# Chunk text
# -------------------------------
def chunk_text(text, chunk_size=400, overlap=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


# -------------------------------
# Main ingestion
# -------------------------------
all_chunks = []
metadata = []

print("Processing papers...")

for category in os.listdir(BASE_PATH):
    category_path = os.path.join(BASE_PATH, category)

    if not os.path.isdir(category_path):
        continue

    # Load metadata.json for year info
    metadata_file = os.path.join(category_path, "metadata.json")
    year_lookup = {}

    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            papers_meta = json.load(f)
            for paper in papers_meta:
                title_key = paper["title"].replace("/", "").replace(" ", "_")[:80]
                year_lookup[title_key] = paper["published"][:4]

    for file in tqdm(os.listdir(category_path), desc=f"{category}"):
        if not file.endswith(".pdf"):
            continue

        pdf_path = os.path.join(category_path, file)

        try:
            text = extract_text(pdf_path)
            text = remove_references(text)
            chunks = chunk_text(text)

            year = year_lookup.get(file.replace(".pdf", ""), "Unknown")

            sections = split_into_sections(text)

            for section_name, section_text in sections.items():

                # Skip references
                if "reference" in section_name:
                    continue

                chunks = chunk_text(section_text, chunk_size=400, overlap=100)

                for idx, chunk in enumerate(chunks):

                    all_chunks.append(chunk)

                    metadata.append({
                        "paper": file,
                        "category": category,
                        "year": year,
                        "section": section_name,
                        "chunk_id": idx,
                        "text": chunk
                    })

        except Exception as e:
            print(f"Failed processing {file}")


print("Creating embeddings...")
embeddings = model.encode(all_chunks, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

os.makedirs("vector_store", exist_ok=True)

faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("Index built successfully!")
