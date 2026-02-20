import gradio as gr
from retriever import retrieve
from llm import generate_answer, tokenizer, model
import fitz
import time
from advanced_evaluator import evaluate_answer
from evaluator import (
    compute_context_precision,
    compute_diversity,
    compute_faithfulness
)


def extract_text_from_upload(file):
    doc = fitz.open(file.name)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chat(query, file, category, year, top_k):

    start_time = time.time()

    if category == "All":
        category = None

    if year == "":
        year = None

    contexts = retrieve(
        query,
        top_k=int(top_k),
        category=category,
        year=year
    )

    if not contexts:
        return "No relevant research papers found.", "", "0 sec", ""

    # Remove math-heavy chunks
    def is_math_heavy(text):
        math_tokens = ["=", "\\", "N(", "q(", "E["]
        return sum(token in text for token in math_tokens) > 3

    contexts = [c for c in contexts if not is_math_heavy(c["text"])]

    # Handle user upload
    if file is not None:
        text = extract_text_from_upload(file)
        contexts.append({
            "paper": "User Upload",
            "category": "Uploaded",
            "year": "N/A",
            "text": text[:1500]
        })

    answer = generate_answer(query, contexts)

    # Advanced evaluation (light version)
    advanced_scores = evaluate_answer(query, answer, contexts, tokenizer, model)
    advanced_metrics = "\n".join(
        [f"{k}: {v}" for k, v in advanced_scores.items()]
    )

    # Format sources
    unique_sources = {}
    for c in contexts:
        key = c["paper"]
        if key not in unique_sources:
            unique_sources[key] = (c["category"], c["year"])

    sources = ""
    for paper, (cat, yr) in unique_sources.items():
        sources += f"- {paper} | {cat} | {yr}\n"

    latency = round(time.time() - start_time, 2)

    precision = compute_context_precision(query, contexts)
    diversity = compute_diversity(contexts)
    faithfulness = compute_faithfulness(answer, contexts)

    metrics = f"""
Context Precision: {precision:.2f}
Diversity Score: {diversity:.2f}
Faithfulness Proxy: {faithfulness:.2f}
"""

    return answer, sources, f"{latency} seconds", metrics + "\n\n" + advanced_metrics


# ---------------- UI ----------------

with gr.Blocks() as demo:

    gr.Markdown("""
# 🧠 ML Research RAG Baseline
CPU-based Retrieval-Augmented Generation system indexing 100+ ML papers.
""")

    with gr.Row():
        with gr.Column(scale=1):

            query_input = gr.Textbox(label="Ask a Question")
            file_input = gr.File(label="Upload PDF (Optional)")

            category_input = gr.Dropdown(
                ["All", "NLP", "CV", "RL"],
                value="All",
                label="Filter by Category"
            )

            year_input = gr.Textbox(label="Filter by Year (Optional)")

            topk_input = gr.Slider(
                minimum=3,
                maximum=6,
                value=5,
                step=1,
                label="Top-K Retrieval"
            )

            submit_btn = gr.Button("🚀 Submit")

        with gr.Column(scale=2):

            answer_output = gr.Textbox(label="Generated Answer", lines=15)
            sources_output = gr.Textbox(label="Sources Used", lines=8)
            latency_output = gr.Textbox(label="Response Time")
            metrics_output = gr.Textbox(label="RAG Evaluation Metrics")

    submit_btn.click(
        chat,
        inputs=[query_input, file_input, category_input, year_input, topk_input],
        outputs=[answer_output, sources_output, latency_output, metrics_output]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
