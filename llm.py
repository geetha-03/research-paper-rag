from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-base"

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()

# ---- Warmup to reduce first query lag ----
with torch.no_grad():
    _ = model.generate(
        **tokenizer("warmup", return_tensors="pt").to(device),
        max_new_tokens=5
    )


def summarize_chunk(chunk_text):

    prompt = f"""
Summarize the key innovation and main contribution from this research excerpt.
Focus only on conceptual contributions.
Avoid equations and detailed math.

Text:
{chunk_text}

Summary:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_answer(query, contexts):

    if not contexts:
        return "No relevant research papers found."

    # Limit to 3 chunks for CPU stability
    contexts = contexts[:3]

    # Trim chunk text
    chunk_texts = [c["text"][:600] for c in contexts]

    # -------- Stage 1: Individual Summaries --------
    intermediate_summaries = []

    for chunk in chunk_texts:
        summary = summarize_chunk(chunk)
        intermediate_summaries.append(summary)

    combined_summary_text = "\n\n".join(intermediate_summaries)

    # -------- Stage 2: Cross-Paper Synthesis --------
    final_prompt = f"""
You are a machine learning research assistant.

The following summaries describe recent research papers related to:
{query}

Your task:
- Identify common themes.
- Group advances into categories (architecture, efficiency, applications).
- Explain how these advances extend earlier diffusion models.
- Do NOT list papers individually.
- Provide 2–3 structured paragraphs.

Summaries:
{combined_summary_text}

Final Answer:
"""

    inputs = tokenizer(
        final_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.7
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
