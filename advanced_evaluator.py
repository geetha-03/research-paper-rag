import torch


def llm_score(prompt, tokenizer, model):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.0
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract first digit 1-5 if present
    for char in result:
        if char in ["1", "2", "3", "4", "5"]:
            return char

    return "N/A"


def evaluate_answer(query, answer, contexts, tokenizer, model):

    # Use less context for evaluation to reduce latency
    context_text = "\n\n".join([c["text"][:300] for c in contexts[:2]])

    # 1️⃣ Faithfulness (LLM judged)
    faithfulness_prompt = f"""
Given the context and answer below,
return ONLY a single number from 1 to 5.

1 = not supported
5 = fully supported

Context:
{context_text}

Answer:
{answer}

Score:
"""

    faithfulness = llm_score(faithfulness_prompt, tokenizer, model)

    # 2️⃣ Answer Relevance
    relevance_prompt = f"""
Question:
{query}

Answer:
{answer}

Return ONLY a single number from 1 to 5 indicating
how well the answer addresses the question.

Score:
"""

    relevance = llm_score(relevance_prompt, tokenizer, model)

    return {
        "Faithfulness_LLM": faithfulness,
        "Answer_Relevance": relevance
    }
