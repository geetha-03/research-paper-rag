import re


def compute_context_precision(query, contexts):
    """
    Measures how many contexts contain query keywords.
    """
    query_words = set(query.lower().split())
    match_count = 0

    for c in contexts:
        text = c["text"].lower()
        if any(word in text for word in query_words):
            match_count += 1

    return match_count / len(contexts) if contexts else 0


def compute_diversity(contexts):
    """
    Measures how many unique papers are used.
    """
    papers = set([c["paper"] for c in contexts])
    return len(papers) / len(contexts) if contexts else 0


def compute_faithfulness(answer, contexts):
    """
    Proxy: checks how many answer words appear in contexts.
    """
    context_text = " ".join([c["text"] for c in contexts]).lower()
    answer_words = set(re.findall(r'\w+', answer.lower()))

    matched = sum(word in context_text for word in answer_words)

    return matched / len(answer_words) if answer_words else 0
