import arxiv
import os
import time
import json

BASE_DIR = "data/papers"

CATEGORIES = {
    "NLP": {
        "query": "cat:cs.CL",
        "count": 150
    },
    "CV": {
        "query": "cat:cs.CV",
        "count": 150
    },
    "RL": {
        "query": "reinforcement learning AND cat:cs.LG",
        "count": 150
    }
}

def fetch_category(category_name, query, max_results):
    save_path = os.path.join(BASE_DIR, category_name)
    os.makedirs(save_path, exist_ok=True)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    client = arxiv.Client()

    downloaded = 0
    metadata_list = []

    print(f"\nFetching {category_name} papers...")

    for result in client.results(search):
        title = result.title.replace("/", "").replace(" ", "_")[:80]
        filename = os.path.join(save_path, f"{title}.pdf")

        if not os.path.exists(filename):
            try:
                result.download_pdf(dirpath=save_path, filename=f"{title}.pdf")
                downloaded += 1

                metadata_list.append({
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "published": str(result.published),
                    "summary": result.summary,
                })

                print(f"Downloaded: {title}")

                time.sleep(1)

            except Exception as e:
                print(f"Failed: {title} | {e}")

        if downloaded >= max_results:
            break

    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata_list, f, indent=2)

    print(f"{category_name}: Downloaded {downloaded} papers.\n")


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    for category, details in CATEGORIES.items():
        fetch_category(category, details["query"], details["count"])

    print("All downloads complete!")


if __name__ == "__main__":
    main()
