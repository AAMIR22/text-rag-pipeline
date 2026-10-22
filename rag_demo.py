"""Run the complete Text-RAG Pipeline classroom demonstration.

The script follows the presentation: load, chunk, embed, store, retrieve,
inspect evidence, generate a grounded answer, and display sources. Users can
ask multiple questions against the same indexed knowledge base in one run.
"""

from pathlib import Path

from utils import (
    answer_question,
    chunk_sections,
    index_chunks,
    load_markdown_sections,
    open_collection,
    print_retrieved_chunks,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STORE_PATH = BASE_DIR / "chroma_store"
DEFAULT_QUESTION = "What is the domestic travel reimbursement limit?"


def main() -> None:
    """Build or reuse the knowledge base and answer user-entered questions."""
    print("STEP 1 - Load source documents by Markdown section")
    sections = load_markdown_sections(DATA_DIR)
    print(f"Loaded {len(sections)} sections.")

    print("\nSTEP 2 - Split sections into retrievable chunks")
    chunks = chunk_sections(sections, chunk_size=300, overlap=50)
    print(f"Created {len(chunks)} chunks with source metadata.")

    print("\nSTEP 3 - Open persistent Chroma storage")
    client, collection = open_collection(STORE_PATH)
    try:
        if index_chunks(collection, chunks):
            print("Embedded and stored the chunks.")
        else:
            print("Reused the saved collection; document embeddings were not rebuilt.")

        print("\nSTEP 4 - Ask your own question")
        print("Type a question about the indexed documents.")
        print("Press Enter to try the example question, or type 'q' to quit.\n")
        next_question = True
        while next_question:
            try:
                entered_question = input("Your question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting the Q&A session.")
                break

            if entered_question.lower() in {"q", "quit", "exit"}:
                break
            question = entered_question or DEFAULT_QUESTION

            print("\nSTEP 5 - Retrieve and inspect evidence")
            result = answer_question(collection, question, top_k=4)
            print(f"Question: {question}\n")
            print_retrieved_chunks(result["retrieved"])

            print("\nSTEP 6 - Grounded answer")
            print(result["answer"])
            print("\n" + "-" * 60)
            try:
                next_question = input("Ask another question? [Y/n]: ").strip().lower() not in {"n", "no", "q", "quit", "exit"}
            except (EOFError, KeyboardInterrupt):
                print("\nExiting the Q&A session.")
                break
    finally:
        client.close()

    print("Q&A session ended.")


if __name__ == "__main__":
    main()
