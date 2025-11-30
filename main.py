import os
from agents import retrieve_top_k, answer_with_gemini


def main():
    print("=== Multimodal Gemini Agent ===")
    print("Make sure you have run:  python ingest.py  at least once.\n")

    while True:
        query = input("Enter your question (or 'quit'): ").strip()
        if query.lower() in {"quit", "exit"}:
            break

        img_path = input("Optional: path to an image (press Enter to skip): ").strip()
        if img_path == "":
            img_path = None
        elif not os.path.exists(img_path):
            print("Image not found, ignoring image.")
            img_path = None

        print("\nRetrieving relevant chunks from MongoDB...")
        chunks = retrieve_top_k(query, k=5)

        if not chunks:
            print("No chunks found. Did you run ingest.py?")
            continue

        print("Asking Gemini...\n")
        answer = answer_with_gemini(query, chunks, img_path)
        print("\n--- ANSWER ---\n")
        print(answer)
        print("\n--------------\n")


if __name__ == "__main__":
    main()
