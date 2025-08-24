from generator_backup import generate_from_rerank
from retriever_backup import search_and_rerank_pipeline

import sys

def run_query(query: str):
    """
    Runs the full retrieval and generation pipeline for a given query.

    Args:
        query: The user's input query string.
    """
    print(f"🔍 Retrieving and reranking documents for: '{query}'...")
    try:
        retrieved_doc = search_and_rerank_pipeline(query)

        if not retrieved_doc:
            print("❌ Could not retrieve any relevant documents. Please try a different query.")
            return

        print("✅ Documents retrieved. Generating answer...")

        result = generate_from_rerank(query, retrieved_doc)

        # --- Display Results ---
        print("\n" + "=" * 60)
        print("🤖 Generated Answer:")
        print("=" * 60)
        print(result.get("answer", "No answer could be generated."))

        print("\n" + "=" * 60)
        print("📚 References Used:")
        print("=" * 60)
        used_refs = result.get("used_refs", [])
        if used_refs:
            for i, ref in enumerate(used_refs, 1):
                if isinstance(ref, dict) and 'url' in ref:
                    print(f"  [{i}] {ref['url']}")
                else:
                    print(f"  [{i}] {ref}")
        else:
            print("  No references were used.")
        print("=" * 60)

    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    """
    The main function to run the interactive CLI loop.
    """
    print("🚀 Welcome to the Pytorch CLI!")
    print("Type your query and press Enter. Type 'exit' or 'quit' to end.")

    while True:
        try:
            # Get user input with a clear prompt
            query = input("\n> ")

            # Check for exit commands
            if query.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            # If input is empty, continue to next iteration
            if not query.strip():
                continue

            # Process the query
            run_query(query)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"A critical error occurred in the main loop: {e}")


if __name__ == "__main__":
    main()
