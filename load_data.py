import chromadb
from pypdf import PdfReader

def main():
    print("Starting data loading process...")

    # 1. Initialize ChromaDB client
    #    (This creates a persistent database in a folder named 'db')
    client = chromadb.PersistentClient(path="./db")

    # 2. Create or get a "collection" (like a table in SQL)
    #    We'll name our collection "documents"
    collection_name = "documents"
    
    # Try to get the collection, if it doesn't exist, create it.
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        print(f"Collection '{collection_name}' not found. Creating...")
        collection = client.create_collection(name=collection_name)
        print("Collection created.")

    # 3. Read and "chunk" the PDF
    print("Reading and chunking PDF...")
    reader = PdfReader("project_management.pdf")
    
    documents = []
    ids = []
    
    doc_id = 1
    # We will treat each page as a "document chunk"
    for page in reader.pages:
        text = page.extract_text()
        if text: # Make sure there is text on the page
            documents.append(text)
            ids.append(f"doc_id_{doc_id}")
            doc_id += 1

    print(f"Successfully chunked PDF into {len(documents)} documents.")

    # 4. Add the documents to the ChromaDB collection
    #    ChromaDB will automatically handle tokenizing and embedding for us!
    try:
        collection.add(
            documents=documents,
            ids=ids
        )
        print(f"Successfully added {len(documents)} documents to the collection.")
    
    except chromadb.errors.IDAlreadyExistsError:
        print("Documents with these IDs already exist in the collection. Skipping.")
    except Exception as e:
        print(f"An error occurred while adding documents: {e}")

    # 5. Check if it worked
    count = collection.count()
    print(f"\nProcess complete. The collection '{collection_name}' now has {count} documents.")
    
    # Optional: Peek at the first item
    if count > 0:
        print("\nPeeking at one item in the database:")
        peek = collection.peek(limit=1)
        print(peek)


if __name__ == "__main__":
    main()