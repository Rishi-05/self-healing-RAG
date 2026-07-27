import argparse
import shutil
from collections import Counter
from pathlib import Path

import chromadb

CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / "data" / "chroma_store")


def get_client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def list_collections():
    client = get_client()
    collections = client.list_collections()
    if not collections:
        print("No collections found.")
        return
    for col in collections:
        c = client.get_collection(col.name)
        print(f"- {col.name}: {c.count()} chunks")


def list_sources(collection_name: str):
    client = get_client()
    c = client.get_collection(collection_name)
    data = c.get(include=["metadatas"])
    sources = Counter(m["source"] for m in data["metadatas"])
    if not sources:
        print(f"Collection '{collection_name}' is empty.")
        return
    print(f"Sources in '{collection_name}':")
    for source, count in sources.items():
        print(f"  - {source}: {count} chunks")


def delete_collection(collection_name: str):
    client = get_client()
    client.delete_collection(collection_name)
    print(f"Deleted collection '{collection_name}'.")


def delete_source(collection_name: str, source: str):
    client = get_client()
    c = client.get_collection(collection_name)
    before = c.count()
    c.delete(where={"source": source})
    after = c.count()
    print(f"Deleted {before - after} chunks from source '{source}' "
          f"in collection '{collection_name}'. ({after} chunks remain)")


def wipe_all():
    path = Path(CHROMA_PERSIST_DIR)
    if path.exists():
        shutil.rmtree(path)
        print(f"Wiped entire vector store at {path}")
    else:
        print("No vector store found — nothing to wipe.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage the ChromaDB vector store.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")

    p_sources = sub.add_parser("sources")
    p_sources.add_argument("--collection", required=True)

    p_delcol = sub.add_parser("delete-collection")
    p_delcol.add_argument("--collection", required=True)

    p_delsrc = sub.add_parser("delete-source")
    p_delsrc.add_argument("--collection", required=True)
    p_delsrc.add_argument("--source", required=True)

    sub.add_parser("wipe-all")

    args = parser.parse_args()

    if args.command == "list":
        list_collections()
    elif args.command == "sources":
        list_sources(args.collection)
    elif args.command == "delete-collection":
        delete_collection(args.collection)
    elif args.command == "delete-source":
        delete_source(args.collection, args.source)
    elif args.command == "wipe-all":
        wipe_all()
