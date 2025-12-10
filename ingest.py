# ingest.py
import os, glob, json
from PyPDF2 import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            texts.append({"page": i+1, "text": txt})
    return texts


def split_and_save(pdf_path, chunk_size=1000, chunk_overlap=200, out_dir="chunks"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(pdf_path)
    pages = pdf_to_text(pdf_path)

    docs = [
        Document(
            page_content=p["text"],
            metadata={"source": base, "page": p["page"]},
        )
        for p in pages
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = splitter.split_documents(docs)

    out = [{"page_content": d.page_content, "metadata": d.metadata} for d in split_docs]

    out_fn = os.path.join(out_dir, base + ".json")
    with open(out_fn, "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved {out_fn} ({len(out)} chunks)")
    return out_fn


def main():
    os.makedirs("chunks", exist_ok=True)
    for p in glob.glob("data/*.pdf"):
        print("Processing", p)
        split_and_save(p)


if __name__ == "__main__":
    main()
