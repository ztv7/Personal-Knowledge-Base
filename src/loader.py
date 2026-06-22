import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file)->list[tuple]:
    doc_bytes = file.read()
    doc = fitz.open(stream=doc_bytes,filetype="pdf")
    pages = []
    for i,page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append((i+1,text))
    doc.close()
    return pages

def chunk_text(page_data:list[tuple], filename:str ,chunk_size=500, overlap=50):
    documents = []
    metadatas = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=overlap)
    for page_num,text in page_data:
        chunk = text_splitter.split_text(text)
        for chunk_idx in chunk:
            documents.append(chunk_idx)
            metadatas.append({"page":page_num,"source":filename})
    ids = [f"{filename}id_{i+1}" for i in range(len(documents))]
    return ids,documents,metadatas