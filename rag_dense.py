import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DenseRAG:
    def __init__(self, pdf_path: str, db_dir: str = "./chroma_db"):
        self.pdf_path = pdf_path
        self.db_dir = db_dir
        
        # Persistent client prevents re-indexing on every benchmark run
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(
            name="class11_math_dense",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Index automatically if database is empty
        if self.collection.count() == 0 and os.path.exists(pdf_path):
            self._build_index()

    def _build_index(self):
        print(f"[DenseRAG] Parsing PDF: {self.pdf_path}...")
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()

        # Chunk overlap keeps formulas spanning line breaks intact
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(pages)

        texts = [doc.page_content for doc in chunks]
        metadatas = [{"page": doc.metadata.get("page", 0)} for doc in chunks]
        ids = [f"doc_chunk_{i}" for i in range(len(chunks))]

        # Batch inserts to handle larger textbooks cleanly
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            self.collection.add(
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                ids=ids[i : i + batch_size]
            )
            
        print(f"[DenseRAG] Indexed {len(chunks)} chunks into vector database.")

    def retrieve(self, query: str, top_k: int = 3) -> str:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No relevant textbook excerpts found."
            
        return "\n\n---\n\n".join(docs)