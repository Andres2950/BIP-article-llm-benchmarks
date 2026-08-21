from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

_vectordb = None
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embeddings

def load_vectorstore(dir_path):
    global _vectordb
    if _vectordb is not None:
        return _vectordb

    documents = []
    for pdf in Path(dir_path).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        for p in pages:
            p.metadata["source"] = pdf.name
        documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"📄 {len(chunks)} fragmentos creados.")

    embeddings = get_embeddings()
    _vectordb = FAISS.from_documents(chunks, embeddings)
    return _vectordb

def load_context(dir_path, question, k=5):
    vectorstore = load_vectorstore(dir_path)
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    sources = {d.metadata.get("source", "desconocido") for d in docs}
    return context

def build_prompt(context, question, question_type):
    type_instructions = {
        "yes_no": "Responde SOLO con la palabra exacta 'sí' o 'no' (sin acentos, sin puntuación, sin texto adicional).",
        "short_answer": "Responde con una frase corta y concisa de máximo 20 palabras.",
        "open_ended": "Responde detalladamente pero sin superar las 200 palabras.",
    }

    return f"""
Eres un asistente que responde preguntas basándose EXCLUSIVAMENTE en el siguiente contexto.

CONTEXTO:
{context}

PREGUNTA: {question}

INSTRUCCIONES:
- Usa ÚNICAMENTE la información del contexto.
- Si no encuentras la respuesta, di "No lo sé".
- Si la pregunta no está relacionada, di "No estoy seguro".
- Tu respuesta debe ser en español, sin mezclar con inglés.
- No incluyas explicaciones adicionales, ni razonamientos.
- Responde directamente a la pregunta.

TIPO DE PREGUNTA: {question_type}
{type_instructions.get(question_type, "")}
"""