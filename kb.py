import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer
import chromadb

st.title("我的本地知识库")

@st.cache_resource
def load():
    return SentenceTransformer('BAAI/bge-small-zh-v1.5'), chromadb.PersistentClient(path="./kb_data")

embedder, client = load()
col = client.get_or_create_collection("docs")

uploaded = st.file_uploader("上传文档", type=["txt","md","pdf","docx"])
if uploaded:
    if uploaded.name.endswith('.pdf'):
        from pypdf import PdfReader
        reader = PdfReader(uploaded)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    elif uploaded.name.endswith('.docx'):
        from docx import Document
        doc = Document(uploaded)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = uploaded.read().decode("utf-8")
    vec = embedder.encode(text).tolist()
    col.add(documents=[text], embeddings=[vec], ids=[uploaded.name])
    st.success("已收录！")

query = st.text_input("想问什么？")
if query:
    qvec = embedder.encode(query).tolist()
    res = col.query(query_embeddings=[qvec], n_results=3)
    context = "\n".join(res['documents'][0])
    prompt = f"基于以下内容回答：{query}\n\n{context}"
    r = ollama.chat(model='qwen3.5:2b', messages=[{'role':'user','content':prompt}])
    st.write(r['message']['content'])
	