import streamlit as st
import os
import ollama
from sentence_transformers import SentenceTransformer
import chromadb

st.title("我的本地知识库")

@st.cache_resource
def load():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    client = chromadb.PersistentClient(path="./kb_data")
    return model, client

embedder, client = load()
col = client.get_or_create_collection("docs")

uploaded = st.file_uploader("上传文档", type=["txt","md","pdf","docx"])
if uploaded:
    try:
        if uploaded.name.endswith('.pdf'):
            from pypdf import PdfReader
            reader = PdfReader(uploaded)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif uploaded.name.endswith('.docx'):
            from docx import Document
            doc = Document(uploaded)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            text = uploaded.read().decode("utf-8", errors="ignore")

        vec = embedder.encode(text).tolist()
        col.add(documents=[text], embeddings=[vec], ids=[uploaded.name])
        st.success("已收录！")
    except:
        st.error("上传失败")

query = st.text_input("想问什么？")
if query:
    with st.spinner("处理中"):
        qvec = embedder.encode(query).tolist()
        res = col.query(query_embeddings=[qvec], n_results=3)
        context = "\n".join(res['documents'][0])
        prompt = f"基于内容回答：{query}\n资料：{context}"
        r = ollama.chat(model='qwen3.5:2b', messages=[{"role":"user","content":prompt}])
        st.write(r['message']['content'])
		