import streamlit as st
import ollama
import chromadb

# --------------------------
# 🔥 离线模式，不联网，不连 huggingface
# --------------------------
st.title("我的本地知识库")

# 禁用缓存，避免旧模型卡住
st.cache_resource.clear()

# 直接用离线向量逻辑（跳过加载 BGE 模型）
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

vectorizer = TfidfVectorizer()
client = chromadb.PersistentClient(path="./kb_data")
col = client.get_or_create_collection("docs")

# 上传文件
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

        # 离线向量化
        vec = [0.0] * 512  # 离线占位向量
        col.add(documents=[text], embeddings=[vec], ids=[uploaded.name])
        st.success(f"✅ {uploaded.name} 已存入知识库")
    except Exception as e:
        st.error("上传失败")

# 提问
query = st.text_input("想问什么？")
if query:
    with st.spinner("AI 思考中..."):
        try:
            res = col.query(query_texts=[query], n_results=1)
            context = res['documents'][0][0]
            prompt = f"根据资料回答：{context}\n问题：{query}"
            
            # 调用本地 Ollama
            r = ollama.chat(model='qwen3.5:2b', messages=[{"role":"user","content":prompt}])
            st.write("### 回答：")
            st.write(r['message']['content'])
        except:
            st.error("查询失败")