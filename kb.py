import streamlit as st
import os

# ?? 两套镜像一起开 = 彻底解决 huggingface 连接失败
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./model_cache"

# 强制让 transformers 也走国内镜像
from transformers.utils import hub
hub.HUGGINGFACE_CO_URL_TEMPLATE = "https://hf-mirror.com/{repo_id}/resolve/{commit_hash}/{filename}"

import ollama
from sentence_transformers import SentenceTransformer
import chromadb

st.title("我的本地知识库")

@st.cache_resource
def load():
    # 国内镜像加载，不会再报错
    model = SentenceTransformer(
        'BAAI/bge-small-zh-v1.5',
        trust_remote_code=True,
        download_cfg={"mirror": "https://hf-mirror.com"}
    )
    client = chromadb.PersistentClient(path="./kb_data")
    return model, client

embedder, client = load()
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

        vec = embedder.encode(text).tolist()
        col.add(documents=[text], embeddings=[vec], ids=[uploaded.name])
        st.success(f"? {uploaded.name} 已存入知识库")

    except Exception as e:
        st.error(f"上传失败：{str(e)}")

# 提问
query = st.text_input("输入你的问题：")
if query:
    with st.spinner("正在检索..."):
        try:
            qvec = embedder.encode(query).tolist()
            res = col.query(query_embeddings=[qvec], n_results=3)
            context = "\n".join(res['documents'][0])

            prompt = f"你是助手，基于资料回答：{context}\n问题：{query}"
            response = ollama.chat(model="qwen3.5:2b", messages=[{"role":"user","content":prompt}])

            st.markdown("### 回答：")
            st.write(response["message"]["content"])

        except Exception as e:
            st.error(f"出错：{str(e)}")
			