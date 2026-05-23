import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer
import chromadb
import os

# 🔥 关键：解决 HuggingFace 连接失败（10060 错误）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

st.title("我的本地知识库")

@st.cache_resource
def load():
    # 从国内镜像加载，不会再报错
    return SentenceTransformer('BAAI/bge-small-zh-v1.5'), chromadb.PersistentClient(path="./kb_data")

embedder, client = load()
col = client.get_or_create_collection("docs")

# 文件上传
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
            # 兼容更多编码
            text = uploaded.read().decode("utf-8", errors="ignore")
        
        # 存入向量库
        vec = embedder.encode(text).tolist()
        col.add(documents=[text], embeddings=[vec], ids=[uploaded.name])
        st.success(f"✅ 文档 {uploaded.name} 已收录！")
    
    except Exception as e:
        st.error(f"读取失败：{str(e)}")

# 提问
query = st.text_input("想问什么？")
if query:
    with st.spinner("思考中..."):
        try:
            qvec = embedder.encode(query).tolist()
            res = col.query(query_embeddings=[qvec], n_results=3)
            context = "\n".join(res['documents'][0])
            
            prompt = f"基于以下内容回答：{query}\n\n{context}"
            r = ollama.chat(model='qwen3.5:2b', messages=[{'role':'user','content':prompt}])
            
            st.write("### 回答：")
            st.write(r['message']['content'])
        
        except Exception as e:
            st.error(f"出错：{str(e)}")