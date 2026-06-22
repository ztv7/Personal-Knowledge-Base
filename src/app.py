from src import loader, store, engine
import streamlit as st

# 页面基础配置
st.set_page_config(page_title="个人知识库RAG", page_icon="📚", layout="wide")
st.title("📚 Personal Knowledge Base 本地知识库问答")

# 初始化数据库（只初始化一次）
if "db" not in st.session_state:
    with st.spinner("正在加载向量数据库..."):
        st.session_state.db = store.Database()
    st.success("向量库加载完成！")

db = st.session_state.db

# ---------------------- 1. PDF上传区域 ----------------------
st.subheader("📤 上传PDF文档入库")
upload_file = st.file_uploader("选择PDF文件", type="pdf", help="支持多份PDF分次上传，自动切片存入向量库")

if upload_file is not None:
    file_name = upload_file.name
    with st.status(f"正在处理文档：{file_name}", expanded=True):
        st.write("1. 解析PDF文本...")
        page_data = loader.load_pdf(upload_file)
        if not page_data:
            st.error("PDF解析失败，未读取到任何文本内容！")
        else:
            st.write("2. 文本分片处理...")
            ids, text_chunks, metadatas = loader.chunk_text(page_data=page_data, filename=file_name)
            st.write(f"✅ 文档成功切分为 **{len(text_chunks)}** 个文本片段")
            st.write("3. 向量化并入库...")
            db.add(ids=ids, documents=text_chunks, metadatas=metadatas)
    st.success(f"【{file_name}】全部片段存入向量库完成！")
    st.divider()

# ---------------------- 2. 对话历史初始化 ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史对话
st.subheader("💬 问答对话区")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------- 侧边栏配置（新增清空对话按钮） ----------------------
with st.sidebar:
    st.header("⚙️ 检索配置")
    # 向量召回TopK
    top_k = st.slider(
        label="召回 Top-K 片段数量",
        min_value=1,
        max_value=20,
        value=8,
        step=1,
        help="向量库初次检索返回的片段总数"
    )
    # Rerank重排后保留条数
    rerank_top = st.slider(
        label="Rerank 重排后保留条数",
        min_value=1,
        max_value=top_k,
        value=3,
        step=1,
        help="对召回结果做相关性重排序，只取分数最高N条送入大模型，数值不能超过召回Top-K"
    )
    st.divider()

    # 清空对话按钮
    if st.button("🧹 清空全部对话历史", type="secondary"):
        st.session_state.messages = []
        st.rerun()

    st.header("📋 已入库文档列表")
    doc_list = list(db.doc_name_map.keys())
    if doc_list:
        for name in doc_list:
            st.markdown(f"- {name}")
    else:
        st.info("暂无入库PDF，请上传文件")
    st.divider()
    st.markdown("### 使用说明")
    st.markdown("1. 上传PDF自动切片入库")
    st.markdown("2. 多轮对话自动识别上下文，补全问句检索")
    st.markdown("3. 回答会标注对应文档名+页码")
    st.markdown("4. 如果没有得到满意的结果可以调整检索配置或者再询问一次")

# ---------------------- 3. 提问+流式回答核心逻辑 ----------------------
user_prompt = st.chat_input("请输入你的问题，无需输入完整文档名，AI自动匹配对应PDF：")
if user_prompt:
    # 展示用户提问并存入会话
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # 转换会话消息为引擎需要的格式：[{"user":"xxx","assistant":"xxx"}, ...]
    chat_history = []
    temp_user = None
    for m in st.session_state.messages:
        if m["role"] == "user":
            temp_user = m["content"]
        elif m["role"] == "assistant" and temp_user is not None:
            chat_history.append({"user": temp_user, "assistant": m["content"]})
            temp_user = None

    # 流式生成回答，所有检索逻辑内部完成
    with st.chat_message("assistant"):
        stream_gen = engine.generate_answer(
            db=db,
            raw_question=user_prompt,
            chat_history=chat_history,
            recall_top_k=top_k,
            rerank_top_n=rerank_top
        )
        full_ans = st.write_stream(stream_gen)
    # 保存助手回答到会话历史
    st.session_state.messages.append({"role": "assistant", "content": full_ans})