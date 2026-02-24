import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# Настройка страницы
st.set_page_config(page_title="Technical Doc Assistant", layout="wide")
st.title("🤖 Технический ассистент (NotebookLM Style)")

# 1. Инициализация API (можно ввести в интерфейсе или через секреты)
api_key = st.sidebar.text_input("Введите Gemini API Key:", type="password")
uploaded_file = st.sidebar.file_uploader("Загрузите технический PDF", type="pdf")

if api_key and uploaded_file:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Сохраняем файл временно
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.get_buffer())

    # 2. Обработка документа (Кешируем, чтобы не пересчитывать при каждом сообщении)
    @st.cache_resource
    def prepare_vector_store():
        loader = PyPDFLoader("temp.pdf")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        chunks = text_splitter.split_documents(data)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return Chroma.from_documents(chunks, embeddings)

    vector_store = prepare_vector_store()
    
    # 3. Настройка чата
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

    # Логика чата
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Задайте вопрос по документу..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            result = qa_chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
            response = result["answer"]
            st.markdown(response)
            
            # Показываем источник (опционально)
            if result['source_documents']:
                with st.expander("Посмотреть источники"):
                    for doc in result['source_documents'][:2]:
                        st.caption(f"Страница {doc.metadata['page']}: {doc.page_content[:200]}...")

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append((prompt, response))
else:
    st.info("Пожалуйста, введите API ключ и загрузите PDF в боковой панели.")
