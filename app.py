import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# --- 1. НАСТРОЙКА СТРАНИЦЫ И БЕЗОПАСНОСТЬ ---
st.set_page_config(page_title="Corporate Doc Assistant", layout="wide")

def check_password():
    def password_entered():
        # Рекомендуется использовать st.secrets["COMPANY_PASSWORD"] для продакшена
        if st.session_state["password"] == "SuperSecret123": 
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Введите пароль доступа", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Неверный пароль", type="password", on_change=password_entered, key="password")
        st.error("Доступ запрещен")
        return False
    return True

if not check_password():
    st.stop()

# --- 2. ИНИЦИАЛИЗАЦИЯ И НАСТРОЙКИ ---
# Берем API ключ из секретов Streamlit
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Ошибка: Настройте GOOGLE_API_KEY в Secrets!")
    st.stop()

st.title("🤖 Корпоративный AI-поиск по документации")

# --- 3. ОБРАБОТКА ДОКУМЕНТОВ (БАЗА ЗНАНИЙ) ---
@st.cache_resource
def load_knowledge_base():
    if not os.path.exists("./docs"):
        os.makedirs("./docs")
        return None
    
    # Загружаем все PDF из папки /docs
    loader = DirectoryLoader('./docs', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

vector_store = load_knowledge_base()

# --- 4. ЛОГИКА ЧАТА ---
if vector_store:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    # Инициализация истории
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Боковая панель
    with st.sidebar:
        st.header("Управление")
        if st.button("🗑️ Очистить историю чата"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        st.info(f"Загружено документов в папке: {len(os.listdir('./docs'))}")

    # Отображение чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Задайте вопрос по документам..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Ищу в документах..."):
                result = qa_chain.invoke({
                    "question": prompt, 
                    "chat_history": st.session_state.chat_history
                })
                
                answer = result["answer"]
                st.markdown(answer)
                
                # Источники
                if result.get('source_documents'):
                    with st.expander("📚 Посмотреть источники"):
                        for doc in result['source_documents'][:3]:
                            fname = os.path.basename(doc.metadata.get('source', 'Unknown'))
                            page = doc.metadata.get('page', '?')
                            st.write(f"**{fname}** (стр. {page})")
                            st.caption(doc.page_content[:250] + "...")

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_history.append((prompt, answer))
            
            if len(st.session_state.chat_history) > 5:
                st.session_state.chat_history.pop(0)
else:
    st.warning("Папка /docs пуста. Пожалуйста, добавьте PDF-файлы в репозиторий и перезапустите.")
