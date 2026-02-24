import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 1. БЕЗОПАСНОСТЬ ---
st.set_page_config(page_title="Corporate Doc Assistant", layout="wide")

def check_password():
    def password_entered():
        if st.session_state["password"] == "SuperSecret123":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input("Введите пароль доступа", type="password", on_change=password_entered, key="password")
        return False
    return st.session_state["password_correct"]

if not check_password():
    st.stop()

# --- 2. ИНИЦИАЛИЗАЦИЯ ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Настройте GOOGLE_API_KEY в Secrets!")
    st.stop()

# --- 3. БАЗА ЗНАНИЙ ---
@st.cache_resource
def load_knowledge_base():
    if not os.path.exists("./docs"):
        os.makedirs("./docs")
    loader = DirectoryLoader('./docs', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents: return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma.from_documents(chunks, embeddings)

vector_store = load_knowledge_base()

# --- 4. ЛОГИКА ЧАТА (План Б) ---
if vector_store:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    
    # Промпт для переформулирования вопроса с учетом истории
    context_q_system_prompt = "С учетом истории чата и последнего вопроса пользователя, сформулируй вопрос, который можно понять без истории."
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    retriever = vector_store.as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

    # Промпт для финального ответа
    qa_system_prompt = "Ты технический ассистент. Отвечай на вопросы, используя только предоставленный контекст: {context}"
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Интерфейс чата
    if "messages" not in st.session_state: st.session_state.messages = []
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Спросите что-нибудь..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            result = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})
            answer = result["answer"]
            st.markdown(answer)
            
            if result.get('context'):
                with st.expander("Источники"):
                    for doc in result['context'][:2]:
                        st.write(f"Стр. {doc.metadata.get('page')} из {os.path.basename(doc.metadata.get('source'))}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append(("human", prompt))
        st.session_state.chat_history.append(("ai", answer))
else:
    st.warning("Добавьте PDF в папку /docs.")
