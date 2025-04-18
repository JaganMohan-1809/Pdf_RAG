# Single PDF RAG App with Ollama
# This is a simple Retrieval-Augmented Generation (RAG) app using Streamlit, Langchain, and Ollama.

# import os
# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# import faiss
# from langchain_ollama import ChatOllama
# from langchain.prompts import PromptTemplate

# # ------------------------------ PROMPT TEMPLATE ------------------------------
# def get_prompt_template():
#     prompt_template = """
#     Human: Use the following pieces of context to provide a 
#     concise answer to the question at the end but summarize with 
#     at least 150 words and detailed explanations. If you don't know the answer, 
#     just say that you don't know. Don't try to make up an answer.
#     Don't give extra information which is not asked in the question.

#     <context>
#     {context}
#     </context>

#     Question: {question}

#     Assistant:
#     """
#     return PromptTemplate(template=prompt_template, input_variables=["context", "question"])


# # ------------------------------ PDF LOADER & EMBEDDINGS ------------------------------
# def load_and_index_pdf(pdf_path, embedding_model):
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     documents = text_splitter.split_documents(docs)

#     texts = [doc.page_content for doc in documents]
#     vectors = embedding_model.encode(texts, convert_to_numpy=True)

#     embedding_dim = embedding_model.get_sentence_embedding_dimension()
#     faiss_index = faiss.IndexFlatL2(embedding_dim)
#     faiss_index.add(vectors)

#     return texts, faiss_index


# # ------------------------------ QUERY HANDLER ------------------------------
# def get_relevant_chunks(query, embedding_model, texts, index, top_k=3):
#     query_vector = embedding_model.encode([query], convert_to_numpy=True)
#     distances, indices = index.search(query_vector, k=top_k)
#     return [texts[i] for i in indices[0]] if len(indices[0]) > 0 else []


# def get_llm_response(context, question, prompt_template):
#     final_prompt = prompt_template.format_prompt(context=context, question=question).to_string()
#     llm = ChatOllama(model="llama3.2:1b", base_url="http://127.0.0.1:11434")
#     response = llm.invoke(final_prompt)
#     return response if isinstance(response, str) else getattr(response, 'content', str(response))


# # ------------------------------ STREAMLIT UI ------------------------------
# def main():
#     st.set_page_config(page_title="Simple RAG with Ollama", layout="centered")
#     st.title("🔍 SIMPLE RAG APP TO QUERY PDF DOCUMENTS")
#     st.write("Query the **'attention.pdf'** document to retrieve relevant info and generate a detailed response.")

#     script_dir = os.path.dirname(os.path.realpath(__file__)) if "__file__" in globals() else os.getcwd()
#     pdf_path = os.path.join(script_dir, "attention.pdf")

#     embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     prompt_template = get_prompt_template()

#     if "pdf_loaded" not in st.session_state:
#         st.session_state.pdf_loaded = False
#         st.session_state.texts = []
#         st.session_state.index = None

#     if not st.session_state.pdf_loaded:
#         if os.path.exists(pdf_path):
#             with st.spinner("Processing 'attention.pdf'..."):
#                 try:
#                     texts, faiss_index = load_and_index_pdf(pdf_path, embedding_model)
#                     st.session_state.texts = texts
#                     st.session_state.index = faiss_index
#                     st.session_state.pdf_loaded = True
#                     st.success("✅ PDF processed and indexed successfully!")
#                 except Exception as e:
#                     st.error(f"❌ Error loading PDF: {e}")
#         else:
#             st.error("❌ 'attention.pdf' file not found!")

#     query = st.text_input("Enter your search query:")

#     if query:
#         if not st.session_state.pdf_loaded:
#             st.error("PDF not loaded. Please try again.")
#         else:
#             relevant_texts = get_relevant_chunks(query, embedding_model, st.session_state.texts, st.session_state.index)

#             if relevant_texts:
#                 context = "\n\n".join(relevant_texts)
#                 try:
#                     response = get_llm_response(context, query, prompt_template)

#                     st.subheader("📄 Relevant Context")
#                     st.write(context)
#                     st.markdown("---")

#                     st.subheader("🤖 LLM Answer (Llama 3.2 1B)")
#                     st.write(response)
#                     st.markdown("---")

#                 except Exception as e:
#                     st.error(f"Error getting response from LLM: {e}")
#             else:
#                 st.warning("No relevant context found for your query.")


# if __name__ == "__main__":
#     main()



# Multi PDF RAG App with Ollama
# This is a simple Retrieval-Augmented Generation (RAG) app using Streamlit, Langchain, and Ollama.

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

# ------------------------------ PROMPT TEMPLATE ------------------------------
def get_prompt_template():
    # prompt_template = """
    # Human: Use the following pieces of context to provide a 
    # concise answer to the question at the end but summarize with 
    # at least 150 words and detailed explanations. If you don't know the answer, 
    # just say that you don't know. Don't try to make up an answer.
    # Don't give extra information which is not asked in the question.

    # <context>
    # {context}
    # </context>

    # Question: {question}

    # Assistant:
    # """
    prompt_template = """
    Human: Use the following pieces of context to provide a 
    concise answer to the question at the end but summarize with 
    at least 150 words and detailed explanations. If you don't know the answer, 
    just say that you don't know. Don't try to make up an answer.
    Don't give extra information which is not asked in the question.

    <context>
    {context}
    </context>

    Question: {question}

    Assistant:
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])


# ------------------------------ PDF LOADER & INDEXER ------------------------------
def process_pdfs(uploaded_files, embedding_model):
    all_texts = []

    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(uploaded_file.name)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)

        texts = [doc.page_content for doc in documents]
        all_texts.extend(texts)

    vectors = embedding_model.encode(all_texts, convert_to_numpy=True)

    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(vectors)

    return all_texts, faiss_index



# ------------------------------ QUERY HANDLER ------------------------------
def get_relevant_chunks(query, embedding_model, texts, index, top_k=3):
    query_vector = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, k=top_k)
    return [texts[i] for i in indices[0]] if len(indices[0]) > 0 else []


def get_llm_response(context, question, prompt_template):
    final_prompt = prompt_template.format_prompt(context=context, question=question).to_string()
    llm = ChatOllama(model="llama3.2:1b", base_url="http://127.0.0.1:11434")
    response = llm.invoke(final_prompt)
    return response if isinstance(response, str) else getattr(response, 'content', str(response))


# ------------------------------ STREAMLIT UI ------------------------------
def main():
    st.set_page_config(page_title="Multi-PDF RAG App", layout="centered")
    st.title("📚 RAG App: Query Multiple PDFs")
    st.write("Upload one or more PDF files to query them with a language model.")

    prompt_template = get_prompt_template()
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("Process PDFs"):
        with st.spinner("Processing uploaded PDFs..."):
            try:
                texts, index = process_pdfs(uploaded_files, embedding_model)
                st.write(f"Processed {len(texts)} text chunks from all PDFs.")
                st.write(f"FAISS index has {index.ntotal} vectors.")

                st.session_state.texts = texts
                st.session_state.index = index
                st.session_state.ready = True
                st.success("✅ PDFs processed and indexed!")
            except Exception as e:
                st.error(f"❌ Error processing PDFs: {e}")

    if st.session_state.get("ready", False):
        query = st.text_input("Enter your search query:")

        if query:
            relevant_texts = get_relevant_chunks(query, embedding_model, st.session_state.texts, st.session_state.index)

            if relevant_texts:
                st.subheader("📑 Retrieved Chunks for this Query")
                for i, text in enumerate(relevant_texts):
                    st.write(f"**Chunk {i+1}:**\n{text}\n\n")

                context = "\n\n".join(relevant_texts)
                try:
                    response = get_llm_response(context, query, prompt_template)

                    st.subheader("📄 Relevant Context")
                    st.write(context)
                    st.markdown("---")

                    st.subheader("🤖 LLM Answer (Llama 3.2 1B)")
                    st.write(response)
                    st.markdown("---")
                except Exception as e:
                    st.error(f"Error getting response from LLM: {e}")
            else:
                st.warning("No relevant context found for your query.")


if __name__ == "__main__":
    main()









