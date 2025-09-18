import streamlit as st
from src.helper import (
    get_pdf_text,
    get_text_chunks,
    get_vector_stores,
    load_tinyllama,
    get_conversational_chain,
)

def main():
    st.set_page_config(page_title="Information Retrieval")
    st.header("📄 Information Retrieval System")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "llm" not in st.session_state:
        # Load TinyLlama once
        snapshot_folder = r"C:\Users\user\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6"
        st.session_state.llm = load_tinyllama(snapshot_folder)

    with st.sidebar:
        st.title("📂 Menu")
        pdf_docs = st.file_uploader(
            "Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("🔄 Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_stores(chunks)

                    # Create conversational chain
                    st.session_state.conversation = get_conversational_chain(
                        vector_store, st.session_state.llm
                    )
                st.success("✅ Processing complete!")
            else:
                st.warning("⚠️ Please upload at least one PDF.")

    # Chat interface
    user_query = st.chat_input("Ask something about your PDFs...")
    if user_query and st.session_state.conversation:
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.conversation.invoke(user_query)
            st.chat_message("user").write(user_query)
            st.chat_message("assistant").write(response)


if __name__ == "__main__":
    main()
