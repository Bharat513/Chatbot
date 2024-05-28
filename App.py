import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrieval
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate



MODEL = "llama2"

model = Ollama(model=MODEL) 


def get_pdf_text(pdf_docs):
    text = " "
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text       



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks



def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model=MODEL)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore


def get_conversational_chain():
    template = """
    Answer the question based on the context below. If you can't
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = OllamaEmbeddings(model=MODEL)
    
    new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization = True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question":user_question}
        , return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config(page_title="AB InBev's WISE BUD", page_icon=":books")

    st.header("AB InBev's WISE BUD")
    user_question = st.text_input("Heyy , I am WISE BUD, ask a question")


    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your pdfs here",accept_multiple_files=True )
        if st.button("Process"):
            with st.spinner("Processing"):
                #get pdf
                raw_text = get_pdf_text(pdf_docs)
                

                #get text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)


                # vector store
                vectorstore = get_vectorstore(text_chunks)

                #success message
                st.success("Done")


if __name__ == '__main__':
    main()


