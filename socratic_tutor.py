import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, question, k):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate

    general_system_template = r""" 
    You are a Socratic Tutor specialized in assisting students with Capture The Flag (CTF) challenges. Use the following principles in responding to students:\nYour role is to provide insightful and thought-provoking guidance to help students navigate and solve CTF challenges effectively. Encourage critical thinking, ask leading questions, and provide hints or explanations as necessary. Assist students in understanding the underlying concepts and techniques involved in CTF challenges, such as cryptography, reverse engineering, web exploitation, and binary exploitation. Tailor your responses to the student's level of expertise, and help them develop their problem-solving skills in the realm of cybersecurity CTFs. Enable a back and forth conversation waiting for the user's response.
    {context}
    """
    general_user_template = question
    messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )
    
    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    
    crc = ConversationalRetrievalChain.from_llm(llm, retriever, combine_docs_chain_kwargs={'prompt': qa_prompt})
    result = crc({'question': question, 'chat_history': st.session_state['history']})
    st.session_state['history'].append((question,result['answer']))
    return result['answer']


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004

# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image('img.jpg')
    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    # user's question text input widget

    q = st.text_input('Ask a question about the content of your file:')
    if q: # if the user entered a question and hit enter
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            if 'history' not in st.session_state:
                st.session_state['history'] = []
            #chat_history = st.session_state.history
            #st.write(f'k: {k}')
            # text area widget for the LLM answer
            answer = ask_and_get_answer(vector_store, q, k)

            st.text_area('LLM Answer: ', value=answer, height=400)

            print(st.session_state)
            st.divider()
            st.write("Chat History:")
            for item in st.session_state['history']:
                st.write(f"Q: {item[0]}")
                st.write(f"A: {item[1]}")
                st.write("-" * 100)
            