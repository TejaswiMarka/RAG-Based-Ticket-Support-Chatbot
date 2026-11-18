# =========================
# LOAD DOCUMENT
# =========================
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path=r"C:\Users\hp\Downloads\support_tickets.csv")
documents = loader.load()


# =========================
# TEXT SPLITTING
# =========================
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)


# =========================
# EMBEDDINGS
# =========================
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# VECTOR DATABASE (FAISS)
# =========================
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)


# =========================
# RETRIEVER
# =========================
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


# =========================
# LLM (LOCAL - NO TOKEN NEEDED)
# =========================
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    max_length=100
)


llm = HuggingFacePipeline(pipeline=pipe)

#  Add your token here


#from langchain_community.llms import HuggingFaceHub
'''
import os
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
os.environ["HF_TOKEN"] = "hf_UKGGJYmbabRtXkiTwkHYMeCUyODwRCyDLApip install langchain-community"
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_length=100
)

llm = HuggingFacePipeline(pipeline=pipe)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    task="text2text-generation",   # ✅ ADD THIS
    model_kwargs={
        "temperature": 0.5,
        "max_length": 100
    },
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)



# convert vector store to retriever

#ask query to retriver
#UI → API → FAISS → HuggingFace → Answer
 #The retriever searches the FAISS vector index and returns the most similar chunks.
       
#Displays the retrieved document chunks

# here it is retrives relvant chunks from FAISS.we need to add generation part to  generate
# final answer from retrived chunks

# generate answer using retrived context
#The LLM takes the retrieved context and the original query to generate a final answer.   

#create prompt for LLM '''

