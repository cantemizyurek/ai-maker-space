import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from tqdm.asyncio import tqdm_asyncio
import asyncio
from tqdm.asyncio import tqdm

# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- # 
"""
This function will load our environment file (.env) if it is present.

NOTE: Make sure that .env is in your .gitignore file - it is by default, but please ensure it remains there.
"""
load_dotenv()

"""
We will load our environment variables here.
"""
HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]

<<<<<<< HEAD
# Set the HuggingFace API token for embeddings
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

=======
>>>>>>> 773786fa013f1300968e0183946b5726f298e4ee
# ---- GLOBAL DECLARATIONS ---- #

# -- RETRIEVAL -- #
"""
1. Load Documents from Text File
2. Split Documents into Chunks
3. Load HuggingFace Embeddings (remember to use the URL we set above)
4. Index Files if they do not exist, otherwise load the vectorstore
"""
### 1. CREATE TEXT LOADER AND LOAD DOCUMENTS
### NOTE: PAY ATTENTION TO THE PATH THEY ARE IN. 
<<<<<<< HEAD
text_loader = TextLoader("data/paul_graham_essays.txt")
documents = text_loader.load()

### 2. CREATE TEXT SPLITTER AND SPLIT DOCUMENTS
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(documents)

### 3. LOAD HUGGINGFACE EMBEDDINGS
hf_embeddings = HuggingFaceEndpointEmbeddings(
    repo_id=HF_EMBED_ENDPOINT,
    task="feature-extraction"
)
=======
text_loader = 
documents = 

### 2. CREATE TEXT SPLITTER AND SPLIT DOCUMENTS
text_splitter = 
split_documents = 

### 3. LOAD HUGGINGFACE EMBEDDINGS
hf_embeddings = 
>>>>>>> 773786fa013f1300968e0183946b5726f298e4ee

async def add_documents_async(vectorstore, documents):
    await vectorstore.aadd_documents(documents)

async def process_batch(vectorstore, batch, is_first_batch, pbar):
    if is_first_batch:
        result = await FAISS.afrom_documents(batch, hf_embeddings)
    else:
        await add_documents_async(vectorstore, batch)
        result = vectorstore
    pbar.update(len(batch))
    return result

async def main():
    print("Indexing Files")
    
    vectorstore = None
    batch_size = 32
    
    batches = [split_documents[i:i+batch_size] for i in range(0, len(split_documents), batch_size)]
    
    async def process_all_batches():
        nonlocal vectorstore
        tasks = []
        pbars = []
        
        for i, batch in enumerate(batches):
            pbar = tqdm(total=len(batch), desc=f"Batch {i+1}/{len(batches)}", position=i)
            pbars.append(pbar)
            
            if i == 0:
                vectorstore = await process_batch(None, batch, True, pbar)
            else:
                tasks.append(process_batch(vectorstore, batch, False, pbar))
        
        if tasks:
            await asyncio.gather(*tasks)
        
        for pbar in pbars:
            pbar.close()
    
    await process_all_batches()
    
    hf_retriever = vectorstore.as_retriever()
    print("\nIndexing complete. Vectorstore is ready for use.")
    return hf_retriever

async def run():
    retriever = await main()
    return retriever

hf_retriever = asyncio.run(run())

# -- AUGMENTED -- #
"""
1. Define a String Template
2. Create a Prompt Template from the String Template
"""
### 1. DEFINE STRING TEMPLATE
<<<<<<< HEAD
RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
=======
RAG_PROMPT_TEMPLATE = 

### 2. CREATE PROMPT TEMPLATE
rag_prompt =
>>>>>>> 773786fa013f1300968e0183946b5726f298e4ee

# -- GENERATION -- #
"""
1. Create a HuggingFaceEndpoint for the LLM
"""
### 1. CREATE HUGGINGFACE ENDPOINT FOR LLM
<<<<<<< HEAD
hf_llm = HuggingFaceEndpoint(
    repo_id=HF_LLM_ENDPOINT,
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN
)
=======
hf_llm = 
>>>>>>> 773786fa013f1300968e0183946b5726f298e4ee

@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message. 

    In this case, we're overriding the 'Assistant' author to be 'Paul Graham Essay Bot'.
    """
    rename_dict = {
        "Assistant" : "Paul Graham Essay Bot"
    }
    return rename_dict.get(original_author, original_author)

@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session. 

    We will build our LCEL RAG chain here, and store it in the user session. 

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    ### BUILD LCEL RAG CHAIN THAT ONLY RETURNS TEXT
<<<<<<< HEAD
    lcel_rag_chain = (
        {"context": itemgetter("query") | hf_retriever, "query": itemgetter("query")}
        | rag_prompt | hf_llm
    )
=======
    lcel_rag_chain = 
>>>>>>> 773786fa013f1300968e0183946b5726f298e4ee

    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    async for chunk in lcel_rag_chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()