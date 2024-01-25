# Load
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader

"""
Negative Example
Response:
Yes, the authors constructed one or more datasets comprised of atomic configurations using DFT or quantum chemical calculations. This is mentioned in the context as follows:

"To this end, we take the final atomic structure obtained with Monte Carlo as the initial configuration for the molecular dynamics simulation."

"In the crystalline phase [Fig. 3(a)] the mean structure factor peaks along a given direction (either x or y depending on the realization) and shows null intensity elsewhere, indicating that the system is long-lived within this orientation. No distinction can be made between the initial and final configurations of the simulation."

Therefore, the authors used DFT or quantum chemical calculations to obtain atomic configurations as the starting point for their molecular dynamics simulation.
"""
# PDF_PATH = "2401.13026.pdf" 
"""
Positive Example (mlearn datasets)
Response:
Yes, the authors constructed a dataset of atomic configurations using DFT or quantum chemical calculations for six elements: Li, Mo, Cu, Ni, Si, and Ge.
"""
PDF_PATH = "1906.08888.pdf" 
loader = PyPDFLoader(PDF_PATH)
data = loader.load()

# Split
# Potentially a hyperparameter to tune
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-private",
    embedding=GPT4AllEmbeddings(),
)
retriever = vectorstore.as_retriever()

# Prompt
template = """Give a concise answer to the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
# Select the LLM that you downloaded
# Model is run in another window using the command "ollama run  llama2:7b-chat"
ollama_llm = "llama2:7b-chat"
model = ChatOllama(model=ollama_llm)
# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
r = chain.invoke("Did the authors construct one or more datasets comprised of atomic configurations using DFT or quantum chemical calculations?"
		)
"""
Summary generation. 
Could also get response back in dictionary form. Something like: 
{'elements: ,
'description:, 
'authors',
'name',
etc}
"""	
# r=chain.invoke("Give a detailed description of any datasets that were used in the scientific work, including the computational method for generated the data and the chemical elements considered.")

# Response
print (r)

