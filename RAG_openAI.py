# !pip install langchain langchain-openai langchain-community chromadb tiktoken -q

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import wget
import os
from dotenv import load_dotenv
load_dotenv()  # ← automatically loads the key

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# === 1. Download the file ===
filename = 'XVnuuEg94sAE4S_xAsGxBA.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/XVnuuEg94sAE4S_xAsGxBA.txt'
if not os.path.exists(filename):
    wget.download(url, out=filename)
    print('file downloaded')
else:
    print('file already exists')

# === 2. Load and split document ===
loader = TextLoader(filename, encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # 200 overlap is better
texts = text_splitter.split_documents(documents)
print(f"Created {len(texts)} chunks")

# === 3. Embeddings + Vector DB (using OpenAI embeddings - still very cheap) ===
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"   # cheapest good embedding model (~$0.02 / 1M tokens)
)

docsearch = Chroma.from_documents(texts, embeddings)
print('Document ingested into Chroma')

# === 4. Use the cheapest good OpenAI model: gpt-4o-mini ===
llm = ChatOpenAI(
    model="gpt-4o-mini",         # ← CHEAPEST high-quality model right now
    temperature=0.3,
    max_tokens=1024
)

# === 5. Strict prompt so the model never hallucinates ===
prompt_template = """
You are a helpful assistant. Answer the question ONLY using the information from the context below.
If the answer is not explicitly present in the context, respond exactly with: "I don't know".
Do NOT use prior knowledge. Do NOT guess. Do NOT make anything up.

Context:
{context}

Question: {question}
Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# === 6. Create the QA chain ===
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 4}),  # retrieve top 4 chunks
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# === 7. Ask questions ===
query = "Summarize the attached document in bullet points."
result = qa.invoke(query)

print("\nAnswer:")
print(result["result"])

print("\nSource chunks used:")
for i, doc in enumerate(result["source_documents"]):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content[:500] + "...")
