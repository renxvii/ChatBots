#from pyscript import Element
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Pinecone
#import pinecone
from tqdm.autonotebook import tqdm
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from huggingface_hub import notebook_login
import textwrap
import sys
import os
import torch
import nltk 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
os.environ['OPENAI_API_KEY']='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'

URLs=[
"https://army.mit.edu"
]


loaders=UnstructuredURLLoader(urls=URLs) 
data=loaders.load()

text_splitter=CharacterTextSplitter(separator='\n',
                            chunk_size=1000,
                            chunk_overlap=200)

text_chunks=text_splitter.split_documents(data)
embeddings=HuggingFaceEmbeddings()
query_result = embeddings.embed_query("Hello world")
vectorstore=FAISS.from_documents(text_chunks, embeddings)

#PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY', 'f5444e56-58db-42db-afd6-d4bd9b2cb40c')
#PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV', 'asia-southeast1-gcp-free')

#pinecone.init(
#    api_key=PINECONE_API_KEY,
##    environment=PINECONE_API_ENV
#)


#index_name='langchainpinecone'

#vectorstore=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

#vectorstore=Pinecone.from_documents(text_chunks, embeddings, index_name=index_name)

llm=ChatOpenAI()
notebook_login()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                    use_auth_token=True,)


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                        device_map='auto',
                                        torch_dtype=torch.float16,
                                        use_auth_token=True,
                                        load_in_8bit=True,
                                        #load_in_4bit=True
                                        )
pipe = pipeline("text-generation",
        model=model,
        tokenizer= tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens = 512,
        do_sample=True,
        top_k=30,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
        )

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
     

result=chain({"question": "How good is Vicuna?"}, return_only_outputs=True)
     

llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})
llm.predict("Please provide a concise summary of the Book Alchemist")

result=chain({"question": "How good is Vicuna?"}, return_only_outputs=True)
wrapped_text = textwrap.fill(result['answer'], width=500)
query=input()
result=chain({'question':query})
