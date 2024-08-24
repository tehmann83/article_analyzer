import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
file_path = "./blogarticle1.txt"

if __name__ == "__main__":
    print("Ingesting...")

    if os.path.exists(file_path):
        print("File found!")

        loader = TextLoader(file_path, encoding='utf-8')
        document = loader.load()

        print("Splitting text...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(document)
        print(f"created {len(texts)} chunks")

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

        print("ingesting...")
        PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
        print("finished")
    else:
        print("File not found:", file_path)


