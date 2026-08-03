from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import SemanticChunker

# Instantiate an OpenAI embeddings model
embedding_model = OpenAIEmbeddings(api_key="<OPENAI_API_TOKEN>", model='text-embedding-3-small')

# Create the semantic text splitter with desired parameters
semantic_splitter = SemanticChunker(
    embeddings=embedding_model, breakpoint_threshold_type="gradient", breakpoint_threshold_amount=0.8
)

# Example document to split
document = "Your document text goes here. Add more sentences to see how the semantic splitter works."

# Split the document
chunks = semantic_splitter.split_text(document)
print(chunks[0])