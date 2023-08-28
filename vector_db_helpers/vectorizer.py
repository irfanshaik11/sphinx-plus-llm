## pip install -U openai pinecone-client jsonlines
## set up pinecone database with 1536 dimensions
import openai
import pinecone


# Initialize OpenAI API
def init_openai(api_key):
    openai.api_key = api_key
    return "text-embedding-ada-002"


# Initialize Pinecone index
def init_pinecone(pinecone_api_key, pinecone_env, index_name, dimension):
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    # Remove disallowed characters
    index_formatted = ''.join([let.lower() for let in index_name if let.isalnum()])
    if index_formatted not in pinecone.list_indexes():
        pinecone.create_index(index_formatted, dimension=dimension)
    return pinecone.Index(index_formatted)


# Create embeddings and populate the index
def create_and_index_embeddings(data, model, index):
    batch_size = 32
    ids_inserted = []
    for i in range(0, len(data), batch_size):
        text_batch = [item["text"] for item in data[i:i+batch_size]]
        ids_batch = [str(n) for n in range(i, i+min(batch_size, len(data)-i))]
        res = openai.Embedding.create(input=text_batch, engine=model)
        embeds = [record["embedding"] for record in res["data"]]
        to_upsert = zip(ids_batch, embeds)
        index.upsert(vectors=list(to_upsert))
        ids_inserted = ids_inserted + ids_batch
    return ids_inserted


def create_pinecone_index(openai_api_key, pinecone_api_key, pinecone_env, index_name):
    # Load the data from train.jsonl
    # train_data = load_data("train.jsonl")

    # Initialize OpenAI Embedding API
    model = init_openai(openai_api_key)

    # Get embeddings dimension
    sample_embedding = openai.Embedding.create(input="sample text", engine=model)["data"][0]["embedding"]
    embedding_dimension = len(sample_embedding)

    # Initialize Pinecone index
    chatgpt_index = init_pinecone(pinecone_api_key, pinecone_env, index_name, embedding_dimension)

    return chatgpt_index


def add_embeddings_to_pinecone_index(openai_api_key, train_data, chatgpt_index):
    model = init_openai(openai_api_key)
    # Create embeddings and populate the index with the train data
    ids_inserted = create_and_index_embeddings(train_data, model, chatgpt_index)
    return ids_inserted


def remove_embeddings_from_pinecone_index(openai_api_key, pinecone_api_key, pinecone_environment, user_id):
    model = init_openai(openai_api_key)
    sample_embedding = openai.Embedding.create(input="sample text", engine=model)["data"][0]["embedding"]
    embedding_dimension = len(sample_embedding)

    index_name = init_pinecone(pinecone_api_key, pinecone_environment, user_id, embedding_dimension)
    index = pinecone.Index(index_name)
    delete_response = index.delete(ids=['vec1', 'vec2'])
