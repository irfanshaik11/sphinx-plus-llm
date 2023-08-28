from google.cloud import vision
from google.oauth2 import service_account
from pymongo import MongoClient
import pinecone
from vector_db_helpers import vectorizer, chunker, pdf_muncher

client = MongoClient("mongodb+srv://doadmin:zoPN05692k3j71YO@db-mongodb-sfo3-82555-d9e16287.mongo.ondigitalocean.com"
                     "/admin?authSource=admin&replicaSet=db-mongodb-sfo3-82555&tls=true")
db = client.admin

# Set up OpenAI and Pinecone API keys
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "f0d1df5f-8fe6-406f-8def-9bd1433c82c8"
PINECONE_ENVIRONMENT="gcp-starter"


# [START vision_text_detection]
def detect_text(content):
    """Detects text in the file."""
    credentials = service_account.Credentials.from_service_account_file('google_service_acc_key.json')
    client = vision.ImageAnnotatorClient(credentials=credentials)
    #
    # # [START vision_python_migration_text_detection]
    # with open(path, "rb") as image_file:
    #     content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    # print("Texts:")
    # print(texts)
    return texts

import re
def formaturl(url):
    if not re.match('(?:http|ftp|https)://', url):
        return 'http://{}'.format(url)
    return url


def extract_data_from_url(url):
    from bs4 import BeautifulSoup
    import requests

    def get_links(url):
        response = requests.get(url)
        data = response.text
        soup = BeautifulSoup(data, 'lxml')

        links = []
        for link in soup.find_all('a'):
            link_url = link.get('href')

            if link_url is not None and link_url.startswith('http'):
                links.append(link_url)
            if len(links) > 100:
                Exception("Attempted to scrape more than 100 sublinks, pausing for memory reasons: " + url)
        return links

    sublinks = get_links(formaturl(url))

    scraped_html_pages = []
    for l in sublinks:
        htmltext = requests.get("http://google.com").text
        scraped_html_pages.append([l, htmltext])

    return scraped_html_pages


def merge_with_saved_data(user_id, train_data):
    data = db.vector_db_metadata.find_one({'user_id': user_id})
    if data == None:
        data = {'train_data': []}
    dedupe_set = set()
    out_data = []
    subset_data = []
    for item in data['train_data']:
        dedupe_set.add(item['id'])
        out_data.append(item)

    for item in train_data:
        if item['id'] in dedupe_set:
            continue
        out_data.append(item)
        subset_data.append(item)

    data['train_data'] = out_data
    db.vector_db_metadata.update_one({'user_id': user_id}, {"$set": data}, upsert=True)
    return subset_data


def scrape_website_sublinks_and_add_to_vector_db(given_url, user_id):
    scraped_html_pages = extract_data_from_url(given_url)
    train_data = chunker.process_html_files(scraped_html_pages)

    # Only process docs that haven't been processed before
    unique_subset = merge_with_saved_data(user_id, train_data)
    chatgpt_index = vectorizer.create_pinecone_index(OPENAI_API_KEY,
                                                     PINECONE_API_KEY,
                                                     PINECONE_ENVIRONMENT,
                                                     user_id)
    index_name = get_index_name(user_id)
    ids_inserted = vectorizer.add_embeddings_to_pinecone_index(OPENAI_API_KEY, unique_subset, chatgpt_index)

    vector_db_metadata = db.vector_db_metadata.find_one({'user_id': user_id})
    vector_db_metadata['index_name'] = index_name
    if 'vector_db_row_ids' not in vector_db_metadata:
        vector_db_metadata['vector_db_row_ids'] = []
    vector_db_metadata['vector_db_row_ids'] += ids_inserted
    vector_db_metadata['vector_db_row_ids'] = list(set(vector_db_metadata['vector_db_row_ids']))
    db.vector_db_metadata.update_one({'user_id': user_id}, {"$set": vector_db_metadata}, upsert=True)

    return vector_db_metadata


def get_index_name(user_id):
    return ''.join([let.lower() for let in user_id if let.isalnum()])


def remove_entry_from_vector_db(string_comp, user_id):
    """ string_comp can be either a filename or a url. This function is super flaky and should be rewritten """
    index_name = get_index_name(user_id)
    index = pinecone.Index(index_name)
    print(f"getting vector metadata: vector_db_metadata = db.vector_db_metadata.find_one {user_id}")
    vector_db_metadata = db.vector_db_metadata.find_one({'user_id': user_id})
    if vector_db_metadata == None:
        return
    ids_to_delete = []
    for item in vector_db_metadata['vector_db_row_ids']:
        if string_comp in item:
            ids_to_delete.append(item)
    print("got ids to delete:")
    print(ids_to_delete)
    delete_response = index.delete(ids=ids_to_delete, namespace=PINECONE_ENVIRONMENT)
    print(f"should delete index.delete({ids_to_delete}, {PINECONE_ENVIRONMENT}")
    return delete_response


def add_pdf_to_vector_db(filename, body_as_bytes, user_id):
    """ Create an pinecone index if doesn't already exist and upload file to it """
    print(f"calling pdf_muncher.process_pdf_files filename: {filename}, body_as_bytes: irfan")
    train_data = pdf_muncher.process_pdf_files(filename, body_as_bytes)
    print(f"calling merge_with_saved_data with vars user_id: {user_id}, train_data: {train_data}")
    unique_subset = merge_with_saved_data(user_id, train_data)
    # print(f"retrieved unique subset: {unique_subset}")
    print(f"calling db.vector_db_metadata.find_one with vars user_id: {user_id}")
    vector_db_metadata = db.vector_db_metadata.find_one({'user_id': user_id})

    index_name = get_index_name(user_id)
    chat_gpt_index = vectorizer.create_pinecone_index(OPENAI_API_KEY,
                                         PINECONE_API_KEY,
                                         PINECONE_ENVIRONMENT,
                                         index_name)
    vector_db_metadata['index_name'] = index_name
    print("irfan-156: adding train data")
    print(train_data)
    ids_inserted = vectorizer.add_embeddings_to_pinecone_index(OPENAI_API_KEY, train_data, chat_gpt_index)
    if 'vector_db_row_ids' not in vector_db_metadata:
        vector_db_metadata['vector_db_row_ids'] = []
    vector_db_metadata['vector_db_row_ids'] += ids_inserted
    db.vector_db_metadata.update_one({'user_id': user_id}, {"$set": vector_db_metadata}, upsert=True)
    return vector_db_metadata

# if __name__ == "__main__":
# scrape_website_sublinks_and_add_to_vector_db('equinoxapi.com', 'test')
