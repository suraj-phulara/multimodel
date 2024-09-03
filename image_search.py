import os
import json
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from opensearchpy import OpenSearch
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Dict

# Load environment variables
load_dotenv()

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Connect to OpenSearch
client = OpenSearch(
    hosts=[{'host': os.getenv('OPENSEARCH_HOST'), 'port': os.getenv('OPENSEARCH_PORT')}],
    http_auth=(os.getenv('OPENSEARCH_USERNAME'), os.getenv('OPENSEARCH_PASSWORD'))
)

# Define the index
index_name = "image_index2"

class Match(BaseModel):
    name: str

class Term(BaseModel):
    mainCategory: Optional[str]
    subCategory: Optional[str]

class RangeValue(BaseModel):
    lte: Optional[float]
    gt: Optional[float]

class Range(BaseModel):
    actualPrice: Optional[RangeValue]
    discountPrice: Optional[RangeValue]
    noOfRatings: Optional[RangeValue]
    ratings: Optional[RangeValue]

class Script(BaseModel):
    source: str
    params: Dict

class FilterClause(BaseModel):
    term: Optional[Term]
    range: Optional[Range]
    # script: Optional[Script]

class BoolQuery(BaseModel):
    must: List[Match]
    filter: List[FilterClause]

class Query(BaseModel):
    bool: BoolQuery

class QueryStructure(BaseModel):
    query: Query

def chat_gpt2(query: str):
    
    openapi_key = os.getenv("OPENAI_API_KEY")

    model = ChatOpenAI(api_key=openapi_key, temperature=0, model="gpt-4o-mini")

    parser = JsonOutputParser(pydantic_object=QueryStructure)

    prompt_template = PromptTemplate(
        template="take the given user query in human language and generate an Open search quey from it according to the given instructions \n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt_template | model | parser
    response = chain.invoke({"query": query})
    print("+++++++++++++++++++", json.dumps(response, indent=4), "++++++++++++++++++")
    return response

    # try:
    # except KeyError as e:
    #     print(f"KeyError: {e}")
    #     return {}



def process_image(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs.detach().numpy().flatten()

def process_text(text):
    inputs = processor(text=text, return_tensors="pt")
    outputs = model.get_text_features(**inputs)
    return outputs.detach().numpy().flatten()

def search_text_and_images(query, k=10):



    open_search_data = {
        "mappings": {
            "properties": {
                "productId": {"type": "keyword"},
                "gender": {"type": "keyword"},
                "category": {"type": "keyword"},
                "subCategory": {"type": "keyword"},
                "productType": {"type": "keyword"},
                "colour": {"type": "keyword"},
                "usage": {"type": "keyword"},
                "productTitle": {"type": "text"},
                "imagePath": {"type": "keyword"},
                "imageURL": {"type": "keyword"},
                "actualPrice": {"type": "integer"},
                "discountPrice": {"type": "integer"},
                "rating": {"type": "float"},
                "reviews": {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 512  # Ensure this matches the dimension of your CLIP embeddings
                }
            }
        },
        "categories": {
            "Apparel": [
                "Topwear",
                "Bottomwear",
                "Dress",
                "Innerwear",
                "Socks",
                "Apparel Set",
            ],
            "Footwear": [
                "Shoes",
                "Flip Flops",
                "Sandal",
            ],
        }
    }
    prompt = f"""
    Act as an OpenSearch query generator whose task is to generate OpenSearch queries from user queries (in human language) for an e-commerce website.

    Below is the mapping of the index in OpenSearch that contains the products:
    mappings : {json.dumps(open_search_data.get('mappings', {}), indent=4)}

    Below JSON data contains categories and their corresponding subcategories from which you have to choose (most important to only choose strictly from given options). The top-level keys represent main categories, and each main category maps to a list of its underlying subcategories. Do not confuse, choose main and subcategory(s) differently and carefully:
    categories_info: {json.dumps(open_search_data.get('categories', {}), indent=4)}
    make sure to only use these categories and sub categories if you cant match it then dont include the category filter in your query

    The available genders are:
        - Girls
        - Boys
        - Men
        - Women

    [Examples]
    Example 1:
    USER_QUERY = "I want to buy a red dress under 50 dollars with good reviews for women."
    Expected Output:
    {{
        "query": {{
            "bool": {{
                "must": [
                    {{"match": {{"productTitle": "red dress"}}}}
                ],
                "filter": [
                    {{"range": {{"actualPrice": {{"lte": 50}}}}}},
                    {{"range": {{"reviews": {{"gt": 0}}}}}},
                    {{"range": {{"rating": {{"gt": 0}}}}}},
                    {{"term": {{"category": "Apparel"}}}},
                    {{"term": {{"subCategory": "Dress"}}}},
                    {{"terms": {{"colour": ["Red"]}}}},
                    {{"term": {{"gender": "Women"}}}}
                ]
            }}
        }}
    }}

    Example 2:
    USER_QUERY = "blue shoes with a rating above 4."
    Expected Output:
    {{
        "query": {{
            "bool": {{
                "must": [
                    {{"match": {{"productTitle": "blue shoes"}}}}
                ],
                "filter": [
                    {{"range": {{"rating": {{"gte": 4}}}}}},
                    {{"term": {{"category": "Footwear"}}}},
                    {{"term": {{"subCategory": "Shoes"}}}},
                    {{"terms": {{"colour": ["Blue"]}}}}
                ]
            }}
        }}
    }}

    Example 3:
    USER_QUERY = "black and red shirts on sale."
    Expected Output:
    {{
        "query": {{
            "bool": {{
                "must": [
                    {{"match": {{"productTitle": "shirts"}}}}
                ],
                "filter": [
                    {{"term": {{"category": "Apparel"}}}},
                    {{"term": {{"subCategory": "Topwear"}}}},
                    {{"terms": {{"colour": ["Black", "Red"]}}}}
                ]
            }}
        }}
    }}

    Example 4:
    USER_QUERY = "green and yellow sneakers under 100 dollars."
    Expected Output:
    {{
        "query": {{
            "bool": {{
                "must": [
                    {{"match": {{"productTitle": "sneakers"}}}}
                ],
                "filter": [
                    {{"range": {{"actualPrice": {{"lte": 100}}}}}},
                    {{"term": {{"category": "Footwear"}}}},
                    {{"term": {{"subCategory": "Sneakers"}}}},
                    {{"terms": {{"colour": ["Green", "Yellow"]}}}}
                ]
            }}
        }}
    }}

    USER_QUERY = {query}

    Now just give me the OpenSearch query for the above USER_QUERY user typed in the searchbox.

    Remember to include colors in the filter block using the `terms` query if multiple colors are specified. The category and subcategory should also be included in the filter block. Ensure that the first character of each color is capitalized (e.g., Red, Blue).

    also dont give any comments in your code
    """

    # Generate the OpenSearch query using GPT
    query_structure = chat_gpt2(prompt)
    
    # Process the text to get its embedding
    text_embedding = process_text(query)

    
    knn = {
            "script_score": {
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": text_embedding.tolist(),  # Convert numpy array to list
                            "k": k
                        }
                    }
                },
                "script": {
                    "source": "_score * 1.5"
                }
            }
        }


    if 'should' not in query_structure['query']['bool']:
        query_structure['query']['bool']['should'] = []

    query_structure['query']['bool']['should'].append(knn)
    
    


    # print("***********************",json.dumps(combined_query, indent=4),"***********************88")

    response = client.search(index=index_name, body=query_structure)
    return response['hits']['hits']

def search_similar_images(query_vector, index_name='image_index2', k=10):
    query = {
        "size": 100,  # Number of results to return
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector.tolist(),  # Convert numpy array to list
                    "k": k
                }
            }
        }
    }

    response = client.search(index=index_name, body=query)
    # print(response)
    return response['hits']['hits']


def search_text(text_query, k=10):
    # Process the text to get its embedding
    text_embedding = process_text(text_query)

    return search_similar_images(text_embedding, index_name='image_index2', k=10)


# Example usage
if __name__ == "__main__":
    query_image_path = "f7aaed18-863f-46bc-83d9-2d270dbb7168.jpg"  # Replace with your query image path
    image = Image.open(query_image_path).convert("RGB")
    image_embedding = process_image(image)
    results = search_similar_images(image_embedding)
    
    print("Similar Images:")
    for result in results:
        image_url = result['_source']['imageURL']
        print(image_url)
