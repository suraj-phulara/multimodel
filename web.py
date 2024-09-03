import streamlit as st
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
from image_search import process_image, search_similar_images, search_text_and_images, search_text

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Set up Streamlit app
st.title("Image Similarity Search")

# Tabs for Image Upload and Text Search
tab1, tab2, tab3 = st.tabs(["Image Search", "Text Search", "Hybrid Search"])

with tab1:
    # Allow user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Searching for similar images...")

        # Convert the uploaded file to an image
        image = Image.open(uploaded_file).convert("RGB")

        # Process the image and get its embedding
        image_embedding = process_image(image)

        # Search for similar images
        results = search_similar_images(image_embedding, k=10)  # Adjust the number of results as needed

        st.write("Top Similar Images:")

        num_cols = 2  # Number of columns per row
        rows = (len(results) + num_cols - 1) // num_cols  # Calculate the number of rows needed

        for row in range(rows):
            cols = st.columns(num_cols)  # Create a row with num_cols columns
            for col_index in range(num_cols):
                result_index = row * num_cols + col_index
                if result_index < len(results):
                    result = results[result_index]
                    product = result['_source']
                    score = result['_score']
                    image_url = product['imageURL']
                    
                    # Display the image and product details in the corresponding column
                    with cols[col_index]:
                        st.image(image_url, use_column_width=True)
                        st.write(f"Product Title: {product['productTitle']}")
                        st.write(f"Category: {product['category']} -> {product['subCategory']}")
                        st.write(f"Actual Price: {product['actualPrice']}")
                        st.write(f"Discount Price: {product['discountPrice']}")
                        st.write(f"Rating: {product['rating']}")
                        st.write(f"Reviews: {product['reviews']}")
                        st.write(f"Score: {score:.2f}")

with tab2:
    # Allow user to input text
    text_query = st.text_input("Enter a text query:", key="text_query")

    if text_query:
        st.write("")
        st.write("Searching for images matching the text query...")

        # Search for similar images using the text query
        results = search_text(text_query, k=10)  # Adjust the number of results as needed

        st.write("Top Similar Images:")

        num_cols = 2  # Number of columns per row
        rows = (len(results) + num_cols - 1) // num_cols  # Calculate the number of rows needed

        for row in range(rows):
            cols = st.columns(num_cols)  # Create a row with num_cols columns
            for col_index in range(num_cols):
                result_index = row * num_cols + col_index
                if result_index < len(results):
                    result = results[result_index]
                    product = result['_source']
                    score = result['_score']
                    image_url = product['imageURL']
                    
                    # Display the image and product details in the corresponding column
                    with cols[col_index]:
                        st.image(image_url, use_column_width=True)
                        for key, value in product.items():
                            if key != 'embedding':
                                st.write(f"{key.replace('_', ' ').title()}: {value}")
                        st.write(f"Score: {score:.2f}")


with tab3:
    # Allow user to input text
    text_query = st.text_input("Enter a text query:", key="text_query3")

    if text_query:
        st.write("")
        st.write("Searching for images matching the text query...")

        # Search for similar images using the text query
        results = search_text_and_images(text_query, k=10)  # Adjust the number of results as needed

        st.write("Top Similar Images:")

        num_cols = 2  # Number of columns per row
        rows = (len(results) + num_cols - 1) // num_cols  # Calculate the number of rows needed

        for row in range(rows):
            cols = st.columns(num_cols)  # Create a row with num_cols columns
            for col_index in range(num_cols):
                result_index = row * num_cols + col_index
                if result_index < len(results):
                    result = results[result_index]
                    product = result['_source']
                    score = result['_score']
                    image_url = product['imageURL']
                    
                    # Display the image and product details in the corresponding column
                    with cols[col_index]:
                        st.image(image_url, use_column_width=True)
                        for key, value in product.items():
                            if key != 'embedding':
                                st.write(f"{key.replace('_', ' ').title()}: {value}")
                        st.write(f"Score: {score:.2f}")
