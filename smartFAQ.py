import streamlit as st
import pandas as pd
import numpy as np
# Import your embedding model and cosine similarity function here

openai.api_key =  st.secrets["mykey"]

def load_data_and_embeddings():
    # Replace 'your_data.csv' with your actual file path
    data = pd.read_csv('qa_dataset_with_embeddings.csv')
    # Load pre-calculated embeddings (replace with your loading logic)
    embeddings = np.load('embeddings.npy')
    return data, embeddings

def generate_embedding(question, model):
    # Replace with your embedding generation logic using the selected model
    embedding = model.encode(question)
    return embedding

def find_answer(user_question, embeddings, data, threshold=0.8):
    # Generate embedding for user question
    user_embedding = generate_embedding(user_question, model)

    # Calculate cosine similarities
    similarities = np.dot(embeddings, user_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(user_embedding))

    # Find the most similar question
    most_similar_index = np.argmax(similarities)
    similarity_score = similarities[most_similar_index]

    if similarity_score >= threshold:
        answer = data['answer'][most_similar_index]
        return answer, similarity_score
    else:
        return None, None

def main():
    st.title("Question Answering App")

    data, embeddings = load_data_and_embeddings()

    user_question = st.text_input("Ask your question:")
    if st.button("Search"):
        if user_question:
            answer, similarity_score = find_answer(user_question, embeddings, data)
            if answer:
                st.write(f"Answer: {answer}")
                st.write(f"Similarity Score: {similarity_score:.2f}")
            else:
                st.write("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")

    # Optional features:
    # st.button("Clear")  # Clear the input field
    # st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.8)  # Adjust threshold
    # st.radio("Rate the answer", options=["Helpful", "Not Helpful"])  # User feedback

if __name__ == "__main__":
    main()
