# !pip install faiss-gpu
# !pip install --user  sentence-transformers
import pathlib
import textwrap
import gdown

import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown


####################



from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

app = Flask(__name__)

# Load the SentenceTransformer model from a file
loaded_model = SentenceTransformer("X:/drive-download-20191524997/My_Profile/Uttowa/Graduation_Project/Team/Abdelrahman/Test/4_Model_Faiss/SentenceTransformer")

# Load the Faiss index from a file
loaded_index = faiss.read_index('X:/drive-download-20191524997/My_Profile/Uttowa/Graduation_Project/Team/Abdelrahman/Test/4_Model_Faiss/faiss_index.index')
csv_path = "X:/drive-download-20191524997/My_Profile/Uttowa/Graduation_Project/Team/Donia/final_data.csv" # Update the path to your CSV file
combined_data = pd.read_csv(csv_path)
# Function to perform k-nearest neighbors search
def perform_knn_search(query_vector, k=3):
    distances, indices = loaded_index.search(np.expand_dims(query_vector, axis=0), k)
    return distances[0], indices[0]

# Test Generative Model Bard Google AI Tools

def enhanceResults(neighbors, skills):
    def to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
    
    # Use your Key
    api_key=""
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-pro')
    
    Saved_data=[]
    # print(neighbors[0])
    for index in range(len(neighbors)):
    # decit={}
        
        case=neighbors.iloc[index]
        
        response = model.generate_content(f'The following is skills remove unnessary words to display only skill name separate each skill with comma make it readable to humans:"{case}".')
        if (len(response.candidates)==0):
            print(f"No candidates were returned for index {index}")
        else :
            Saved_data.append({response.text})
    
    df_sentiment = pd.DataFrame(Saved_data, columns=[skills])
    
    return df_sentiment
    


# Home page with a simple form for user input
@app.route('/')
def index():
    return render_template('Form.html')

# Handle the form submission
@app.route('/search', methods=['POST'])
def search():
    # user_query = request.form['query']
    # query_vector = loaded_model.encode(user_query)

    # Perform k-nearest neighbors search
    # distances, indices = perform_knn_search(query_vector)

    # Get the corresponding records from your data
    # neighbors_data = [combined_data.loc[ind] for ind in indices]

    # Retrieve form data
    job_title = request.form.get('Job-Title', '')
    skills = request.form.get('Skills', '')
    prefer_job_title = request.form.get('pref-job', '')
    industry = request.form.get('Industry', '')
    career_level = request.form.get('Career-level', 'not-spec')

    # Combine the values with <s> separator job_title skills prefer_job_title industry career_level
    combined_values = f"{skills}<s>{career_level}<s>{industry}<s>{job_title}<s>''<s>{prefer_job_title}"

    query_vector = loaded_model.encode(combined_values)

    distances, indices = perform_knn_search(query_vector)
    neighbors_data = [combined_data.loc[ind] for ind in indices]
    # results=neighbors_data[['job_titles', 'text_without_stopwords']]
    neighbors = [combined_data.loc[ind, ['js_skills','job_skills', 'job_titles']] for ind in indices]
    
    neighbors_df =pd.DataFrame(neighbors)
    # Reset the index if needed
    neighbors_df = neighbors_df.reset_index(drop=True)
    
    # print(neighbors_df)
    neighbors_df['js_skills'] = enhanceResults(neighbors_df['js_skills'],'js_skills')
    # print(neighbors_df['js_skills'])
    neighbors_df['job_skills'] = enhanceResults(neighbors_df['job_skills'],'job_skills')

    # print(neighbors_df)
    # neighbors = [(index, row.to_dict()) for index, row in neighbors_df.iterrows()]
    # neighbors = [neighbors_df.iloc[ind, ['js_skills','job_skills', 'job_titles']] for ind in indices]
    
    return render_template('job.html', query=combined_values, neighbors=neighbors_df)


if __name__ == '__main__':
    app.run(debug=True)
