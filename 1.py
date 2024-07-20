from flask import Flask, request, jsonify, render_template
import pandas as pd
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load JSON data
try:
    with open('tasks.json') as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"Error reading JSON: {e}")
    data = {
        "InitialPreference": "Default initial preference",
        "Tasks": []
    }

initialPreference = data['InitialPreference']

# Normalize and create DataFrame
td = pd.json_normalize(data['Tasks'])
td.columns = ['Tasks', 'Descriptions', 'Deadline']

# Add initial preference task
date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df = pd.DataFrame([{
    'Tasks': 'init',
    'Descriptions': initialPreference,
    'Deadline': date,
}])
td = pd.concat([df, td], ignore_index=True)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer()
td['Descriptions'] = td['Descriptions'].fillna('')
tfidf_matrix = tfidf.fit_transform(td['Descriptions'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to calculate time score
def calc_time(TaskDeadline):
    now = datetime.now()
    task_deadline = datetime.strptime(TaskDeadline, "%Y-%m-%d %H:%M:%S")
    timeDiff = (task_deadline - now).total_seconds() / 3600
    return (1 / (timeDiff + 1)) / 10

# Function to get recommendations
def get_rec(title, cosine_sim=cosine_similarity, dataFrame=td):
    indices = pd.Series(dataFrame.index, index=dataFrame['Tasks']).drop_duplicates()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]

    sim_scores_with_time = []
    for i in sim_scores:
        deadline_str = dataFrame["Deadline"].iloc[i[0]]
        time_score = calc_time(deadline_str)
        total_score = i[1] + time_score
        sim_scores_with_time.append((i[0], total_score))

    sim_scores_with_time = sorted(sim_scores_with_time, key=lambda x: x[1], reverse=True)
    tasks_indices = [i[0] for i in sim_scores_with_time]

    return dataFrame.iloc[tasks_indices]

@app.route('/')
def index():
    return render_template('index.html', tasks=td.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form['title']
    recommendations = get_rec(title)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
