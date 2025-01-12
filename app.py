import streamlit as st
import pandas as pd
import pickle
import random
# Load data and similarity matrix
df = pickle.load(open("df.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

# Utility function for truncating job description
def truncate_description(description, length=100):
    return description if len(description) <= length else description[:length] + "..."

# Content-based Recommendation
def content_based_recommendation(title, similarity, df, top_n=20):
    if title not in df['Title'].values:
        return []

    idx = df[df['Title'] == title].index[0]
    idx = df.index.get_loc(idx)
    distances = sorted(
        list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True
    )[1:top_n + 1]

    recommendations = [
        {
            "Job Title": df.iloc[i[0]].Title,
            "Company": df.iloc[i[0]].Company,
            "Description": truncate_description(df.iloc[i[0]]['Job.Description'], length=100),
        }
        for i in distances
    ]

    return recommendations

# Simulate User Interactions
def generate_user_interaction_data(df):
    interactions = []
    for index, row in df.iterrows():
        click_probability = random.random()
        interactions.append((index, 1 if click_probability > 0.4 else 0))
    return interactions

interactions = generate_user_interaction_data(df)

# Hybrid Recommendation
def hybrid_recommendation(title, similarity, interactions, df, top_n=20):
    # Get content-based recommendations
    content_jobs = content_based_recommendation(title, similarity, df, top_n=top_n)

    # Generate the interaction matrix and extract collaborative recommendations
    interaction_matrix = pd.DataFrame(interactions, columns=['Job Index', 'Interaction Score'])
    collaborative_jobs = interaction_matrix[interaction_matrix['Interaction Score'] == 1]['Job Index'].tolist()

    # Ensure that collaborative recommendations are within bounds
    collaborative_recommendations = []
    for job_idx in collaborative_jobs[:top_n]:
        if 0 <= job_idx < len(df):  # Boundary check
            collaborative_recommendations.append({
                "Job Title": df.iloc[job_idx].Title,
                "Company": df.iloc[job_idx].Company,
                "Description": truncate_description(df.iloc[job_idx]['Job.Description'], length=100),
            })

    # Combine content-based and collaborative recommendations, ensuring uniqueness
    combined_jobs = {job['Job Title']: job for job in content_jobs + collaborative_recommendations}.values()

    return list(combined_jobs)[:top_n]


# Streamlit UI
st.title('Job Recommendation System')

title = st.selectbox('Search job', df['Title'])
if title:
    st.subheader(f"Recommendations for '{title}'")
    recommendations = hybrid_recommendation(title, similarity, interactions, df, top_n=10)

    if recommendations:
        for job in recommendations:
            with st.expander(f"{job['Job Title']} at {job['Company']}"):
                st.markdown(f"**Job Title:** {job['Job Title']}")
                st.markdown(f"**Company:** {job['Company']}")
                st.markdown(f"**Description:** {job['Description']}")
                st.markdown("---")
    else:
        st.write("No recommendations found.")
