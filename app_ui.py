import streamlit as st
from recommender import ShoppingRecommender
import pandas as pd


# App Title and Description

st.set_page_config(page_title="Shopping Recommendation System", layout="centered")
st.title(" Smart Shopping Recommendation System")
st.markdown("""
This app clusters customers based on their shopping behavior and uses **cosine similarity** within each cluster 
to recommend product categories that similar shoppers love.
""")


@st.cache_data
def load_recommender():
    rec = ShoppingRecommender()
    df = rec.load_data()
    rec.cluster_customers()
    return rec, df

try:
    rec, df = load_recommender()
except FileNotFoundError:
    st.error(" Dataset not found. Please make sure your CSV file is inside the `/data` folder.")
    st.stop()

#Customer selection
st.sidebar.header(" Select Customer")
customer_ids = df["CustomerID"].unique().tolist()
selected_customer = st.sidebar.selectbox("Choose a Customer ID:", customer_ids)


#  Customer Info

st.subheader(" Customer Profile")
customer_data = df[df["CustomerID"] == selected_customer]

if not customer_data.empty:
    st.dataframe(customer_data[["CustomerID", "Gender", "Tenure_Months", "Online_Spend", "Offline_Spend", "Discount_pct", "Product_Category"]])
else:
    st.warning("No data found for this customer.")


if st.button("Get Recommendations"):
    with st.spinner("Analyzing similar shoppers..."):
        recommendations = rec.recommend(selected_customer)
        st.success("Personalized Recommendations Ready!")

        st.subheader(" Top Product Categories You May Like:")
        for idx, prod in enumerate(recommendations, start=1):
            st.write(f"**{idx}.** {prod}")


st.subheader(" Cluster Overview")
cluster_summary = rec.get_cluster_summary()
st.dataframe(cluster_summary)

st.markdown("---")
