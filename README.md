Smart Shopping Recommendation System:
Personalized Recommendations Using K-Means Clustering + Cosine Similarity

This project is a machine learning–based recommendation system that analyzes customer shopping behavior and suggests product categories similar customers are likely to prefer. It combines unsupervised clustering (K-Means) with cosine similarity for hybrid, personalized recommendations.

Features:

1) Data-driven customer segmentation using K-Means

2) Personalized recommendations using similarity between customers within the same cluster

3) Cluster insights – view average spend, tenure, and discount behavior

4) Interactive UI built with Streamlit

5) Ready for deployment or portfolio demonstration

Project Structure
shopping-recommender/
│
├── app_ui.py               # Streamlit frontend app
├── recommender.py          # Core recommendation logic
├── data/
│   └── online_shoppers.csv # Dataset (add here manually)
├── requirements.txt        # Dependencies
├── .gitignore              # Ignore unnecessary files
└── README.md               # Project documentation

Working:

Data Loading & Cleaning
The dataset (from Kaggle) is preprocessed to retain key attributes:

CustomerID, Gender, Tenure_Months

Online_Spend, Offline_Spend, Discount_pct

Product_Category

Customer Clustering (K-Means)
Customers are grouped based on spending habits and discount behavior.

Cosine Similarity Filtering
Within each cluster, cosine similarity identifies the most similar customers.

Recommendation Generation
The system recommends the top 3 most common product categories among similar shoppers.

Tech Stack:
Component	Technology Used
Programming Language	Python 3
Data Analysis	Pandas, Scikit-learn
Clustering	K-Means
Similarity Metric	Cosine Similarity
Frontend	Streamlit
IDE	Visual Studio Code


Installation & Setup
a) Clone the Repository
git clone https://github.com/<your-username>/shopping-recommender.git
cd shopping-recommender

b) Install Dependencies

Make sure you have Python 3.8+ installed. Then run:

pip install -r requirements.txt

c) Add Dataset

Place your Kaggle dataset CSV inside the data/ folder, e.g.:

data/online_shoppers.csv

 Run the Streamlit App
streamlit run app_ui.py


Then open the displayed local URL in your browser (usually http://localhost:8501).

d) Example Output

Cluster-level summary showing customer segmentation

Recommended top 3 product categories personalized per customer

Dynamic, interactive visualization through Streamlit


Dataset

1 Source: Online Shopping Dataset – Kaggle

2 Attributes used:

3 CustomerID: Unique customer identifier

4 Gender: Male/Female

5 Tenure_Months: Number of months with the platform

6 Online_Spend / Offline_Spend: Amount spent

7 Discount_pct: Applied discount percentage

8 Product_Category: Purchased product category


Future Improvements:

Integrate collaborative filtering with user-item matrices

Add time-based trend analysis (e.g., seasonality)

Display visual analytics with bar/pie charts in Streamlit

Option to recommend specific products, not just categories

