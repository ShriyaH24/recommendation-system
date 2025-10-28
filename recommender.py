import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class ShoppingRecommender:
    def __init__(self, data_path="data"):
        # Auto-detect CSV file in /data folder
        for file in os.listdir(data_path):
            if file.endswith(".csv"):
                self.data_path = os.path.join(data_path, file)
                break
        else:
            raise FileNotFoundError("No CSV file found in 'data' folder.")

        self.df = None
        self.model = None

    def load_data(self):
        """Load and clean dataset"""
        df = pd.read_csv(self.data_path)
        df.columns = df.columns.str.strip()

        # Keep relevant columns
        needed_cols = [
            "CustomerID", "Gender", "Tenure_Months", "Online_Spend",
            "Offline_Spend", "Discount_pct", "Product_Category"
        ]
        df = df[needed_cols].dropna()

        # Aggregate per customer
        customer_df = df.groupby(["CustomerID", "Gender", "Tenure_Months"]).agg({
            "Online_Spend": "mean",
            "Offline_Spend": "mean",
            "Discount_pct": "mean",
            "Product_Category": lambda x: x.mode()[0] if not x.mode().empty else None
        }).reset_index()

        self.df = customer_df
        return customer_df

    def cluster_customers(self, n_clusters=5):
        """Cluster customers using K-Means"""
        features = ["Tenure_Months", "Online_Spend", "Offline_Spend", "Discount_pct"]
        X = self.df[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df["Cluster"] = kmeans.fit_predict(X_scaled)
        self.model = kmeans

        return self.df

    def recommend(self, customer_id, top_n=3):
        """Hybrid: Cluster + Cosine Similarity Recommendations"""
        if self.df is None or "Cluster" not in self.df.columns:
            raise ValueError("You must run cluster_customers() before recommending.")

        customer_row = self.df[self.df["CustomerID"] == customer_id]
        if customer_row.empty:
            return ["Customer not found"]

        cluster = customer_row.iloc[0]["Cluster"]
        cluster_df = self.df[self.df["Cluster"] == cluster]

        # Compute similarity inside the same cluster
        features = ["Tenure_Months", "Online_Spend", "Offline_Spend", "Discount_pct"]
        X_cluster = cluster_df[features]
        sim_matrix = cosine_similarity(X_cluster)

        # Find the index of the current customer
        cust_idx = cluster_df.index[cluster_df["CustomerID"] == customer_id][0]

        # Get top 5 most similar customers
        sim_scores = list(enumerate(sim_matrix[cust_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

        similar_customers = cluster_df.iloc[[i[0] for i in sim_scores]]

        # Recommend top categories among similar customers
        top_products = (
            similar_customers["Product_Category"]
            .value_counts()
            .head(top_n)
            .index.tolist()
        )

        return top_products

    def get_cluster_summary(self):
        """Return average metrics per cluster"""
        summary = self.df.groupby("Cluster")[["Tenure_Months", "Online_Spend", "Offline_Spend", "Discount_pct"]].mean().round(2)
        return summary

