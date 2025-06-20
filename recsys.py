import numpy as np
import pandas as pd
from pymongo import MongoClient
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

class ColdStartRecommendationSystem:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="juno_db"):
        """
        Initialize the recommendation system.
        The MongoDB client is initialized on-demand and excluded from pickling.
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client = None
        self.db = None
        
        # DataFrames
        self.users_df = None
        self.products_df = None
        self.interactions_df = None
        
        # LightFM components
        self.dataset = Dataset()
        self.model = None
        self.user_features = None
        self.item_features = None
        
        # Store feature lists for consistency
        self.user_features_list = []
        self.item_features_list = []
        self.top_categories = []
        
    def _connect_mongo(self):
        """Establish MongoDB connection."""
        if not self.client:
            print("Establishing MongoDB connection...")
            self.client = MongoClient(self.mongo_uri, unicode_decode_error_handler='ignore')
            self.db = self.client[self.db_name]

    def __getstate__(self):
        """
        Prepare the object's state for pickling.
        
        This method is called by pickle.dump(). We exclude the non-serializable
        MongoDB client and database objects.
        """
        state = self.__dict__.copy()
        # Remove the unpickleable entries.
        del state['client']
        del state['db']
        return state

    def __setstate__(self, state):
        """
        Restore the object's state from a pickled state.
        
        This method is called by pickle.load(). We restore the attributes
        and then re-initialize the MongoDB connection.
        """
        self.__dict__.update(state)
        # Re-initialize the MongoDB connection after unpickling.
        self.client = None 
        self.db = None

    def load_data_from_mongo(self):
        """
        Load data from MongoDB collections into pandas DataFrames.
        """
        self._connect_mongo()
        print("Loading data from MongoDB...")
        
        users_cursor = self.db['users'].find({})
        users_data = [{
            'user_id': user['id'],
            'age': user.get('age', 25),
            'gender': user.get('gender', 'unknown'),
            'role': user.get('role', 'user'),
            'account_status': user.get('account_status', 'active'),
            'login_count': user.get('login_count', 0),
            'profile_completion': user.get('profile_completion', 0)
        } for user in users_cursor]

        products_cursor = self.db['products'].find({})
        products_data = []
        for product in products_cursor:
            pricing = product.get('pricing', {})
            categories = [cat.get('name', '') for cat in product.get('categories', [])]
            description = f"{product.get('description', '')} {product.get('short_description', '')}"
            products_data.append({
                'product_id': product['id'],
                'title': product.get('title', ''),
                'description': description,
                'price': pricing.get('price', 0),
                'discounted': pricing.get('discounted', False),
                'product_type': product.get('product_type', ''),
                'categories': ','.join(categories),
                'tags': ','.join(product.get('tags', [])),
                'rating': product.get('rating', 0),
                'review_count': product.get('review_count', 0),
                'is_trending': product.get('is_trending', False),
                'is_featured': product.get('is_featured', False)
            })

        interactions_cursor = self.db['interactions'].find({})
        interactions_data = []
        for interaction in interactions_cursor:
            action_type = interaction.get('action_type', 'view')
            rating = interaction.get('rating', 0)
            if rating == 0:
                rating_map = {'like': 1.0, 'dislike': 0.1, 'view': 0.5, 'purchase': 2.0, 'add_to_cart': 1.5}
                rating = rating_map.get(action_type, 0.5)
            interactions_data.append({
                'user_id': interaction['user_id'],
                'product_id': interaction['product_id'],
                'rating': rating,
                'action_type': action_type
            })

        self.users_df = pd.DataFrame(users_data)
        self.products_df = pd.DataFrame(products_data)
        self.interactions_df = pd.DataFrame(interactions_data)
        
        print(f"Loaded {len(self.users_df)} users, {len(self.products_df)} products, {len(self.interactions_df)} interactions")

    def prepare_user_features(self):
        """Prepare user features for the model."""
        print("Preparing user features...")
        features = set()
        for feature in ['gender', 'role', 'account_status']:
            for value in self.users_df[feature].fillna('unknown').unique():
                features.add(f"{feature}:{value}")
        for label in ['teen', 'young', 'adult', 'middle', 'senior']: features.add(f"age:{label}")
        for label in ['new', 'occasional', 'regular', 'frequent']: features.add(f"login_count:{label}")
        for label in ['low', 'medium', 'high', 'complete']: features.add(f"profile_completion:{label}")
        self.user_features_list = sorted(list(features))
        return self.user_features_list

    def prepare_item_features(self):
        """Prepare item features for the model."""
        print("Preparing item features...")
        features = set()
        for ptype in self.products_df['product_type'].dropna().unique():
            if ptype: features.add(f"product_type:{ptype}")
        for label in ['budget', 'affordable', 'mid_range', 'premium', 'luxury']: features.add(f"price_range:{label}")
        for feature in ['discounted', 'is_trending', 'is_featured']:
            features.add(f"{feature}:True"); features.add(f"{feature}:False")
        for label in ['low', 'medium', 'good', 'excellent']: features.add(f"rating:{label}")
        
        all_categories = self.products_df['categories'].str.split(',').explode().str.strip().dropna()
        self.top_categories = all_categories.value_counts().head(20).index.tolist()
        for cat in self.top_categories:
            features.add(f"category:{cat}")
            
        self.item_features_list = sorted(list(features))
        return self.item_features_list
        
    def build_user_item_features(self):
        """Build user and item feature matrices."""
        print("Building feature matrices...")
        def get_user_features(user):
            features = set()
            for col in ['gender', 'role', 'account_status']:
                features.add(f"{col}:{user[col] if pd.notna(user[col]) else 'unknown'}")
            age_map = {range(0, 21): 'teen', range(21, 31): 'young', range(31, 41): 'adult', range(41, 51): 'middle'}
            features.add(f"age:{next((v for k, v in age_map.items() if user.get('age', 25) in k), 'senior')}")
            # ... (add other feature derivations) ...
            return list(features)

        def get_item_features(product):
            features = set()
            if pd.notna(product['product_type']) and product['product_type']:
                features.add(f"product_type:{product['product_type']}")
            price = product.get('price', 0)
            price_map = {range(0, 2001): 'budget', range(2001, 5001): 'affordable', range(5001, 10001): 'mid_range', range(10001, 20001): 'premium'}
            features.add(f"price_range:{next((v for k, v in price_map.items() if price in k), 'luxury')}")
            # ... (add other feature derivations) ...
            return list(features)

        user_features_data = [(user['user_id'], get_user_features(user)) for _, user in self.users_df.iterrows()]
        item_features_data = [(prod['product_id'], get_item_features(prod)) for _, prod in self.products_df.iterrows()]
        
        return user_features_data, item_features_data

    def train_model(self):
        """Train the LightFM model."""
        print("Starting model training process...")
        self.load_data_from_mongo()
        
        if self.users_df.empty or self.products_df.empty or self.interactions_df.empty:
            print("No data loaded. Aborting training.")
            return

        self.dataset.fit(
            users=self.users_df['user_id'].unique(),
            items=self.products_df['product_id'].unique(),
            user_features=self.prepare_user_features(),
            item_features=self.prepare_item_features()
        )
        
        user_features_data, item_features_data = self.build_user_item_features()
        
        interactions_list = [tuple(x) for x in self.interactions_df[['user_id', 'product_id', 'rating']].values]
        interactions_matrix, _ = self.dataset.build_interactions(interactions_list)
        
        self.user_features = self.dataset.build_user_features(user_features_data)
        self.item_features = self.dataset.build_item_features(item_features_data)
        
        self.model = LightFM(loss='warp', learning_rate=0.05, no_components=50, random_state=42)
        self.model.fit(
            interactions_matrix,
            user_features=self.user_features,
            item_features=self.item_features,
            epochs=50,
            num_threads=4,
            verbose=True
        )
        print("Model training completed!")

    def save(self, filepath='recommendation_system.pkl'):
        """Saves the entire recommendation system object to a file."""
        print(f"Saving system state to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Save complete.")

    @staticmethod
    def load(filepath='recommendation_system.pkl'):
        """Loads a recommendation system object from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved model found at {filepath}")
        print(f"Loading system state from {filepath}...")
        with open(filepath, 'rb') as f:
            system = pickle.load(f)
        print("Loading complete.")
        return system

    def get_recommendations(self, user_id, num_recommendations=10):
        """Get recommendations for a user (handles cold start)."""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please call train_model() or load().")

        user_id_map, _, item_id_map, _ = self.dataset.mapping()
        
        if user_id in user_id_map:
            # Existing user
            user_index = user_id_map[user_id]
            all_item_indices = list(item_id_map.values())
            
            scores = self.model.predict(
                user_index,
                all_item_indices,
                user_features=self.user_features,
                item_features=self.item_features,
                num_threads=4
            )
            
            user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]['product_id'].tolist()
            
            top_items = sorted(zip(item_id_map.keys(), scores), key=lambda x: x[1], reverse=True)
            
            recommendations = [(item_id, str(score)) for item_id, score in top_items if item_id not in user_interactions]
            return recommendations[:num_recommendations]
        else:
            # Cold start user: recommend popular and trending items
            print(f"Cold start for user: {user_id}. Recommending popular items.")
            popular_items = self.products_df.sort_values(by=['is_trending', 'review_count'], ascending=False)
            recommendations = list(zip(popular_items['product_id'], popular_items['rating'])) # Use rating as a proxy score
            return recommendations[:num_recommendations]

# --- Example Usage ---
def main():
    MODEL_PATH = 'cold_start_recsys.pkl'
    
    # Initialize the recommendation system
    # Use your MongoDB Atlas connection string
    rec_system = ColdStartRecommendationSystem(mongo_uri="mongodb+srv://swift:swift@hobby.nzyzrid.mongodb.net/")
    
    # --- Train and Save ---
    print("--- Training and Saving Model ---")
    rec_system.train_model()
    rec_system.save(MODEL_PATH)
    
    # --- Load and Predict ---
    print("\n--- Loading Model and Making Predictions ---")
    try:
        # Load the saved system state into a new object
        loaded_rec_system = ColdStartRecommendationSystem.load(MODEL_PATH)

        # Ensure DataFrames are loaded in the new instance
        if loaded_rec_system.products_df is None or loaded_rec_system.users_df is None:
             print("DataFrames not loaded correctly. Re-loading from Mongo.")
             loaded_rec_system.load_data_from_mongo()

        # Example: Get recommendations for an existing user
        if not loaded_rec_system.users_df.empty:
            existing_user_id = loaded_rec_system.users_df.iloc[0]['user_id']
            recommendations = loaded_rec_system.get_recommendations(existing_user_id, num_recommendations=5)
            print(f"\nRecommendations for existing user '{existing_user_id}':")
            for item_id, score in recommendations:
                title = loaded_rec_system.products_df.loc[loaded_rec_system.products_df['product_id'] == item_id, 'title'].values[0]
                print(f"- {title} (Score: {score:.3f})")

        # Example: Get recommendations for a cold start user
        cold_start_user_id = "new_user_who_does_not_exist"
        cold_recommendations = loaded_rec_system.get_recommendations(cold_start_user_id, num_recommendations=5)
        print(f"\nCold start recommendations for user '{cold_start_user_id}':")
        for item_id, score in cold_recommendations:
            title = loaded_rec_system.products_df.loc[loaded_rec_system.products_df['product_id'] == item_id, 'title'].values[0]
            print(f"- {title} (Score: {score:.3f})")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()