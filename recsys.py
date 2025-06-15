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
warnings.filterwarnings('ignore')

class ColdStartRecommendationSystem:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="juno_db"):
        """
        Initialize the recommendation system with MongoDB connection
        """
        self.client = MongoClient(mongo_uri, unicode_decode_error_handler='ignore')
        self.db = self.client[db_name]
        self.users_collection = self.db['users']
        self.products_collection = self.db['products']
        self.interactions_collection = self.db['interactions']
        
        # Initialize components
        self.dataset = Dataset()
        self.model = None
        self.user_features = None
        self.item_features = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        
        # Feature encoders
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Store feature lists for consistency
        self.user_features_list = []
        self.item_features_list = []
        self.top_categories = []
        
    def load_data_from_mongo(self):
        """
        Load data from MongoDB collections
        """
        print("Loading data from MongoDB...")
        
        # Load users
        users_cursor = self.users_collection.find({})
        users_data = []
        for user in users_cursor:
            users_data.append({
                'user_id': user['id'],
                'age': user.get('age', 25),
                'gender': user.get('gender', 'unknown'),
                'role': user.get('role', 'user'),
                'account_status': user.get('account_status', 'active'),
                'login_count': user.get('login_count', 0),
                'profile_completion': user.get('profile_completion', 0)
            })
        
        # Load products
        products_cursor = self.products_collection.find({})
        products_data = []
        for product in products_cursor:
            # Handle pricing
            pricing = product.get('pricing', {})
            price = pricing.get('price', 0)
            discounted = pricing.get('discounted', False)
            
            # Handle categories
            categories = product.get('categories', [])
            category_names = [cat.get('name', '') for cat in categories]
            
            # Handle tags
            tags = product.get('tags', [])
            
            # Create description for content-based features
            description = product.get('description', '') + ' ' + product.get('short_description', '')
            
            products_data.append({
                'product_id': product['id'],
                'title': product.get('title', ''),
                'description': description,
                'price': price,
                'discounted': discounted,
                'product_type': product.get('product_type', ''),
                'categories': ','.join(category_names),
                'tags': ','.join(tags),
                'rating': product.get('rating', 0),
                'review_count': product.get('review_count', 0),
                'is_trending': product.get('is_trending', False),
                'is_featured': product.get('is_featured', False)
            })
        
        # Load interactions
        interactions_cursor = self.interactions_collection.find({})
        interactions_data = []
        for interaction in interactions_cursor:
            # Convert action types to ratings
            action_type = interaction.get('action_type', 'view')
            rating = interaction.get('rating', 0)
            
            # Map action types to implicit ratings if rating is 0
            if rating == 0:
                if action_type == 'like':
                    rating = 1.0
                elif action_type == 'dislike':
                    rating = 0.1  # Small positive value for implicit feedback
                elif action_type == 'view':
                    rating = 0.5
                elif action_type == 'purchase':
                    rating = 2.0
                elif action_type == 'add_to_cart':
                    rating = 1.5
                else:
                    rating = 0.5
            
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
        """
        Prepare user features for the model
        """
        print("Preparing user features...")
        
        user_features_list = []
        
        # Categorical features
        categorical_features = ['gender', 'role', 'account_status']
        for feature in categorical_features:
            unique_values = self.users_df[feature].fillna('unknown').unique()
            for value in unique_values:
                user_features_list.append(f"{feature}:{value}")
        
        # Age features
        age_labels = ['teen', 'young', 'adult', 'middle', 'senior']
        for label in age_labels:
            user_features_list.append(f"age:{label}")
        
        # Login count features
        login_labels = ['new', 'occasional', 'regular', 'frequent']
        for label in login_labels:
            user_features_list.append(f"login_count:{label}")
        
        # Profile completion features
        completion_labels = ['low', 'medium', 'high', 'complete']
        for label in completion_labels:
            user_features_list.append(f"profile_completion:{label}")
        
        self.user_features_list = user_features_list
        return user_features_list
    
    def prepare_item_features(self):
        """
        Prepare item features for the model
        """
        print("Preparing item features...")
        
        item_features_list = []
        
        # Product type features
        product_types = self.products_df['product_type'].dropna().unique()
        for ptype in product_types:
            if ptype != '':
                item_features_list.append(f"product_type:{ptype}")
        
        # Price range features
        price_labels = ['budget', 'affordable', 'mid_range', 'premium', 'luxury']
        for label in price_labels:
            item_features_list.append(f"price_range:{label}")
        
        # Boolean features
        boolean_features = ['discounted', 'is_trending', 'is_featured']
        for feature in boolean_features:
            item_features_list.extend([f"{feature}:True", f"{feature}:False"])
        
        # Rating features
        rating_labels = ['low', 'medium', 'good', 'excellent']
        for label in rating_labels:
            item_features_list.append(f"rating:{label}")
        
        # Category features (determine top categories once and store)
        all_categories = []
        for cats in self.products_df['categories'].fillna(''):
            if cats:
                all_categories.extend([cat.strip() for cat in cats.split(',') if cat.strip()])
        
        # Get top 20 categories and store them
        self.top_categories = pd.Series(all_categories).value_counts().head(20).index.tolist()
        for cat in self.top_categories:
            item_features_list.append(f"category:{cat}")
        
        self.item_features_list = item_features_list
        return item_features_list
    
    def build_user_item_features(self):
        """
        Build user and item feature matrices using the prepared feature lists
        """
        print("Building feature matrices...")
        
        # Build user features data
        user_features_data = []
        for idx, user in self.users_df.iterrows():
            features = []
            
            # Categorical features
            for feature in ['gender', 'role', 'account_status']:
                value = user[feature] if pd.notna(user[feature]) else 'unknown'
                features.append(f"{feature}:{value}")
            
            # Age features
            age = user['age'] if pd.notna(user['age']) else 25
            if age <= 20:
                features.append("age:teen")
            elif age <= 30:
                features.append("age:young")
            elif age <= 40:
                features.append("age:adult")
            elif age <= 50:
                features.append("age:middle")
            else:
                features.append("age:senior")
            
            # Login count features
            login_count = user['login_count'] if pd.notna(user['login_count']) else 0
            if login_count <= 5:
                features.append("login_count:new")
            elif login_count <= 20:
                features.append("login_count:occasional")
            elif login_count <= 50:
                features.append("login_count:regular")
            else:
                features.append("login_count:frequent")
            
            # Profile completion features
            profile_completion = user['profile_completion'] if pd.notna(user['profile_completion']) else 0
            if profile_completion <= 25:
                features.append("profile_completion:low")
            elif profile_completion <= 50:
                features.append("profile_completion:medium")
            elif profile_completion <= 75:
                features.append("profile_completion:high")
            else:
                features.append("profile_completion:complete")
            
            user_features_data.append((user['user_id'], features))
        
        # Build item features data
        item_features_data = []
        for idx, product in self.products_df.iterrows():
            features = []
            
            # Product type
            if pd.notna(product['product_type']) and product['product_type']:
                features.append(f"product_type:{product['product_type']}")
            
            # Price range
            price = product['price'] if pd.notna(product['price']) else 0
            if price <= 2000:
                features.append("price_range:budget")
            elif price <= 5000:
                features.append("price_range:affordable")
            elif price <= 10000:
                features.append("price_range:mid_range")
            elif price <= 20000:
                features.append("price_range:premium")
            else:
                features.append("price_range:luxury")
            
            # Boolean features
            features.append(f"discounted:{product['discounted']}")
            features.append(f"is_trending:{product['is_trending']}")
            features.append(f"is_featured:{product['is_featured']}")
            
            # Rating
            rating = product['rating'] if pd.notna(product['rating']) else 0
            if rating <= 2:
                features.append("rating:low")
            elif rating <= 3:
                features.append("rating:medium")
            elif rating <= 4:
                features.append("rating:good")
            else:
                features.append("rating:excellent")
            
            # Categories (only use top categories that were determined earlier)
            categories = product['categories'] if pd.notna(product['categories']) else ''
            if categories:
                for cat in categories.split(','):
                    cat = cat.strip()
                    if cat and cat in self.top_categories:  # Only use top categories
                        features.append(f"category:{cat}")
            
            item_features_data.append((product['product_id'], features))
        
        return user_features_data, item_features_data
    
    def train_model(self):
        """
        Train the LightFM model
        """
        print("Training LightFM model...")
        
        # Load data
        self.load_data_from_mongo()
        
        # Prepare features
        user_features_list = self.prepare_user_features()
        item_features_list = self.prepare_item_features()
        
        # Get unique users and items
        all_users = self.users_df['user_id'].unique()
        all_items = self.products_df['product_id'].unique()
        
        # Fit the dataset
        self.dataset.fit(
            users=all_users,
            items=all_items,
            user_features=user_features_list,
            item_features=item_features_list
        )
        
        # Build feature data
        user_features_data, item_features_data = self.build_user_item_features()
        
        # Build interactions matrix
        interactions_list = []
        for _, interaction in self.interactions_df.iterrows():
            interactions_list.append((interaction['user_id'], interaction['product_id'], interaction['rating']))
        
        # Build matrices
        interactions_matrix, weights = self.dataset.build_interactions(interactions_list)
        user_features_matrix = self.dataset.build_user_features(user_features_data)
        item_features_matrix = self.dataset.build_item_features(item_features_data)
        
        # Store feature matrices
        self.user_features = user_features_matrix
        self.item_features = item_features_matrix
        
        # Create mappings
        self.user_id_map = self.dataset.mapping()[0]
        self.item_id_map = self.dataset.mapping()[2]
        self.reverse_user_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_item_map = {v: k for k, v in self.item_id_map.items()}
        
        # Initialize and train model
        self.model = LightFM(
            loss='warp',  # Good for implicit feedback
            learning_rate=0.05,
            item_alpha=1e-6,
            user_alpha=1e-6,
            no_components=50,  # Reduced for cold start scenarios
            random_state=42
        )
        
        # Train the model
        self.model.fit(
            interactions_matrix,
            user_features=user_features_matrix,
            item_features=item_features_matrix,
            epochs=50,
            num_threads=4,
            verbose=True
        )
        
        print("Model training completed!")
        
    def get_recommendations(self, user_id, num_recommendations=10, include_cold_start=True):
        """
        Get recommendations for a user (handles cold start)
        """
        if self.model is None:
            raise ValueError("Model not trained. Please call train_model() first.")
        
        # Check if user exists in our system
        user_in_training = user_id in self.user_id_map
        
        if not user_in_training and not include_cold_start:
            return []
        
        # Get all items
        all_items = list(self.item_id_map.keys())
        item_indices = [self.item_id_map[item] for item in all_items]
        
        if user_in_training:
            # Existing user - use collaborative filtering with features
            user_index = self.user_id_map[user_id]
            
            # Get user's existing interactions to exclude them
            user_interactions = self.interactions_df[
                self.interactions_df['user_id'] == user_id
            ]['product_id'].tolist()
            
            # Predict scores
            scores = self.model.predict(
                user_index,
                item_indices,
                user_features=self.user_features,
                item_features=self.item_features,
                num_threads=4
            )
            
            # Create recommendations
            recommendations = []
            for i, item_id in enumerate(all_items):
                if item_id not in user_interactions:  # Exclude already interacted items
                    recommendations.append((item_id, float(scores[i])))
            
        else:
            # Cold start user - use content-based approach
            # For cold start, we'll recommend popular items in different categories
            # and trending/featured items
            
            # Get popular items
            popular_items = self.products_df.nlargest(20, 'review_count')['product_id'].tolist()
            trending_items = self.products_df[self.products_df['is_trending'] == True]['product_id'].tolist()
            featured_items = self.products_df[self.products_df['is_featured'] == True]['product_id'].tolist()
            
            # Combine and score cold start recommendations
            cold_start_items = list(set(popular_items + trending_items + featured_items))
            
            # Simple scoring for cold start
            recommendations = []
            for item_id in cold_start_items:
                if item_id in self.item_id_map:
                    product_info = self.products_df[self.products_df['product_id'] == item_id].iloc[0]
                    
                    # Simple scoring based on product features
                    score = 0.0
                    if product_info['is_trending']:
                        score += 0.3
                    if product_info['is_featured']:
                        score += 0.2
                    if product_info['discounted']:
                        score += 0.1
                    
                    # Add rating influence
                    score += product_info['rating'] * 0.1
                    
                    # Add review count influence (normalized)
                    review_count_normalized = min(product_info['review_count'] / 100, 1.0)
                    score += review_count_normalized * 0.2
                    
                    recommendations.append((item_id, score))
        
        # Sort and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:num_recommendations]
    
    def get_similar_items(self, item_id, num_similar=5):
        """
        Get similar items to a given item
        """
        if self.model is None:
            raise ValueError("Model not trained. Please call train_model() first.")
        
        if item_id not in self.item_id_map:
            return []
        
        item_index = self.item_id_map[item_id]
        
        # Get item representations
        item_biases, item_embeddings = self.model.get_item_representations(
            features=self.item_features
        )
        
        # Calculate similarities
        target_embedding = item_embeddings[item_index]
        similarities = np.dot(item_embeddings, target_embedding)
        
        # Get top similar items (excluding the item itself)
        similar_indices = np.argsort(similarities)[::-1][1:num_similar + 1]
        
        similar_items = []
        for idx in similar_indices:
            item_id_similar = self.reverse_item_map[idx]
            similarity_score = float(similarities[idx])
            similar_items.append((item_id_similar, similarity_score))
        
        return similar_items
    
    def add_new_interaction(self, user_id, product_id, action_type='view', rating=None):
        """
        Add a new interaction to the database and update recommendations
        """
        # Convert action to rating if not provided
        if rating is None:
            if action_type == 'like':
                rating = 1.0
            elif action_type == 'dislike':
                rating = 0.1
            elif action_type == 'view':
                rating = 0.5
            elif action_type == 'purchase':
                rating = 2.0
            elif action_type == 'add_to_cart':
                rating = 1.5
            else:
                rating = 0.5
        
        # Add to MongoDB
        interaction_doc = {
            'id': f"{user_id}_{product_id}_{action_type}",
            'user_id': user_id,
            'product_id': product_id,
            'rating': rating,
            'action_type': action_type,
            'timestamp': pd.Timestamp.now()
        }
        
        self.interactions_collection.insert_one(interaction_doc)
        print(f"Added interaction: {user_id} -> {product_id} ({action_type})")

# Example usage and testing
def main():
    # Initialize the recommendation system
    rec_system = ColdStartRecommendationSystem(mongo_uri="mongodb+srv://swift:swift@hobby.nzyzrid.mongodb.net/")
    
    # Train the model
    rec_system.train_model()
    
    # Example: Get recommendations for an existing user
    try:
        existing_user_id = rec_system.users_df.iloc[0]['user_id']
        recommendations = rec_system.get_recommendations(existing_user_id, num_recommendations=5)
        print(f"\nRecommendations for existing user {existing_user_id}:")
        for item_id, score in recommendations:
            product_info = rec_system.products_df[
                rec_system.products_df['product_id'] == item_id
            ].iloc[0]
            print(f"- {product_info['title']} (Score: {score:.3f})")
    except Exception as e:
        print(f"Error getting recommendations for existing user: {e}")
    
    # Example: Get recommendations for a cold start user
    cold_start_user_id = "new_user_123"
    cold_recommendations = rec_system.get_recommendations(
        cold_start_user_id, 
        num_recommendations=5, 
        include_cold_start=True
    )
    print(f"\nCold start recommendations for user {cold_start_user_id}:")
    for item_id, score in cold_recommendations:
        try:
            product_info = rec_system.products_df[
                rec_system.products_df['product_id'] == item_id
            ].iloc[0]
            print(f"- {product_info['title']} (Score: {score:.3f})")
        except:
            print(f"- Product {item_id} (Score: {score:.3f})")
    
    # Example: Get similar items
    try:
        sample_product_id = rec_system.products_df.iloc[0]['product_id']
        similar_items = rec_system.get_similar_items(sample_product_id, num_similar=3)
        print(f"\nSimilar items to {sample_product_id}:")
        for item_id, similarity in similar_items:
            product_info = rec_system.products_df[
                rec_system.products_df['product_id'] == item_id
            ].iloc[0]
            print(f"- {product_info['title']} (Similarity: {similarity:.3f})")
    except Exception as e:
        print(f"Error getting similar items: {e}")

if __name__ == "__main__":
    main()