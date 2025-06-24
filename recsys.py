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
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class ColdStartRecommendationSystem:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="juno_db",
                 freshness_weight=0.2, diversity_weight=0.3, recency_weight=0.1):
        """
        Initialize the recommendation system.
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client = None
        self.db = None
        
        self.freshness_weight = freshness_weight
        self.diversity_weight = diversity_weight
        self.recency_weight = recency_weight

        self.dataset = Dataset()
        self.model = None
        
        self.user_features_list = []
        self.item_features_list = []
        self.top_categories = []

    def _connect_mongo(self):
        """Establish MongoDB connection if not already established."""
        if not self.client:
            print("Establishing MongoDB connection...")
            self.client = MongoClient(self.mongo_uri, unicode_decode_error_handler='ignore')
            self.db = self.client[self.db_name]
            print("MongoDB connection established.")

    def __getstate__(self):
        """
        Prepare the object's state for pickling.
        We now KEEP the feature matrices and item metadata for fast prediction times after loading.
        """
        state = self.__dict__.copy()
        del state['client']
        del state['db']
        return state

    def __setstate__(self, state):
        """Restore the object's state from a pickled state."""
        self.__dict__.update(state)
        self.client = None 
        self.db = None

    def _cache_item_metadata(self):
        """Caches item metadata. Runs only if the attribute doesn't exist (for backward compatibility)."""
        if hasattr(self, 'item_metadata') and self.item_metadata:
            return

        self._connect_mongo()
        print("Backward compatibility: Caching item metadata for re-ranking...")
        self.item_metadata = {
            p['id']: {
                'created_at': p.get('createdAt', datetime.now() - timedelta(days=365)),
                'categories': [cat.get('name', '') for cat in p.get('categories', [])]
            }
            for p in self.db['products'].find({}, {'id': 1, 'createdAt': 1, 'categories.name': 1, '_id': 0})
        }
        print(f"Cached metadata for {len(self.item_metadata)} items.")

    def _ensure_feature_matrices(self):
        """Builds feature matrices. Runs only if attributes don't exist (for backward compatibility)."""
        if hasattr(self, 'user_features_matrix') and self.user_features_matrix is not None:
            return

        print("Backward compatibility: Rebuilding feature matrices from dataset...")
        self._connect_mongo()
        self.user_features_matrix = self.dataset.build_user_features(self._user_features_generator())
        self.item_features_matrix = self.dataset.build_item_features(self._item_features_generator())
        print("Feature matrices rebuilt successfully.")

    def _fetch_and_prepare_features(self):
        """Efficiently prepares features and caches metadata during training."""
        self._connect_mongo()
        print("Preparing user and item features from MongoDB...")

        user_features = set()
        for feature in ['gender', 'role', 'account_status']:
            for value in self.db['users'].distinct(feature):
                user_features.add(f"{feature}:{value or 'unknown'}")
        for label in ['teen', 'young', 'adult', 'middle', 'senior']: user_features.add(f"age:{label}")
        self.user_features_list = sorted(list(user_features))

        item_features = set()
        for ptype in self.db['products'].distinct('product_type'):
            if ptype: item_features.add(f"product_type:{ptype}")
        for label in ['budget', 'affordable', 'mid_range', 'premium', 'luxury']: item_features.add(f"price_range:{label}")
        for feature in ['discounted', 'is_trending', 'is_featured']:
            item_features.add(f"{feature}:True"); item_features.add(f"{feature}:False")
        for label in ['low', 'medium', 'good', 'excellent']: item_features.add(f"rating:{label}")
        
        pipeline = [
            {"$unwind": "$categories"},
            {"$group": {"_id": "$categories.name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 20}
        ]
        self.top_categories = [doc['_id'] for doc in self.db['products'].aggregate(pipeline)]
        for cat in self.top_categories:
            item_features.add(f"category:{cat}")
        self.item_features_list = sorted(list(item_features))
        
        # We still cache metadata during training.
        self._cache_item_metadata()

    # The generator methods remain unchanged
    def _user_features_generator(self):
        """A generator that yields user features one by one from the DB."""
        for user in self.db['users'].find({}, {'_id': 0}):
            features = set()
            for col in ['gender', 'role', 'account_status']:
                features.add(f"{col}:{user.get(col, 'unknown')}")
            age_map = {range(0, 21): 'teen', range(21, 31): 'young', range(31, 41): 'adult', range(41, 51): 'middle'}
            features.add(f"age:{next((v for k, v in age_map.items() if user.get('age', 25) in k), 'senior')}")
            yield (user['id'], list(features))

    def _item_features_generator(self):
        """A generator that yields item features one by one from the DB."""
        for prod in self.db['products'].find({}, {'_id': 0}):
            features = set()
            if prod.get('product_type'):
                features.add(f"product_type:{prod['product_type']}")
            price = prod.get('pricing', {}).get('price', 0)
            price_map = {range(0, 2001): 'budget', range(2001, 5001): 'affordable', range(5001, 10001): 'mid_range', range(10001, 20001): 'premium'}
            features.add(f"price_range:{next((v for k, v in price_map.items() if price in k), 'luxury')}")
            yield (prod['id'], list(features))
    
    def _interactions_generator(self):
        """A generator that yields interaction tuples from the DB."""
        rating_map = {'like': 1.0, 'dislike': 0.1, 'view': 0.5, 'purchase': 2.0, 'add_to_cart': 1.5}
        for interaction in self.db['interactions'].find({}, {'_id': 0}):
            rating = interaction.get('rating', 0)
            if rating == 0:
                rating = rating_map.get(interaction.get('action_type', 'view'), 0.5)
            yield (interaction['user_id'], interaction['product_id'], rating)

    def train_model(self):
        """Train the LightFM model by streaming data from MongoDB."""
        print("Starting memory-efficient model training process...")
        self._fetch_and_prepare_features()

        self.dataset.fit(
            users=(u['id'] for u in self.db['users'].find({}, {'id': 1, '_id': 0})),
            items=(p['id'] for p in self.db['products'].find({}, {'id': 1, '_id': 0})),
            user_features=self.user_features_list,
            item_features=self.item_features_list
        )
        
        print("Building interaction matrix from stream...")
        interactions_matrix, _ = self.dataset.build_interactions(self._interactions_generator())
        
        # Build the feature matrices and they will be saved with the model
        self._ensure_feature_matrices()
        
        self.model = LightFM(loss='warp', learning_rate=0.05, no_components=50, random_state=42)
        self.model.fit(
            interactions_matrix,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix,
            epochs=50,
            num_threads=4,
            verbose=True
        )
        print("Model training completed!")

    def save(self, filepath='recommendation_system.pkl'):
        """Saves the recommendation system object to a file."""
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

    # The reranking and interaction fetching methods remain unchanged
    def _get_realtime_user_interactions(self, user_id):
        self._connect_mongo()
        cursor = self.db['interactions'].find({'user_id': user_id}, {'product_id': 1, '_id': 0})
        return {i['product_id'] for i in cursor}

    def _apply_reranking(self, recommendations, num_recommendations):
        if not recommendations:
            return []
        # ... logic is the same ...
        max_score = recommendations[0][1]
        if max_score == 0: max_score = 1.0
        final_recommendations = []
        for item_id, score in recommendations:
            product_info = self.item_metadata.get(item_id)
            if not product_info: continue
            created_at = product_info['created_at']
            now = datetime.now(created_at.tzinfo)
            days_since_creation = (now - created_at).days
            freshness_score = max(0, 1 - (days_since_creation / 365))
            product_categories = set(product_info['categories'])
            diversity_penalty = sum(0.1 for _, _, rec_cats in final_recommendations if product_categories.intersection(rec_cats))
            diversity_score = 1 - diversity_penalty
            normalized_score = score / max_score
            combined_score = ((1 - self.freshness_weight - self.diversity_weight) * normalized_score +
                               self.freshness_weight * freshness_score +
                               self.diversity_weight * diversity_score)
            final_recommendations.append((item_id, combined_score, product_categories))
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        return [(item_id, score) for item_id, score, _ in final_recommendations[:num_recommendations]]

    def _get_recommendations_for_warm_user(self, user_interactions, num_recommendations):
        """
        Handles the 'warm start' scenario. The user has interactions but is not in the trained model.
        Generates recommendations by creating an inferred user profile from their interacted items.
        """
        print(f"Warm start for user. Generating recommendations based on {len(user_interactions)} interactions.")
        
        # 1. Get the mapping and the model's item embeddings
        # We need the item_id_map to link product IDs to the model's internal indices.
        _, _, item_id_map, _ = self.dataset.mapping()
        _, item_embeddings = self.model.get_item_representations(features=self.item_features_matrix)

        # 2. Find the indices for the items the user has interacted with
        interacted_item_indices = [item_id_map[item_id] for item_id in user_interactions if item_id in item_id_map]

        if not interacted_item_indices:
            # This can happen if the user interacted with items that have since been removed or were not in training.
            # We fall back to a standard cold start.
            print("Warm start user's items not found in model. Falling back to cold start.")
            return self._get_recommendations_for_cold_user(num_recommendations)

        # 3. Create the user's inferred 'taste' vector by averaging the embeddings of items they've interacted with.
        inferred_user_embedding = np.mean(item_embeddings[interacted_item_indices], axis=0)

        # 4. Calculate scores for all items by taking the dot product of the inferred user vector and all item embeddings.
        all_scores = np.dot(item_embeddings, inferred_user_embedding)

        # 5. Get top items, excluding those already interacted with
        # Use the pre-calculated inverse map for efficiency
        top_indices = np.argsort(all_scores)[::-1]
        
        # We should have this pre-calculated now, but just in case.
        if not hasattr(self, 'inverse_item_map'):
            self.inverse_item_map = {v: k for k, v in item_id_map.items()}
            
        predictions = []
        for index in top_indices:
            item_id = self.inverse_item_map[index]
            if item_id not in user_interactions:
                predictions.append((item_id, float(all_scores[index])))
            if len(predictions) >= max(num_recommendations * 3, 50):
                break
                
        # Apply the same re-ranking logic for diversity, freshness etc.
        reranked_predictions = self._apply_reranking(predictions, num_recommendations)
        product_ids = [pid for pid, score in reranked_predictions]
        
        return product_ids

    # Optional: Refactor the cold-start logic into its own method for clarity
    def _get_recommendations_for_cold_user(self, num_recommendations):
        """
        Handles the true 'cold start' scenario for users with no interactions.
        """
        print(f"True cold start. Recommending popular items.")
        popular_items_cursor = self.db['products'].find(
            {}, {'id': 1, 'rating': 1, '_id': 0}
        ).sort([('is_trending', -1), ('review_count', -1), ('createdAt', -1)]).limit(100)
        
        predictions = [(item['id'], item.get('rating', 0)) for item in popular_items_cursor]
        reranked_predictions = self._apply_reranking(predictions, num_recommendations)
        product_ids = [pid for pid, score in reranked_predictions]
        return product_ids

    # This is the MODIFIED get_recommendations function

    def get_recommendations(self, user_id, num_recommendations=10):
        """
        Get recommendations for a user. Handles Hot, Warm, and Cold start scenarios.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please call train_model() or load().")

        self._cache_item_metadata()
        self._ensure_feature_matrices()
        self._connect_mongo() # Connect once at the beginning
        
        user_id_map, _, item_id_map, _ = self.dataset.mapping()

        # --- HOT START: User is in the trained model ---
        if user_id in user_id_map:
            user_index = user_id_map[user_id]
            all_item_indices = np.arange(len(item_id_map))
            
            scores = self.model.predict(
                user_index,
                all_item_indices,
                user_features=self.user_features_matrix,
                item_features=self.item_features_matrix,
                num_threads=4
            )

            
            if not hasattr(self, 'inverse_item_map'):
                self.inverse_item_map = {v: k for k, v in item_id_map.items()}

            user_interactions = self._get_realtime_user_interactions(user_id)
            
            top_items = sorted(
                [(self.inverse_item_map[i], scores[i]) for i in all_item_indices], 
                key=lambda x: x[1], 
                reverse=True
            )

            predictions = [(item_id, float(score)) for item_id, score in top_items if item_id not in user_interactions]

            candidates_for_reranking = predictions[:max(num_recommendations * 3, 50)]
            reranked_predictions = self._apply_reranking(candidates_for_reranking, num_recommendations)
            product_ids = [pid for pid, score in reranked_predictions]

        # --- WARM/COLD START: User is NOT in the trained model ---
        else:
            # Check if this "unknown" user has any interactions in the DB
            user_interactions = self._get_realtime_user_interactions(user_id)
            if user_interactions:
                # WARM START: User has interactions, so we can infer their preferences
                product_ids = self._get_recommendations_for_warm_user(user_interactions, num_recommendations)
            else:
                # TRUE COLD START: User has no interactions, recommend popular items
                product_ids = self._get_recommendations_for_cold_user(num_recommendations)

        # --- Final Step: Fetch product details for the recommended IDs ---
        if not product_ids:
            return []

        products_cursor = self.db['products'].find({'id': {'$in': product_ids}}, {'_id': 0})
        product_map = {p['id']: p for p in products_cursor}
        ordered_products = [product_map[pid] for pid in product_ids if pid in product_map]
        
        return ordered_products


# --- Example Usage ---
def main():
    MODEL_PATH = 'cold_start_recsys_mem_efficient.pkl'
    
    rec_system = ColdStartRecommendationSystem(mongo_uri="mongodb+srv://swift:swift@hobby.nzyzrid.mongodb.net/")
    
    # --- Train and Save ---
    print("--- Training and Saving Model ---")
    rec_system.train_model()
    rec_system.save(MODEL_PATH)
    
    # --- Load and Predict ---
    print("\n--- Loading Model and Making Predictions ---")
    try:
        loaded_rec_system = ColdStartRecommendationSystem.load(MODEL_PATH)

        loaded_rec_system._connect_mongo()
        user_doc = loaded_rec_system.db['users'].find_one()
        if user_doc:
            existing_user_id = user_doc['id']
            recommendations = loaded_rec_system.get_recommendations(existing_user_id, num_recommendations=5)
            print(f"\nRecommendations for existing user '{existing_user_id}':")
            for product in recommendations:
                print(f"- {product.get('title', 'N/A')} (ID: {product.get('id', 'N/A')})")
        else:
            print("No users found in the database to test existing user recommendations.")

        cold_start_user_id = "new_user_who_does_not_exist"
        cold_recommendations = loaded_rec_system.get_recommendations(cold_start_user_id, num_recommendations=5)
        print(f"\nCold start recommendations for user '{cold_start_user_id}':")
        for product in cold_recommendations:
            print(f"- {product.get('title', 'N/A')} (ID: {product.get('id', 'N/A')})")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()