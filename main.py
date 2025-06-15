from flask import Flask, request, jsonify
import os
import pickle
import threading
import time
import schedule
from datetime import datetime
import logging
from functools import wraps
import traceback
from recsys import ColdStartRecommendationSystem  # Import your recommendation system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://swift:swift@hobby.nzyzrid.mongodb.net/')
MODEL_PATH = os.getenv('MODEL_PATH', '/tmp/recommendation_model.pkl')
RETRAIN_INTERVAL_HOURS = int(os.getenv('RETRAIN_INTERVAL_HOURS', '6'))  # Retrain every 6 hours
PORT = int(os.getenv('PORT', 8080))

# Global variables
rec_system = None
model_lock = threading.Lock()
last_training_time = None

def handle_errors(f):
    """Decorator to handle API errors gracefully"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Internal server error',
                'message': str(e)
            }), 500
    return decorated_function

def load_or_train_model():
    """Load existing model or train a new one"""
    global rec_system, last_training_time
    
    logger.info("Initializing recommendation system...")
    
    with model_lock:
        try:
            # Try to load existing model
            if os.path.exists(MODEL_PATH):
                logger.info(f"Loading existing model from {MODEL_PATH}")
                with open(MODEL_PATH, 'rb') as f:
                    rec_system = pickle.load(f)
                logger.info("Model loaded successfully")
            else:
                logger.info("No existing model found, training new model...")
                train_and_save_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Training new model due to loading error...")
            train_and_save_model()

def train_and_save_model():
    """Train the model and save it"""
    global rec_system, last_training_time
    
    logger.info("Starting model training...")
    start_time = time.time()
    
    try:
        # Initialize and train the recommendation system
        rec_system = ColdStartRecommendationSystem(mongo_uri=MONGO_URI)
        rec_system.train_model()
        
        # Save the trained model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(rec_system, f)
        
        last_training_time = datetime.now()
        training_time = time.time() - start_time
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        logger.info(f"Model saved to {MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def periodic_training():
    """Function to run periodic model retraining"""
    logger.info("Starting periodic training...")
    
    with model_lock:
        try:
            train_and_save_model()
            logger.info("Periodic training completed successfully")
        except Exception as e:
            logger.error(f"Error during periodic training: {e}")

def start_scheduler():
    """Start the background scheduler for periodic training"""
    def run_scheduler():
        schedule.every(RETRAIN_INTERVAL_HOURS).hours.do(periodic_training)
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info(f"Scheduler started - model will retrain every {RETRAIN_INTERVAL_HOURS} hours")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': rec_system is not None,
        'last_training_time': last_training_time.isoformat() if last_training_time else None
    })

@app.route('/products', methods=['GET'])
@handle_errors
def get_recommendations():
    """
    Get product recommendations for a user
    Query parameters:
    - user_id (required): User ID
    - num_products (optional): Number of products to return (default: 10)
    - include_cold_start (optional): Whether to handle cold start users (default: true)
    """
    # Validate required parameters
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id parameter is required'}), 400
    
    # Get optional parameters
    try:
        num_products = int(request.args.get('num_products', 10))
        if num_products <= 0 or num_products > 100:
            return jsonify({'error': 'num_products must be between 1 and 100'}), 400
    except ValueError:
        return jsonify({'error': 'num_products must be a valid integer'}), 400
    
    include_cold_start = request.args.get('include_cold_start', 'true').lower() == 'true'
    
    # Check if model is loaded
    if rec_system is None:
        return jsonify({'error': 'Recommendation model not loaded'}), 503
    
    # Get recommendations
    with model_lock:
        recommendations = rec_system.get_recommendations(
            user_id=user_id,
            num_recommendations=num_products,
            include_cold_start=include_cold_start
        )
    
    # Format response
    products = []
    for product_id, score in recommendations:
        try:
            # Get product details
            product_info = rec_system.products_df[
                rec_system.products_df['product_id'] == product_id
            ].iloc[0]
            
            products.append({
                'product_id': product_id,
                'title': product_info['title'],
                'price': float(product_info['price']),
                'rating': float(product_info['rating']),
                'review_count': int(product_info['review_count']),
                'is_trending': bool(product_info['is_trending']),
                'is_featured': bool(product_info['is_featured']),
                'discounted': bool(product_info['discounted']),
                'categories': product_info['categories'],
                'recommendation_score': float(score)
            })
        except Exception as e:
            logger.warning(f"Error getting details for product {product_id}: {e}")
            products.append({
                'product_id': product_id,
                'recommendation_score': float(score),
                'error': 'Product details not available'
            })
    
    return jsonify({
        'user_id': user_id,
        'num_products_requested': num_products,
        'num_products_returned': len(products),
        'products': products,
        'is_cold_start': user_id not in rec_system.user_id_map if rec_system else False
    })

@app.route('/similar', methods=['GET'])
@handle_errors
def get_similar_products():
    """
    Get similar products for a given product
    Query parameters:
    - product_id (required): Product ID
    - num_similar (optional): Number of similar products to return (default: 5)
    """
    # Validate required parameters
    product_id = request.args.get('product_id')
    if not product_id:
        return jsonify({'error': 'product_id parameter is required'}), 400
    
    # Get optional parameters
    try:
        num_similar = int(request.args.get('num_similar', 5))
        if num_similar <= 0 or num_similar > 50:
            return jsonify({'error': 'num_similar must be between 1 and 50'}), 400
    except ValueError:
        return jsonify({'error': 'num_similar must be a valid integer'}), 400
    
    # Check if model is loaded
    if rec_system is None:
        return jsonify({'error': 'Recommendation model not loaded'}), 503
    
    # Check if product exists
    if product_id not in rec_system.item_id_map:
        return jsonify({'error': f'Product {product_id} not found in the system'}), 404
    
    # Get similar products
    with model_lock:
        similar_items = rec_system.get_similar_items(
            item_id=product_id,
            num_similar=num_similar
        )
    
    # Format response
    similar_products = []
    for similar_product_id, similarity_score in similar_items:
        try:
            # Get product details
            product_info = rec_system.products_df[
                rec_system.products_df['product_id'] == similar_product_id
            ].iloc[0]
            
            similar_products.append({
                'product_id': similar_product_id,
                'title': product_info['title'],
                'price': float(product_info['price']),
                'rating': float(product_info['rating']),
                'review_count': int(product_info['review_count']),
                'is_trending': bool(product_info['is_trending']),
                'is_featured': bool(product_info['is_featured']),
                'discounted': bool(product_info['discounted']),
                'categories': product_info['categories'],
                'similarity_score': float(similarity_score)
            })
        except Exception as e:
            logger.warning(f"Error getting details for similar product {similar_product_id}: {e}")
            similar_products.append({
                'product_id': similar_product_id,
                'similarity_score': float(similarity_score),
                'error': 'Product details not available'
            })
    
    # Get original product details
    try:
        original_product = rec_system.products_df[
            rec_system.products_df['product_id'] == product_id
        ].iloc[0]
        original_product_info = {
            'product_id': product_id,
            'title': original_product['title'],
            'price': float(original_product['price']),
            'rating': float(original_product['rating']),
            'categories': original_product['categories']
        }
    except Exception as e:
        logger.warning(f"Error getting original product details: {e}")
        original_product_info = {'product_id': product_id}
    
    return jsonify({
        'original_product': original_product_info,
        'num_similar_requested': num_similar,
        'num_similar_returned': len(similar_products),
        'similar_products': similar_products
    })

@app.route('/retrain', methods=['POST'])
@handle_errors
def manual_retrain():
    """Manually trigger model retraining"""
    if rec_system is None:
        return jsonify({'error': 'No model to retrain'}), 503
    
    # Start retraining in background
    def background_retrain():
        try:
            periodic_training()
        except Exception as e:
            logger.error(f"Background retraining failed: {e}")
    
    retrain_thread = threading.Thread(target=background_retrain)
    retrain_thread.start()
    
    return jsonify({
        'message': 'Model retraining started in background',
        'last_training_time': last_training_time.isoformat() if last_training_time else None
    })

@app.route('/model/info', methods=['GET'])
@handle_errors
def get_model_info():
    """Get information about the current model"""
    if rec_system is None:
        return jsonify({'error': 'No model loaded'}), 503
    
    with model_lock:
        info = {
            'model_loaded': True,
            'last_training_time': last_training_time.isoformat() if last_training_time else None,
            'num_users': len(rec_system.users_df) if hasattr(rec_system, 'users_df') else 0,
            'num_products': len(rec_system.products_df) if hasattr(rec_system, 'products_df') else 0,
            'num_interactions': len(rec_system.interactions_df) if hasattr(rec_system, 'interactions_df') else 0,
            'retrain_interval_hours': RETRAIN_INTERVAL_HOURS,
            'model_path': MODEL_PATH
        }
    
    return jsonify(info)

@app.before_first_request
def initialize():
    """Initialize the application"""
    load_or_train_model()
    start_scheduler()

if __name__ == '__main__':
    # Initialize the model and scheduler
    load_or_train_model()
    start_scheduler()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)