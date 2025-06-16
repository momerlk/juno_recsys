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
from recsys import ColdStartRecommendationSystem  # Your recommendation system class

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://swift:swift@hobby.nzyzrid.mongodb.net/')
MODEL_PATH = os.getenv('MODEL_PATH', '/tmp/recsystem.pkl')
RETRAIN_INTERVAL_HOURS = int(os.getenv('RETRAIN_INTERVAL_HOURS', '2'))
PORT = int(os.getenv('PORT', 8080))

# Global variables
rec_system = None
model_lock = threading.Lock()
last_training_time = None

def handle_errors(f):
    """API error handling decorator"""
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
    """Load or train the model"""
    global rec_system, last_training_time

    logger.info("Initializing recommendation system...")
    with model_lock:
        try:
            rec_system = ColdStartRecommendationSystem(mongo_uri=MONGO_URI)
            rec_system = rec_system.load(MODEL_PATH)
            logger.info("Model loaded successfully")
           
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Training new model due to loading error...")
            train_and_save_model()

def train_and_save_model():
    """Train and save model"""
    global rec_system, last_training_time

    logger.info("Starting model training...")
    start_time = time.time()

    try:
        rec_system = ColdStartRecommendationSystem(mongo_uri=MONGO_URI)
        rec_system.train_model()
        rec_system.save(MODEL_PATH)
        last_training_time = datetime.now()
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f} seconds. Saved to {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

def periodic_training():
    """Periodic retraining task"""
    logger.info("Starting periodic retraining...")
    with model_lock:
        try:
            train_and_save_model()
            logger.info("Periodic retraining completed")
        except Exception as e:
            logger.error(f"Error during periodic retraining: {e}")

def start_scheduler():
    """Background scheduler thread"""
    def scheduler_worker():
        schedule.every(RETRAIN_INTERVAL_HOURS).hours.do(periodic_training)
        while True:
            schedule.run_pending()
            time.sleep(60)

    threading.Thread(target=scheduler_worker, daemon=True).start()
    logger.info(f"Scheduler started. Retraining every {RETRAIN_INTERVAL_HOURS} hours.")

@app.get('/health')
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': rec_system is not None,
        'last_training_time': last_training_time.isoformat() if last_training_time else None
    })

@app.get('/products')
@handle_errors
def get_recommendations():
    global rec_system 

    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id parameter is required'}), 400

    try:
        num_products = int(request.args.get('num_products', 10))
        if not (1 <= num_products <= 100):
            return jsonify({'error': 'num_products must be between 1 and 100'}), 400
    except ValueError:
        return jsonify({'error': 'num_products must be an integer'}), 400



    rec_system = rec_system.load(MODEL_PATH)

    with model_lock:
        recommendations = rec_system.get_recommendations(
            user_id=user_id,
            num_recommendations=num_products,
        )

    products = []
    # for pid, score in recommendations:
    #     try:
    #         info = rec_system.products_df[rec_system.products_df['product_id'] == pid].iloc[0]
    #         products.append({
    #             'product_id': pid,
    #             'title': info['title'],
    #             'price': float(info['price']),
    #             'rating': float(info['rating']),
    #             'review_count': int(info['review_count']),
    #             'is_trending': bool(info['is_trending']),
    #             'is_featured': bool(info['is_featured']),
    #             'discounted': bool(info['discounted']),
    #             'categories': info['categories'],
    #             'recommendation_score': float(score)
    #         })
    #     except Exception as e:
    #         logger.warning(f"Missing product details for {pid}: {e}")
    #         products.append({
    #             'product_id': pid,
    #             'recommendation_score': float(score),
    #             'error': 'Product details not available'
    #         })

    return jsonify({
        'user_id': user_id,
        'num_products_requested': num_products,
        'num_products_returned': len(recommendations),
        'products': recommendations,
    })

@app.get('/similar')
@handle_errors
def get_similar_products():
    product_id = request.args.get('product_id')
    if not product_id:
        return jsonify({'error': 'product_id parameter is required'}), 400

    try:
        num_similar = int(request.args.get('num_similar', 5))
        if not (1 <= num_similar <= 50):
            return jsonify({'error': 'num_similar must be between 1 and 50'}), 400
    except ValueError:
        return jsonify({'error': 'num_similar must be an integer'}), 400

    if rec_system is None:
        return jsonify({'error': 'Recommendation model not loaded'}), 503
    if product_id not in rec_system.item_id_map:
        return jsonify({'error': f'Product {product_id} not found'}), 404

    with model_lock:
        similar_items = rec_system.get_similar_items(product_id, num_similar)

    similar_products = []
    for sid, score in similar_items:
        try:
            info = rec_system.products_df[rec_system.products_df['product_id'] == sid].iloc[0]
            similar_products.append({
                'product_id': sid,
                'title': info['title'],
                'price': float(info['price']),
                'rating': float(info['rating']),
                'review_count': int(info['review_count']),
                'is_trending': bool(info['is_trending']),
                'is_featured': bool(info['is_featured']),
                'discounted': bool(info['discounted']),
                'categories': info['categories'],
                'similarity_score': float(score)
            })
        except Exception as e:
            logger.warning(f"Missing similar product details for {sid}: {e}")
            similar_products.append({
                'product_id': sid,
                'similarity_score': float(score),
                'error': 'Product details not available'
            })

    return jsonify({
        'original_product': product_id,
        'num_similar_requested': num_similar,
        'num_similar_returned': len(similar_products),
        'similar_products': similar_products
    })

@app.post('/retrain')
@handle_errors
def manual_retrain():
    if rec_system is None:
        return jsonify({'error': 'No model to retrain'}), 503

    def retrain_worker():
        try:
            periodic_training()
        except Exception as e:
            logger.error(f"Background retraining failed: {e}")

    threading.Thread(target=retrain_worker).start()
    return jsonify({
        'message': 'Model retraining started in background',
        'last_training_time': last_training_time.isoformat() if last_training_time else None
    })

@app.get('/model/info')
@handle_errors
def get_model_info():
    if rec_system is None:
        return jsonify({'error': 'No model loaded'}), 503

    with model_lock:
        return jsonify({
            'model_loaded': True,
            'last_training_time': last_training_time.isoformat() if last_training_time else None,
            'num_users': len(rec_system.users_df) if hasattr(rec_system, 'users_df') else 0,
            'num_products': len(rec_system.products_df) if hasattr(rec_system, 'products_df') else 0,
            'num_interactions': len(rec_system.interactions_df) if hasattr(rec_system, 'interactions_df') else 0,
            'retrain_interval_hours': RETRAIN_INTERVAL_HOURS,
            'model_path': MODEL_PATH
        })


if __name__ == '__main__':
    # Initialize the model and scheduler *before* starting the app
    load_or_train_model()
    start_scheduler()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)

