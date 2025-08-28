"""
Fraud Detection API Server
Simple Flask API for serving fraud detection predictions

Usage:
    python api_server.py

Endpoints:
    POST /predict - Single transaction prediction
    POST /batch_predict - Multiple transaction predictions
    GET /health - Health check
    GET /model_info - Model information
"""

from flask import Flask, request, jsonify
from fraud_detector import FraudDetector
import logging
import traceback
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize fraud detector
try:
    detector = FraudDetector()
    logger.info("Fraud detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize fraud detector: {str(e)}")
    detector = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if detector is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Fraud detector not initialized',
            'timestamp': datetime.now().isoformat()
        }), 500
    
    return jsonify({
        'status': 'healthy',
        'message': 'Fraud detection API is running',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': detector.model is not None
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if detector is None:
        return jsonify({'error': 'Fraud detector not initialized'}), 500
    
    try:
        info = detector.get_model_info()
        return jsonify({
            'status': 'success',
            'model_info': info,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
def predict_single():
    """Predict fraud for a single transaction"""
    if detector is None:
        return jsonify({'error': 'Fraud detector not initialized'}), 500
    
    try:
        # Get transaction data from request
        transaction = request.get_json()
        
        if not transaction:
            return jsonify({
                'status': 'error',
                'message': 'No transaction data provided'
            }), 400
        
        # Make prediction
        result = detector.predict_fraud(transaction)
        
        return jsonify({
            'status': 'success',
            'prediction': result
        })
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/batch_predict', methods=['POST'])
def predict_batch():
    """Predict fraud for multiple transactions"""
    if detector is None:
        return jsonify({'error': 'Fraud detector not initialized'}), 500
    
    try:
        # Get transactions data from request
        data = request.get_json()
        transactions = data.get('transactions', [])
        
        if not transactions:
            return jsonify({
                'status': 'error',
                'message': 'No transactions provided'
            }), 400
        
        if len(transactions) > 100:
            return jsonify({
                'status': 'error',
                'message': 'Maximum 100 transactions per batch'
            }), 400
        
        # Make predictions
        results = detector.batch_predict(transactions)
        
        return jsonify({
            'status': 'success',
            'predictions': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error making batch predictions: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    # Example usage and testing
    print("Starting Fraud Detection API Server...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /model_info - Model information")
    print("  POST /predict - Single transaction prediction")
    print("  POST /batch_predict - Batch transaction predictions")
    print()
    print("Example curl commands:")
    print("curl -X GET http://localhost:5000/health")
    print("curl -X GET http://localhost:5000/model_info")
    print("""curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "transaction_id": "TXN_001",
    "step": 1,
    "type": "TRANSFER",
    "amount": 9000.60,
    "nameOrig": "C1231006815",
    "oldbalanceOrg": 9000.60,
    "newbalanceOrig": 0.00,
    "nameDest": "C1666544295",
    "oldbalanceDest": 0.00,
    "newbalanceDest": 0.00
  }'""")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)