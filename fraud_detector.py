"""
Fraud Detection Model - Production Ready Script
Accredian Internship Task - Data Science & Machine Learning

This script provides a production-ready fraud detection system
with real-time prediction capabilities.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from typing import Dict, List, Union, Tuple
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

class FraudDetector:
    """
    Production-ready fraud detection system
    """
    
    def __init__(self, model_path: str = 'fraud_detection_model.pkl'):
        """
        Initialize the fraud detector
        
        Args:
            model_path: Path to the saved model artifacts
        """
        self.model_path = model_path
        self.model_artifacts = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoders = None
        self.performance_metrics = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model artifacts
        self.load_model()
    
    def load_model(self) -> None:
        """Load the trained model and preprocessing objects"""
        try:
            self.model_artifacts = joblib.load(self.model_path)
            self.model = self.model_artifacts['model']
            self.scaler = self.model_artifacts['scaler']
            self.feature_names = self.model_artifacts['feature_names']
            self.label_encoders = self.model_artifacts['label_encoders']
            self.performance_metrics = self.model_artifacts['performance_metrics']
            
            self.logger.info(f"Model loaded successfully: {self.model_artifacts['model_name']}")
            self.logger.info(f"Model performance - F1: {self.performance_metrics['f1_score']:.4f}")
            
        except FileNotFoundError:
            self.logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_transaction(self, transaction: Dict) -> np.ndarray:
        """
        Preprocess a single transaction for prediction
        
        Args:
            transaction: Dictionary containing transaction features
            
        Returns:
            Preprocessed feature array
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([transaction])
            
            # Feature engineering (adapt based on your actual features)
            if 'amount' in df.columns:
                df['amount_bin'] = pd.cut(df['amount'], 
                                        bins=5, labels=[0, 1, 2, 3, 4])
            
            if 'step' in df.columns:
                df['hour'] = df['step'] % 24
                df['day'] = df['step'] // 24
                df['is_weekend'] = (df['day'] % 7).isin([5, 6]).astype(int)
            
            # Balance change features
            if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
                df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
                df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
            
            # Ratio features
            if 'amount' in df.columns and 'oldbalanceOrg' in df.columns:
                df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
            
            # Apply label encoders for categorical variables
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    df[col] = encoder.transform(df[col].astype(str))
            
            # Select only the features used in training
            df_selected = df[self.feature_names]
            
            # Scale features
            features_scaled = self.scaler.transform(df_selected)
            
            return features_scaled
            
        except Exception as e:
            self.logger.error(f"Error preprocessing transaction: {str(e)}")
            raise
    
    def predict_fraud(self, transaction: Dict) -> Dict:
        """
        Predict fraud probability for a single transaction
        
        Args:
            transaction: Dictionary with transaction features
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess transaction
            features = self.preprocess_transaction(transaction)
            
            # Make prediction
            fraud_probability = self.model.predict_proba(features)[0][1]
            is_fraud = self.model.predict(features)[0]
            
            # Determine risk level
            if fraud_probability >= 0.8:
                risk_level = "HIGH"
            elif fraud_probability >= 0.5:
                risk_level = "MEDIUM"
            elif fraud_probability >= 0.2:
                risk_level = "LOW"
            else:
                risk_level = "VERY_LOW"
            
            # Create result
            result = {
                'transaction_id': transaction.get('transaction_id', 'N/A'),
                'is_fraud': bool(is_fraud),
                'fraud_probability': float(fraud_probability),
                'risk_level': risk_level,
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_artifacts['model_name']
            }
            
            # Add recommendation
            if is_fraud:
                result['recommendation'] = "BLOCK_TRANSACTION"
                result['action'] = "Immediate review required"
            elif fraud_probability >= 0.5:
                result['recommendation'] = "MANUAL_REVIEW"
                result['action'] = "Flag for manual verification"
            else:
                result['recommendation'] = "APPROVE"
                result['action'] = "Process normally"
            
            self.logger.info(f"Prediction completed - Fraud Prob: {fraud_probability:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def batch_predict(self, transactions: List[Dict]) -> List[Dict]:
        """
        Predict fraud for multiple transactions
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for transaction in transactions:
            try:
                result = self.predict_fraud(transaction)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing transaction {transaction.get('transaction_id', 'N/A')}: {str(e)}")
                # Add error result
                results.append({
                    'transaction_id': transaction.get('transaction_id', 'N/A'),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_artifacts['model_name'],
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'model_type': type(self.model).__name__
        }

def main():
    """
    Example usage of the FraudDetector
    """
    # Initialize detector
    detector = FraudDetector()
    
    # Example transaction (adapt based on your actual data structure)
    sample_transaction = {
        'transaction_id': 'TXN_001',
        'step': 1,
        'type': 'TRANSFER',
        'amount': 9000.60,
        'nameOrig': 'C1231006815',
        'oldbalanceOrg': 9000.60,
        'newbalanceOrig': 0.00,
        'nameDest': 'C1666544295',
        'oldbalanceDest': 0.00,
        'newbalanceDest': 0.00
    }
    
    # Make prediction
    try:
        result = detector.predict_fraud(sample_transaction)
        print("Fraud Detection Result:")
        print(f"Transaction ID: {result['transaction_id']}")
        print(f"Is Fraud: {result['is_fraud']}")
        print(f"Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Action: {result['action']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get model information
    model_info = detector.get_model_info()
    print(f"\nModel Information:")
    print(f"Model: {model_info['model_name']}")
    print(f"Features: {model_info['feature_count']}")
    print(f"F1-Score: {model_info['performance_metrics']['f1_score']:.4f}")

if __name__ == "__main__":
    main()