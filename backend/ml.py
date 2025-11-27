"""Machine learning module for SVM training and prediction."""
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from typing import Dict, Any, Tuple

FAULT_CLASSES = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']

def train_svm(X: np.ndarray, y: np.ndarray) -> Tuple[SVC, Dict[str, Any]]:
    """
    Train SVM classifier for fault detection.
    
    Following IEEE paper: RBF kernel SVM with probability support.
    
    Args:
        X: Feature matrix (n_samples, 12)
        y: Labels (n_samples,) with values 0-5
    
    Returns:
        (trained_model, metrics_dict)
    """
    print(f"\n{'='*60}")
    print("Training SVM classifier")
    print(f"{'='*60}\n")
    
    # Split data (80/20 train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train SVM with RBF kernel
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("✓ Training complete")
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=FAULT_CLASSES, output_dict=True)
    
    # Extract per-class F1 scores
    per_class = {}
    for cls in FAULT_CLASSES:
        if cls in class_report:
            per_class[cls] = class_report[cls]['f1-score']
    
    metrics = {
        'accuracy': float(accuracy),
        'per_class': per_class,
        'confusion': conf_matrix.tolist()
    }
    
    print(f"\n{'='*60}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Per-class F1 scores:")
    for cls, score in per_class.items():
        print(f"  {cls}: {score:.3f}")
    print(f"{'='*60}\n")
    
    # Save model and metrics
    save_model(model, metrics)
    
    return model, metrics

def save_model(model: SVC, metrics: Dict[str, Any]) -> None:
    """Save trained model and metrics."""
    os.makedirs('backend/models', exist_ok=True)
    
    joblib.dump(model, 'backend/models/svm_model.joblib')
    joblib.dump(metrics, 'backend/models/metrics.joblib')
    
    print("✓ Model saved to backend/models/svm_model.joblib")
    print("✓ Metrics saved to backend/models/metrics.joblib")

def load_model() -> SVC:
    """Load trained model from file."""
    path = 'backend/models/svm_model.joblib'
    
    if os.path.exists(path):
        print(f"✓ Loading model from {path}")
        return joblib.load(path)
    
    return None

def load_metrics() -> Dict[str, Any]:
    """Load metrics from file."""
    path = 'backend/models/metrics.joblib'
    
    if os.path.exists(path):
        return joblib.load(path)
    
    return None

def predict_fault(model: SVC, features: np.ndarray) -> Dict[str, Any]:
    """
    Predict fault class from features.
    
    Args:
        model: Trained SVM model
        features: Feature vector (12 features)
    
    Returns:
        Dict with label and confidence
    """
    # Reshape for prediction
    features = features.reshape(1, -1)
    
    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    return {
        "label": FAULT_CLASSES[prediction],
        "confidence": float(probabilities[prediction])
    }
