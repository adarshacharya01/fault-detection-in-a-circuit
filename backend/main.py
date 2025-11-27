"""FastAPI main application - NO DEMO VALUES."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
from typing import Dict, Any

from .sim import simulate_netlist, get_one_cycle
from .features import extract_features
from .ml import train_svm, load_model, load_metrics, predict_fault
from .demo_data import (
    generate_training_dataset,
    load_distribution,
    generate_per_class_waveforms
)

app = FastAPI(title="Fault Detection API - IEEE Paper Implementation")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassifyRequest(BaseModel):
    netlist: str

@app.get("/")
def read_root():
    return {
        "message": "Fault Detection API - IEEE Paper Implementation",
        "version": "2.0.0",
        "status": "NO DEMO VALUES - Real simulation and SVM classification only"
    }

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/classify")
def classify_fault(request: ClassifyRequest) -> Dict[str, Any]:
    """
    Classify fault based on netlist simulation.
    
    NO DEMO VALUES. Returns HTTP 500 on error.
    
    Process:
    1. Validate netlist
    2. Load or train SVM model
    3. Simulate input netlist
    4. Extract features and classify
    5. Generate per-class waveforms
    6. Return complete response
    """
    try:
        # Step 1: Validate
        if not request.netlist or not request.netlist.strip():
            raise HTTPException(status_code=400, detail="Netlist cannot be empty")
        
        print(f"\n{'='*70}")
        print("CLASSIFICATION REQUEST")
        print(f"{'='*70}")
        
        # Step 2: Load or train model
        print("\n[1/5] Loading model and distribution...")
        model = load_model()
        metrics = load_metrics()
        distribution = load_distribution()
        
        if model is None or metrics is None or distribution is None:
            print("  Model not found. Generating training dataset...")
            
            X, y, distribution = generate_training_dataset(request.netlist, N=60)
            model, metrics = train_svm(X, y)
        else:
            print("  ✓ Loaded existing model and distribution")
        
        # Step 3: Simulate input netlist
        print("\n[2/5] Simulating input netlist...")
        measured_sim = simulate_netlist(request.netlist)
        print("  ✓ Simulation complete")
        
        # Step 4: Extract features and classify
        print("\n[3/5] Classifying...")
        features = extract_features(measured_sim)
        classification = predict_fault(model, features)
        print(f"  ✓ Classification: {classification['label']} ({classification['confidence']:.1%} confidence)")
        
        # Step 5: Generate per-class waveforms
        print("\n[4/5] Generating per-class waveforms...")
        class_waveforms = generate_per_class_waveforms(request.netlist)
        print("  ✓ Waveforms generated")
        
        # Step 6: Build response
        print("\n[5/5] Building response...")
        
        # Get time array
        t = measured_sim['t']
        
        # Build waveforms object
        waveforms = {
            "t": t,
            "vin": measured_sim['vin']
        }
        
        # Add V0 waveforms for each class
        for fault_class in ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']:
            key = f"v0_{fault_class}" if fault_class != 'E0' else "v0_healthy"
            if fault_class in class_waveforms:
                waveforms[key] = class_waveforms[fault_class]['v0']
            else:
                # Should not happen, but use measured as fallback
                waveforms[key] = measured_sim['v0']
        
        # Add V1 waveforms for each class
        for fault_class in ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']:
            key = f"v1_{fault_class}" if fault_class != 'E0' else "v1_healthy"
            if fault_class in class_waveforms:
                waveforms[key] = class_waveforms[fault_class]['v1']
            else:
                waveforms[key] = measured_sim['v1']
        
        response = {
            "classification": classification,
            "waveforms": waveforms,
            "metrics": metrics,
            "distribution": distribution
        }
        
        print("  ✓ Response ready")
        print(f"\n{'='*70}")
        print(f"SUCCESS: {classification['label']} ({classification['confidence']:.1%})")
        print(f"{'='*70}\n")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        # Log error and return HTTP 500 - NO DEMO FALLBACK
        print(f"\n{'='*70}")
        print("ERROR:")
        print(traceback.format_exc())
        print(f"{'='*70}\n")
        
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
