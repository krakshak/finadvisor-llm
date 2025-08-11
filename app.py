"""
Flask frontend for FinAdvisor - Financial Advisory Tool
"""

from flask import Flask, render_template, request, jsonify, session
import logging
import os
from inference_pipeline.inference_engine import get_inference_engine, switch_model
from inference_pipeline.model_loader import get_available_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'finadvisor-secret-key-change-in-production')

@app.route('/')
def index():
    """Main page with the financial advisory interface."""
    available_models = get_available_models()
    return render_template('index.html', available_models=available_models)

@app.route('/api/generate', methods=['POST'])
def generate_advice():
    """API endpoint to generate financial advice."""
    try:
        data = request.get_json()
        
        # Extract parameters from request
        instruction = data.get('instruction', '').strip()
        input_text = data.get('input', '').strip()
        model_name = data.get('model', 'LLaMa 7B')
        temperature = float(data.get('temperature', 0.6))
        top_p = float(data.get('top_p', 0.95))
        max_tokens = int(data.get('max_tokens', 128))
        repetition_penalty = float(data.get('repetition_penalty', 1.15))
        
        # Validate instruction
        if not instruction:
            return jsonify({
                'error': 'Instruction is required',
                'success': False
            }), 400
        
        logger.info(f"Generating advice using {model_name} for instruction: {instruction[:100]}...")
        
        # Get inference engine for the selected model
        inference_engine = get_inference_engine(model_name)
        
        # Generate response using inference engine
        response = inference_engine.evaluate(
            instruction=instruction,
            input_text=input_text if input_text and input_text.lower() != 'none' else None,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        )
        
        return jsonify({
            'response': response,
            'success': True,
            'model': model_name,
            'parameters': {
                'temperature': temperature,
                'top_p': top_p,
                'max_tokens': max_tokens,
                'repetition_penalty': repetition_penalty
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating advice: {str(e)}")
        return jsonify({
            'error': f'Error generating advice: {str(e)}',
            'success': False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check if inference engine is initialized
        if not inference_engine._initialized:
            return jsonify({
                'status': 'initializing',
                'message': 'Model is still loading...'
            }), 202
        
        return jsonify({
            'status': 'healthy',
            'message': 'FinAdvisor API is running'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/initialize', methods=['POST'])
def initialize_model():
    """Initialize the model (can be called to warm up the model)."""
    try:
        logger.info("Initializing model...")
        inference_engine.initialize()
        return jsonify({
            'success': True,
            'message': 'Model initialized successfully'
        })
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Initialize the model on startup (optional - can be done on first request)
    logger.info("Starting FinAdvisor Flask application...")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
