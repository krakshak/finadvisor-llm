# FinAdvisor Inference Pipeline & Flask Frontend

This project provides a modular inference pipeline and Flask-based web frontend for the FinAdvisor LLM, migrated from the original Gradio notebook interface.

## 🏗️ Project Structure

```
finadvisor-llm/
├── inference_pipeline/          # Modular inference pipeline
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── model_loader.py         # Model loading utilities
│   ├── prompt_utils.py         # Prompt formatting functions
│   └── inference_engine.py     # Main inference engine
├── templates/                  # Flask HTML templates
│   └── index.html             # Main web interface
├── static/                    # Static web assets
│   ├── css/
│   │   └── style.css          # Custom styling
│   └── js/
│       └── app.js             # Frontend JavaScript
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
└── README_INFERENCE.md        # This file
```

## 🚀 Features

### Inference Pipeline
- **Modular Design**: Separated concerns into distinct modules
- **Model Loading**: Automatic LLaMA 7B + LoRA weights loading
- **Device Detection**: Automatic CUDA/MPS/CPU device selection
- **Optimization**: 8-bit quantization, model compilation with torch.compile
- **Flexible Parameters**: Temperature, top-p, max tokens, repetition penalty

### Flask Frontend
- **Modern UI**: Bootstrap-based responsive design with gradient backgrounds
- **Real-time Parameter Control**: Interactive sliders for generation parameters
- **Model Selection**: Dropdown for different model variants
- **Error Handling**: Comprehensive error messages and loading states
- **Health Monitoring**: API health checks and model initialization status
- **Example Prompts**: Built-in example financial questions

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM (for model loading)
- Dependencies listed in `requirements.txt`

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Model Access**:
   Ensure you have access to:
   - Base model: `baffo32/decapoda-research-llama-7B-hf`
   - LoRA weights: `kunchum/capstone-llama-finetuned`

## 🎯 Usage

### Running the Flask Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   Open your browser to `http://localhost:5000`

3. **Generate Financial Advice**:
   - Enter your financial question in the "Instruction" field
   - Optionally provide additional context in the "Input" field
   - Adjust generation parameters using the sliders
   - Click "Generate Advice" to get your response

### Using the Inference Pipeline Directly

```python
from inference_pipeline.inference_engine import get_inference_engine

# Initialize the engine
engine = get_inference_engine()
engine.initialize()

# Generate advice
response = engine.generate_response(
    instruction="When should I get a second credit card?",
    input_text="I currently have one credit card with a $5000 limit",
    temperature=0.6,
    max_new_tokens=128
)

print(response)
```

## 🔧 Configuration

### Model Configuration (`inference_pipeline/config.py`)

- **BASE_MODEL**: HuggingFace model identifier
- **LORA_WEIGHTS**: LoRA adapter weights identifier
- **DEVICE**: Auto-detected (cuda/mps/cpu)
- **DEFAULT_GENERATION_CONFIG**: Default generation parameters

### Flask Configuration (`app.py`)

- **Host**: `0.0.0.0` (accessible from network)
- **Port**: `5000`
- **Debug**: `True` (disable in production)
- **Threading**: `True` (for concurrent requests)

## 🌐 API Endpoints

### POST `/api/generate`
Generate financial advice based on user input.

**Request Body**:
```json
{
    "instruction": "When should I get a second credit card?",
    "input": "Additional context (optional)",
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 128,
    "repetition_penalty": 1.15
}
```

**Response**:
```json
{
    "success": true,
    "response": "Generated financial advice...",
    "parameters": {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 128,
        "repetition_penalty": 1.15
    }
}
```

### GET `/api/health`
Check the health status of the API and model.

**Response**:
```json
{
    "status": "healthy",
    "message": "FinAdvisor API is running"
}
```

### POST `/api/initialize`
Manually initialize the model (useful for warming up).

## 🎨 Frontend Features

### Interactive Parameters
- **Temperature**: Controls randomness (0-1)
- **Top-p**: Nucleus sampling threshold (0-1)
- **Max Tokens**: Maximum response length (1-512)
- **Repetition Penalty**: Prevents repetitive text (0.1-2.0)

### User Experience
- **Loading States**: Visual feedback during generation
- **Error Handling**: Clear error messages for troubleshooting
- **Responsive Design**: Works on desktop and mobile devices
- **Example Prompts**: One-click example financial questions

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `max_new_tokens` parameter
   - Enable 8-bit quantization (default)
   - Close other GPU applications

2. **Model Loading Errors**:
   - Check internet connection for model downloads
   - Verify HuggingFace access tokens if needed
   - Ensure sufficient disk space (>20GB)

3. **Slow Generation**:
   - First generation is slower due to model compilation
   - Subsequent generations should be faster
   - Consider using GPU if available

### Performance Tips

- **GPU Usage**: CUDA provides significant speedup
- **Model Compilation**: torch.compile optimizes inference (PyTorch 2.0+)
- **Batch Processing**: Process multiple requests efficiently
- **Caching**: Model stays loaded between requests

## 🔒 Security Considerations

- **Input Validation**: All user inputs are validated
- **Error Handling**: Sensitive information is not exposed in errors
- **Rate Limiting**: Consider implementing for production use
- **HTTPS**: Use HTTPS in production environments

## 📈 Performance Metrics

### Expected Performance (NVIDIA RTX 3080)
- **Model Loading**: ~30-60 seconds (first time)
- **Generation Speed**: ~10-20 tokens/second
- **Memory Usage**: ~8-12GB VRAM
- **Response Time**: ~5-15 seconds per request

## 🤝 Migration from Gradio

This Flask frontend provides equivalent functionality to the original Gradio interface with these improvements:

- **Better UI/UX**: Modern, responsive design
- **API Structure**: RESTful API for integration
- **Error Handling**: More robust error management
- **Customization**: Easier to customize and extend
- **Production Ready**: Better suited for deployment

## 📝 Example Financial Questions

- "When should I get a second credit card?"
- "How much should I save for retirement each month?"
- "What's the best way to pay off high-interest debt?"
- "Should I invest in stocks or bonds for long-term growth?"
- "How can I improve my credit score quickly?"
- "What emergency fund amount do I need?"
- "Is it better to rent or buy a home in my situation?"
- "How should I diversify my investment portfolio?"
