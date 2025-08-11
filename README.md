# FinAdvisor - AI-Powered Financial Advisory Tool

FinAdvisor is an AI-driven financial advisory tool that provides contextually-aware personal finance assistance using fine-tuned Large Language Models (LLMs). The project combines advanced machine learning techniques with a modern web interface to deliver personalized financial advice.

## Research Questions and Hypotheses

- How can LLMs improve the accuracy and timeliness of financial advice?

Hypothesis: Fine-tuning on financial datasets enhances the model’s contextual relevance, enabling it to offer precise, tailored recommendations.

- Can QLoRA (Quantized Low-Rank Adaptation) achieve efficient fine-tuning without high computational costs?

Hypothesis: Using QLoRA enables rapid adaptation to financial contexts, providing accurate, resource-efficient advice and making the model scalable.

## Related Work and Datasets

The project builds on recent research in financial AI, particularly in real-time data retrieval and language processing, using advanced methods like Low-Rank Adaptation (LoRA) to improve accuracy and responsiveness. Two primary datasets serve as the foundation:

- Alpaca News API: Real-time financial news data that keeps the model up-to-date with market trends.
- Hugging Face Finance-Alpaca Dataset: A question-answer dataset focused on finance to train the model on common financial inquiries, providing it with essential background knowledge.

## Methods

The tool’s development involves data collection, preprocessing, fine-tuning, and deployment. After data cleaning and embedding, the processed data is stored in a Vector Database (Pinecone) for efficient retrieval. LoRA is used to fine-tune the model to specialize in financial topics, ensuring responses remain relevant and contextually accurate. The model is then deployed using a RESTful API, with Flask providing an intuitive user interface for interactions.

**Fine-Tuning with LoRA (Low-Rank Adaptation):**

- **LoRA Overview:** LoRA is used to fine-tune the language model, adjusting a minimal set of parameters that allow it to specialize in financial contexts without requiring complete retraining.
![adapters](https://github.com/user-attachments/assets/a7c5600d-48b8-426f-be95-6d4aea2cb0fb)

- **Training:** Through LoRA adaptation, the model learns to interpret financial data, allowing it to deliver accurate and context-sensitive financial advice.
![LoRA and QLoRA](https://github.com/user-attachments/assets/49bc20cc-eccd-481b-847f-8beea9ed4a6e)

## Overview

This project addresses the gap in personal finance management by providing real-time, personalized financial insights through:
- **Multiple LLM Support**: LLaMA 7B and Mistral 7B models fine-tuned with LoRA (Low-Rank Adaptation)
- **Flask-based web interface** with dynamic model selection
- **Modular inference pipeline** for scalable deployment
- **Real-time financial advice** generation


## Project Structure

```
finadvisor-llm/
├── training_pipeline/          # Model training components
│   ├── config.py              # Training configuration
│   ├── data.py                # Data loading and processing
│   ├── model.py               # Model architecture
│   ├── trainer.py             # Training logic
│   ├── evaluator.py           # Model evaluation
│   └── utils.py               # Utility functions
├── inference_pipeline/         # Inference components
│   ├── config.py              # Inference configuration
│   ├── model_loader.py        # Model loading utilities
│   ├── prompt_utils.py        # Prompt formatting
│   └── inference_engine.py    # Main inference engine
├── templates/                  # Flask HTML templates
│   └── index.html             # Web interface
├── static/                     # Static web assets
│   ├── css/style.css          # Custom styling
│   └── js/app.js              # Frontend JavaScript
├── scripts/                    # Utility scripts
├── app.py                      # Flask web application
├── requirements.txt            # Dependencies
└── *.ipynb                     # Jupyter notebooks
```

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd finadvisor-llm
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application:**
   ```bash
   python app.py
   ```

4. **Access the web interface:**
   Open your browser to `http://localhost:5000`

## Training Pipeline

### Data Sources
- **Alpaca News API**: Real-time financial news data
- **Hugging Face Finance-Alpaca Dataset**: Financial Q&A pairs

### Training Process
1. **Data Collection**: Financial data from multiple sources
2. **Preprocessing**: Text normalization, tokenization, and cleaning
3. **Fine-tuning**: LoRA adaptation on selected base model (LLaMA 7B or Mistral 7B)
4. **Evaluation**: Performance tracking with MLFlow

### Supported Models

#### LLaMA 7B
- **Base Model**: `baffo32/decapoda-research-llama-7B-hf`
- **LoRA Weights**: `kunchum/capstone-llama-finetuned`
- **Target Modules**: `['q_proj', 'v_proj', 'k_proj', 'o_proj']`
- **Output Directory**: `/kaggle/working/llama_7b_tuned`

#### Mistral 7B
- **Base Model**: `mistralai/Mistral-7B-v0.1`
- **LoRA Weights**: `kunchum/capstone-mistral-finetuned`
- **Target Modules**: `['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']`
- **Output Directory**: `/kaggle/working/mistral_7b_tuned`

### Training Method
- **QLoRA (Quantized Low-Rank Adaptation)**: Efficient fine-tuning with reduced memory requirements
- **8-bit Quantization**: Reduces memory usage during training
- **Gradient Checkpointing**: Further memory optimization

## Inference Pipeline

### Architecture
The inference pipeline is designed for production use with:
- **Multi-Model Support**: Dynamic switching between LLaMA 7B and Mistral 7B
- **Modular Design**: Separated concerns for maintainability
- **Device Detection**: Automatic CUDA/MPS/CPU selection
- **Model Optimization**: 8-bit quantization and torch.compile
- **Error Handling**: Comprehensive error management

### Model Selection
The system supports dynamic model selection at runtime:

```python
from inference_pipeline.inference_engine import get_inference_engine, get_available_models
from inference_pipeline.model_loader import get_available_models

# Get available models
print(get_available_models())  # ['LLaMa 7B', 'Mistral 7B']

# Use LLaMA 7B
llama_engine = get_inference_engine("LLaMa 7B")
llama_engine.initialize()

# Use Mistral 7B
mistral_engine = get_inference_engine("Mistral 7B")
mistral_engine.initialize()
```

### Usage Example
```python
from inference_pipeline.inference_engine import get_inference_engine

# Initialize the engine with specific model
engine = get_inference_engine("Mistral 7B")  # or "LLaMa 7B"
engine.initialize()

# Generate advice
response = engine.generate_response(
    instruction="When should I get a second credit card?",
    temperature=0.6,
    max_new_tokens=128
)

print(response)
```

## Web Interface

### Features
- **Dynamic Model Selection**: Choose between LLaMA 7B and Mistral 7B models
- **Interactive Parameters**: Real-time slider controls for temperature, top-p, max tokens, and repetition penalty
- **Model-Aware Processing**: Automatic model switching based on user selection
- **Error Handling**: User-friendly error messages and loading states
- **Example Prompts**: Built-in financial question examples

### API Endpoints

#### `POST /api/generate`
Generate financial advice based on user input.

**Request:**
```json
{
    "instruction": "How much should I save for retirement?",
    "input": "I'm 30 years old with $50k income",
    "model": "Mistral 7B",
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 128,
    "repetition_penalty": 1.15
}
```

**Response:**
```json
{
    "success": true,
    "response": "Generated financial advice...",
    "model": "Mistral 7B",
    "parameters": {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 128,
        "repetition_penalty": 1.15
    }
}
```

#### `GET /api/health`
Check API and model status.

## Performance

### Expected Performance (NVIDIA RTX 3080)
- **Model Loading**: 30-60 seconds (first time)
- **Generation Speed**: 10-20 tokens/second
- **Memory Usage**: 8-12GB VRAM
- **Response Time**: 5-15 seconds per request

## Configuration

### Model Configuration
```python
# inference_pipeline/config.py
BASE_MODEL = "baffo32/decapoda-research-llama-7B-hf"
LORA_WEIGHTS = "kunchum/capstone-llama-finetuned"
DEVICE = "cuda"  # Auto-detected

DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens": 128,
    "repetition_penalty": 1.15
}
```

## Example Financial Questions

- "When should I get a second credit card?"
- "How much should I save for retirement each month?"
- "What's the best way to pay off high-interest debt?"
- "Should I invest in stocks or bonds for long-term growth?"
- "How can I improve my credit score quickly?"
- "What emergency fund amount do I need?"
- "Is it better to rent or buy a home?"
- "How should I diversify my investment portfolio?"

## Development

### Training a New Model
1. Prepare your financial dataset
2. Update `training_pipeline/config.py`
3. Run the training notebook or script
4. Monitor training with MLFlow
5. Update inference config with new model path

### Customizing the Interface
1. Modify `templates/index.html` for UI changes
2. Update `static/css/style.css` for styling
3. Extend `static/js/app.js` for new functionality
4. Add new API endpoints in `app.py`

## Troubleshooting

### Common Issues
- **CUDA Out of Memory**: Reduce `max_new_tokens` or enable 8-bit quantization
- **Model Loading Errors**: Check internet connection and HuggingFace access
- **Slow Generation**: First generation is slower due to model compilation

### Performance Tips
- Use GPU for significant speedup
- Enable model compilation (PyTorch 2.0+)
- Keep model loaded between requests
- Consider batch processing for multiple requests


