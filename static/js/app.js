// FinAdvisor Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Get form elements
    const form = document.getElementById('advisoryForm');
    const generateBtn = document.getElementById('generateBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const output = document.getElementById('output');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');

    // Parameter sliders and their value displays
    const sliders = {
        temperature: {
            slider: document.getElementById('temperature'),
            display: document.getElementById('tempValue')
        },
        top_p: {
            slider: document.getElementById('top_p'),
            display: document.getElementById('topPValue')
        },
        max_tokens: {
            slider: document.getElementById('max_tokens'),
            display: document.getElementById('maxTokensValue')
        },
        repetition_penalty: {
            slider: document.getElementById('repetition_penalty'),
            display: document.getElementById('repPenaltyValue')
        }
    };

    // Update slider value displays
    Object.keys(sliders).forEach(key => {
        const slider = sliders[key].slider;
        const display = sliders[key].display;
        
        slider.addEventListener('input', function() {
            display.textContent = this.value;
        });
    });

    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Get form values
        const instruction = document.getElementById('instruction').value.trim();
        const input = document.getElementById('input').value.trim();
        const model = document.getElementById('model').value;
        const temperature = parseFloat(document.getElementById('temperature').value);
        const top_p = parseFloat(document.getElementById('top_p').value);
        const max_tokens = parseInt(document.getElementById('max_tokens').value);
        const repetition_penalty = parseFloat(document.getElementById('repetition_penalty').value);

        // Validate instruction
        if (!instruction) {
            showError('Please enter an instruction or question.');
            return;
        }

        // Show loading state
        showLoading();

        try {
            // Make API request
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    instruction: instruction,
                    input: input || 'none',
                    model: model,
                    temperature: temperature,
                    top_p: top_p,
                    max_tokens: max_tokens,
                    repetition_penalty: repetition_penalty
                })
            });

            const data = await response.json();

            if (data.success) {
                showResponse(data.response);
            } else {
                showError(data.error || 'An error occurred while generating advice.');
            }

        } catch (error) {
            console.error('Error:', error);
            showError('Network error: Unable to connect to the server.');
        }
    });

    // Show loading state
    function showLoading() {
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Generating...';
        loadingSpinner.classList.remove('d-none');
        output.classList.add('d-none');
        errorMessage.classList.add('d-none');
    }

    // Show response
    function showResponse(responseText) {
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Generate Advice';
        loadingSpinner.classList.add('d-none');
        output.classList.remove('d-none');
        errorMessage.classList.add('d-none');
        
        output.textContent = responseText;
        output.classList.add('has-content');
        
        // Scroll to output
        output.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // Show error
    function showError(message) {
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Generate Advice';
        loadingSpinner.classList.add('d-none');
        output.classList.add('d-none');
        errorMessage.classList.remove('d-none');
        errorText.textContent = message;
        
        // Scroll to error
        errorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // Check API health on page load
    checkApiHealth();

    async function checkApiHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            if (data.status === 'initializing') {
                showInfo('Model is loading... This may take a few minutes on first startup.');
                
                // Poll for readiness
                const pollInterval = setInterval(async () => {
                    try {
                        const healthResponse = await fetch('/api/health');
                        const healthData = await healthResponse.json();
                        
                        if (healthData.status === 'healthy') {
                            clearInterval(pollInterval);
                            hideInfo();
                        }
                    } catch (error) {
                        console.error('Health check error:', error);
                    }
                }, 5000); // Check every 5 seconds
            }
        } catch (error) {
            console.error('Initial health check failed:', error);
        }
    }

    // Show info message
    function showInfo(message) {
        // Create info banner if it doesn't exist
        let infoBanner = document.getElementById('infoBanner');
        if (!infoBanner) {
            infoBanner = document.createElement('div');
            infoBanner.id = 'infoBanner';
            infoBanner.className = 'alert alert-info alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
            infoBanner.style.zIndex = '9999';
            infoBanner.innerHTML = `
                <i class="fas fa-info-circle me-2"></i>
                <span id="infoText">${message}</span>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            document.body.appendChild(infoBanner);
        } else {
            document.getElementById('infoText').textContent = message;
            infoBanner.classList.add('show');
        }
    }

    // Hide info message
    function hideInfo() {
        const infoBanner = document.getElementById('infoBanner');
        if (infoBanner) {
            infoBanner.classList.remove('show');
        }
    }

    // Add some example prompts
    const examplePrompts = [
        "When should I get a second credit card?",
        "How much should I save for retirement each month?",
        "What's the best way to pay off high-interest debt?",
        "Should I invest in stocks or bonds for long-term growth?",
        "How can I improve my credit score quickly?",
        "What emergency fund amount do I need?",
        "Is it better to rent or buy a home in my situation?",
        "How should I diversify my investment portfolio?"
    ];

    // Add example button functionality
    const instructionField = document.getElementById('instruction');
    
    // Create example prompts dropdown
    const exampleButton = document.createElement('button');
    exampleButton.type = 'button';
    exampleButton.className = 'btn btn-outline-secondary btn-sm mt-2';
    exampleButton.innerHTML = '<i class="fas fa-lightbulb me-1"></i>Try an example';
    
    exampleButton.addEventListener('click', function() {
        const randomPrompt = examplePrompts[Math.floor(Math.random() * examplePrompts.length)];
        instructionField.value = randomPrompt;
        instructionField.focus();
    });
    
    // Insert example button after instruction field
    instructionField.parentNode.appendChild(exampleButton);
});
