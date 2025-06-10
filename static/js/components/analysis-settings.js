/**
 * Analysis Settings Component
 * A web component for managing analysis model settings
 */
class AnalysisSettings extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.render();
    }

    connectedCallback() {
        // Initialize the component when connected to DOM
        this.initializeSettings();
        this.setupEventListeners();
    }

    initializeSettings() {
        // Load current settings from the API
        this.fetchSettings();
    }

    setupEventListeners() {
        // Provider selection
        const providerSelect = this.shadowRoot.querySelector('#analysisProviderSelect');
        if (providerSelect) {
            providerSelect.addEventListener('change', () => {
                this.updateModelSelectionVisibility();
                this.saveSettings();
            });
        }

        // Ollama model selection
        const ollamaModelSelect = this.shadowRoot.querySelector('#analysisOllamaModelSelect');
        if (ollamaModelSelect) {
            ollamaModelSelect.addEventListener('change', () => {
                this.saveSettings();
            });
        }

        // OpenRouter model selection
        const openrouterModelSelect = this.shadowRoot.querySelector('#analysisOpenrouterModelSelect');
        if (openrouterModelSelect) {
            openrouterModelSelect.addEventListener('change', () => {
                this.saveSettings();
            });
        }

        // Temperature slider
        const temperatureSlider = this.shadowRoot.querySelector('#analysisTemperature');
        const temperatureValue = this.shadowRoot.querySelector('#analysisTemperatureValue');
        if (temperatureSlider && temperatureValue) {
            temperatureSlider.addEventListener('input', () => {
                temperatureValue.textContent = temperatureSlider.value;
            });
            temperatureSlider.addEventListener('change', () => {
                this.saveSettings();
            });
        }

        // Max tokens input
        const maxTokensInput = this.shadowRoot.querySelector('#analysisMaxTokens');
        if (maxTokensInput) {
            maxTokensInput.addEventListener('change', () => {
                this.saveSettings();
            });
        }

        // Test connection button
        const testButton = this.shadowRoot.querySelector('#testAnalysisConnection');
        if (testButton) {
            testButton.addEventListener('click', () => {
                this.testConnection();
            });
        }
    }

    updateModelSelectionVisibility() {
        const provider = this.shadowRoot.querySelector('#analysisProviderSelect').value;
        const ollamaSection = this.shadowRoot.querySelector('#ollamaModelSection');
        const openrouterSection = this.shadowRoot.querySelector('#openrouterModelSection');

        if (provider === 'ollama') {
            ollamaSection.classList.remove('hidden');
            openrouterSection.classList.add('hidden');
        } else {
            ollamaSection.classList.add('hidden');
            openrouterSection.classList.remove('hidden');
        }
    }

    async fetchSettings() {
        try {
            const response = await fetch('/api/analysis/settings');
            if (response.ok) {
                const data = await response.json();
                if (data.success && data.settings) {
                    this.updateUI(data.settings);
                } else {
                    this.showMessage('Failed to load settings: ' + (data.message || 'Unknown error'), 'error');
                }
            } else {
                this.showMessage('Failed to load settings from server', 'error');
            }
        } catch (error) {
            console.error('Error fetching analysis settings:', error);
            this.showMessage('Error loading settings: ' + error.message, 'error');
        }
    }

    async saveSettings() {
        try {
            const provider = this.shadowRoot.querySelector('#analysisProviderSelect').value;
            const ollamaModel = this.shadowRoot.querySelector('#analysisOllamaModelSelect').value;
            const openrouterModel = this.shadowRoot.querySelector('#analysisOpenrouterModelSelect').value;
            const temperature = parseFloat(this.shadowRoot.querySelector('#analysisTemperature').value);
            const maxTokens = parseInt(this.shadowRoot.querySelector('#analysisMaxTokens').value);

            const settings = {
                provider: provider,
                model_name: provider === 'ollama' ? ollamaModel : undefined,
                openrouter_model: provider === 'openrouter' ? openrouterModel : undefined,
                temperature: temperature,
                max_tokens: maxTokens
            };

            const response = await fetch('/api/analysis/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });

            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    this.showMessage('Settings saved successfully', 'success');
                } else {
                    this.showMessage('Failed to save settings: ' + (data.message || 'Unknown error'), 'error');
                }
            } else {
                this.showMessage('Failed to save settings to server', 'error');
            }
        } catch (error) {
            console.error('Error saving analysis settings:', error);
            this.showMessage('Error saving settings: ' + error.message, 'error');
        }
    }

    async testConnection() {
        try {
            const testButton = this.shadowRoot.querySelector('#testAnalysisConnection');
            const statusEl = this.shadowRoot.querySelector('#analysisConnectionStatus');
            
            testButton.disabled = true;
            testButton.textContent = 'Testing...';
            statusEl.textContent = 'Testing connection...';
            statusEl.className = 'text-blue-500';

            const response = await fetch('/api/analysis/test-connection', {
                method: 'POST'
            });

            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    statusEl.textContent = 'Connection successful';
                    statusEl.className = 'text-green-500';
                    this.showMessage('Successfully connected to analysis model', 'success');
                } else {
                    statusEl.textContent = 'Connection failed';
                    statusEl.className = 'text-red-500';
                    this.showMessage('Failed to connect: ' + (data.message || 'Unknown error'), 'error');
                }
            } else {
                statusEl.textContent = 'Connection error';
                statusEl.className = 'text-red-500';
                this.showMessage('Connection test failed', 'error');
            }
        } catch (error) {
            console.error('Error testing analysis connection:', error);
            this.showMessage('Error testing connection: ' + error.message, 'error');
            
            const statusEl = this.shadowRoot.querySelector('#analysisConnectionStatus');
            statusEl.textContent = 'Connection error';
            statusEl.className = 'text-red-500';
        } finally {
            const testButton = this.shadowRoot.querySelector('#testAnalysisConnection');
            testButton.disabled = false;
            testButton.textContent = 'Test Connection';
        }
    }

    updateUI(settings) {
        // Set provider selection
        const providerSelect = this.shadowRoot.querySelector('#analysisProviderSelect');
        if (providerSelect) {
            providerSelect.value = settings.provider || 'ollama';
        }

        // Set model selections
        const ollamaModelSelect = this.shadowRoot.querySelector('#analysisOllamaModelSelect');
        if (ollamaModelSelect) {
            // Fetch available Ollama models if needed
            this.fetchOllamaModels().then(models => {
                ollamaModelSelect.innerHTML = '';
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    ollamaModelSelect.appendChild(option);
                });
                
                // Set current selection
                if (settings.model_name) {
                    ollamaModelSelect.value = settings.model_name;
                }
            });
        }

        const openrouterModelSelect = this.shadowRoot.querySelector('#analysisOpenrouterModelSelect');
        if (openrouterModelSelect) {
            // Fetch available OpenRouter models if needed
            this.fetchOpenRouterModels().then(models => {
                openrouterModelSelect.innerHTML = '';
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = model.name;
                    openrouterModelSelect.appendChild(option);
                });
                
                // Set current selection
                if (settings.openrouter_model) {
                    openrouterModelSelect.value = settings.openrouter_model;
                }
            });
        }

        // Set temperature slider
        const temperatureSlider = this.shadowRoot.querySelector('#analysisTemperature');
        const temperatureValue = this.shadowRoot.querySelector('#analysisTemperatureValue');
        if (temperatureSlider && temperatureValue && settings.temperature !== undefined) {
            temperatureSlider.value = settings.temperature;
            temperatureValue.textContent = settings.temperature;
        }

        // Set max tokens
        const maxTokensInput = this.shadowRoot.querySelector('#analysisMaxTokens');
        if (maxTokensInput && settings.max_tokens !== undefined) {
            maxTokensInput.value = settings.max_tokens;
        }

        // Update visibility based on provider
        this.updateModelSelectionVisibility();
    }

    async fetchOllamaModels() {
        try {
            const response = await fetch('/api/ollama/models');
            if (response.ok) {
                const data = await response.json();
                if (data && data.models) {
                    return data.models.map(model => model.name);
                }
            }
            return ['mistral:7b-instruct-v0.2-q4_0']; // Default model if fetch fails
        } catch (error) {
            console.error('Error fetching Ollama models:', error);
            return ['mistral:7b-instruct-v0.2-q4_0']; // Default model on error
        }
    }

    async fetchOpenRouterModels() {
        try {
            const response = await fetch('/api/openrouter/models');
            if (response.ok) {
                const data = await response.json();
                if (data && data.models) {
                    return data.models;
                }
            }
            return [{ id: 'anthropic/claude-3-haiku', name: 'Claude 3 Haiku' }]; // Default
        } catch (error) {
            console.error('Error fetching OpenRouter models:', error);
            return [{ id: 'anthropic/claude-3-haiku', name: 'Claude 3 Haiku' }]; // Default on error
        }
    }

    showMessage(message, type = 'info') {
        const messageElement = this.shadowRoot.querySelector('#analysisMessage');
        if (messageElement) {
            messageElement.textContent = message;
            messageElement.className = 'mt-2 p-2 rounded';
            
            switch (type) {
                case 'success':
                    messageElement.classList.add('bg-green-800', 'text-green-200');
                    break;
                case 'error':
                    messageElement.classList.add('bg-red-800', 'text-red-200');
                    break;
                default:
                    messageElement.classList.add('bg-blue-800', 'text-blue-200');
            }
            
            setTimeout(() => {
                messageElement.textContent = '';
                messageElement.className = 'mt-2 p-2 rounded hidden';
            }, 5000);
        }
    }

    render() {
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen-Sans, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
                }
                
                .container {
                    background-color: #1f2937;
                    border-radius: 0.75rem;
                    padding: 1.5rem;
                    border: 1px solid #374151;
                    transition: border-color 0.2s ease;
                }
                
                .container:hover {
                    border-color: #6366f1;
                }
                
                h3 {
                    color: #f9fafb;
                    margin-top: 0;
                    border-bottom: 1px solid #374151;
                    padding-bottom: 0.5rem;
                }
                
                .form-group {
                    margin-bottom: 1rem;
                }
                
                label {
                    display: block;
                    margin-bottom: 0.5rem;
                    color: #d1d5db;
                }
                
                select, input[type="number"] {
                    width: 100%;
                    padding: 0.5rem;
                    border-radius: 0.375rem;
                    background-color: #111827;
                    color: #f9fafb;
                    border: 1px solid #374151;
                    margin-bottom: 1rem;
                }
                
                select:focus, input[type="number"]:focus {
                    outline: none;
                    border-color: #6366f1;
                    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
                }
                
                .row {
                    display: flex;
                    gap: 1rem;
                }
                
                .col {
                    flex: 1;
                }
                
                button {
                    background-color: #4f46e5;
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 0.375rem;
                    border: none;
                    cursor: pointer;
                    transition: background-color 0.2s ease;
                }
                
                button:hover {
                    background-color: #4338ca;
                }
                
                button:disabled {
                    background-color: #6b7280;
                    cursor: not-allowed;
                }
                
                .hidden {
                    display: none;
                }
                
                .slider-container {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                }
                
                .slider {
                    flex-grow: 1;
                }
                
                .slider-value {
                    min-width: 2.5rem;
                    text-align: center;
                }
                
                .connection-test {
                    margin-top: 1.5rem;
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }
                
                .text-green-500 {
                    color: #10b981;
                }
                
                .text-red-500 {
                    color: #ef4444;
                }
                
                .text-blue-500 {
                    color: #3b82f6;
                }
                
                .bg-green-800 {
                    background-color: #065f46;
                }
                
                .bg-red-800 {
                    background-color: #991b1b;
                }
                
                .bg-blue-800 {
                    background-color: #1e40af;
                }
                
                .text-green-200 {
                    color: #a7f3d0;
                }
                
                .text-red-200 {
                    color: #fecaca;
                }
                
                .text-blue-200 {
                    color: #bfdbfe;
                }
                
                .rounded {
                    border-radius: 0.25rem;
                }
                
                .mt-2 {
                    margin-top: 0.5rem;
                }
                
                .p-2 {
                    padding: 0.5rem;
                }
            </style>
            
            <div class="container">
                <h3>Analysis Model Settings</h3>
                
                <div class="form-group">
                    <label for="analysisProviderSelect">Provider</label>
                    <select id="analysisProviderSelect">
                        <option value="ollama">Ollama (Local)</option>
                        <option value="openrouter">OpenRouter (Cloud)</option>
                    </select>
                </div>
                
                <div id="ollamaModelSection" class="form-group">
                    <label for="analysisOllamaModelSelect">Ollama Model</label>
                    <select id="analysisOllamaModelSelect">
                        <option value="mistral:7b-instruct-v0.2-q4_0">Loading models...</option>
                    </select>
                </div>
                
                <div id="openrouterModelSection" class="form-group hidden">
                    <label for="analysisOpenrouterModelSelect">OpenRouter Model</label>
                    <select id="analysisOpenrouterModelSelect">
                        <option value="anthropic/claude-3-haiku">Loading models...</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="analysisTemperature">Temperature</label>
                    <div class="slider-container">
                        <input type="range" id="analysisTemperature" class="slider" min="0" max="1" step="0.1" value="0.7">
                        <span id="analysisTemperatureValue" class="slider-value">0.7</span>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="analysisMaxTokens">Max Tokens</label>
                    <input type="number" id="analysisMaxTokens" min="500" max="10000" step="100" value="2000">
                </div>
                
                <div class="connection-test">
                    <button id="testAnalysisConnection">Test Connection</button>
                    <div id="analysisConnectionStatus" class="text-blue-500"></div>
                </div>
                
                <div id="analysisMessage" class="mt-2 p-2 rounded hidden"></div>
            </div>
        `;
    }
}

// Define the custom element
customElements.define('analysis-settings', AnalysisSettings);