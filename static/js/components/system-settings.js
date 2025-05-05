/**
 * SystemSettings Component
 * A web component for managing system settings, integrating with Ollama and OpenRouter widgets
 */
class SystemSettings extends HTMLElement {
    constructor() {
        super();
        this.provider = 'ollama'; // Default provider
        this.systemPrompt = '';
        this.training = {
            start: '00:00',
            stop: '06:00'
        };
    }

    /**
     * Load settings from backend API
     * @param {string} backendUrl - The backend URL
     * @param {string} groupId - Group ID for settings scope
     */
    async loadSettings(backendUrl = '', groupId = '') {
        try {
            console.log('Loading system settings');
            
            // Make API request to get system settings
            const response = await fetch(`/api/system-settings${groupId ? `?group_id=${groupId}` : ''}`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to load settings: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Received system settings:', data);

            // Update form with received data
            this.updateForm({
                provider: data.provider || 'ollama',
                model_name: data.model_name || 'llama3.2:1b',
                openrouter_model: data.openrouter_model || 'anthropic/claude-3-haiku',
                system_prompt: data.system_prompt || '',
                training: data.training || {
                    start: '00:00',
                    stop: '06:00'
                }
            });
            
            // Also update model select dropdowns
            this.updateModelSelections(data.model_name, data.openrouter_model);

        } catch (error) {
            console.error('Error loading system settings:', error);
            // Still show default values if we can't load from backend
            this.updateForm({
                provider: 'ollama',
                model_name: 'llama3.2:1b',
                openrouter_model: 'anthropic/claude-3-haiku',
                system_prompt: 'You are a Knowledge Base Assistant that helps with retrieving and explaining information.',
                training: {
                    start: '00:00',
                    stop: '06:00'
                }
            });
        }
    }
    
    /**
     * Update model selections in the dropdown selects
     * @param {string} ollamaModel - The selected Ollama model
     * @param {string} openrouterModel - The selected OpenRouter model
     */
    updateModelSelections(ollamaModel, openrouterModel) {
        let modelUpdated = false;
        
        // Helper function to update a select element with a value
        const updateSelectWithValue = (select, value) => {
            if (select && value) {
                // If the model exists in the dropdown, select it
                for (let i = 0; i < select.options.length; i++) {
                    if (select.options[i].value === value) {
                        select.selectedIndex = i;
                        return true;
                    }
                }
                
                // Model not found in the options
                return false;
            }
            return false;
        };
        
        // Update embedded Ollama model selection
        const embeddedOllamaSelect = this.querySelector('#embeddedOllamaModelSelect');
        if (embeddedOllamaSelect) {
            modelUpdated = updateSelectWithValue(embeddedOllamaSelect, ollamaModel);
            
            // If we couldn't find the model immediately, try again after a delay
            if (!modelUpdated && ollamaModel) {
                setTimeout(() => {
                    updateSelectWithValue(embeddedOllamaSelect, ollamaModel);
                    this.updateSelectedModelDisplay();
                }, 1000);
            }
        }
        
        // Update embedded OpenRouter model selection
        const embeddedOpenrouterSelect = this.querySelector('#embeddedOpenrouterModelSelect');
        if (embeddedOpenrouterSelect) {
            modelUpdated = updateSelectWithValue(embeddedOpenrouterSelect, openrouterModel);
            
            // If we couldn't find the model immediately, try again after a delay
            if (!modelUpdated && openrouterModel) {
                setTimeout(() => {
                    updateSelectWithValue(embeddedOpenrouterSelect, openrouterModel);
                    this.updateSelectedModelDisplay();
                }, 1000);
            }
        }
        
        // Also update global selects for compatibility
        const ollamaSelect = document.getElementById('ollamaModelSelect');
        if (ollamaSelect) {
            updateSelectWithValue(ollamaSelect, ollamaModel);
            
            // If we couldn't find the exact model, try again after a delay
            if (!modelUpdated && ollamaModel) {
                setTimeout(() => {
                    updateSelectWithValue(ollamaSelect, ollamaModel);
                }, 1000);
            }
        }
        
        const openrouterSelect = document.getElementById('openrouterModelSelect');
        if (openrouterSelect) {
            updateSelectWithValue(openrouterSelect, openrouterModel);
            
            // If we couldn't find the exact model, try again after a delay
            if (!modelUpdated && openrouterModel) {
                setTimeout(() => {
                    updateSelectWithValue(openrouterSelect, openrouterModel);
                }, 1000);
            }
        }
        
        // Update the model display
        this.updateSelectedModelDisplay();
    }

    /**
     * Get current model information from the selected provider
     * @returns {Object} Model information with provider, name, and display name
     */
    getSelectedModelInfo() {
        const providerType = this.querySelector('input[name="providerType"]:checked').value;
        let modelInfo = {
            provider: providerType,
            name: null,
            displayName: null,
            isValid: false
        };
        
        if (providerType === 'ollama') {
            // First try the embedded widget
            const embeddedOllamaModelSelect = this.querySelector('#embeddedOllamaModelSelect');
            if (embeddedOllamaModelSelect && embeddedOllamaModelSelect.value && embeddedOllamaModelSelect.value !== 'loading') {
                const selectedOption = embeddedOllamaModelSelect.options[embeddedOllamaModelSelect.selectedIndex];
                modelInfo.name = embeddedOllamaModelSelect.value;
                modelInfo.displayName = selectedOption.textContent || embeddedOllamaModelSelect.value;
                modelInfo.isValid = true;
            } else {
                // Fall back to the global widget if embedded one isn't available or selected
                const ollamaModelSelect = document.getElementById('ollamaModelSelect');
                if (ollamaModelSelect && ollamaModelSelect.value && ollamaModelSelect.value !== 'loading') {
                    const selectedOption = ollamaModelSelect.options[ollamaModelSelect.selectedIndex];
                    modelInfo.name = ollamaModelSelect.value;
                    modelInfo.displayName = selectedOption.textContent || ollamaModelSelect.value;
                    modelInfo.isValid = true;
                }
            }
        } else {
            // First try the embedded widget
            const embeddedOpenrouterModelSelect = this.querySelector('#embeddedOpenrouterModelSelect');
            if (embeddedOpenrouterModelSelect && embeddedOpenrouterModelSelect.value && embeddedOpenrouterModelSelect.value !== 'loading') {
                const selectedOption = embeddedOpenrouterModelSelect.options[embeddedOpenrouterModelSelect.selectedIndex];
                modelInfo.name = embeddedOpenrouterModelSelect.value;
                modelInfo.displayName = selectedOption.textContent || embeddedOpenrouterModelSelect.value;
                modelInfo.isValid = true;
            } else {
                // Fall back to the global widget if embedded one isn't available or selected
                const openrouterModelSelect = document.getElementById('openrouterModelSelect');
                if (openrouterModelSelect && openrouterModelSelect.value && openrouterModelSelect.value !== 'loading') {
                    const selectedOption = openrouterModelSelect.options[openrouterModelSelect.selectedIndex];
                    modelInfo.name = openrouterModelSelect.value;
                    modelInfo.displayName = selectedOption.textContent || openrouterModelSelect.value;
                    modelInfo.isValid = true;
                }
            }
        }
        
        return modelInfo;
    }
    
    /**
     * Save settings to backend API
     */
    async saveSettings() {
        try {
            // Show saving state
            const saveButton = this.querySelector('#saveSystemSettings');
            const originalButtonText = saveButton.innerHTML;
            saveButton.innerHTML = `
                <div class="animate-spin mr-2 h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                Saving...
            `;
            saveButton.disabled = true;
            
            // Get the provider from the radio button
            const providerType = this.querySelector('input[name="providerType"]:checked').value;
            
            // Get model information
            const modelInfo = this.getSelectedModelInfo();
            if (!modelInfo.isValid) {
                this.showMessage(`Please select a valid ${providerType === 'ollama' ? 'Ollama' : 'OpenRouter'} model`, 'error');
                
                // Reset button
                saveButton.innerHTML = originalButtonText;
                saveButton.disabled = false;
                return;
            }

            // Prepare settings object based on the selected provider
            const settings = {
                provider: providerType,
                system_prompt: this.querySelector('#systemPrompt').value,
                
                // Always include both model fields to maintain compatibility
                model_name: providerType === 'ollama' ? modelInfo.name : 'llama3.2:1b',
                openrouter_model: providerType === 'openrouter' ? modelInfo.name : 'anthropic/claude-3-haiku',
                
                // Add training schedule
                training: {
                    start: this.querySelector('#trainingStartTime').value,
                    stop: this.querySelector('#trainingStopTime').value
                },
                
                // Include default parameters from the ModelSettings class
                temperature: 0.7,
                max_tokens: 2000,
                top_p: 0.9,
                top_k: 40,
                num_ctx: 256,
                num_thread: 4
            };

            // Log what we're sending
            console.log('Saving settings:', settings);
            
            // Send settings to backend
            const response = await fetch('/api/system-settings', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Failed to save settings: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Save response:', data);

            // Show success message with model information
            this.showMessage(`Settings saved successfully - ${modelInfo.displayName || modelInfo.name} model selected`, 'success');
            
            // Update the selected model display
            this.updateSelectedModelDisplay();
            
            return data;
        } catch (error) {
            console.error('Error saving system settings:', error);
            this.showMessage(`Error: ${error.message}`, 'error');
        } finally {
            // Reset save button state
            const saveButton = this.querySelector('#saveSystemSettings');
            if (saveButton) {
                saveButton.innerHTML = `
                    <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                    Save Settings
                `;
                saveButton.disabled = false;
            }
        }
    }

    /**
     * Update the form with settings data
     * @param {Object} data - The settings data
     */
    updateForm(data) {
        // Set provider radio button
        const providerType = data.provider || 'ollama';
        const providerRadio = this.querySelector(`input[name="providerType"][value="${providerType}"]`);
        if (providerRadio) {
            providerRadio.checked = true;
        }

        // Update visibility of provider sections
        this.toggleProviderSections(providerType);

        // Set system prompt
        const systemPromptElem = this.querySelector('#systemPrompt');
        if (systemPromptElem) {
            systemPromptElem.value = data.system_prompt || '';
        }

        // Set training schedule times
        const startTimeElem = this.querySelector('#trainingStartTime');
        const stopTimeElem = this.querySelector('#trainingStopTime');
        
        if (startTimeElem && data.training?.start) {
            startTimeElem.value = data.training.start;
        }
        
        if (stopTimeElem && data.training?.stop) {
            stopTimeElem.value = data.training.stop;
        }
        
        // Hide loading indicator
        const loadingIndicator = this.querySelector('#settingsLoadingIndicator');
        if (loadingIndicator) {
            loadingIndicator.classList.add('hidden');
        }
        
        // Update the model display
        setTimeout(() => this.updateSelectedModelDisplay(), 500);
    }

    /**
     * Toggle visibility of provider-specific sections
     * @param {string} providerType - The selected provider type
     */
    toggleProviderSections(providerType) {
        // Toggle description visibility
        const ollamaDesc = this.querySelector('.ollama-description');
        const openrouterDesc = this.querySelector('.openrouter-description');
        
        if (ollamaDesc && openrouterDesc) {
            if (providerType === 'ollama') {
                ollamaDesc.classList.remove('hidden');
                openrouterDesc.classList.add('hidden');
            } else {
                ollamaDesc.classList.add('hidden');
                openrouterDesc.classList.remove('hidden');
            }
        }
        
        // Toggle our embedded widgets
        const embeddedOllamaWidget = this.querySelector('#embeddedOllamaWidget');
        const embeddedOpenrouterWidget = this.querySelector('#embeddedOpenrouterWidget');
        
        if (embeddedOllamaWidget && embeddedOpenrouterWidget) {
            if (providerType === 'ollama') {
                embeddedOllamaWidget.classList.remove('hidden');
                embeddedOpenrouterWidget.classList.add('hidden');
            } else {
                embeddedOllamaWidget.classList.add('hidden');
                embeddedOpenrouterWidget.classList.remove('hidden');
            }
        }
        
        // Update the provider type in any hidden elements for backward compatibility
        // We don't need to toggle visibility since these are now always hidden
        document.dispatchEvent(new CustomEvent('providerTypeChanged', {
            detail: { provider: providerType }
        }));
    }

    /**
     * Show a status message to the user
     * @param {string} message - The message to show
     * @param {string} type - The message type (success, error)
     */
    showMessage(message, type = 'info') {
        // Find or create message element
        let messageElement = this.querySelector('.settings-message');
        if (!messageElement) {
            messageElement = document.createElement('div');
            messageElement.className = 'settings-message mt-4 px-4 py-2 rounded text-sm';
            this.appendChild(messageElement);
        }

        // Set message content and style
        messageElement.textContent = message;
        messageElement.className = `settings-message mt-4 px-4 py-2 rounded text-sm ${
            type === 'error' ? 'bg-red-900 text-red-100' : 'bg-green-900 text-green-100'
        }`;

        // Auto-hide after a delay
        setTimeout(() => {
            messageElement.style.opacity = '0';
            setTimeout(() => {
                messageElement.remove();
            }, 300);
        }, 3000);
    }

    /**
     * Called when element is connected to the DOM
     */
    connectedCallback() {
        this.render();
        
        // Initialize event listeners
        this.initEventListeners();
        
        // Load initial settings
        this.loadSettings();
    }

    /**
     * Update the selected model display
     */
    updateSelectedModelDisplay() {
        const modelInfo = this.getSelectedModelInfo();
        const currentModelNameElem = this.querySelector('#currentModelName');
        
        if (currentModelNameElem) {
            if (modelInfo.isValid) {
                currentModelNameElem.textContent = modelInfo.displayName || modelInfo.name;
                currentModelNameElem.classList.remove('text-neutral-400');
                currentModelNameElem.classList.add('text-indigo-400');
            } else {
                currentModelNameElem.textContent = `No ${modelInfo.provider === 'ollama' ? 'Ollama' : 'OpenRouter'} model selected`;
                currentModelNameElem.classList.remove('text-indigo-400');
                currentModelNameElem.classList.add('text-neutral-400');
            }
        }
    }

    /**
     * Initialize the embedded Ollama widget
     */
    initEmbeddedOllamaWidget() {
        // Map UI elements
        const statusElem = this.querySelector('#embeddedOllamaStatus');
        const statusIcon = this.querySelector('#embeddedOllamaStatusIcon');
        const modelSelect = this.querySelector('#embeddedOllamaModelSelect');
        const modelList = this.querySelector('#embeddedOllamaModelList');
        const refreshBtn = this.querySelector('#embeddedOllamaRefreshBtn');
        
        if (!statusElem || !statusIcon || !modelSelect || !refreshBtn) return;
        
        // Set up refresh button
        refreshBtn.addEventListener('click', async () => {
            try {
                // Show loading state
                statusElem.textContent = 'Checking connection...';
                statusIcon.classList.remove('bg-green-500', 'bg-red-500');
                statusIcon.classList.add('bg-yellow-500');
                
                // Fetch models from the API
                const response = await fetch('/api/ollama/models');
                
                if (response.ok) {
                    const data = await response.json();
                    if (data.models && Array.isArray(data.models)) {
                        // Update status UI
                        statusElem.textContent = 'Connected';
                        statusElem.classList.remove('text-neutral-400', 'text-red-500');
                        statusElem.classList.add('text-green-500');
                        statusIcon.classList.remove('bg-yellow-500', 'bg-red-500');
                        statusIcon.classList.add('bg-green-500');
                        
                        // Update model dropdown
                        modelSelect.innerHTML = '';
                        data.models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model.name;
                            option.textContent = model.name;
                            modelSelect.appendChild(option);
                        });
                        
                        // Set selected model if available
                        const mainModelSelect = document.getElementById('ollamaModelSelect');
                        if (mainModelSelect && mainModelSelect.value !== 'loading') {
                            modelSelect.value = mainModelSelect.value;
                        }
                        
                        // Update model list
                        if (modelList && data.models.length > 0) {
                            modelList.innerHTML = `<p>${data.models.length} model${data.models.length > 1 ? 's' : ''} available</p>`;
                        }
                        
                        // Update display
                        this.updateSelectedModelDisplay();
                    }
                } else {
                    throw new Error('Failed to load models');
                }
            } catch (error) {
                console.error('Error refreshing Ollama:', error);
                statusElem.textContent = 'Disconnected';
                statusElem.classList.remove('text-neutral-400', 'text-green-500');
                statusElem.classList.add('text-red-500');
                statusIcon.classList.remove('bg-yellow-500', 'bg-green-500');
                statusIcon.classList.add('bg-red-500');
            }
        });
        
        // Set up model select change handler
        modelSelect.addEventListener('change', () => {
            // Notify any listeners about the model change
            document.dispatchEvent(new CustomEvent('embeddedModelChanged', {
                detail: {
                    provider: 'ollama',
                    modelName: modelSelect.value
                }
            }));
            
            // Update our display directly
            this.updateSelectedModelDisplay();
        });
        
        // Initial check
        refreshBtn.click();
    }
    
    /**
     * Initialize the embedded OpenRouter widget
     */
    initEmbeddedOpenrouterWidget() {
        // Map UI elements
        const statusElem = this.querySelector('#embeddedOpenrouterStatus');
        const statusIcon = this.querySelector('#embeddedOpenrouterStatusIcon');
        const modelSelect = this.querySelector('#embeddedOpenrouterModelSelect');
        const modelList = this.querySelector('#embeddedOpenrouterModelList');
        const refreshBtn = this.querySelector('#embeddedOpenrouterRefreshBtn');
        const tokenInput = this.querySelector('#embeddedOpenrouterTokenInput');
        const saveTokenBtn = this.querySelector('#embeddedOpenrouterSaveToken');
        
        if (!statusElem || !statusIcon || !modelSelect || !refreshBtn || !tokenInput || !saveTokenBtn) return;
        
        // Check if token exists
        const checkToken = async () => {
            try {
                const response = await fetch('/api/openrouter/status');
                if (response.ok) {
                    const data = await response.json();
                    
                    if (data.status === 'connected') {
                        statusElem.textContent = 'Connected';
                        statusElem.classList.remove('text-neutral-400', 'text-red-500');
                        statusElem.classList.add('text-green-500');
                        statusIcon.classList.remove('bg-yellow-500', 'bg-red-500');
                        statusIcon.classList.add('bg-green-500');
                        
                        // Load models
                        loadModels();
                    } else {
                        statusElem.textContent = 'Needs API Key';
                        statusElem.classList.remove('text-green-500');
                        statusElem.classList.add('text-neutral-400');
                        statusIcon.classList.remove('bg-green-500', 'bg-red-500');
                        statusIcon.classList.add('bg-yellow-500');
                    }
                }
            } catch (error) {
                console.error('Error checking OpenRouter token:', error);
                statusElem.textContent = 'Disconnected';
                statusElem.classList.add('text-red-500');
                statusIcon.classList.add('bg-red-500');
            }
        };
        
        // Load models function
        const loadModels = async () => {
            try {
                const response = await fetch('/api/openrouter/models');
                if (response.ok) {
                    const data = await response.json();
                    
                    if (data.models && Array.isArray(data.models)) {
                        // Update model dropdown
                        modelSelect.innerHTML = '';
                        data.models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model.id;
                            option.textContent = model.name;
                            modelSelect.appendChild(option);
                        });
                        
                        // Set selected model if available
                        const mainModelSelect = document.getElementById('openrouterModelSelect');
                        if (mainModelSelect && mainModelSelect.value !== 'loading') {
                            modelSelect.value = mainModelSelect.value;
                        }
                        
                        // Update model list
                        if (modelList && data.models.length > 0) {
                            modelList.innerHTML = `<p>${data.models.length} model${data.models.length > 1 ? 's' : ''} available</p>`;
                        }
                        
                        // Update display
                        this.updateSelectedModelDisplay();
                    }
                } else {
                    throw new Error('Failed to load models');
                }
            } catch (error) {
                console.error('Error loading OpenRouter models:', error);
                modelList.innerHTML = '<p>Error loading models</p>';
            }
        };
        
        // Set up refresh button
        refreshBtn.addEventListener('click', () => {
            checkToken();
        });
        
        // Set up save token button
        saveTokenBtn.addEventListener('click', async () => {
            const token = tokenInput.value.trim();
            if (!token) {
                this.showMessage('Please enter a valid API token', 'error');
                return;
            }
            
            try {
                // Update UI state
                saveTokenBtn.disabled = true;
                statusElem.textContent = 'Connecting...';
                statusIcon.classList.remove('bg-green-500', 'bg-red-500');
                statusIcon.classList.add('bg-yellow-500');
                
                // Save the token
                const response = await fetch('/api/openrouter/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ api_key: token })
                });
                
                if (response.ok) {
                    // Update the main token input if it exists
                    const mainTokenInput = document.getElementById('openrouterTokenInput');
                    if (mainTokenInput) {
                        mainTokenInput.value = token;
                    }
                    
                    this.showMessage('API key saved successfully', 'success');
                    checkToken(); // Refresh status
                } else {
                    const data = await response.json();
                    throw new Error(data.error || 'Failed to save token');
                }
            } catch (error) {
                console.error('Error saving OpenRouter token:', error);
                this.showMessage(`Error: ${error.message}`, 'error');
                statusElem.textContent = 'Connection failed';
                statusElem.classList.add('text-red-500');
                statusIcon.classList.remove('bg-yellow-500');
                statusIcon.classList.add('bg-red-500');
            } finally {
                saveTokenBtn.disabled = false;
            }
        });
        
        // Set up model select change handler
        modelSelect.addEventListener('change', () => {
            // Notify any listeners about the model change
            document.dispatchEvent(new CustomEvent('embeddedModelChanged', {
                detail: {
                    provider: 'openrouter',
                    modelName: modelSelect.value
                }
            }));
            
            // Update our display directly
            this.updateSelectedModelDisplay();
        });
        
        // Initial check
        checkToken();
    }
    
    /**
     * Initialize event listeners
     */
    initEventListeners() {
        // Add save button listener
        const saveButton = this.querySelector('#saveSystemSettings');
        if (saveButton) {
            saveButton.addEventListener('click', () => this.saveSettings());
        }
        
        // Provider type radio listeners
        const providerRadios = this.querySelectorAll('input[name="providerType"]');
        providerRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.toggleProviderSections(e.target.value);
                this.updateSelectedModelDisplay();
            });
        });
        
        // Initialize embedded widgets
        this.initEmbeddedOllamaWidget();
        this.initEmbeddedOpenrouterWidget();
        
        // Listen for model selection changes in global widgets as well
        document.addEventListener('DOMContentLoaded', () => {
            const ollamaModelSelect = document.getElementById('ollamaModelSelect');
            const openrouterModelSelect = document.getElementById('openrouterModelSelect');
            
            if (ollamaModelSelect) {
                ollamaModelSelect.addEventListener('change', () => {
                    // Update embedded widget selection
                    const embeddedSelect = this.querySelector('#embeddedOllamaModelSelect');
                    if (embeddedSelect && embeddedSelect.value !== ollamaModelSelect.value) {
                        embeddedSelect.value = ollamaModelSelect.value;
                    }
                    this.updateSelectedModelDisplay();
                });
            }
            
            if (openrouterModelSelect) {
                openrouterModelSelect.addEventListener('change', () => {
                    // Update embedded widget selection
                    const embeddedSelect = this.querySelector('#embeddedOpenrouterModelSelect');
                    if (embeddedSelect && embeddedSelect.value !== openrouterModelSelect.value) {
                        embeddedSelect.value = openrouterModelSelect.value;
                    }
                    this.updateSelectedModelDisplay();
                });
            }
        });
    }

    /**
     * Render the component
     */
    render() {
        this.innerHTML = `
            <div class="space-y-6 mb-6">
                <div id="settingsLoadingIndicator" class="flex items-center">
                    <div class="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-500"></div>
                    <span class="ml-2 text-sm text-neutral-400">Loading settings...</span>
                </div>
                
                <!-- Provider Selection -->
                <div class="mb-4">
                    <h5 class="text-sm uppercase text-neutral-300 font-bold mb-4">Choose a Model Provider</h5>
                    <div class="flex space-x-6 mb-4">
                        <div class="flex items-center p-2 rounded hover:bg-neutral-900">
                            <input type="radio" id="providerLocal" name="providerType" value="ollama" 
                                   class="mr-2" checked>
                            <label for="providerLocal" class="text-white cursor-pointer">Local (Ollama)</label>
                        </div>
                        <div class="flex items-center p-2 rounded hover:bg-neutral-900">
                            <input type="radio" id="providerAPI" name="providerType" value="openrouter"
                                   class="mr-2">
                            <label for="providerAPI" class="text-white cursor-pointer">API (OpenRouter)</label>
                        </div>
                    </div>
                    
                    <!-- Provider Descriptions -->
                    <div class="text-sm text-neutral-400 mb-2">
                        <p class="provider-description ollama-description">Local models run on your hardware with Ollama. CPU requirements vary by model size.</p>
                        <p class="provider-description openrouter-description hidden">OpenRouter provides access to various AI models through their API with usage credits.</p>
                    </div>
                    
                    <!-- Model Selection Widgets -->
                    <div class="mt-4 pt-4 border-t border-neutral-800">
                        <!-- Ollama Widget Section (Embedded) -->
                        <div id="embeddedOllamaWidget" class="ollama-section">
                            <div class="flex items-center justify-between mb-4">
                                <div class="flex items-center">
                                    <div class="w-8 h-8 mr-3">
                                        <img src="/static/icons/OLLAMALOGO.png" class="w-full h-full" alt="Ollama Logo" />
                                    </div>
                                    <div>
                                        <h3 class="text-sm font-medium">Ollama Models</h3>
                                        <div class="flex items-center mt-1">
                                            <div id="embeddedOllamaStatusIcon" class="w-2 h-2 rounded-full bg-yellow-500 mr-2"></div>
                                            <p id="embeddedOllamaStatus" class="text-xs text-neutral-400">Checking status...</p>
                                        </div>
                                    </div>
                                </div>
                                <button id="embeddedOllamaRefreshBtn" class="text-neutral-400 hover:text-white transition-colors">
                                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                                    </svg>
                                </button>
                            </div>
                            
                            <div class="mb-4">
                                <label class="block text-sm font-medium text-neutral-400 mb-2">Available Models</label>
                                <select id="embeddedOllamaModelSelect" class="w-full bg-neutral-700 text-white p-2 rounded border border-neutral-600 focus:border-indigo-500 focus:outline-none">
                                    <option value="loading">Loading models...</option>
                                </select>
                            </div>
                            
                            <div id="embeddedOllamaModelsContainer" class="mt-2">
                                <div id="embeddedOllamaModelList" class="text-xs text-neutral-400">
                                    <p>Select a model from the dropdown above.</p>
                                </div>
                            </div>
                        </div>
                        
                        <!-- OpenRouter Widget Section (Embedded) -->
                        <div id="embeddedOpenrouterWidget" class="openrouter-section hidden">
                            <div class="flex items-center justify-between mb-4">
                                <div class="flex items-center">
                                    <div class="w-8 h-8 mr-3">
                                        <img src="/static/icons/OPENROUTERLOGO.png" class="w-full h-full" alt="OpenRouter Logo" 
                                             onerror="this.src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBhdGggZD0iTTEyIDJMMiA3bDEwIDVNMTIgMmwxMCA1LTEwIDVNMiAxN2wxMCA1IDEwLTUiLz48L3N2Zz4='" />
                                    </div>
                                    <div>
                                        <h3 class="text-sm font-medium">OpenRouter Models</h3>
                                        <div class="flex items-center mt-1">
                                            <div id="embeddedOpenrouterStatusIcon" class="w-2 h-2 rounded-full bg-yellow-500 mr-2"></div>
                                            <p id="embeddedOpenrouterStatus" class="text-xs text-neutral-400">Checking status...</p>
                                        </div>
                                    </div>
                                </div>
                                <button id="embeddedOpenrouterRefreshBtn" class="text-neutral-400 hover:text-white transition-colors">
                                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                                    </svg>
                                </button>
                            </div>
                            
                            <!-- API Token input field -->
                            <div class="mb-4">
                                <label class="block text-sm font-medium text-neutral-400 mb-2">API Token</label>
                                <div class="flex space-x-2">
                                    <input id="embeddedOpenrouterTokenInput" type="password" placeholder="Enter your OpenRouter API Token" 
                                           class="flex-1 bg-neutral-700 px-3 py-2 text-sm rounded border border-neutral-600 focus:border-indigo-500 focus:outline-none">
                                    <button id="embeddedOpenrouterSaveToken" 
                                            class="bg-indigo-600 hover:bg-indigo-700 px-3 py-2 rounded flex items-center">
                                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                                        </svg>
                                    </button>
                                </div>
                                <p class="text-xs text-neutral-400 mt-1"><a href="https://openrouter.ai/keys" target="_blank" class="text-indigo-400 hover:underline">Get your API key here</a></p>
                            </div>
                            
                            <!-- Model selection -->
                            <div class="mb-4">
                                <label class="block text-sm font-medium text-neutral-400 mb-2">Available Models</label>
                                <select id="embeddedOpenrouterModelSelect" class="w-full bg-neutral-700 text-white p-2 rounded border border-neutral-600 focus:border-indigo-500 focus:outline-none">
                                    <option value="loading">Loading models...</option>
                                </select>
                            </div>
                            
                            <div id="embeddedOpenrouterModelsContainer" class="mt-2">
                                <div id="embeddedOpenrouterModelList" class="text-xs text-neutral-400">
                                    <p>Select a model from the dropdown above.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Currently Selected Model Display -->
                    <div id="selectedModelDisplay" class="mt-4 pt-4 border-t border-neutral-800 text-sm">
                        <div class="flex justify-between items-center">
                            <span class="text-neutral-400">Currently Selected Model:</span>
                            <span id="currentModelName" class="text-indigo-400 font-medium">Loading...</span>
                        </div>
                    </div>
                </div>
                
                <!-- System Prompt -->
                <div class="mb-4">
                    <h5 class="text-sm uppercase text-neutral-300 font-bold mb-4">System Prompt</h5>
                    <textarea id="systemPrompt"
                            class="w-full p-3 bg-neutral-800 rounded border border-neutral-700 text-white h-32"
                            placeholder="You are a Knowledge Base Assistant that helps with retrieving and explaining information."></textarea>
                    <div class="text-xs text-neutral-400 mt-2">
                        The system prompt defines the AI's behavior and capabilities.
                    </div>
                </div>

                <!-- Training Schedule -->
                <div>
                    <h5 class="text-sm uppercase text-neutral-300 font-bold mb-4">Daily Training Schedule</h5>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm text-neutral-400 mb-2" for="trainingStartTime">Start Time</label>
                            <select id="trainingStartTime" class="w-full p-2 bg-neutral-800 rounded border border-neutral-700 text-white">
                                ${Array.from({ length: 24 }, (_, i) => {
                                    const hour = i.toString().padStart(2, '0');
                                    return `<option value="${hour}:00">${hour}:00</option>`;
                                }).join('')}
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm text-neutral-400 mb-2" for="trainingStopTime">Stop Time</label>
                            <select id="trainingStopTime" class="w-full p-2 bg-neutral-800 rounded border border-neutral-700 text-white">
                                ${Array.from({ length: 24 }, (_, i) => {
                                    const hour = i.toString().padStart(2, '0');
                                    return `<option value="${hour}:00">${hour}:00</option>`;
                                }).join('')}
                            </select>
                        </div>
                    </div>
                    <p class="text-xs text-neutral-400 mt-2">Training will run automatically between these hours every day.</p>
                </div>
                
                <!-- Save Button -->
                <div class="mt-6 flex justify-end">
                    <button id="saveSystemSettings" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-md flex items-center justify-center">
                        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                        </svg>
                        Save Settings
                    </button>
                </div>
            </div>
        `;
    }
}

// Register the custom element
customElements.define('system-settings', SystemSettings);