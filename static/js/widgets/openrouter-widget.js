/**
 * OpenRouter Widget Module
 * Handles all OpenRouter-related functionality
 */

const OpenRouterWidget = (() => {
    // Cache DOM elements
    const elements = {
        statusElement: () => document.getElementById('openrouterStatus'),
        statusIcon: () => document.getElementById('openrouterStatusIcon'),
        modelSelect: () => document.getElementById('openrouterModelSelect'),
        modelListContainer: () => document.getElementById('openrouterModelList'),
        refreshBtn: () => document.getElementById('openrouterRefreshBtn'),
        tokenInput: () => document.getElementById('openrouterTokenInput')
    };

    /**
     * Initialize the OpenRouter widget
     */
    function initialize() {
        // Set up event listeners
        if (elements.refreshBtn()) {
            elements.refreshBtn().onclick = refreshOpenRouter;
        }
        
        // Set up model select change handler
        if (elements.modelSelect()) {
            elements.modelSelect().onchange = changeOpenRouterModel;
        }
        
        // Check token status on initialization
        checkTokenStatus();
    }

    /**
     * Update the OpenRouter status display
     * @param {boolean} isConnected - Whether OpenRouter is connected
     */
    function updateStatus(isConnected) {
        const statusElement = elements.statusElement();
        const statusIcon = elements.statusIcon();
        
        if (!statusElement || !statusIcon) return;
        
        if (isConnected) {
            statusElement.textContent = 'Connected';
            statusElement.classList.remove('text-neutral-400', 'text-red-500');
            statusElement.classList.add('text-green-500');
            
            statusIcon.classList.remove('bg-yellow-500', 'bg-red-500');
            statusIcon.classList.add('bg-green-500');
            
            // Now we can try to load the available models
            loadModels();
        } else {
            statusElement.textContent = 'API Key Required';
            statusElement.classList.remove('text-neutral-400', 'text-green-500');
            statusElement.classList.add('text-red-500');
            
            statusIcon.classList.remove('bg-yellow-500', 'bg-green-500');
            statusIcon.classList.add('bg-red-500');
        }
    }

    /**
     * Load available OpenRouter models
     */
    async function loadModels() {
        try {
            const modelSelect = elements.modelSelect();
            const modelListContainer = elements.modelListContainer();
            
            if (!modelSelect || !modelListContainer) return;
            
            // Add loading state
            modelSelect.innerHTML = '<option value="loading">Loading models...</option>';
            modelListContainer.innerHTML = '<p class="text-neutral-400">Loading models...</p>';
            
            // Try to fetch available models from our API endpoint
            const response = await fetch('/api/openrouter/models');
            if (!response.ok) {
                throw new Error(`Failed to fetch models: ${response.statusText}`);
            }
            
            const data = await response.json();
            if (!data.success) {
                throw new Error(data.message || 'Unknown error');
            }
            
            const models = data.models || [];
            const currentModel = data.current_model || '';
            
            console.log('Current model from settings:', currentModel);
            
            // Clear and populate the select
            modelSelect.innerHTML = '';
            
            if (models.length === 0) {
                modelSelect.innerHTML = '<option value="">No models found</option>';
                modelListContainer.innerHTML = '<p class="text-neutral-400">No models available</p>';
                return;
            }
            
            // Add each model to the select
            models.forEach(model => {
                const option = document.createElement('option');
                const modelId = model.id || model;
                
                option.value = modelId;
                // Format display name nicely
                const displayName = model.name || modelId.split('/').pop();
                option.textContent = displayName + (model.is_free ? ' (Free)' : '');
                
                // Set as selected if it matches the current model
                if (currentModel && (modelId === currentModel)) {
                    option.selected = true;
                }
                
                modelSelect.appendChild(option);
            });
            
            // Create model list with better visual indicators for selection
            modelListContainer.innerHTML = '';
            const modelList = document.createElement('ul');
            modelList.className = 'space-y-2 mt-3';
            
            models.slice(0, 5).forEach(model => { // Show only top 5 models in list
                const modelId = model.id || model;
                const displayName = model.name || modelId.split('/').pop();
                const isFree = model.is_free || modelId.includes(':free');
                const isSelected = modelId === currentModel;
                
                const listItem = document.createElement('li');
                listItem.className = `flex items-center p-1.5 rounded ${isSelected ? 'bg-neutral-700' : ''}`;
                
                // Selection indicator
                const indicator = document.createElement('span');
                indicator.className = 'w-2 h-2 rounded-full mr-2.5';
                indicator.classList.add(isSelected ? 'bg-green-500' : 'bg-neutral-500');
                
                // Model name with provider
                const nameSpan = document.createElement('span');
                nameSpan.className = `flex-1 ${isSelected ? 'font-medium' : ''}`;
                nameSpan.textContent = displayName;
                
                // Free badge
                if (isFree) {
                    const badge = document.createElement('span');
                    badge.className = 'text-xs px-1.5 py-0.5 bg-green-900 text-green-300 rounded ml-1';
                    badge.textContent = 'Free';
                    nameSpan.appendChild(badge);
                }
                
                listItem.appendChild(indicator);
                listItem.appendChild(nameSpan);
                modelList.appendChild(listItem);
            });
            
            // Add a note if there are more models
            if (models.length > 5) {
                const moreItem = document.createElement('li');
                moreItem.className = 'text-xs text-neutral-500 text-center mt-1';
                moreItem.textContent = `+${models.length - 5} more models available`;
                modelList.appendChild(moreItem);
            }
            
            modelListContainer.appendChild(modelList);
            
        } catch (error) {
            console.error('Error loading OpenRouter models:', error);
            const modelSelect = elements.modelSelect();
            const modelListContainer = elements.modelListContainer();
            
            if (modelSelect) {
                modelSelect.innerHTML = '<option value="">Error loading models</option>';
            }
            
            if (modelListContainer) {
                modelListContainer.innerHTML = 
                    '<p class="text-red-500">Error loading models: ' + error.message + '</p>';
            }
        }
    }

    /**
     * Change the active OpenRouter model
     */
    async function changeOpenRouterModel() {
        const modelSelect = elements.modelSelect();
        
        if (!modelSelect) return;
        
        const selectedModel = modelSelect.value;
        
        if (!selectedModel || selectedModel === 'loading') return;
        
        try {
            // Call the backend API to update the model setting
            const response = await fetch('/api/openrouter/set-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: selectedModel })
            });
            
            const result = await response.json();
            
            if (result.success) {
                console.log(`OpenRouter model changed to ${selectedModel}`);
                // Reload the models to refresh the UI with the new selection
                loadModels();
            } else {
                console.error('Failed to change model:', result.message);
                alert(`Failed to change model: ${result.message || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error changing model:', error);
            alert(`Error changing model: ${error.message}`);
        }
    }

    /**
     * Refresh OpenRouter status and model list
     */
    async function refreshOpenRouter() {
        const refreshBtn = elements.refreshBtn();
        
        if (!refreshBtn) return;
        
        refreshBtn.classList.add('opacity-50');
        
        try {
            // Check status first
            const statusResponse = await fetch('/api/openrouter/status');
            if (statusResponse.ok) {
                const statusData = await statusResponse.json();
                updateStatus(statusData.status === 'connected');
            } else {
                updateStatus(false);
            }
            
            // Then check token and load models if we're connected
            await checkTokenStatus();
        } catch (error) {
            console.error('Error refreshing OpenRouter status:', error);
        } finally {
            setTimeout(() => {
                refreshBtn.classList.remove('opacity-50');
            }, 500);
        }
    }

    /**
     * Check if an OpenRouter token is set
     */
    async function checkTokenStatus() {
        try {
            const response = await fetch('/api/openrouter/token-status');
            const data = await response.json();
            
            const tokenInput = elements.tokenInput();
            
            if (!tokenInput) return;
            
            if (data.has_token) {
                tokenInput.placeholder = data.masked_token || "API token is set";
                
                // Update connection status
                updateStatus(data.valid);
            } else {
                tokenInput.placeholder = "Enter your OpenRouter API Token";
                updateStatus(false);
            }
        } catch (error) {
            console.error('Error checking token status:', error);
            updateStatus(false);
        }
    }

    /**
     * Save the OpenRouter API token
     */
    async function saveToken() {
        const tokenInput = elements.tokenInput();
        
        if (!tokenInput) return;
        
        const token = tokenInput.value.trim();
        if (!token) {
            alert('Please enter a valid OpenRouter API Token');
            return;
        }

        try {
            console.log("Sending token to server...");
            // Save the token to the server
            const response = await fetch('/api/openrouter/set-token', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ token: token })
            });

            console.log("Server response status:", response.status);
            
            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }

            const result = await response.json();
            console.log("Server response:", result);
            
            // Show success or failure feedback
            if (result.success) {
                // Show success feedback
                tokenInput.value = '';
                tokenInput.placeholder = "API token saved successfully";
                tokenInput.classList.add('border-green-500');
                
                setTimeout(() => {
                    tokenInput.placeholder = "API token is set";
                    tokenInput.classList.remove('border-green-500');
                }, 3000);
                
                // Update status and load models
                updateStatus(true);
                loadModels();
            } else {
                // Show error feedback
                alert(`Failed to save token: ${result.message || 'Unknown error'}`);
                updateStatus(false);
            }
        } catch (error) {
            console.error('Error saving token:', error);
            alert('Error saving token: ' + error.message);
        }
    }

    // Public API
    return {
        initialize,
        updateStatus,
        loadModels,
        refreshOpenRouter,
        checkTokenStatus,
        saveToken
    };
})();

// Initialize on script load
document.addEventListener('DOMContentLoaded', OpenRouterWidget.initialize);
