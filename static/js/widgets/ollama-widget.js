/**
 * Ollama Widget Module
 * Handles all Ollama-related functionality
 */

const OllamaWidget = (() => {
    // Cache DOM elements
    const elements = {
        statusElement: () => document.getElementById('ollamaStatus'),
        statusIcon: () => document.getElementById('ollamaStatusIcon'),
        modelSelect: () => document.getElementById('ollamaModelSelect'),
        modelListContainer: () => document.getElementById('ollamaModelList'),
        refreshBtn: () => document.getElementById('ollamaRefreshBtn')
    };

    /**
     * Initialize the Ollama widget
     */
    function initialize() {
        // Set up event listeners
        if (elements.refreshBtn()) {
            elements.refreshBtn().onclick = refreshOllama;
        }
        
        // Set up model select change handler
        if (elements.modelSelect()) {
            elements.modelSelect().onchange = changeOllamaModel;
        }
    }

    /**
     * Update the Ollama status display
     * @param {boolean} isConnected - Whether Ollama is connected
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
            statusElement.textContent = 'Disconnected';
            statusElement.classList.remove('text-neutral-400', 'text-green-500');
            statusElement.classList.add('text-red-500');
            
            statusIcon.classList.remove('bg-yellow-500', 'bg-green-500');
            statusIcon.classList.add('bg-red-500');
        }
    }

    /**
     * Load available Ollama models
     */
    async function loadModels() {
        try {
            const modelSelect = elements.modelSelect();
            const modelListContainer = elements.modelListContainer();
            
            if (!modelSelect || !modelListContainer) return;
            
            // Add loading state
            modelSelect.innerHTML = '<option value="loading">Loading models...</option>';
            modelListContainer.innerHTML = '<p class="text-neutral-400">Loading models...</p>';
            
            // Try to fetch available models - handle potential 404 error
            const response = await fetch('/api/ollama/models');
            if (response.status === 404) {
                console.warn('Ollama models API endpoint not found. Using fallback.');
                // Try alternative endpoint if available
                const altResponse = await fetch('/api/status');
                
                if (!altResponse.ok) {
                    throw new Error('Failed to fetch models from alternative endpoint');
                }
                
                const statusData = await altResponse.json();
                if (statusData.ollama_models && Array.isArray(statusData.ollama_models)) {
                    populateModels(statusData.ollama_models, statusData.current_model || null);
                    return;
                }
                
                throw new Error('No models data in status response');
            }
            
            if (!response.ok) {
                throw new Error(`Failed to fetch models: ${response.statusText}`);
            }
            
            const data = await response.json();
            if (!data.models || !Array.isArray(data.models)) {
                throw new Error('Invalid response format');
            }
            
            populateModels(data.models, data.current_model || null);
        } catch (error) {
            console.error('Error loading Ollama models:', error);
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
     * Populate models in UI
     * @param {Array} models - Array of model objects or strings
     * @param {string|null} currentModel - Currently selected model name
     */
    function populateModels(models, currentModel) {
        const modelSelect = elements.modelSelect();
        const modelListContainer = elements.modelListContainer();
        
        if (!modelSelect || !modelListContainer) return;
        
        // Clear and populate the select
        modelSelect.innerHTML = '';
        
        if (models.length === 0) {
            modelSelect.innerHTML = '<option value="">No models found</option>';
            modelListContainer.innerHTML = '<p class="text-neutral-400">No models installed</p>';
            return;
        }
        
        // Add each model to the select
        models.forEach(model => {
            const modelName = model.name || model;
            
            // Add to dropdown
            const option = document.createElement('option');
            option.value = modelName;
            option.textContent = modelName;
            
            // Set as selected if it matches the current model
            if (currentModel && (modelName === currentModel)) {
                option.selected = true;
            }
            
            modelSelect.appendChild(option);
        });
        
        // Create model list
        modelListContainer.innerHTML = '';
        const modelListElement = document.createElement('ul');
        modelListElement.className = 'space-y-1';
        
        models.forEach(model => {
            const modelName = model.name || model;
            const listItem = document.createElement('li');
            listItem.className = 'flex items-center';
            
            // Add active indicator
            const indicator = document.createElement('span');
            indicator.className = 'w-1.5 h-1.5 rounded-full mr-2';
            indicator.classList.add(modelName === currentModel ? 'bg-green-500' : 'bg-neutral-500');
            
            // Add model name
            const nameSpan = document.createElement('span');
            nameSpan.textContent = modelName;
            
            listItem.appendChild(indicator);
            listItem.appendChild(nameSpan);
            modelListElement.appendChild(listItem);
        });
        
        modelListContainer.appendChild(modelListElement);
    }

    /**
     * Change the active Ollama model
     */
    async function changeOllamaModel() {
        const modelSelect = elements.modelSelect();
        
        if (!modelSelect) return;
        
        const selectedModel = modelSelect.value;
        
        if (!selectedModel || selectedModel === 'loading') return;
        
        try {
            const response = await fetch('/api/ollama/set-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: selectedModel })
            });
            
            const result = await response.json();
            
            if (result.success) {
                console.log(`Model changed to ${selectedModel}`);
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
     * Refresh Ollama status
     */
    async function refreshOllama() {
        const refreshBtn = elements.refreshBtn();
        
        if (!refreshBtn) return;
        
        refreshBtn.classList.add('opacity-50');
        
        try {
            // Call the global status update function if available
            if (typeof updateAllStatus === 'function') {
                await updateAllStatus();
            } else {
                // Direct update if global function not available
                try {
                    const response = await fetch('/api/status');
                    if (response.ok) {
                        const statusData = await response.json();
                        updateStatus(statusData.ollama === 'connected');
                    }
                } catch (e) {
                    console.error('Error updating Ollama status:', e);
                }
            }
        } finally {
            setTimeout(() => {
                refreshBtn.classList.remove('opacity-50');
            }, 500);
        }
    }

    // Public API
    return {
        initialize,
        updateStatus,
        loadModels,
        refreshOllama
    };
})();

// Initialize on script load
document.addEventListener('DOMContentLoaded', OllamaWidget.initialize);
