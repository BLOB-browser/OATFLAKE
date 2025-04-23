class ResponseBox extends HTMLElement {
    constructor() {
        super();
        this.responseText = '';
        this.metadata = null;
    }

    connectedCallback() {
        this.render();
    }

    render() {
        this.innerHTML = `
            <div id="responseContainer" class="min-h-[100px] mb-4 md:mb-6 hidden">
            <div class="bg-black p-3 md:p-4 border-stone-800 border-2 rounded-3xl">
                <!-- Metadata header -->
                <div id="responseMetadataHeader" class="flex justify-between items-center mb-2 pb-2 border-b border-stone-700 text-sm">
                <div class="flex items-center gap-2 text-stone-400">
                    <!-- Improved layout: provider icon, provider name, >, model icon, model name -->
                    <div class="flex items-center gap-1">
                    <img id="providerIcon" src="/icons/models/default.png" alt="Provider" class="w-5 h-5" />
                    <span id="providerInfo" class="font-mono"></span>
                    <span class="mx-1 text-stone-600">›</span>
                    <img id="modelIcon" src="/icons/models/default.png" alt="Model" class="w-5 h-5" />
                    <span id="modelInfo" class="font-mono"></span>
                    </div>
                </div>
                <div class="flex items-center gap-2 text-stone-400">
                    <img src="/icons/24/timer.png" alt="Timer" class="w-5 h-5" />
                    <span id="timingInfo" class="font-mono"></span>
                </div>
                </div>
                
                <div class="text-base md:text-lg font-bold mb-2 text-white">AI Response:</div>
                <div id="responseText" class="text-gray-200 font-mono whitespace-pre-wrap text-sm md:text-base"></div>
                
                <!-- Metadata footer -->
                <div id="responseMetadataFooter" class="flex flex-wrap gap-3 mt-4 pt-2 border-t border-stone-700 text-xs text-stone-400">
                <!-- Will be populated with metadata -->
                </div>
            </div>
            </div>
        `;
    }

    setResponse(text, metadata = null) {
        console.log('Response box setting response:', text ? text.substring(0, 50) + '...' : 'empty',
            'Metadata:', metadata ? 'present' : 'none');

        if (!text || text.trim() === '') {
            this.clear();
            return;
        }

        this.responseText = text;
        this.metadata = metadata;
        this.updateDisplay();
    }

    updateDisplay() {
        const container = this.querySelector('#responseContainer');
        const responseText = this.querySelector('#responseText');

        if (!container || !responseText) {
            console.error('Response container or text element not found');
            return;
        }

        // First make sure container is visible
        container.classList.remove('hidden');

        // Then set the response text
        responseText.textContent = this.responseText;

        // Handle metadata display
        if (this.metadata) {
            this.displayResponseMetadata(this.metadata);
        } else {
            // Hide metadata sections if no metadata available
            const header = this.querySelector('#responseMetadataHeader');
            const footer = this.querySelector('#responseMetadataFooter');
            if (header) header.style.display = 'none';
            if (footer) footer.style.display = 'none';
        }
    }

    displayResponseMetadata(metadata) {
        // Update header metadata
        const providerInfoEl = this.querySelector('#providerInfo');
        const modelInfoEl = this.querySelector('#modelInfo');
        const timingInfoEl = this.querySelector('#timingInfo');
        const metadataFooter = this.querySelector('#responseMetadataFooter');
        const providerIconImg = this.querySelector('#providerIcon');
        const modelIconImg = this.querySelector('#modelIcon');

        if (!metadata || !providerInfoEl || !modelInfoEl || !timingInfoEl || !metadataFooter) return;

        // Ensure metadata header is visible
        const header = this.querySelector('#responseMetadataHeader');
        if (header) header.style.display = 'flex';

        // Format model info
        const model = metadata.model || {};
        let provider = model.provider ?
            model.provider.charAt(0).toUpperCase() + model.provider.slice(1) : 'Unknown';
        let modelName = model.model_name || 'Unknown';
        let originalProvider = provider; // Keep track of original provider for icon selection

        // Handle OpenRouter models differently - they often come in "provider/model" format
        // For OpenRouter, check either openrouter_model or model_name field
        let openRouterSubProvider = null;
        let modelPath = null;

        if (provider.toLowerCase() === 'openrouter') {
            // Check either field, prefer openrouter_model if exists
            modelPath = model.openrouter_model || model.model_name;

            if (modelPath && modelPath.includes('/')) {
                const parts = modelPath.split('/');
                if (parts.length > 1) {
                    openRouterSubProvider = parts[0];
                    provider = parts[0].charAt(0).toUpperCase() + parts[0].slice(1);
                    modelName = parts[1].replace(/-/g, ' ');
                }
            }
        }

        // Set the provider and model info text separately
        providerInfoEl.textContent = originalProvider;
        modelInfoEl.textContent = modelName;

        // Set both provider and model icons
        if (providerIconImg && modelIconImg) {
            // Get provider icon
            let providerIconPath = '/static/icons/models/default.png';

            if (originalProvider.toLowerCase() === 'openrouter') {
                // For OpenRouter, we'll show the OpenRouter icon as provider
                providerIconPath = '/static/icons/models/openrouter.png';

                // And the specific model provider as the model icon
                if (openRouterSubProvider) {
                    switch (openRouterSubProvider.toLowerCase()) {
                        case 'anthropic':
                            modelIconImg.src = '/static/icons/models/antrophic.png';
                            break;
                        case 'mistralai':
                            modelIconImg.src = '/static/icons/models/mistral.png';
                            break;
                        case 'google':
                            modelIconImg.src = '/static/icons/models/google.png';
                            break;
                        case 'meta':
                            modelIconImg.src = '/static/icons/models/meta.png';
                            break;
                        case 'cohere':
                            modelIconImg.src = '/static/icons/models/cohere.png';
                            break;
                        default:
                            // Try to infer from model name
                            modelIconImg.src = this.getModelSpecificIcon(modelName.toLowerCase());
                    }
                } else {
                    // If we can't determine the provider from the path, try to infer from model name
                    modelIconImg.src = this.getModelSpecificIcon(modelName.toLowerCase());
                }
            } else {
                // For other providers, set the provider icon
                switch (originalProvider.toLowerCase()) {
                    case 'ollama':
                        providerIconPath = '/static/icons/models/ollama.png';
                        break;
                    case 'openai':
                        providerIconPath = '/static/icons/models/openai.png';
                        break;
                    case 'anthropic':
                        providerIconPath = '/static/icons/models/antrophic.png';
                        break;
                    case 'meta':
                        providerIconPath = '/static/icons/models/meta.png';
                        break;
                    case 'google':
                        providerIconPath = '/static/icons/models/google.png';
                        break;
                    case 'mistral':
                        providerIconPath = '/static/icons/models/mistral.png';
                        break;
                    case 'cohere':
                        providerIconPath = '/static/icons/models/cohere.png';
                        break;
                    default:
                        providerIconPath = '/static/icons/models/default.png';
                }

                // And set the model-specific icon
                modelIconImg.src = this.getModelSpecificIcon(modelName.toLowerCase());
            }

            // Set the provider icon
            providerIconImg.src = providerIconPath;
            providerIconImg.alt = `${originalProvider} Provider`;

            // Set alt text for model icon
            modelIconImg.alt = `${modelName} Model`;

            // Add error handlers
            providerIconImg.onerror = function () {
                this.src = '/static/icons/models/default.png';
            };

            modelIconImg.onerror = function () {
                this.src = '/static/icons/models/default.png';
            };
        }

        // Format timing info
        const timing = metadata.timing || {};
        const totalTime = timing.total_seconds != null ?
            Number(timing.total_seconds).toFixed(2) + 's' : 'Unknown';
        timingInfoEl.textContent = totalTime;

        // Populate detailed footer metadata
        const retrievalTime = timing.retrieval_seconds != null ?
            Number(timing.retrieval_seconds).toFixed(2) + 's' : 'Unknown';
        const generationTime = timing.generation_seconds != null ?
            Number(timing.generation_seconds).toFixed(2) + 's' : 'Unknown';

        metadataFooter.innerHTML = `
            <div class="flex items-center gap-1">
                <span class="font-semibold">Retrieval:</span>
                <span class="font-mono">${retrievalTime}</span>
            </div>
            <div class="flex items-center gap-1">
                <span class="font-semibold">Generation:</span>
                <span class="font-mono">${generationTime}</span>
            </div>
            <div class="flex items-center gap-1">
                <span class="font-semibold">Words:</span>
                <span class="font-mono">${metadata.word_count || 0}</span>
            </div>
        `;
    }

    // New helper method to get model-specific icons
    getModelSpecificIcon(modelName) {
        // The following checks the model name for specific patterns
        // and returns the appropriate icon

        if (modelName.includes('llama') || modelName.includes('llama3') || modelName.includes('tinyllama')) {
            return '/icons/models/meta.png';
        }
        else if (modelName.includes('mistral')) {
            return '/icons/models/mistral.png';
        }
        else if (modelName.includes('gemma') || modelName.includes('gemini')) {
            return '/icons/models/google.png';
        }
        else if (modelName.includes('phi')) {
            return '/icons/models/microsoft.png';
        }
        else if (modelName.includes('qwen')) {
            return '/icons/models/qwen.png';
        }
        else if (modelName.includes('claude')) {
            return '/icons/models/antrophic.png';
        }
        else if (modelName.includes('command')) {
            return '/icons/models/cohere.png';
        }

        // Default icon if no specific match
        return '/icons/models/default.png';
    }

    getModelIcon(provider, modelName) {
        // Default icon path
        let iconPath = '/icons/models/default.png';  // Updated to use /icons/models/ path

        // Normalize inputs for matching
        const providerLower = provider?.toLowerCase() || '';
        const modelLower = modelName?.toLowerCase() || '';

        // Handle different providers
        if (providerLower === 'ollama') {
            // Ollama model mapping - match patterns from system-settings.js
            if (modelLower.includes('llama3.2') || modelLower.includes('llama3:') || modelLower.includes('llama-3') || modelLower.includes('tinyllama')) {
                iconPath = '/icons/models/meta.png';
            }
            else if (modelLower.includes('mistral')) {
                iconPath = '/icons/models/mistral.png';
            }
            else if (modelLower.includes('gemma')) {
                iconPath = '/icons/models/google.png';
            }
            else if (modelLower.includes('phi')) {
                iconPath = '/icons/models/microsoft.png';
            }
            else if (modelLower.includes('qwen')) {
                iconPath = '/icons/models/qwen.png';
            }
            else {
                // Default Ollama icon if we can't determine the specific model
                iconPath = '/icons/models/ollama.png';
            }
        }
        else if (providerLower === 'openrouter') {
            // For OpenRouter, extract the provider from the model name format "provider/model"
            const openRouterProvider = modelLower.split('/')[0];

            switch (openRouterProvider) {
                case 'anthropic':
                    iconPath = '/icons/models/antrophic.png';
                    break;
                case 'mistralai':
                    iconPath = '/icons/models/mistral.png';
                    break;
                case 'google':
                    iconPath = '/icons/models/google.png';
                    break;
                case 'meta':
                    iconPath = '/icons/models/meta.png';
                    break;
                case 'cohere':
                    iconPath = '/icons/models/cohere.png';
                    break;
                default:
                    // If provider not recognized, try to determine from model name
                    if (modelLower.includes('claude') || modelLower.includes('anthropic')) {
                        iconPath = '/icons/models/antrophic.png';
                    }
                    else if (modelLower.includes('llama') || modelLower.includes('meta')) {
                        iconPath = '/icons/models/meta.png';
                    }
                    else if (modelLower.includes('gemini') || modelLower.includes('google')) {
                        iconPath = '/icons/models/google.png';
                    }
                    else if (modelLower.includes('mistral')) {
                        iconPath = '/icons/models/mistral.png';
                    }
                    else if (modelLower.includes('command')) {
                        iconPath = '/icons/models/cohere.png';
                    }
            }
        }
        else if (providerLower === 'openai') {
            iconPath = '/icons/models/openai.png';
        }
        else if (providerLower === 'anthropic') {
            iconPath = '/icons/models/antrophic.png';
        }
        else if (providerLower === 'meta') {
            iconPath = '/icons/models/meta.png';
        }
        else if (providerLower === 'google') {
            iconPath = '/icons/models/google.png';
        }
        else if (providerLower === 'mistral') {
            iconPath = '/icons/models/mistral.png';
        }
        else if (providerLower === 'cohere') {
            iconPath = '/icons/models/cohere.png';
        }

        // Add fallback to handle any icon loading errors
        return iconPath;
    }

    clear() {
        const container = this.querySelector('#responseContainer');
        if (container) container.classList.add('hidden');
    }
}

customElements.define('response-box', ResponseBox);
