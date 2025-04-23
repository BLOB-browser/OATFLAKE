/**
 * SearchSlide - Handles the generation of the search interface
 * Uses our custom search-box component
 */
const SearchSlide = (() => {
    // HTML template for the search interface using web components
    const template = `
        <div class="h-full flex flex-col p-0 overflow-hidden w-full">
            <!-- Search Container - similar to your previous frontend -->
            <div class="min-h-[60vh] mb-4 md:mb-6 w-full">
                <div class="bg-black min-h-[60vh] md:min-h-[70vh] border-stone-800 border-2 rounded-3xl content-center h-full w-full" id="searchBoxContainer">
                    <!-- search-box will be added here directly -->
                </div>
            </div>
            
            <!-- Response Container -->
            <div id="responseContainer" class="bg-neutral-800 rounded-lg p-6 border border-neutral-700 shadow-lg w-full mb-8 hidden">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-xl font-medium">Summary</h3>
                    <div class="text-xs text-neutral-400" id="responseMetadataHeader">
                        <span id="timingInfo"></span>
                    </div>
                </div>
                <div id="responseText" class="prose prose-invert max-w-none mb-4"></div>
                <div id="responseMetadataFooter" class="flex items-center justify-end text-xs text-neutral-400">
                    <div class="mr-2">Powered by:</div>
                    <img id="providerIcon" class="h-4 w-4 mr-1" src="" alt="" />
                    <span id="providerInfo" class="mr-2"></span>
                    <img id="modelIcon" class="h-4 w-4 mr-1" src="" alt="" />
                    <span id="modelInfo"></span>
                </div>
            </div>
        </div>
    `;

    /**
     * Render the search interface in the container
     * @param {HTMLElement} container - The container element to render the search interface in
     */
    function render(container) {
        if (!container) return;
        
        // Insert template into container
        container.innerHTML = template;
        
        // Just load the search-box component directly - no need for the widget intermediary
        loadScript('/static/js/components/search-box.js')
            .then(() => {
                console.log('Search box component loaded successfully');
                
                // Wait a brief moment to ensure the component is fully registered
                setTimeout(() => {
                    try {
                        // Explicitly create the search-box element and add it to the container
                        const searchBoxContainer = container.querySelector('#searchBoxContainer');
                        if (searchBoxContainer) {
                            // Create the HTML directly instead of using createElement
                            searchBoxContainer.innerHTML = '<search-box id="searchBox"></search-box>';
                            console.log('Search box added to DOM using innerHTML');
                            
                            // Set up event listener for search responses
                            document.addEventListener('search-response', function(event) {
                                if (event.detail) {
                                    const responseData = event.detail;
                                    handleSearchResponse(responseData);
                                }
                            });
                        }
                    } catch (error) {
                        console.error('Error creating search-box:', error);
                    }
                }, 100);
            })
            .catch(error => {
                console.error('Failed to load search components:', error);
            });
    }
    
    /**
     * Handle the search response and update the UI using the response-box component
     * @param {Object} responseData - The response data from the search
     */
    function handleSearchResponse(responseData) {
        // Find existing response-box or create a new one
        let responseBox = document.querySelector('response-box');
        
        if (!responseBox) {
            console.log('Creating new response-box component');
            // Create the response-box component
            responseBox = document.createElement('response-box');
            responseBox.id = 'responseBox';
            
            // Find where to insert the response box (after the search container)
            const searchContainer = document.getElementById('searchContainer');
            const responseContainer = document.getElementById('responseContainer');
            
            // If the response container exists, replace it with our response-box
            if (responseContainer) {
                responseContainer.parentNode.replaceChild(responseBox, responseContainer);
            } 
            // Otherwise insert after the search container
            else if (searchContainer && searchContainer.parentNode) {
                searchContainer.parentNode.insertBefore(responseBox, searchContainer.nextSibling);
            }
            // Fallback - append to the main content
            else {
                const mainContent = document.getElementById('mainContent') || document.body;
                mainContent.appendChild(responseBox);
            }
        }
        
        console.log('Setting response in response-box component');
        
        // Prepare the metadata
        const metadata = responseData.metadata || {};
        const formattedMetadata = {
            timing: {
                total_seconds: metadata.timing?.total_seconds || 0,
                retrieval_seconds: metadata.timing?.retrieval_seconds || 0,
                generation_seconds: metadata.timing?.generation_seconds || 0
            },
            model: {
                provider: metadata.model?.provider || 'Ollama',
                model_name: metadata.model?.model_name || 'Unknown'
            },
            word_count: responseData.response ? responseData.response.split(/\s+/).length : 0
        };
        
        // Set the response in the response-box component
        responseBox.setResponse(responseData.response || responseData.text || '', formattedMetadata);
    }
    
    /**
     * Update response metadata in the UI
     * @param {Object} metadata - The metadata object with timing and model info
     */
    function updateResponseMetadata(metadata) {
        const timingInfo = document.getElementById('timingInfo');
        const providerInfo = document.getElementById('providerInfo');
        const modelInfo = document.getElementById('modelInfo');
        const providerIcon = document.getElementById('providerIcon');
        const modelIcon = document.getElementById('modelIcon');
        
        // Update timing information
        if (timingInfo && metadata.timing) {
            const totalTime = metadata.timing.total_seconds || 0;
            timingInfo.textContent = `${totalTime.toFixed(1)}s`;
        }
        
        // Update provider info
        if (providerInfo && metadata.model && metadata.model.provider) {
            providerInfo.textContent = metadata.model.provider;
        }
        
        // Update model info
        if (modelInfo && metadata.model && metadata.model.model_name) {
            modelInfo.textContent = metadata.model.model_name;
        }
        
        // Update icons if available
        if (providerIcon && metadata.model && metadata.model.provider) {
            const provider = metadata.model.provider.toLowerCase();
            
            // Set icon based on provider
            if (provider.includes('openai')) {
                providerIcon.src = '/static/icons/openai.svg';
            } else if (provider.includes('ollama')) {
                providerIcon.src = '/static/icons/ollama.svg';
            } else if (provider.includes('openrouter')) {
                providerIcon.src = '/static/icons/openrouter.svg';
            } else {
                providerIcon.src = '/static/icons/ai.svg';
            }
            
            providerIcon.alt = metadata.model.provider;
        }
        
        // Set model icon
        if (modelIcon) {
            modelIcon.src = '/static/icons/model.svg';
            modelIcon.alt = 'Model';
        }
    }
    
    /**
     * Helper to dynamically load script
     * @param {string} src - Script source URL
     * @returns {Promise} - Promise that resolves when the script is loaded
     */
    function loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.type = 'module'; // Add module type for ES module scripts
            script.onload = () => resolve();
            script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
            document.head.appendChild(script);
        });
    }

    // Return public API
    return {
        render
    };
})();

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    const searchContainer = document.getElementById('searchContainer');
    if (searchContainer) {
        SearchSlide.render(searchContainer);
    }
});
