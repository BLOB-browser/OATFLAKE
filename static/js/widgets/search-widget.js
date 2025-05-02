// SearchWidget - Uses web components for search functionality
// Connects search-box and search-container to search.py

// Load the component scripts
function loadSearchComponents() {
    return Promise.all([
        loadScript('/static/js/components/search-box.js'),
        loadScript('/static/js/components/search-container.js')
    ]).catch(error => {
        console.error('Failed to load search components:', error);
    });
}

// Helper function to load scripts
function loadScript(src) {
    return new Promise((resolve, reject) => {
        // Check if already loaded
        if (document.querySelector(`script[src="${src}"]`)) {
            resolve();
            return;
        }
        
        const script = document.createElement('script');
        script.src = src;
        script.type = 'module';
        script.onload = () => resolve();
        script.onerror = (err) => reject(new Error(`Failed to load ${src}`));
        document.head.appendChild(script);
    });
}

document.addEventListener('DOMContentLoaded', async function() {
    // First load the component scripts
    await loadSearchComponents();
    
    // Initialize search components
    initializeSearch();
});

function initializeSearch() {
    console.log('Initializing search with web components');
    
    // Get container elements
    const searchBoxContainer = document.querySelector('.search-box');
    const searchResultsContainer = document.querySelector('.search-results');
    const searchContainer = document.querySelector('.search-container');
    
    if (!searchBoxContainer || !searchContainer) {
        console.error('Search containers not found in DOM');
        return;
    }
    
    // Clear any existing content from containers and add web components
    if (!searchBoxContainer.querySelector('search-box')) {
        // Create search-box component
        const searchBox = document.createElement('search-box');
        
        // Replace the existing content with the component
        searchBoxContainer.innerHTML = '';
        searchBoxContainer.appendChild(searchBox);
    }
    
    if (!searchContainer.querySelector('search-container') && searchResultsContainer) {
        // Connect the search results to the search-box component
        document.addEventListener('search-results', function(event) {
            if (event.detail && event.detail.results) {
                displayResults(event.detail.results);
            }
        });
    }
    
    // Connect to process button for summary generation
    const processButton = document.getElementById('processButton');
    const processContainer = document.getElementById('process-container');
    
    if (processButton) {
        processButton.addEventListener('click', function() {
            // This will be handled by the search-box component's internal logic
            const searchBoxElement = document.querySelector('search-box');
            if (searchBoxElement) {
                // Trigger the summarize method on the search-box component
                if (typeof searchBoxElement.summarize === 'function') {
                    searchBoxElement.summarize();
                } else {
                    // Dispatch a custom event for the search-box to handle
                    searchBoxElement.dispatchEvent(new CustomEvent('summarize-request'));
                }
            }
        });
    }
    
    // Helper function to display search results
    function displayResults(results) {
        // Show the process button when we have results
        if (results && results.length > 0 && processContainer) {
            processContainer.classList.remove('hidden');
        } else if (processContainer) {
            processContainer.classList.add('hidden');
        }
    }
    
    // Override fetch to ensure all API calls go to local search.py
    const originalFetch = window.fetch;
    window.fetch = function(url, options) {
        // Redirect API calls to local endpoints
        if (typeof url === 'string') {
            // For references API
            if (url.includes('/api/references')) {
                console.log('Redirecting to local search.py: /api/references');
                return originalFetch('/api/references', options);
            }
            
            // For summary generation API
            if (url.includes('/api/web')) {
                console.log('Redirecting to local web API: /api/web');
                return originalFetch('/api/web', options);
            }
        }
        
        // For all other requests, use original fetch
        return originalFetch.apply(this, arguments);
    };
    
    // Listen for search responses to display in the response container
    document.addEventListener('search-response', function(event) {
        const responseData = event.detail;
        const responseContainer = document.getElementById('responseContainer');
        const responseText = document.getElementById('responseText');
        
        if (!responseContainer || !responseText || !responseData) return;
        
        // Display response
        responseContainer.classList.remove('hidden');
        responseText.innerHTML = responseData.response || responseData.text || '';
        
        // Update metadata if available
        if (responseData.metadata) {
            updateResponseMetadata(responseData.metadata);
        }
    });
    
    // Update response metadata display
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
        
        // Update provider and model info
        if (metadata.model) {
            // Provider name
            if (providerInfo) {
                providerInfo.textContent = metadata.model.provider || 'Unknown';
            }
            
            // Model name
            if (modelInfo) {
                modelInfo.textContent = metadata.model.model_name || 'Unknown';
            }
            
            // Provider icon
            if (providerIcon) {
                let iconPath = '/static/icons/ai.svg';
                const provider = metadata.model.provider?.toLowerCase() || '';
                
                if (provider === 'openrouter') {
                    iconPath = '/static/icons/openrouter.svg';
                } else if (provider === 'ollama') {
                    iconPath = '/static/icons/ollama.svg';
                }
                
                providerIcon.src = iconPath;
                providerIcon.alt = `${metadata.model.provider || 'AI'} Provider`;
            }
            
            // Model icon
            if (modelIcon) {
                modelIcon.src = '/static/icons/model.svg';
                modelIcon.alt = metadata.model.model_name || 'Model';
            }
        }
    }
}
