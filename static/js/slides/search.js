/**
 * SearchSlide - Handles the generation of the search interface
 * Uses our custom search-box component
 * CACHE BUSTER: v1.3.0 - FIXED GENERATE BUTTON REFERENCES STORAGE
 */
const SearchSlide = (() => {
    // HTML template for the search interface using web components
    const template = `
        <div class="fixed inset-0 w-full h-full overflow-hidden">
            <!-- Search Container - response-box component will be added above this when needed -->
            <div class="w-full h-full">
                <div class="bg-black w-full h-full" id="searchBoxContainer">
                    <!-- search-box will be added here directly -->
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
        container.innerHTML = template;        // Just load the search-box component directly - no need for the widget intermediary
        loadScript('/static/js/components/search-box.js?v=' + Date.now())
            .then(() => {
                console.log('🔧 Search box script loaded with timestamp cache-bust');
                console.log('🔧 searchBoxRegistered flag:', window.searchBoxRegistered);
                console.log('🔧 customElements.get("search-box"):', !!customElements.get('search-box'));
                
                // Wait a brief moment to ensure the component is fully registered
                setTimeout(() => {
                    try {
                        // Explicitly create the search-box element and add it to the container
                        const searchBoxContainer = container.querySelector('#searchBoxContainer');
                        console.log('🔧 Found searchBoxContainer:', !!searchBoxContainer);                        if (searchBoxContainer) {
                            // Create the search-box element using createElement to trigger web component lifecycle
                            const searchBoxElement = document.createElement('search-box');
                            searchBoxElement.id = 'searchBox';
                            console.log('🔧 Created element:', searchBoxElement);
                            console.log('🔧 Element tag name:', searchBoxElement.tagName);
                            console.log('🔧 Element constructor:', searchBoxElement.constructor.name);
                            
                            searchBoxContainer.appendChild(searchBoxElement);
                            console.log('🔧 Search box added to DOM using createElement - CACHE BUST v1.2.3');
                            
                            // Check if the element was actually created
                            const createdElement = searchBoxContainer.querySelector('search-box');
                            console.log('🔧 Created element found:', !!createdElement);
                            
                            // Wait a moment and check if connectedCallback was called
                            setTimeout(() => {
                                console.log('🔧 Checking if element is connected...');
                                console.log('🔧 Element isConnected:', createdElement?.isConnected);
                                console.log('🔧 Element parentNode:', !!createdElement?.parentNode);
                            }, 500);
                            
                            // Set up event listener for search responses
                            document.addEventListener('search-response', function(event) {
                                if (event.detail) {
                                    const responseData = event.detail;
                                    handleSearchResponse(responseData);
                                }
                            });
                        } else {
                            console.error('🔧 searchBoxContainer not found in DOM');
                        }
                    } catch (error) {
                        console.error('🔧 Error creating search-box:', error);
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
            
            // Insert before the search container (above search for better UX)
            const searchBoxContainer = document.getElementById('searchBoxContainer');
            
            if (searchBoxContainer && searchBoxContainer.parentNode) {
                searchBoxContainer.parentNode.insertBefore(responseBox, searchBoxContainer);
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
    }    // Return public API
    return {
        render
    };
})();

// Removed automatic DOMContentLoaded initialization to prevent duplicate search boxes
// Search slide is now only rendered by app.js initializeSlides()
