// Basic search-box web component with inline search functionality
// No external dependencies required

// Define the component immediately
class SearchBox extends HTMLElement {
    constructor() {
        super();
        this.currentQuery = '';
        this.references = [];
    }

    connectedCallback() {
        // Render the component when connected to the DOM
        this.render();
        
        // Set up event listeners
        const form = this.querySelector('form');
        const input = this.querySelector('input');
        const processButton = this.querySelector('#processButton');

        if (form && input) {
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const query = input.value.trim();
                if (query) {
                    this.currentQuery = query;
                    await this.fetchReferences(query);
                }
            });
        }

        if (processButton) {
            processButton.addEventListener('click', async () => {
                if (this.currentQuery && this.references.length > 0) {
                    await this.processWithLLM(this.currentQuery, this.references);
                }
            });
        }
    }

    render() {
        this.innerHTML = `
            <style>
                :host {
                    display: block;
                    width: 100%;
                    height: 100%;
                }
                
                .search-box-container {
                    width: 100%;
                    height: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .floating-container {
                    position: relative;
                    overflow: hidden;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 100%;
                    min-height: 60vh;
                    border-radius: 1rem;
                }
                
                .floating-references {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    pointer-events: none;
                }
                
                .floating-card {
                    position: absolute;
                    transition: transform 0.5s ease-out;
                    animation-name: float;
                    animation-timing-function: ease-in-out;
                    animation-iteration-count: infinite;
                    animation-direction: alternate;
                    transform: scale(0.8);
                    pointer-events: auto;
                    width: 280px;
                    box-sizing: border-box;
                }
                
                .floating-card:hover {
                    opacity: 1;
                    transform: scale(1);
                    z-index: 100 !important;
                }
                
                @keyframes float {
                    0% {
                        transform: translate(0, 0) scale(0.8);
                    }
                    100% {
                        transform: translate(var(--float-x), var(--float-y)) scale(0.8);
                    }
                }
                
                #process-container {
                    position: absolute;
                    right: 0.5rem;
                    top: 50%;
                    transform: translateY(-50%);
                    z-index: 51;
                }
                
                .search-controls {
                    position: relative;
                    z-index: 50;
                    width: 100%;
                    max-width: 700px;
                    background-color: rgba(0, 0, 0, 0.5);
                    backdrop-filter: blur(8px);
                    padding: 1rem;
                    border-radius: 1rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                
                .search-input {
                    display: block;
                    width: 100%;
                    padding: 1rem;
                    background-color: #0c0c0c;
                    color: white;
                    border: 2px solid #222;
                    border-radius: 0.75rem;
                    font-size: 1.125rem;
                    line-height: 1.5;
                    transition: all 0.3s;
                }
                
                .search-input:hover {
                    border-color: #4f46e5;
                }
                
                .search-input:focus {
                    outline: none;
                    border-color: rgba(79, 70, 229, 0.7);
                    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.3);
                }
            </style>
            
            <div class="search-box-container">
                <div class="floating-container">
                    <!-- Floating references will appear here -->
                    <div class="floating-references"></div>
                    
                    <!-- Search Input (centered) -->
                    <div class="search-controls">
                        <form class="search-input-container">
                            <div class="relative w-full">
                                <input type="text" 
                                      class="search-input"
                                      placeholder="Ask anything...">
                                <div class="search-status absolute right-5 top-1/2 transform -translate-y-1/2 text-sm"></div>
                                
                                <!-- Process Button (inside input field) -->
                                <div id="process-container" class="hidden">
                                    <button id="processButton" 
                                            class="text-white font-medium text-sm py-2 px-4 bg-indigo-600/90 hover:bg-indigo-600 rounded-lg transition-all duration-200">
                                        Generate
                                    </button>
                                </div>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        `;
    }

    showLoadingState(message = 'Searching...') {
        const status = this.querySelector('.search-status');
        if (status) {
            status.innerHTML = `
                <div class="flex items-center gap-2 text-indigo-400">
                    <div class="animate-spin h-4 w-4 border-2 border-indigo-500 rounded-full border-t-transparent"></div>
                    <span>${message}</span>
                </div>
            `;
        }
    }

    hideLoadingState() {
        const status = this.querySelector('.search-status');
        if (status) {
            status.innerHTML = '';
        }
    }

    async fetchReferences(query) {
        try {
            this.showLoadingState('Finding references...');
            
            // Call the actual local API endpoint
            console.log('Searching for:', query);
            
            // Make the API call to the local backend
            const response = await fetch('/api/references', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    context_k: 5
                })
            });

            if (!response.ok) {
                throw new Error(`Search failed with status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Search results:', data);
            
            // Store references for later use
            this.references = data.references || [];
            
            // Display results
            this.displayReferences(data.references || [], data.content || []);
            
            // Dispatch event with search results
            this.dispatchEvent(new CustomEvent('search-results', {
                bubbles: true,
                detail: {
                    query: query,
                    results: [...(data.references || []), ...(data.content || [])]
                }
            }));
            
            this.hideLoadingState();
            
        } catch (error) {
            console.error('Error in search:', error);
            this.hideLoadingState();
            
            // Display error in the floating references area
            const container = this.querySelector('.floating-references');
            if (container) {
                container.innerHTML = `
                    <div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-black/70 p-4 rounded-lg text-red-500 text-center max-w-md">
                        <div class="text-xl mb-2">Search Error</div>
                        <div>${error.message}</div>
                    </div>
                `;
            }
        }
    }

    displayReferences(references, contentItems) {
        const container = this.querySelector('.floating-references');
        const processContainer = this.querySelector('#process-container');
        
        if (!container) return;

        // Combine references and content items
        const allItems = [
            ...(references || []),
            ...(contentItems || [])
        ];

        if (allItems.length === 0) {
            container.innerHTML = '<div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-yellow-400 p-3 text-center">No relevant references found</div>';
            if (processContainer) processContainer.classList.add('hidden');
            return;
        }

        // Show the process button when we have results
        if (processContainer) processContainer.classList.remove('hidden');

        // Clear previous results
        container.innerHTML = '';

        // Place cards in different areas of the container
        allItems.forEach((item, index) => {
            // Calculate a score for positioning (use relevance score or just index as fallback)
            const score = item.relevance_score || item.hybrid_score || (1 - index/allItems.length);
            
            // Create card element
            const card = document.createElement('div');
            card.className = 'floating-card bg-black/90 border border-stone-800 rounded-lg p-3';
            
            // Position card with some randomness
            const row = Math.floor(index / 3);
            const col = index % 3;
            const randX = Math.random() * 10 - 5;
            const randY = Math.random() * 10 - 5;
            
            card.style.left = `${15 + col * 30 + randX}%`;
            card.style.top = `${15 + row * 30 + randY}%`;
            card.style.zIndex = Math.floor(score * 10) + 5;
            
            // Add animation
            const floatX = (Math.random() * 20 - 10) + 'px';
            const floatY = (Math.random() * 20 - 10) + 'px';
            card.style.setProperty('--float-x', floatX);
            card.style.setProperty('--float-y', floatY);
            card.style.animationDuration = (2 + Math.random() * 3) + 's';
            
            // Card content
            const title = item.title || item.term || item.name || 'Reference';
            const content = item.content || item.description || item.text || '';
            const type = item.type || item.source_category || '';
            
            card.innerHTML = `
                <div class="font-medium mb-2 text-white">${title}</div>
                ${type ? `<div class="text-xs text-indigo-400 mb-2">${type}</div>` : ''}
                <div class="text-sm text-gray-300">${content.substring(0, 150)}${content.length > 150 ? '...' : ''}</div>
                <div class="text-xs text-indigo-400 mt-2">Relevance: ${Math.round(score * 100)}%</div>
            `;
            
            container.appendChild(card);
        });
    }
    
    async processWithLLM(query, references) {
        try {
            this.showLoadingState('Processing with local LLM...');
            
            // Format the references into a good prompt for the LLM
            const formattedReferences = references.map(ref => {
                const title = ref.title || ref.term || ref.name || 'Reference';
                const content = ref.content || ref.description || ref.text || '';
                const type = ref.type || ref.source_category || '';
                return `${type ? `[${type}] ` : ''}${title}:\n${content}\n`;
            }).join('\n\n');
            
            // Create the full prompt with references and query
            const fullPrompt = `Please answer this question based on the references provided:\n\nQuestion: ${query}\n\nReferences:\n${formattedReferences}\n\nAnswer:`;
            
            console.log('Sending LLM request with prompt length:', fullPrompt.length);
            
            // Call the local backend through the /api/web endpoint which exists in your API
            const response = await fetch('/api/web', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    prompt: fullPrompt,
                    phase: 1
                })
            });

            if (!response.ok) {
                throw new Error(`Processing failed with status: ${response.status}`);
            }
            
            // The web endpoint returns a request_id for phase 1
            const initialResult = await response.json();
            console.log('Initial processing result:', initialResult);
            
            if (initialResult.status === 'processing' && initialResult.request_id) {
                // Phase 2: Poll for the completed response
                await this.pollForResponse(query, initialResult.request_id);
            } else if (initialResult.status === 'success') {
                // Find or create the response-box component
                this.displayFinalResponse(query, initialResult);
            } else {
                // Error or unexpected response
                throw new Error(`Unexpected response: ${JSON.stringify(initialResult)}`);
            }
            
        } catch (error) {
            console.error('Error processing:', error);
            this.hideLoadingState();
            
            // Display error to the user
            this.displayError(error.message);
        }
    }
    
    // Display the final response using the response-box component
    displayFinalResponse(query, result) {
        this.hideLoadingState();
        
        // Prepare metadata for the response-box component
        const metadata = {
            timing: result.timing || { 
                total_seconds: 0,
                retrieval_seconds: result.retrieval_time || 0,
                generation_seconds: result.generation_time || 0
            },
            model: result.model_info || { 
                provider: 'Ollama', 
                model_name: result.model || 'Unknown' 
            },
            word_count: result.response ? result.response.split(/\s+/).length : 0
        };
        
        console.log('Using response-box component for display');
        
        // Find the response-box component or create it if not found
        let responseBox = document.querySelector('response-box');
        if (!responseBox) {
            // Create the response-box component if it doesn't exist
            responseBox = document.createElement('response-box');
            responseBox.id = 'responseBox';
            
            // Find a suitable place to insert it
            const searchContainer = document.getElementById('searchContainer');
            if (searchContainer && searchContainer.parentNode) {
                searchContainer.parentNode.insertBefore(responseBox, searchContainer.nextSibling);
            } else {
                // Fallback - append to the main content
                const mainContent = document.getElementById('mainContent') || document.body;
                mainContent.appendChild(responseBox);
            }
        }
        
        // Set the response in the response-box component
        responseBox.setResponse(
            result.response || `No results generated for "${query}"`,
            metadata
        );
    }
    
    // Display a fallback response without the response-box component
    displayFallbackResponse(responseText) {
        console.log('Displaying fallback response');
        
        // Create a container for the response if it doesn't exist
        let responseContainer = document.getElementById('searchResponseContainer');
        if (!responseContainer) {
            responseContainer = document.createElement('div');
            responseContainer.id = 'searchResponseContainer';
            responseContainer.className = 'mt-8 p-4 bg-black/80 border border-stone-700 rounded-lg';
            
            // Find a place to insert it
            const searchContainer = document.getElementById('searchContainer');
            if (searchContainer && searchContainer.parentNode) {
                searchContainer.parentNode.insertBefore(responseContainer, searchContainer.nextSibling);
            } else {
                document.body.appendChild(responseContainer);
            }
        }
        
        // Set the content
        responseContainer.innerHTML = `
            <div class="text-lg font-medium mb-4 text-white">Response:</div>
            <div class="text-gray-200 whitespace-pre-wrap">${responseText}</div>
        `;
    }
    
    // Display an error message
    displayError(message) {
        console.log('Displaying error:', message);
        
        // Try to use response-box if available and registered
        if (window.customElements.get('response-box')) {
            let responseBox = document.querySelector('response-box');
            if (responseBox && typeof responseBox.setResponse === 'function') {
                responseBox.setResponse(`Error: ${message}`, null);
                return;
            }
        }
        
        // Fallback to built-in error display
        this.createResponseDisplay(`<div class="text-red-500 font-medium">Error: ${message}</div>`);
    }
    
    // Create a simple response display when the response-box component is not available
    createResponseDisplay(responseText, metadata = null) {
        console.log('Creating fallback response display');
        
        // Create a container for the response if it doesn't exist
        let responseContainer = document.getElementById('searchResponseContainer');
        if (!responseContainer) {
            responseContainer = document.createElement('div');
            responseContainer.id = 'searchResponseContainer';
            responseContainer.className = 'mt-8 mb-4 bg-black/80 border-2 border-stone-800 rounded-3xl';
            
            // Find a place to insert it
            const searchContainer = document.getElementById('searchContainer');
            if (searchContainer && searchContainer.parentNode) {
                searchContainer.parentNode.insertBefore(responseContainer, searchContainer.nextSibling);
            } else {
                document.body.appendChild(responseContainer);
            }
        }
        
        // Prepare the metadata display if available
        let metadataHeader = '';
        let metadataFooter = '';
        
        if (metadata) {
            // Format model info for header
            const model = metadata.model || {};
            const provider = model.provider || 'Unknown';
            const modelName = model.model_name || 'Unknown';
            
            // Format timing info
            const timing = metadata.timing || {};
            const totalTime = timing.total_seconds != null ? 
                Number(timing.total_seconds).toFixed(2) + 's' : 'Unknown';
            const retrievalTime = timing.retrieval_seconds != null ?
                Number(timing.retrieval_seconds).toFixed(2) + 's' : 'Unknown';
            const generationTime = timing.generation_seconds != null ?
                Number(timing.generation_seconds).toFixed(2) + 's' : 'Unknown';
            
            // Create header with model info and timing
            metadataHeader = `
                <div class="flex justify-between items-center mb-2 pb-2 border-b border-stone-700 text-sm">
                    <div class="flex items-center gap-2 text-stone-400">
                        <div class="flex items-center gap-1">
                            <span class="font-mono">${provider}</span>
                            <span class="mx-1 text-stone-600">›</span>
                            <span class="font-mono">${modelName}</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-2 text-stone-400">
                        <span class="font-mono">${totalTime}</span>
                    </div>
                </div>
            `;
            
            // Create footer with detailed timing info
            metadataFooter = `
                <div class="flex flex-wrap gap-3 mt-4 pt-2 border-t border-stone-700 text-xs text-stone-400">
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
                </div>
            `;
        }
        
        // Set the content with metadata if available
        responseContainer.innerHTML = `
            <div class="p-3 md:p-4">
                ${metadataHeader}
                <div class="text-base md:text-lg font-bold mb-2 text-white">AI Response:</div>
                <div class="text-gray-200 font-mono whitespace-pre-wrap text-sm md:text-base">${responseText}</div>
                ${metadataFooter}
            </div>
        `;
    }
    
    // Poll for the response when using the two-phase approach
    async pollForResponse(query, requestId, attempts = 0) {
        try {
            // Show polling status
            this.showLoadingState(`Processing with LLM... (${attempts + 1})`);
            
            if (attempts > 120) { // Increased to 120 attempts (about 2 minutes)
                throw new Error("LLM processing timed out. Please try again.");
            }
            
            // Call phase 2 endpoint with the request_id
            const response = await fetch('/api/web', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    phase: 2,
                    request_id: requestId
                })
            });
            
            if (!response.ok) {
                throw new Error(`Polling failed with status: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('Polling result:', result);
            
            if (result.status === 'processing') {
                // Still processing, wait and try again
                await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
                return this.pollForResponse(query, requestId, attempts + 1);
            } else if (result.status === 'success' && result.complete) {
                // Process completed successfully
                this.displayFinalResponse(query, result);
                return result;
            } else if (result.status === 'error') {
                // Error in processing
                throw new Error(result.error || "Unknown processing error");
            } else {
                // Wait and try again
                await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
                return this.pollForResponse(query, requestId, attempts + 1);
            }
        } catch (error) {
            console.error('Polling error:', error);
            this.hideLoadingState();
            throw error;
        }
    }
}

// Register the component immediately
customElements.define('search-box', SearchBox);
console.log('SearchBox component registered successfully');
