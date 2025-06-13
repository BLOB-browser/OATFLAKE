// Basic search-box web component with inline search functionality
// No external dependencies required

// Add immediate debugging
console.log('🔧 === search-box.js file loading - CACHE BUST v1.7.0 - SKIP SEARCH FLAG ADDED - ' + Date.now() + ' ===');

// Define the component immediately
class SearchBox extends HTMLElement {    constructor() {
        super();
        console.log('🔧 SearchBox constructor called');
        this.currentQuery = '';
        this.references = [];
    }    connectedCallback() {
        console.log('🔧 === SearchBox connectedCallback called ===');
        // Render the component when connected to the DOM
        this.render();
          console.log('✅ SearchBox connected to DOM and rendered');
        
        // Add a simple click test to the entire component
        this.addEventListener('click', (e) => {
            console.log('🔧 Click detected on search-box component:', e.target);
        });
        
        // Set up event listeners
        const form = this.querySelector('form');
        const input = this.querySelector('input');
        const processButton = this.querySelector('#processButton');

        console.log('Elements found:');
        console.log('- Form:', !!form);
        console.log('- Input:', !!input);
        console.log('- Process Button:', !!processButton);

        if (form && input) {
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const query = input.value.trim();
                if (query) {
                    this.currentQuery = query;
                    await this.fetchReferences(query);
                }
            });
        }        if (processButton) {
            console.log('🔧 Found process button, attaching event listener');
            processButton.addEventListener('click', async (e) => {
                console.log('🔥 === GENERATE BUTTON CLICKED ===');
                console.log('🔥 Event target:', e.target);
                console.log('🔥 Event type:', e.type);
                console.log('🔥 Button ID:', e.target.id);
                console.log('🔥 Button text:', e.target.textContent.trim());
                console.log('🔥 Current query:', this.currentQuery);
                console.log('🔥 References length:', this.references ? this.references.length : 'undefined');
                console.log('🔥 References:', this.references);
                console.log('🔥 About to call processWithLLM...');
                
                // EXTRA DEBUGGING - Check if form submission is also happening
                console.log('🔥 🚨 === NETWORK DEBUGGING ===');
                console.log('🔥 🚨 This should ONLY call /api/web');
                console.log('🔥 🚨 If you see /api/references in network tab, there\'s a form submission conflict');
                console.log('🔥 🚨 Button moved outside form to prevent conflicts');
                
                // Prevent any default behavior or propagation
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation(); // Extra safety
                  console.log('🔥 === CHECKING CONDITIONS FOR GENERATE ===');
                console.log('🔥 - Has currentQuery:', !!this.currentQuery, '(', this.currentQuery, ')');
                console.log('🔥 - Has references array:', !!this.references);
                console.log('🔥 - References length:', this.references ? this.references.length : 'undefined');
                console.log('🔥 - References type:', typeof this.references);
                console.log('🔥 - References:', this.references);
                console.log('🔥 - Array.isArray(this.references):', Array.isArray(this.references));
                  if (this.currentQuery && this.references && this.references.length > 0) {
                    console.log('🔥 ✅ ALL CONDITIONS MET - Calling processWithLLM...');
                    console.log('🔥 ✅ Query:', this.currentQuery);
                    console.log('🔥 ✅ References count:', this.references.length);
                    await this.processWithLLM(this.currentQuery, this.references);
                } else {
                    console.log('🔥 ❌ CONDITIONS NOT MET FOR processWithLLM');
                    console.log('🔥 ❌ - Has query:', !!this.currentQuery);
                    console.log('🔥 ❌ - Has references:', !!(this.references && this.references.length > 0));
                    
                    // Let's try to search again if we don't have results
                    if (this.currentQuery && (!this.references || this.references.length === 0)) {
                        console.log('🔥 🔄 No references found, trying to search again...');
                        await this.fetchReferences(this.currentQuery);
                        
                        // Try again after search
                        if (this.references && this.references.length > 0) {
                            console.log('🔥 ✅ After re-search, calling processWithLLM...');
                            await this.processWithLLM(this.currentQuery, this.references);
                        } else {
                            console.log('🔥 ❌ Still no references after re-search');
                        }
                    }
                }
            });
            console.log('🔧 Event listener attached to process button');
        } else {
            console.log('❌ Process button not found!');
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
                    <div class="search-controls">                        <form class="search-input-container">
                            <div class="relative w-full">
                                <input type="text" 
                                      class="search-input"
                                      placeholder="Ask anything...">
                                <div class="search-status absolute right-5 top-1/2 transform -translate-y-1/2 text-sm"></div>
                            </div>
                        </form>
                        
                        <!-- Process Button (OUTSIDE form to prevent submission conflicts) -->
                        <div id="process-container" class="hidden mt-4">
                            <button id="processButton" 
                                    type="button"
                                    class="text-white font-medium text-sm py-2 px-4 bg-indigo-600/90 hover:bg-indigo-600 rounded-lg transition-all duration-200">
                                Generate
                            </button>
                        </div>
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
    }    async fetchReferences(query) {
        console.log('🔍 === fetchReferences called ===');
        console.log('🔍 THIS METHOD CALLS /api/references (SEARCH)');
        console.log('🔍 Query:', query);
        
        try {
            this.showLoadingState('Finding references...');
            
            // Call the actual local API endpoint
            console.log('🔍 Searching for:', query);
            console.log('🔍 📤 CALLING /api/references - NOT /api/web');
            console.log('🔍 📤 Endpoint URL: /api/references');
            console.log('🔍 📤 Method: POST');
            
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
            }            const data = await response.json();
            console.log('Search results:', data);
            console.log('🔍 References found:', (data.references || []).length);
            console.log('🔍 Content items found:', (data.content || []).length);
              // Store ALL results for later use (both references and content)
            this.references = [
                ...(data.references || []),
                ...(data.content || [])
            ];
            console.log('🔍 Total stored results:', this.references.length);
            console.log('🔍 ✅ REFERENCES STORED SUCCESSFULLY:', this.references);
            console.log('🔍 ✅ this.references is now available for Generate button');
            
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
            
            // Card content with richer metadata display
            const title = item.title || item.term || item.name || 'Reference';
            const content = item.content || item.description || item.text || '';
            const type = item.type || item.source_category || '';
            const category = item.category || '';
            const tags = item.tags || [];
            const origin_url = item.origin_url || item.resource_url || '';
            
            // Build additional info sections
            let additionalInfo = '';
            
            // Add category if available
            if (category) {
                additionalInfo += `<div class="text-xs text-amber-400 mb-1">Category: ${category}</div>`;
            }
            
            // Add tags if available
            if (tags && tags.length > 0) {
                const tagList = Array.isArray(tags) ? tags : (typeof tags === 'string' ? JSON.parse(tags) : []);
                if (tagList.length > 0) {
                    const tagDisplay = tagList.slice(0, 3).map(tag => `<span class="px-1 py-0.5 bg-gray-700 rounded text-xs">${tag}</span>`).join(' ');
                    additionalInfo += `<div class="mb-1">${tagDisplay}</div>`;
                }
            }
            
            // Add origin URL if available
            if (origin_url) {
                const shortUrl = origin_url.length > 30 ? origin_url.substring(0, 30) + '...' : origin_url;
                additionalInfo += `<div class="text-xs text-blue-400 mb-1">Source: ${shortUrl}</div>`;
            }
            
            card.innerHTML = `
                <div class="font-medium mb-2 text-white">${title}</div>
                ${type ? `<div class="text-xs text-indigo-400 mb-2">${type}</div>` : ''}
                ${additionalInfo}
                <div class="text-sm text-gray-300 mb-2">${content.substring(0, 120)}${content.length > 120 ? '...' : ''}</div>
                <div class="text-xs text-indigo-400">Relevance: ${Math.round(score * 100)}%</div>
            `;
            
            container.appendChild(card);
        });    }    async processWithLLM(query, references) {
        console.log('🚀 === processWithLLM called ===');
        console.log('🚀 THIS METHOD SHOULD CALL /api/web NOT /api/references');
        console.log('🚀 Query:', query);
        console.log('🚀 References count:', references ? references.length : 'undefined');
        console.log('🚀 References:', references);
        
        try {
            this.showLoadingState('Processing with local LLM...');
            
            // Format the references into a rich prompt for the LLM
            const formattedReferences = references.map(ref => {
                const title = ref.title || ref.term || ref.name || 'Reference';
                const content = ref.content || ref.description || ref.text || '';
                const type = ref.type || ref.source_category || '';
                const category = ref.category || '';
                const tags = ref.tags || [];
                const source = ref.origin_url || ref.resource_url || ref.source || '';
                
                // Build a rich reference entry
                let refEntry = `${type ? `[${type}] ` : ''}${title}`;
                
                // Add category if available
                if (category) {
                    refEntry += ` (Category: ${category})`;
                }
                
                // Add tags if available  
                if (tags && tags.length > 0) {
                    const tagList = Array.isArray(tags) ? tags : (typeof tags === 'string' ? JSON.parse(tags) : []);
                    if (tagList.length > 0) {
                        refEntry += ` [Tags: ${tagList.slice(0, 3).join(', ')}]`;
                    }
                }
                
                refEntry += `:\n${content}`;
                
                // Add source if available
                if (source) {
                    refEntry += `\nSource: ${source}`;
                }
                
                return refEntry + '\n';
            }).join('\n\n');
            
            // Create the full prompt with references and query
            const fullPrompt = `Please answer this question based on the references provided:\n\nQuestion: ${query}\n\nReferences:\n${formattedReferences}\n\nAnswer:`;
              console.log('📤 Sending LLM request to /api/web with prompt length:', fullPrompt.length);
            console.log('📤 🚨 CALLING /api/web - NOT /api/references 🚨');
            console.log('📤 Endpoint URL: /api/web');
            console.log('📤 Method: POST');
              // Call the local backend through the /api/web endpoint which exists in your API
            const response = await fetch('/api/web', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    query: query,                    // Original search query
                    prompt: fullPrompt,              // Formatted prompt with references
                    phase: 1,
                    skip_search: true,               // Flag to bypass search since we already have references
                    references_provided: true        // Additional flag for clarity
                })
            });

            console.log('📥 Response status:', response.status);
            console.log('📥 Response headers:', Object.fromEntries(response.headers.entries()));

            if (!response.ok) {
                throw new Error(`Processing failed with status: ${response.status}`);
            }
            
            // The web endpoint returns a request_id for phase 1
            const initialResult = await response.json();
            console.log('📥 Initial processing result:', initialResult);
            
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

// Register the custom element
if (!customElements.get('search-box')) {
    customElements.define('search-box', SearchBox);
    console.log('✅ SearchBox component registered successfully - CACHE BUST v1.5.0 - BUTTON TYPE=BUTTON FIX');
} else {
    console.log('⚠️ SearchBox component already registered');
}
