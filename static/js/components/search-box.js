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
                    position: relative;
                }
                
                .search-box-container {
                    width: 100%;
                    height: 100%;
                    position: relative;
                    overflow: hidden;
                }
                
                .floating-container {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    overflow: visible;
                    z-index: 1;
                }
                
                .floating-references {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    pointer-events: none;
                    z-index: 1;
                    overflow: visible;
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
                    max-width: 280px;
                    box-sizing: border-box;
                    opacity: 0.85;
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
                
                .process-container {
                    display: flex;
                    justify-content: center;
                    opacity: 0;
                    visibility: hidden;
                    transition: opacity 0.3s ease, visibility 0.3s ease;
                }
                
                .process-container.visible {
                    opacity: 1;
                    visibility: visible;
                }
                
                .controls-container {
                    margin-top: 1rem;
                    min-height: 40px;
                    display: flex;
                    justify-content: flex-start;
                    align-items: center;
                }
                
                .search-depth-container {
                    opacity: 1;
                    visibility: visible;
                    transition: opacity 0.3s ease, visibility 0.3s ease;
                }
                
                .search-depth-container.hidden {
                    opacity: 0;
                    visibility: hidden;
                }
                
                .search-depth-input {
                    width: 60px;
                    padding: 0.375rem 0.5rem;
                    background: rgba(31, 41, 55, 0.8);
                    border: 1px solid rgba(107, 114, 128, 0.4);
                    border-radius: 0.375rem;
                    color: white;
                    text-align: center;
                    font-size: 0.875rem;
                    transition: border-color 0.2s ease, box-shadow 0.2s ease;
                }
                
                .search-depth-input:focus {
                    outline: none;
                    border-color: rgba(99, 102, 241, 0.6);
                    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
                }
                
                .search-controls {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    z-index: 10;
                    width: 50%;
                    min-width: 400px;
                    max-width: 600px;
                    background-color: rgba(0, 0, 0, 0.7);
                    backdrop-filter: blur(12px);
                    padding: 1.5rem;
                    border-radius: 1rem;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                .search-summary {
                    background-color: rgba(15, 15, 15, 0.6);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 0.5rem;
                    padding: 0.75rem;
                    margin-top: 1rem;
                    display: none;
                    font-size: 0.875rem;
                }
                
                .search-summary.visible {
                    display: block;
                }
                
                .response-container {
                    margin-top: 1rem;
                    display: none;
                    max-height: 400px;
                    overflow-y: auto;
                    font-size: 0.875rem;
                }
                
                .response-container.visible {
                    display: block;
                }
                
                .response-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 0.75rem;
                    padding-bottom: 0.5rem;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                    font-size: 0.75rem;
                }
                
                .response-content {
                    color: #e5e7eb;
                    line-height: 1.6;
                    white-space: pre-wrap;
                    font-family: 'Inter', system-ui, sans-serif;
                }
                
                .response-footer {
                    margin-top: 0.75rem;
                    padding-top: 0.5rem;
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                    display: flex;
                    gap: 1rem;
                    font-size: 0.75rem;
                    color: #9ca3af;
                }
                
                .summary-stats {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 1rem;
                }
                
                .summary-count {
                    color: #a3a3a3;
                    font-weight: 500;
                }
                
                .summary-relevance {
                    color: #6366f1;
                    font-weight: 500;
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
                    
                    <!-- Search Input with simple summary -->
                    <div class="search-controls">
                        <form class="search-input-container">
                            <div class="relative w-full">
                                <input type="text" 
                                      class="search-input"
                                      placeholder="Ask anything...">
                                <div class="search-status absolute right-5 top-1/2 transform -translate-y-1/2 text-sm"></div>
                            </div>
                        </form>
                        
                        <!-- Controls Container (shows either search depth or process button) -->
                        <div class="controls-container" id="controls-container">
                            <!-- Search Depth Input (shown when process is hidden) -->
                            <div class="search-depth-container" id="search-depth-container">
                                <div class="flex items-center gap-3 justify-start">
                                    <label for="searchDepth" class="text-sm text-gray-300">Depth:</label>
                                    <input type="number" 
                                           id="searchDepth" 
                                           class="search-depth-input"
                                           value="5" 
                                           min="1" 
                                           max="50"
                                           title="Number of results to retrieve (1-50)">
                                </div>
                            </div>

                            <!-- Process Button (inside search controls) -->
                            <div class="process-container" id="process-container">
                                <button id="processButton" 
                                        type="button"
                                        class="text-white font-medium text-sm py-2 px-6 bg-indigo-600/90 hover:bg-indigo-600 rounded-lg transition-all duration-200 shadow-lg">
                                    Generate Response
                                </button>
                            </div>
                        </div>
                        
                        <!-- Search Summary (below generate button) -->
                        <div class="search-summary" id="searchSummary">
                            <div class="summary-stats">
                                <span class="summary-count" id="summaryCount">0 results found</span>
                                <span class="summary-relevance" id="summaryRelevance">Avg: 0%</span>
                            </div>
                        </div>
                        
                        <!-- Response Container (shows LLM response) -->
                        <div class="response-container" id="responseContainer">
                            <!-- Response content will be inserted here -->
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
            
            // Hide any previous response
            const responseContainer = this.querySelector('#responseContainer');
            if (responseContainer) responseContainer.classList.remove('visible');
            
            // Call the actual local API endpoint
            console.log('🔍 Searching for:', query);
            console.log('🔍 📤 CALLING /api/references - NOT /api/web');
            console.log('🔍 📤 Endpoint URL: /api/references');
            console.log('🔍 📤 Method: POST');
            
            // Get search depth from input
            const searchDepthInput = this.querySelector('#searchDepth');
            const contextK = searchDepthInput ? parseInt(searchDepthInput.value) || 5 : 5;
            
            // Make the API call to the local backend
            const requestBody = {
                query: query,
                context_k: contextK
            };
            
            console.log('🔍 📤 Request body:', JSON.stringify(requestBody, null, 2));
            console.log('🔍 📤 Using search depth (k):', contextK);
            
            const response = await fetch('/api/references', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`Search failed with status: ${response.status}`);
            }            const data = await response.json();
            console.log('🔍 === SEARCH RESPONSE ANALYSIS ===');
            console.log('🔍 Full response data:', data);
            console.log('🔍 References found:', (data.references || []).length);
            console.log('🔍 Content items found:', (data.content || []).length);
            console.log('🔍 Total items:', (data.references || []).length + (data.content || []).length);
            console.log('🔍 Request was for context_k:', requestBody.context_k);
            
            // Log first few items to see what we're getting
            if (data.references && data.references.length > 0) {
                console.log('🔍 First reference:', data.references[0]);
            }
            if (data.content && data.content.length > 0) {
                console.log('🔍 First content item:', data.content[0]);
            }
            
            // Check for metadata about the search
            if (data.metadata) {
                console.log('🔍 Search metadata:', data.metadata);
            }
              // Store ALL results for later use (both references and content)
            this.references = [
                ...(data.references || []),
                ...(data.content || [])
            ];
            console.log('🔍 === STORED REFERENCES FOR GENERATE ===');
            console.log('🔍 Total stored results:', this.references.length);
            console.log('🔍 Breakdown by type:');
            console.log('🔍 - References:', (data.references || []).length);
            console.log('🔍 - Content items:', (data.content || []).length);
            console.log('🔍 ✅ REFERENCES STORED SUCCESSFULLY');
            console.log('🔍 ✅ this.references is now available for Generate button');
            
            // Log sample of what's stored
            if (this.references.length > 0) {
                console.log('🔍 Sample stored item (first):', {
                    title: this.references[0].title || this.references[0].term || this.references[0].name,
                    content_length: (this.references[0].content || this.references[0].description || '').length,
                    type: this.references[0].type || this.references[0].source_category,
                    relevance: this.references[0].relevance_score || this.references[0].hybrid_score
                });
            }
            
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
            
            // Display error in the summary area
            const searchSummary = this.querySelector('#searchSummary');
            const summaryCount = this.querySelector('#summaryCount');
            const summaryRelevance = this.querySelector('#summaryRelevance');
            const processContainer = this.querySelector('#process-container');
            const searchDepthContainer = this.querySelector('#search-depth-container');
            const responseContainer = this.querySelector('#responseContainer');
            
            if (searchSummary && summaryCount) {
                searchSummary.classList.add('visible');
                summaryCount.textContent = 'Search Error';
                summaryCount.style.color = '#ef4444';
                if (summaryRelevance) {
                    summaryRelevance.textContent = error.message;
                    summaryRelevance.style.color = '#ef4444';
                }
            }
            
            // Hide process button and response container, show search depth on error
            if (processContainer) processContainer.classList.remove('visible');
            if (searchDepthContainer) searchDepthContainer.classList.remove('hidden');
            if (responseContainer) responseContainer.classList.remove('visible');
        }
    }

    displayReferences(references, contentItems) {
        const container = this.querySelector('.floating-references');
        const processContainer = this.querySelector('#process-container');
        const searchDepthContainer = this.querySelector('#search-depth-container');
        const searchSummary = this.querySelector('#searchSummary');
        const summaryCount = this.querySelector('#summaryCount');
        const summaryRelevance = this.querySelector('#summaryRelevance');
        
        if (!container) return;

        // Combine references and content items
        const allItems = [
            ...(references || []),
            ...(contentItems || [])
        ];

        if (allItems.length === 0) {
            container.innerHTML = '<div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-yellow-400 p-3 text-center">No relevant references found</div>';
            if (processContainer) processContainer.classList.remove('visible');
            if (searchDepthContainer) searchDepthContainer.classList.remove('hidden');
            if (searchSummary) searchSummary.classList.remove('visible');
            return;
        }

        // Show the process button and summary when we have results, keep search depth visible
        if (processContainer) processContainer.classList.add('visible');
        // Don't hide search depth - keep it visible for subsequent searches
        // if (searchDepthContainer) searchDepthContainer.classList.add('hidden');
        if (searchSummary) {
            searchSummary.classList.add('visible');
            
            // Update summary stats
            if (summaryCount) {
                summaryCount.textContent = `${allItems.length} result${allItems.length !== 1 ? 's' : ''} found`;
            }
            
            if (summaryRelevance) {
                // Calculate average relevance score
                const scores = allItems.map(item => 
                    item.relevance_score || item.hybrid_score || 0
                ).filter(score => score > 0);
                
                const avgScore = scores.length > 0 
                    ? scores.reduce((sum, score) => sum + score, 0) / scores.length 
                    : 0;
                
                summaryRelevance.textContent = `Avg: ${Math.round(avgScore * 100)}%`;
            }
        }

        // Use the new search results display component
        this.displayResultsWithComponent(references, contentItems);
    }

    /**
     * Display results using the new SearchResultsDisplay component
     */
    displayResultsWithComponent(references, contentItems) {
        const container = this.querySelector('.floating-references');
        if (!container) return;

        // Create search data object that matches the API response format
        const searchData = {
            query: this.currentQuery,
            references: references || [],
            content: contentItems || [],
            metadata: {
                reference_count: (references || []).length,
                content_count: (contentItems || []).length,
                k_value: 10,
                context_words: this.estimateContextWords(references, contentItems),
                timestamp: new Date().toISOString(),
                stores_info: {
                    reference_store_loaded: (references || []).length > 0,
                    content_store_loaded: (contentItems || []).length > 0,
                    topic_stores_loaded: this.countTopicStores(references)
                }
            }
        };

        // Clear container and add the search results component
        container.innerHTML = '';
        container.style.position = 'absolute';
        container.style.top = '0';
        container.style.left = '0';
        container.style.width = '100%';
        container.style.height = '100%';
        container.style.pointerEvents = 'none';
        container.style.overflow = 'visible';
        container.style.zIndex = '1';

        // Load the search results display component
        this.loadSearchResultsComponent().then(() => {
            const resultsDisplay = document.createElement('search-results-display');
            resultsDisplay.setSearchData(searchData);
            
            // Listen for generate response events
            resultsDisplay.addEventListener('generate-response', (e) => {
                if (this.currentQuery && this.references && this.references.length > 0) {
                    this.processWithLLM(this.currentQuery, this.references);
                }
            });

            container.appendChild(resultsDisplay);
        }).catch(error => {
            console.error('Failed to load search results component:', error);
            // Fallback to the original floating display
            this.displayFloatingReferences(references, contentItems);
        });
    }

    /**
     * Estimate context words from references and content
     */
    estimateContextWords(references, contentItems) {
        const allItems = [...(references || []), ...(contentItems || [])];
        return allItems.reduce((total, item) => {
            const content = item.content || item.description || item.text || '';
            return total + content.split(/\s+/).length;
        }, 0);
    }

    /**
     * Count topic stores in references
     */
    countTopicStores(references) {
        const topicStores = new Set();
        (references || []).forEach(ref => {
            if (ref.topic_store) {
                topicStores.add(ref.topic_store);
            }
        });
        return topicStores.size;
    }

    /**
     * Load the search results display component dynamically
     */
    async loadSearchResultsComponent() {
        if (customElements.get('search-results-display')) {
            return; // Already loaded
        }

        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = '/static/js/components/search-results-display.js?v=' + Date.now();
            script.type = 'module';
            script.onload = () => resolve();
            script.onerror = () => reject(new Error('Failed to load search results display component'));
            document.head.appendChild(script);
        });
    }

    /**
     * Fallback floating display (original implementation)
     */
    displayFloatingReferences(references, contentItems) {
        const container = this.querySelector('.floating-references');
        
        // Add debug visualization (temporary)
        console.log('🎯 Starting floating references display with', references.length, 'references and', contentItems.length, 'content items');
        
        // Combine references and content items
        const allItems = [
            ...(references || []),
            ...(contentItems || [])
        ];

        // Clear previous results
        container.innerHTML = '';
        
        console.log('🎯 Total items to display:', allItems.length);

        // Place cards in different areas of the container
        allItems.forEach((item, index) => {
            // Calculate a score for positioning (use relevance score or just index as fallback)
            const score = item.relevance_score || item.hybrid_score || (1 - index/allItems.length);
            
            // Create card element
            const card = document.createElement('div');
            card.className = 'floating-card bg-black/90 border border-stone-800 rounded-lg p-3';
            
            // Add debug styling
            card.style.border = `2px solid rgba(99, 102, 241, 0.3)`; // Subtle blue border for debugging
            
            console.log(`🎯 Creating card ${index + 1}/${allItems.length} for item:`, (item.title || item.term || item.name || 'Reference').substring(0, 30));
            
            // Position cards using improved grid-based distribution with avoidance zones
            const totalItems = allItems.length;
            let posX, posY;
            
            // Define zones to avoid (search controls area)
            const searchControlsX = { min: 35, max: 65 }; // Avoid center area
            const searchControlsY = { min: 35, max: 65 }; // Avoid center area
            
            // Create a more distributed approach using different positioning strategies
            if (totalItems <= 6) {
                // For few items, use corner + edge positions
                const positions = [
                    { x: 15, y: 15 },  // Top-left
                    { x: 85, y: 15 },  // Top-right
                    { x: 15, y: 85 },  // Bottom-left
                    { x: 85, y: 85 },  // Bottom-right
                    { x: 15, y: 50 },  // Mid-left
                    { x: 85, y: 50 }   // Mid-right
                ];
                const pos = positions[index % positions.length];
                posX = pos.x + (Math.random() * 10 - 5); // Add some randomization
                posY = pos.y + (Math.random() * 10 - 5);
            } else {
                // For many items, use a combination of grid and radial distribution
                const gridCols = Math.ceil(Math.sqrt(totalItems));
                const gridRows = Math.ceil(totalItems / gridCols);
                
                const gridX = (index % gridCols) / (gridCols - 1);
                const gridY = Math.floor(index / gridCols) / (gridRows - 1);
                
                // Map grid position to viewport (0-1 to percentage)
                posX = gridX * 80 + 10; // Use 10%-90% of viewport width
                posY = gridY * 80 + 10; // Use 10%-90% of viewport height
                
                // Adjust positions that fall in the search controls area
                if (posX >= searchControlsX.min && posX <= searchControlsX.max && 
                    posY >= searchControlsY.min && posY <= searchControlsY.max) {
                    // Push to nearest edge
                    if (posX < 50) {
                        posX = Math.min(posX, searchControlsX.min - 5);
                    } else {
                        posX = Math.max(posX, searchControlsX.max + 5);
                    }
                }
            }
            
            // Ensure cards stay within viewport bounds
            posX = Math.max(5, Math.min(95, posX));
            posY = Math.max(5, Math.min(95, posY));
            
            console.log(`Grid positioning: Card ${index + 1}/${totalItems}, strategy: ${totalItems <= 6 ? 'corner/edge' : 'grid'}, final: (${posX.toFixed(1)}%, ${posY.toFixed(1)}%)`);
            
            // Add small random offset for organic feel
            const randX = (Math.random() * 4 - 2); // ±2% random variation
            const randY = (Math.random() * 4 - 2);
            
            card.style.left = `${Math.max(2, Math.min(98, posX + randX))}%`;
            card.style.top = `${Math.max(2, Math.min(98, posY + randY))}%`;
            card.style.zIndex = Math.floor(score * 10) + 5;
            
            // Debug logging for positioning
            console.log(`Card ${index + 1}: positioned at ${card.style.left}, ${card.style.top}`);
            
            // Add animation
            const floatX = (Math.random() * 10 - 5) + 'px'; // Smaller float range
            const floatY = (Math.random() * 10 - 5) + 'px';
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
            console.log('📥 === INITIAL RESULT ANALYSIS ===');
            console.log('📥 Full initial result:', initialResult);
            console.log('📥 Status:', initialResult.status);
            console.log('📥 Request ID:', initialResult.request_id);
            console.log('📥 Has response:', !!initialResult.response);
            console.log('📥 Response length:', initialResult.response ? initialResult.response.length : 'N/A');
            console.log('📥 All keys:', Object.keys(initialResult));
            
            if (initialResult.status === 'processing' && initialResult.request_id) {
                console.log('📥 🔄 Status is processing, polling for completion...');
                // Phase 2: Poll for the completed response
                await this.pollForResponse(query, initialResult.request_id);
            } else if (initialResult.status === 'success') {
                console.log('📥 ✅ Status is success, displaying response immediately...');
                // Find or create the response-box component
                this.displayFinalResponse(query, initialResult);
            } else {
                console.log('📥 ❌ Unexpected status:', initialResult.status);
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
    
    // Display the final response using the internal response container
    displayFinalResponse(query, result) {
        console.log('🎉 === displayFinalResponse called ===');
        console.log('🎉 Query:', query);
        console.log('🎉 Result object:', result);
        console.log('🎉 Result.response:', result.response);
        console.log('🎉 Result keys:', Object.keys(result));
        
        this.hideLoadingState();
        
        // Get the internal response container
        const responseContainer = this.querySelector('#responseContainer');
        if (!responseContainer) {
            console.log('🎉 ❌ Response container not found!');
            return;
        }
        
        // Prepare metadata
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
        
        console.log('🎉 Prepared metadata:', metadata);
        
        // Format timing info
        const timing = metadata.timing;
        const totalTime = timing.total_seconds != null ? 
            Number(timing.total_seconds).toFixed(2) + 's' : 'Unknown';
        const retrievalTime = timing.retrieval_seconds != null ?
            Number(timing.retrieval_seconds).toFixed(2) + 's' : 'Unknown';
        const generationTime = timing.generation_seconds != null ?
            Number(timing.generation_seconds).toFixed(2) + 's' : 'Unknown';
        
        // Format model info
        const model = metadata.model;
        const provider = model.provider || 'Unknown';
        const modelName = model.model_name || 'Unknown';
        
        // Get response text
        const responseText = result.response || `No results generated for "${query}"`;
        
        // Create the response HTML
        responseContainer.innerHTML = `
            <div class="response-header">
                <div class="flex items-center gap-2 text-gray-400">
                    <span class="font-mono">${provider}</span>
                    <span class="mx-1 text-gray-600">›</span>
                    <span class="font-mono">${modelName}</span>
                </div>
                <div class="text-gray-400">
                    <span class="font-mono">${totalTime}</span>
                </div>
            </div>
            <div class="response-content">${responseText}</div>
            <div class="response-footer">
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
                    <span class="font-mono">${metadata.word_count}</span>
                </div>
            </div>
        `;
        
        // Show the response container
        responseContainer.classList.add('visible');
        
        console.log('🎉 ✅ Response displayed successfully in internal container');
    }
    
    // Display a fallback response without the response-box component
    displayFallbackResponse(responseText) {
        console.log('Displaying fallback response');
        
        // Create a container for the response if it doesn't exist
        let responseContainer = document.getElementById('searchResponseContainer');
        if (!responseContainer) {
            responseContainer = document.createElement('div');
            responseContainer.id = 'searchResponseContainer';
            responseContainer.className = 'mt-8 p-4 bg-stone-800 border border-stone-700 rounded-lg relative z-10';
            
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
            <div class="text-stone-200 whitespace-pre-wrap">${responseText}</div>
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
            responseContainer.className = 'mt-8 mb-4 bg-stone-800 border-2 border-stone-700 rounded-3xl relative z-10';
            
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
            console.log('🔄 === POLLING RESULT ANALYSIS ===');
            console.log('🔄 Attempt:', attempts + 1);
            console.log('🔄 Full polling result:', result);
            console.log('🔄 Status:', result.status);
            console.log('🔄 Complete flag:', result.complete);
            console.log('🔄 Has response:', !!result.response);
            console.log('🔄 Response length:', result.response ? result.response.length : 'N/A');
            console.log('🔄 All keys:', Object.keys(result));
            
            if (result.status === 'processing') {
                console.log('🔄 Still processing, waiting 1 second...');
                // Still processing, wait and try again
                await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
                return this.pollForResponse(query, requestId, attempts + 1);
            } else if (result.status === 'success' && result.complete) {
                console.log('🔄 ✅ Processing completed successfully!');
                // Process completed successfully
                this.displayFinalResponse(query, result);
                return result;
            } else if (result.status === 'error') {
                console.log('🔄 ❌ Error in processing:', result.error);
                // Error in processing
                throw new Error(result.error || "Unknown processing error");
            } else {
                console.log('🔄 ⏳ Status unclear, waiting and trying again...');
                console.log('🔄 Status was:', result.status, 'Complete was:', result.complete);
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
