/**
 * Component for displaying search results with metadata
 */

class SearchResult extends HTMLElement {
    constructor() {
        super();
        console.log('SearchResult constructor called');
        this.render();

        // Explicitly bind methods to this instance
        this.displayResult = this.displayResult.bind(this);
        this.handleSearchResponse = this.handleSearchResponse.bind(this);
        this.showResponse = this.showResponse.bind(this);
        this.showMetadata = this.showMetadata.bind(this);
        this.hideMetadata = this.hideMetadata.bind(this);
        this.updateResultHeader = this.updateResultHeader.bind(this);

        // Mark this component as ready for use
        this.isReady = true;
        console.log('SearchResult constructor completed');
    }

    connectedCallback() {
        console.log('SearchResult connected to DOM');

        // Make sure the method is available immediately 
        window.searchResultComponent = this;

        // Verify methods are available
        console.log('Methods available on SearchResult:',
            'displayResult:', typeof this.displayResult === 'function',
            'showResponse:', typeof this.showResponse === 'function',
            'updateResultHeader:', typeof this.updateResultHeader === 'function'
        );

        // Keep event-based approach as backup
        document.querySelector('search-box')?.addEventListener('search-response', this.handleSearchResponse);
    }

    disconnectedCallback() {
        document.querySelector('search-box')?.removeEventListener('search-response', this.handleSearchResponse);

        // Clean up global reference
        if (window.searchResultComponent === this) {
            window.searchResultComponent = null;
        }
    }

    // New public method for direct calling
    displayResult(response, metadata) {
        console.log('Displaying result:', { responseLength: response?.length || 0, hasMetadata: !!metadata });

        if (!response) {
            this.showResponse("No response received from the server.");
            return;
        }

        // Display the text response
        this.showResponse(response);

        // Check if metadata exists and has expected structure
        if (metadata && typeof metadata === 'object') {
            try {
                console.log('Metadata available:', metadata);
                // Force clear and rebuild the header
                const headerEl = this.querySelector('.result-header');
                if (headerEl) {
                    headerEl.innerHTML = '';
                    headerEl.style.display = 'flex';
                }

                // Update the result header with model and time info
                this.updateResultHeader(metadata);

                // Show detailed metadata footer
                this.showMetadata(metadata);
            } catch (err) {
                console.error('Error displaying metadata:', err);
                this.hideMetadata();
            }
        } else {
            this.hideMetadata();
            // Clear the header if no metadata
            const headerEl = this.querySelector('.result-header');
            if (headerEl) headerEl.style.display = 'none';
        }

        // Show the results container and ensure it's visible
        const container = this.querySelector('.search-result-container');
        if (container) {
            container.style.display = 'block';

            // Ensure the container is visible by scrolling to it
            setTimeout(() => {
                this.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 100);
        }
    }

    // Keep the event handler method to handle the older approach
    handleSearchResponse(event) {
        console.log('Event-based handleSearchResponse called with:', event.detail);

        // Use the new display method
        const { response, metadata } = event.detail;
        this.displayResult(response, metadata);
    }

    showResponse(text) {
        const responseEl = this.querySelector('.search-response');
        if (responseEl) {
            // Format the text to handle line breaks
            const formattedText = typeof text === 'string' ? text.replace(/\n/g, '<br>') : String(text);
            responseEl.innerHTML = formattedText;
        }
    }

    showMetadata(metadata) {
        const metadataEl = this.querySelector('.search-metadata');
        if (!metadataEl) return;

        metadataEl.style.display = 'flex';

        // Create placeholder values with safe access
        const timing = metadata.timing || {};
        const model = metadata.model || {};

        // Format the timing information with safe fallbacks
        const totalTime = (timing.total_seconds != null) ? Number(timing.total_seconds).toFixed(2) : '0.00';
        const retrievalTime = (timing.retrieval_seconds != null) ? Number(timing.retrieval_seconds).toFixed(2) : '0.00';
        const generationTime = (timing.generation_seconds != null) ? Number(timing.generation_seconds).toFixed(2) : '0.00';

        // Format the provider/model information
        const provider = model.provider
            ? (String(model.provider).charAt(0).toUpperCase() + String(model.provider).slice(1))
            : 'Unknown';
        const modelName = model.model_name || 'Unknown';

        // Create the HTML content
        metadataEl.innerHTML = `
            <div class="metadata-item">
                <span class="metadata-label">Model:</span>
                <span class="metadata-value">${provider} / ${modelName}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Total time:</span>
                <span class="metadata-value">${totalTime}s</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Retrieval:</span>
                <span class="metadata-value">${retrievalTime}s</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Generation:</span>
                <span class="metadata-value">${generationTime}s</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Words:</span>
                <span class="metadata-value">${metadata.word_count || 0}</span>
            </div>
        `;
    }

    hideMetadata() {
        const metadataEl = this.querySelector('.search-metadata');
        if (metadataEl) {
            metadataEl.style.display = 'none';
        }
    }

    updateResultHeader(metadata) {
        const headerEl = this.querySelector('.result-header');
        if (!headerEl) return;

        headerEl.style.display = 'flex';

        // Create placeholder values with safe access
        const timing = metadata.timing || {};
        const model = metadata.model || {};

        // Format the timing information with safe fallbacks
        const totalTime = (timing.total_seconds != null) ? Number(timing.total_seconds).toFixed(2) : '0.00';

        // Format the provider/model information
        const provider = model.provider
            ? (String(model.provider).charAt(0).toUpperCase() + String(model.provider).slice(1))
            : 'Unknown';
        const modelName = model.model_name || 'Unknown';

        // Update the header content
        headerEl.innerHTML = `
            <div class="header-model-info">
                <span class="model-icon">🤖</span>
                <span class="model-name">${provider} / ${modelName}</span>
            </div>
            <div class="header-time-info">
                <span class="time-icon">⏱️</span>
                <span class="processing-time">${totalTime}s</span>
            </div>
        `;
    }

    render() {
        this.innerHTML = `
            <style>
                .search-result-container {
                    display: none;
                    margin-top: 2rem;
                    background: rgba(0, 0, 0, 0.5);
                    border-radius: 1rem;
                    padding: 1.5rem;
                    backdrop-filter: blur(10px);
                    width: 100%;
                    margin-left: auto;
                    margin-right: auto;
                }
                
                .result-header {
                    display: none;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1rem;
                    padding-bottom: 0.75rem;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                    font-size: 0.875rem;
                    color: rgba(255, 255, 255, 0.7);
                    width: 100%; /* Ensure full width */
                }
                
                /* Enhanced styling for better visibility */
                .header-model-info, .header-time-info {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    background-color: rgba(50, 50, 50, 0.4);
                    padding: 0.3rem 0.6rem;
                    border-radius: 0.5rem;
                }
                
                .model-icon, .time-icon {
                    font-size: 1rem;
                }
                
                .model-name, .processing-time {
                    font-family: ui-monospace, monospace;
                    font-weight: 500;
                }
                
                .search-response {
                    font-size: 1rem;
                    line-height: 1.6;
                    color: #fff;
                    white-space: pre-wrap;
                }
                
                .search-metadata {
                    margin-top: 1rem;
                    padding-top: 0.75rem;
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.75rem;
                    font-size: 0.75rem;
                    color: rgba(255, 255, 255, 0.6);
                }
                
                .metadata-item {
                    display: inline-flex;
                    align-items: center;
                }
                
                .metadata-label {
                    font-weight: bold;
                    margin-right: 0.25rem;
                }
                
                .metadata-value {
                    font-family: ui-monospace, monospace;
                }
            </style>
            
            <div class="search-result-container">
                <div class="result-header"></div>
                <div class="search-response"></div>
                <div class="search-metadata"></div>
            </div>
        `;
    }

    // Add a static property to check if the component is defined
    static get isReady() {
        return true;
    }
}

// Define the custom element
customElements.define('search-result', SearchResult);

// Export the component class
export default SearchResult;

// Provide a global reference to verify the component is loaded
window.SearchResultComponent = SearchResult;
console.log('search-result component defined and exported');
