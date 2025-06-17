/**
 * BrowserModal - A modal component that displays web content in an iframe
 * Acts like a basic browser window with header, URL display, and close button
 */

class BrowserModal extends HTMLElement {
    constructor() {
        super();
        this.currentUrl = '';
        this.isVisible = false;
    }

    connectedCallback() {
        this.render();
        this.setupEventListeners();
        
        // Make this component globally accessible
        window.browserModal = this;
        console.log('BrowserModal connected to DOM');
    }

    disconnectedCallback() {
        // Clean up global reference
        if (window.browserModal === this) {
            window.browserModal = null;
        }
    }

    render() {
        this.innerHTML = `
            <!-- Modal Overlay -->
            <div id="browserModalOverlay" class="fixed inset-0 bg-black bg-opacity-80 z-50 hidden">
                <!-- Modal Container -->
                <div class="fixed inset-4 md:inset-8 bg-neutral-900 rounded-lg shadow-2xl flex flex-col max-w-7xl mx-auto">
                    
                    <!-- Browser Header -->
                    <div class="bg-neutral-800 rounded-t-lg px-4 py-3 flex items-center justify-between border-b border-neutral-700">
                        <!-- URL Display -->
                        <div class="flex-1 flex items-center gap-3">
                            <div class="flex gap-2">
                                <!-- Browser Window Controls -->
                                <div class="w-3 h-3 rounded-full bg-red-500 hover:bg-red-400 cursor-pointer transition-colors"></div>
                                <div class="w-3 h-3 rounded-full bg-yellow-500 hover:bg-yellow-400 cursor-pointer transition-colors"></div>
                                <div class="w-3 h-3 rounded-full bg-green-500 hover:bg-green-400 cursor-pointer transition-colors"></div>
                            </div>
                            
                            <!-- Navigation Controls -->
                            <div class="flex gap-2 ml-4">
                                <button id="browserBackBtn" class="p-1 text-gray-400 hover:text-white hover:bg-neutral-700 rounded transition-colors" title="Back">
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path>
                                    </svg>
                                </button>
                                <button id="browserForwardBtn" class="p-1 text-gray-400 hover:text-white hover:bg-neutral-700 rounded transition-colors" title="Forward">
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                                    </svg>
                                </button>
                                <button id="browserRefreshBtn" class="p-1 text-gray-400 hover:text-white hover:bg-neutral-700 rounded transition-colors" title="Refresh">
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                                    </svg>
                                </button>
                            </div>
                            
                            <!-- URL Bar -->
                            <div class="flex-1 bg-neutral-700 rounded-md px-3 py-2 text-sm text-gray-300 font-mono truncate mx-3">
                                <span class="text-gray-500 mr-2">🔒</span>
                                <span id="browserUrlDisplay">about:blank</span>
                            </div>
                        </div>
                        
                        <!-- Close Button -->
                        <button id="browserCloseBtn" 
                                class="ml-4 p-2 text-gray-400 hover:text-white hover:bg-neutral-700 rounded-md transition-colors">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </button>
                    </div>
                    
                    <!-- Loading Indicator -->
                    <div id="browserLoadingIndicator" class="bg-neutral-800 px-4 py-2 border-b border-neutral-700 hidden">
                        <div class="flex items-center gap-2 text-sm text-gray-400">
                            <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-500"></div>
                            <span>Loading...</span>
                        </div>
                    </div>
                    
                    <!-- Error Display -->
                    <div id="browserErrorDisplay" class="bg-red-900 bg-opacity-50 px-4 py-3 border-b border-red-700 hidden">
                        <div class="flex items-center gap-2 text-sm text-red-300">
                            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
                            </svg>
                            <span id="browserErrorText">Failed to load content</span>
                        </div>
                    </div>
                    
                    <!-- Iframe Content -->
                    <div class="flex-1 overflow-hidden">
                        <iframe id="browserIframe" 
                                class="w-full h-full border-0" 
                                src="about:blank"
                                allow="clipboard-read; clipboard-write"
                                sandbox="allow-same-origin allow-scripts allow-popups allow-forms allow-top-navigation">
                        </iframe>
                    </div>
                    
                    <!-- Footer with additional controls (optional) -->
                    <div class="bg-neutral-800 rounded-b-lg px-4 py-2 flex items-center justify-between border-t border-neutral-700">
                        <div class="text-xs text-gray-500">
                            Press ESC to close • Use ⌘+Click to open links in new tabs
                        </div>
                        <div class="flex gap-2">
                            <button id="browserOpenExternalBtn" 
                                    class="px-3 py-1 text-xs bg-indigo-600 hover:bg-indigo-700 rounded text-white transition-colors flex items-center gap-1">
                                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                                </svg>
                                Open in Browser
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        const overlay = this.querySelector('#browserModalOverlay');
        const closeBtn = this.querySelector('#browserCloseBtn');
        const refreshBtn = this.querySelector('#browserRefreshBtn');
        const backBtn = this.querySelector('#browserBackBtn');
        const forwardBtn = this.querySelector('#browserForwardBtn');
        const openExternalBtn = this.querySelector('#browserOpenExternalBtn');
        const iframe = this.querySelector('#browserIframe');

        // Close button
        closeBtn?.addEventListener('click', () => this.close());

        // Refresh button
        refreshBtn?.addEventListener('click', () => this.refresh());

        // Back button (basic implementation)
        backBtn?.addEventListener('click', () => {
            try {
                iframe.contentWindow.history.back();
            } catch (e) {
                console.log('Cannot go back in iframe');
            }
        });

        // Forward button (basic implementation)
        forwardBtn?.addEventListener('click', () => {
            try {
                iframe.contentWindow.history.forward();
            } catch (e) {
                console.log('Cannot go forward in iframe');
            }
        });

        // Open external button
        openExternalBtn?.addEventListener('click', () => this.openExternal());

        // Close on overlay click (but not on modal content)
        overlay?.addEventListener('click', (e) => {
            if (e.target === overlay) {
                this.close();
            }
        });

        // ESC key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isVisible) {
                this.close();
            }
        });

        // Iframe load events
        iframe?.addEventListener('load', () => {
            this.hideLoading();
            this.hideError();
            
            // Try to update URL display with actual loaded URL
            try {
                const iframeUrl = iframe.contentWindow.location.href;
                if (iframeUrl && iframeUrl !== 'about:blank') {
                    const urlDisplay = this.querySelector('#browserUrlDisplay');
                    if (urlDisplay) urlDisplay.textContent = iframeUrl;
                }
            } catch (e) {
                // Cross-origin restriction, can't access iframe URL
                console.log('Cannot access iframe URL due to cross-origin policy');
            }
        });

        iframe?.addEventListener('error', () => {
            this.hideLoading();
            this.showError('Failed to load the webpage');
        });
    }

    /**
     * Open the modal with a specific URL
     * @param {string} url - The URL to load in the iframe
     * @param {string} title - Optional title for the modal
     */
    open(url, title = null) {
        if (!url || url.trim() === '') {
            console.error('BrowserModal: No URL provided');
            return;
        }

        this.currentUrl = url;
        const overlay = this.querySelector('#browserModalOverlay');
        const urlDisplay = this.querySelector('#browserUrlDisplay');
        const iframe = this.querySelector('#browserIframe');

        // Update URL display
        urlDisplay.textContent = url;

        // Show loading
        this.showLoading();
        this.hideError();

        // Load the URL in iframe
        iframe.src = url;

        // Show the modal
        overlay.classList.remove('hidden');
        this.isVisible = true;

        // Add smooth animation
        overlay.style.opacity = '0';
        setTimeout(() => {
            overlay.style.opacity = '1';
            overlay.style.transition = 'opacity 0.2s ease-in-out';
        }, 10);

        console.log('BrowserModal opened with URL:', url);
    }

    /**
     * Close the modal
     */
    close() {
        const overlay = this.querySelector('#browserModalOverlay');
        const iframe = this.querySelector('#browserIframe');

        // Animate out
        overlay.style.opacity = '0';
        setTimeout(() => {
            overlay.classList.add('hidden');
            overlay.style.transition = '';
            
            // Clear iframe to stop loading
            iframe.src = 'about:blank';
            
            this.isVisible = false;
            this.currentUrl = '';
            
            this.hideLoading();
            this.hideError();
        }, 200);

        console.log('BrowserModal closed');
    }

    /**
     * Refresh the current URL
     */
    refresh() {
        if (this.currentUrl) {
            const iframe = this.querySelector('#browserIframe');
            this.showLoading();
            this.hideError();
            iframe.src = this.currentUrl;
        }
    }

    /**
     * Open the current URL in an external browser
     */
    openExternal() {
        if (this.currentUrl) {
            window.open(this.currentUrl, '_blank');
        }
    }

    /**
     * Show loading indicator
     */
    showLoading() {
        const loadingIndicator = this.querySelector('#browserLoadingIndicator');
        loadingIndicator?.classList.remove('hidden');
    }

    /**
     * Hide loading indicator
     */
    hideLoading() {
        const loadingIndicator = this.querySelector('#browserLoadingIndicator');
        loadingIndicator?.classList.add('hidden');
    }

    /**
     * Show error message
     * @param {string} message - Error message to display
     */
    showError(message) {
        const errorDisplay = this.querySelector('#browserErrorDisplay');
        const errorText = this.querySelector('#browserErrorText');
        
        if (errorText) errorText.textContent = message;
        errorDisplay?.classList.remove('hidden');
    }

    /**
     * Hide error message
     */
    hideError() {
        const errorDisplay = this.querySelector('#browserErrorDisplay');
        errorDisplay?.classList.add('hidden');
    }

    /**
     * Check if the modal is currently visible
     * @returns {boolean}
     */
    isOpen() {
        return this.isVisible;
    }

    /**
     * Get the current URL
     * @returns {string}
     */
    getCurrentUrl() {
        return this.currentUrl;
    }
}

// Register the custom element
if (!customElements.get('browser-modal')) {
    customElements.define('browser-modal', BrowserModal);
    console.log('BrowserModal component registered');
}

// Global convenience function to open browser modal
window.openBrowserModal = function(url, title = null) {
    console.log('Global openBrowserModal called with URL:', url);
    
    // Find or create browser modal component
    let browserModal = document.querySelector('browser-modal');
    
    if (!browserModal) {
        console.log('Creating new browser-modal element');
        browserModal = document.createElement('browser-modal');
        document.body.appendChild(browserModal);
    }
    
    // Open the modal
    browserModal.open(url, title);
};

// Test function for demonstration - can be called from console
window.testBrowserModal = function() {
    console.log('Testing browser modal with example URL...');
    window.openBrowserModal('https://example.com', 'Test Website');
};

// Another test function with a different URL
window.testBrowserModalWithGithub = function() {
    console.log('Testing browser modal with GitHub...');
    window.openBrowserModal('https://github.com', 'GitHub');
};

// Test function for search results
window.testSearchResultModal = function() {
    console.log('Testing browser modal with search result simulation...');
    // Find a search results display component
    const resultsDisplay = document.querySelector('search-results-display');
    if (resultsDisplay) {
        resultsDisplay.openUrlInModal('https://docs.github.com', 'GitHub Documentation');
    } else {
        console.log('No search results display found, using global function');
        window.openBrowserModal('https://docs.github.com', 'GitHub Documentation');
    }
};

// Test function to inject URLs into existing search results for demonstration
window.injectTestUrls = function() {
    console.log('🧪 Injecting test URLs into search results...');
    const resultsDisplay = document.querySelector('search-results-display');
    if (resultsDisplay && resultsDisplay.references) {
        // Add URLs to some of the existing search results
        resultsDisplay.references.forEach((item, index) => {
            if (index < 3) { // Add URLs to first 3 items
                const testUrls = [
                    'https://github.com/features',
                    'https://docs.github.com/en/get-started',
                    'https://github.com/about'
                ];
                item.origin_url = testUrls[index];
                console.log(`Added URL to item "${item.title || item.term}": ${item.origin_url}`);
            }
        });
        
        // Re-render to show the new URLs
        resultsDisplay.render();
        console.log('✅ Test URLs injected and search results re-rendered');
    } else {
        console.log('❌ No search results display or references found');
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BrowserModal;
}
