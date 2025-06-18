/**
 * PDFViewerModal - Specialized modal component for viewing local PDF materials
 * Designed specifically for materials stored in the data/materials folder
 * Provides PDF-specific controls and metadata display
 */

class PDFViewerModal extends HTMLElement {
    constructor() {
        super();
        this.isVisible = false;
        this.currentPdfUrl = '';
        this.currentTitle = '';
        this.currentMetadata = null;
        this.currentPage = 1;
        this.totalPages = 0;
        this.zoomLevel = 1.0;
        
        // Bind methods
        this.show = this.show.bind(this);
        this.hide = this.hide.bind(this);
        this.handleKeyDown = this.handleKeyDown.bind(this);
    }

    connectedCallback() {
        this.render();
        this.addEventListeners();
        console.log('📚 PDFViewerModal component initialized');
    }

    render() {
        this.innerHTML = `
            <div id="pdfViewerModalOverlay" class="fixed inset-0 bg-black bg-opacity-75 z-50 hidden flex items-center justify-center p-4">
                <div class="bg-gray-900 rounded-xl shadow-2xl w-full h-full max-w-7xl max-h-[95vh] flex flex-col border border-gray-700">
                    
                    <!-- PDF Viewer Header -->
                    <div class="flex items-center justify-between p-4 border-b border-gray-700">
                        <!-- Title and Metadata -->
                        <div class="flex items-center space-x-4 flex-1 min-w-0">
                            <div class="flex items-center space-x-2">
                                <svg class="w-6 h-6 text-red-500" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
                                </svg>
                                <span class="text-white font-medium text-lg" id="pdfViewerTitle">PDF Document</span>
                            </div>
                            <div class="text-sm text-gray-400" id="pdfViewerMetadata"></div>
                        </div>

                        <!-- PDF Controls -->
                        <div class="flex items-center space-x-3">
                            <!-- Download -->
                            <button id="pdfDownload" class="p-2 hover:bg-gray-800 rounded transition-colors" title="Download PDF">
                                <svg class="w-4 h-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                                </svg>
                            </button>

                            <!-- Close -->
                            <button id="pdfViewerClose" class="p-2 hover:bg-gray-800 rounded transition-colors" title="Close">
                                <svg class="w-5 h-5 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                                </svg>
                            </button>
                        </div>
                    </div>

                    <!-- PDF Content Area -->
                    <div class="flex-1 overflow-hidden relative">
                        <!-- Loading Indicator -->
                        <div id="pdfLoadingIndicator" class="absolute inset-0 flex items-center justify-center bg-gray-800">
                            <div class="text-center">
                                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-white mb-4"></div>
                                <div id="pdfLoadingStatus" class="text-white">Loading PDF...</div>
                            </div>
                        </div>
                        
                        <!-- Error Fallback -->
                        <div id="pdfErrorFallback" class="absolute inset-0 flex items-center justify-center bg-gray-800 hidden">
                            <div class="text-center max-w-md">
                                <div class="text-red-400 mb-4">
                                    <svg class="w-16 h-16 mx-auto" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M12,2L13.09,8.26L22,9L13.09,9.74L12,16L10.91,9.74L2,9L10.91,8.26L12,2Z" />
                                    </svg>
                                </div>
                                <h3 class="text-white text-lg font-medium mb-2">PDF Display Issue</h3>
                                <p class="text-gray-300 mb-4">The PDF cannot be displayed inline. You can download it instead.</p>
                                <button id="pdfFallbackDownload" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded transition-colors">
                                    Download PDF
                                </button>
                            </div>
                        </div>
                        
                        <!-- PDF Iframe -->
                        <iframe id="pdfViewerFrame" 
                                class="w-full h-full border-none" 
                                frameborder="0"
                                allowfullscreen
                                sandbox="allow-same-origin allow-scripts allow-popups allow-forms"
                                title="PDF Document Viewer">
                        </iframe>
                    </div>

                    <!-- PDF Footer with Metadata -->
                    <div id="pdfViewerFooter" class="border-t border-gray-700 p-3 bg-gray-800">
                        <div class="flex items-center justify-between text-xs text-gray-400">
                            <div class="flex items-center space-x-4">
                                <span id="pdfFileSize"></span>
                                <span id="pdfLastModified"></span>
                                <span id="pdfMaterialType"></span>
                            </div>
                            <div class="flex items-center space-x-4">
                                <span id="pdfLoadingStatus">Ready</span>
                                <div id="pdfLoadingIndicator" class="hidden">
                                    <div class="animate-spin h-3 w-3 border border-gray-400 rounded-full border-t-transparent"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    addEventListeners() {
        // Close modal
        const closeBtn = this.querySelector('#pdfViewerClose');
        const overlay = this.querySelector('#pdfViewerModalOverlay');
        
        if (closeBtn) {
            closeBtn.addEventListener('click', this.hide);
        }

        if (overlay) {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    this.hide();
                }
            });
        }

        // Download
        const downloadBtn = this.querySelector('#pdfDownload');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadPdf());
        }

        // Fallback download
        const fallbackDownloadBtn = this.querySelector('#pdfFallbackDownload');
        if (fallbackDownloadBtn) {
            fallbackDownloadBtn.addEventListener('click', () => this.downloadPdf());
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyDown);

        // Monitor iframe load status
        const iframe = this.querySelector('#pdfViewerFrame');
        if (iframe) {
            iframe.addEventListener('load', () => this.onPdfLoaded());
            iframe.addEventListener('error', () => this.onPdfError());
        }
    }

    /**
     * Show the PDF viewer modal
     * @param {string} pdfPath - Path to the PDF file (relative to materials folder)
     * @param {string} title - Title for the PDF
     * @param {Object} metadata - Additional metadata about the PDF
     */
    show(pdfPath, title = 'PDF Document', metadata = null) {
        console.log('📚 Opening PDF viewer for:', { pdfPath, title, metadata });

        this.currentTitle = title;
        this.currentMetadata = metadata;
        this.currentPage = 1;
        this.zoomLevel = 1.0;

        // Update UI elements
        const titleElement = this.querySelector('#pdfViewerTitle');
        const metadataElement = this.querySelector('#pdfViewerMetadata');
        const overlay = this.querySelector('#pdfViewerModalOverlay');
        const iframe = this.querySelector('#pdfViewerFrame');
        const loadingStatus = this.querySelector('#pdfLoadingStatus');
        const loadingIndicator = this.querySelector('#pdfLoadingIndicator');

        if (titleElement) titleElement.textContent = title;
        
        if (metadataElement && metadata) {
            const metaText = [];
            if (metadata.fields) metaText.push(`Fields: ${metadata.fields}`);
            if (metadata.created_at) {
                const date = new Date(metadata.created_at).toLocaleDateString();
                metaText.push(`Created: ${date}`);
            }
            metadataElement.textContent = metaText.join(' • ');
        }

        // Show loading state
        if (loadingStatus) loadingStatus.textContent = 'Loading PDF...';
        if (loadingIndicator) loadingIndicator.classList.remove('hidden');

        // Hide error fallback
        const errorFallback = this.querySelector('#pdfErrorFallback');
        if (errorFallback) errorFallback.classList.add('hidden');

        // Construct PDF URL - serve through the backend API
        this.currentPdfUrl = `/api/data/materials/pdf/${encodeURIComponent(pdfPath)}`;
        
        if (iframe) {
            // Clear any previous content
            iframe.src = 'about:blank';
            
            // Set up timeout for loading
            this.loadingTimeout = setTimeout(() => {
                console.warn('📚 PDF loading timeout, showing fallback');
                this.onPdfError();
            }, 10000); // 10 second timeout
            
            // Load PDF with proper headers
            iframe.src = this.currentPdfUrl + '#toolbar=1&navpanes=1&scrollbar=1';
        }

        // Show modal
        if (overlay) {
            overlay.classList.remove('hidden');
            this.isVisible = true;
            
            // Apply animation
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.style.opacity = '1';
                overlay.style.transition = 'opacity 0.3s ease-in-out';
            }, 10);
        }

        // Update footer metadata
        this.updateFooterMetadata(metadata);
    }

    /**
     * Hide the PDF viewer modal
     */
    hide() {
        // Clear any loading timeout
        if (this.loadingTimeout) {
            clearTimeout(this.loadingTimeout);
            this.loadingTimeout = null;
        }
        
        const overlay = this.querySelector('#pdfViewerModalOverlay');
        if (overlay && this.isVisible) {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.classList.add('hidden');
                this.isVisible = false;
                
                // Clear iframe to stop loading
                const iframe = this.querySelector('#pdfViewerFrame');
                if (iframe) {
                    iframe.src = 'about:blank';
                }
            }, 300);
        }
    }

    /**
     * Handle keyboard shortcuts
     */
    handleKeyDown(e) {
        if (!this.isVisible) return;

        switch (e.key) {
            case 'Escape':
                this.hide();
                e.preventDefault();
                break;
        }
    }

    /**
     * Adjust zoom level
     */
    adjustZoom(delta) {
        this.zoomLevel = Math.max(0.25, Math.min(3.0, this.zoomLevel + delta));
        this.updateZoomDisplay();
        this.applyZoom();
    }

    /**
     * Apply zoom to PDF viewer
     */
    applyZoom() {
        const iframe = this.querySelector('#pdfViewerFrame');
        if (iframe && iframe.contentWindow) {
            try {
                // Send zoom command to PDF.js viewer
                iframe.contentWindow.postMessage({
                    type: 'setZoom',
                    zoom: this.zoomLevel
                }, '*');
            } catch (e) {
                console.warn('Could not apply zoom to PDF viewer:', e);
            }
        }
    }

    /**
     * Update zoom display
     */
    updateZoomDisplay() {
        const zoomDisplay = this.querySelector('#pdfZoomLevel');
        if (zoomDisplay) {
            zoomDisplay.textContent = `${Math.round(this.zoomLevel * 100)}%`;
        }
    }

    /**
     * Change page
     */
    changePage(delta) {
        const newPage = this.currentPage + delta;
        if (newPage >= 1 && newPage <= this.totalPages) {
            this.goToPage(newPage);
        }
    }

    /**
     * Go to specific page
     */
    goToPage(page) {
        this.currentPage = page;
        const currentPageInput = this.querySelector('#pdfCurrentPage');
        if (currentPageInput) {
            currentPageInput.value = page;
        }

        const iframe = this.querySelector('#pdfViewerFrame');
        if (iframe && iframe.contentWindow) {
            try {
                iframe.contentWindow.postMessage({
                    type: 'goToPage',
                    page: page
                }, '*');
            } catch (e) {
                console.warn('Could not navigate to page:', e);
            }
        }
    }

    /**
     * Fit PDF to width
     */
    fitToWidth() {
        const iframe = this.querySelector('#pdfViewerFrame');
        if (iframe && iframe.contentWindow) {
            try {
                iframe.contentWindow.postMessage({
                    type: 'fitToWidth'
                }, '*');
            } catch (e) {
                console.warn('Could not fit to width:', e);
            }
        }
    }

    /**
     * Fit PDF to page
     */
    fitToPage() {
        const iframe = this.querySelector('#pdfViewerFrame');
        if (iframe && iframe.contentWindow) {
            try {
                iframe.contentWindow.postMessage({
                    type: 'fitToPage'
                }, '*');
            } catch (e) {
                console.warn('Could not fit to page:', e);
            }
        }
    }

    /**
     * Download the current PDF
     */
    downloadPdf() {
        if (this.currentPdfUrl) {
            // Use the download parameter to force download
            const downloadUrl = this.currentPdfUrl + (this.currentPdfUrl.includes('?') ? '&' : '?') + 'download=true';
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = this.currentTitle.replace(/[^a-z0-9]/gi, '_').toLowerCase() + '.pdf';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    /**
     * Handle PDF load completion
     */
    onPdfLoaded() {
        console.log('📚 PDF loaded successfully');
        
        // Clear timeout
        if (this.loadingTimeout) {
            clearTimeout(this.loadingTimeout);
            this.loadingTimeout = null;
        }
        
        const loadingStatus = this.querySelector('#pdfLoadingStatus');
        const loadingIndicator = this.querySelector('#pdfLoadingIndicator');
        
        if (loadingStatus) loadingStatus.textContent = 'Ready';
        if (loadingIndicator) loadingIndicator.classList.add('hidden');
    }

    /**
     * Handle PDF load error
     */
    onPdfError() {
        console.error('📚 Error loading PDF, showing fallback');
        
        // Clear timeout
        if (this.loadingTimeout) {
            clearTimeout(this.loadingTimeout);
            this.loadingTimeout = null;
        }
        
        const loadingIndicator = this.querySelector('#pdfLoadingIndicator');
        const errorFallback = this.querySelector('#pdfErrorFallback');
        
        if (loadingIndicator) loadingIndicator.classList.add('hidden');
        if (errorFallback) errorFallback.classList.remove('hidden');
    }

    /**
     * Update footer metadata
     */
    updateFooterMetadata(metadata) {
        if (!metadata) return;

        const fileSizeElement = this.querySelector('#pdfFileSize');
        const lastModifiedElement = this.querySelector('#pdfLastModified');
        const materialTypeElement = this.querySelector('#pdfMaterialType');

        if (fileSizeElement && metadata.file_size) {
            const size = parseInt(metadata.file_size);
            const sizeStr = size > 1024 * 1024 
                ? `${(size / 1024 / 1024).toFixed(1)} MB`
                : `${(size / 1024).toFixed(0)} KB`;
            fileSizeElement.textContent = sizeStr;
        }

        if (lastModifiedElement && metadata.created_at) {
            const date = new Date(metadata.created_at);
            lastModifiedElement.textContent = date.toLocaleDateString();
        }

        if (materialTypeElement && metadata.fields) {
            materialTypeElement.textContent = `Type: ${metadata.fields}`;
        }
    }

    disconnectedCallback() {
        document.removeEventListener('keydown', this.handleKeyDown);
    }
}

// Register the custom element
customElements.define('pdf-viewer-modal', PDFViewerModal);

// Global function to open PDF viewer
window.openPDFViewer = function(pdfPath, title = 'PDF Document', metadata = null) {
    console.log('🌍 Global openPDFViewer called:', { pdfPath, title, metadata });
    
    let pdfViewer = document.querySelector('pdf-viewer-modal');
    if (!pdfViewer) {
        pdfViewer = document.createElement('pdf-viewer-modal');
        document.body.appendChild(pdfViewer);
    }
    
    pdfViewer.show(pdfPath, title, metadata);
};

console.log('✅ PDFViewerModal component and global openPDFViewer function registered');
