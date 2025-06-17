/**
 * SearchResultsDisplay - Advanced component for displaying search results
 * Handles the universal data structure with expandable cards, type-based styling, and click actions
 * Based on analysis from test_search_data_structure.py
 */

class SearchResultsDisplay extends HTMLElement {
    constructor() {
        super();
        this.references = [];
        this.content = [];
        this.expandedItems = new Set();
        this.itemDataCache = {}; // Cache for storing item data
    }

    connectedCallback() {
        console.log('🔄 SearchResultsDisplay connected to DOM');
        this.render();
    }

    /**
     * Set the search results data
     * @param {Array} references - Array of reference items
     * @param {Array} content - Array of content items  
     * @param {Object} metadata - Search metadata
     */
    setResults(references = [], content = [], metadata = {}) {
        this.references = references;
        this.content = content;
        this.metadata = metadata;
        this.render();
    }

    /**
     * Set search results data and update display (legacy compatibility)
     * @param {Object} data - Search results from /api/references
     */
    setSearchData(data) {
        console.log('🔄 SearchResultsDisplay.setSearchData called with:', {
            references: data.references?.length || 0,
            content: data.content?.length || 0,
            hasMetadata: !!data.metadata
        });
        
        this.setResults(
            data.references || [],
            data.content || [],
            data.metadata || {}
        );
    }

    /**
     * Clear all results
     */
    clear() {
        this.references = [];
        this.content = [];
        this.metadata = {};
        this.expandedItems.clear();
        this.render();
    }

    /**
     * Get the type-based color scheme
     * @param {string} type - The item type
     * @returns {Object} Color scheme object
     */
    getTypeColors(type) {
        const schemes = {
            'method': {
                bg: 'bg-blue-900/40',
                border: 'border-blue-500/60',
                text: 'text-blue-200',
                accent: 'text-blue-400'
            },
            'topic_reference': {
                bg: 'bg-purple-900/40',
                border: 'border-purple-500/60', 
                text: 'text-purple-200',
                accent: 'text-purple-400'
            },
            'definition': {
                bg: 'bg-indigo-900/40',
                border: 'border-indigo-500/60',
                text: 'text-indigo-200',
                accent: 'text-indigo-400'
            },
            'resources': {
                bg: 'bg-green-900/40',
                border: 'border-green-500/60',
                text: 'text-green-200', 
                accent: 'text-green-400'
            },
            'project': {
                bg: 'bg-orange-900/40',
                border: 'border-orange-500/60',
                text: 'text-orange-200',
                accent: 'text-orange-400'
            },
            'default': {
                bg: 'bg-gray-900/40',
                border: 'border-gray-500/60',
                text: 'text-gray-200',
                accent: 'text-gray-400'
            }
        };
        return schemes[type] || schemes.default;
    }

    /**
     * Parse tags from various formats
     * @param {*} tags - Tags in string, array, or other format
     * @returns {Array} Parsed tags array
     */
    parseTags(tags) {
        if (!tags) return [];
        if (Array.isArray(tags)) return tags;
        if (typeof tags === 'string') {
            try {
                return JSON.parse(tags);
            } catch {
                return tags.split(',').map(t => t.trim());
            }
        }
        return [];
    }

    /**
     * Check if a URL is valid and can be opened
     * @param {string} url - URL to validate
     * @returns {boolean} True if valid URL
     */
    isValidUrl(url) {
        if (!url || typeof url !== 'string') return false;
        
        try {
            // Handle relative URLs by assuming they're valid
            if (url.startsWith('/') || url.startsWith('./') || url.startsWith('../')) {
                return true;
            }
            
            // Validate absolute URLs
            const urlObj = new URL(url);
            return ['http:', 'https:'].includes(urlObj.protocol);
        } catch {
            return false;
        }
    }

    /**
     * Get the best clickable link from an item
     * @param {Object} item - The search result item
     * @returns {string|null} URL or null
     */
    getClickableLink(item) {
        // First check direct URL fields (both lowercase and uppercase)
        const linkCandidates = [
            'origin_url', 'ORIGIN_URL',
            'resource_url', 'RESOURCE_URL', 
            'source_url', 'SOURCE_URL',
            'url', 'URL',
            'link', 'LINK'
        ];
        
        for (const candidate of linkCandidates) {
            if (item[candidate]) {
                return item[candidate];
            }
        }
        
        // If no direct URL field found, try to extract URL from content/description
        const contentFields = ['content', 'description', 'text', 'definition'];
        for (const field of contentFields) {
            if (item[field]) {
                const content = item[field];
                
                // Look for ORIGIN_URL: pattern (new format)
                const originUrlMatch = content.match(/ORIGIN_URL:\s*(https?:\/\/[^\s\n]+)/i);
                if (originUrlMatch) {
                    return originUrlMatch[1];
                }
                
                // Look for SOURCE_URL: pattern (backwards compatibility)
                const sourceUrlMatch = content.match(/SOURCE_URL:\s*(https?:\/\/[^\s\n]+)/i);
                if (sourceUrlMatch) {
                    return sourceUrlMatch[1];
                }
                
                // Look for other URL patterns in content
                const urlMatch = content.match(/(?:URL|LINK):\s*(https?:\/\/[^\s\n]+)/i);
                if (urlMatch) {
                    return urlMatch[1];
                }
                
                // Look for any standalone URLs
                const standaloneUrlMatch = content.match(/(https?:\/\/[^\s\n]+)/);
                if (standaloneUrlMatch) {
                    return standaloneUrlMatch[1];
                }
            }
        }
        
        return null;
    }

    /**
     * Get display title from an item
     * @param {Object} item - The search result item
     * @returns {string} Display title
     */
    getDisplayTitle(item) {
        const titleCandidates = ['title', 'name', 'term'];
        for (const candidate of titleCandidates) {
            if (item[candidate]) {
                return item[candidate];
            }
        }
        return 'Untitled';
    }

    /**
     * Get clean display content from an item (without labels)
     * @param {Object} item - The search result item
     * @returns {string} Clean display content
     */
    getCleanDisplayContent(item) {
        // Try to get clean description first
        const description = item.description || item.text || item.definition;
        if (description && description.length > 10 && !description.startsWith('TITLE:') && !description.startsWith('DESCRIPTION:')) {
            return description;
        }
        
        // If we have formatted content with labels, try to extract clean description
        const content = item.content || '';
        if (content.includes('DESCRIPTION:')) {
            const descMatch = content.match(/DESCRIPTION:\s*([^\n]+)/i);
            if (descMatch && descMatch[1]) {
                return descMatch[1].trim();
            }
        }
        
        // Try to extract purpose or other meaningful content
        if (content.includes('PURPOSE:')) {
            const purposeMatch = content.match(/PURPOSE:\s*([^\n]+)/i);
            if (purposeMatch && purposeMatch[1]) {
                return purposeMatch[1].trim();
            }
        }
        
        // Fallback to raw content but clean it up
        if (content) {
            // Remove label-style formatting and take first meaningful line
            const lines = content.split('\n').filter(line => line.trim());
            for (const line of lines) {
                if (!line.match(/^[A-Z_]+:\s/) && line.length > 20) {
                    return line.trim();
                }
            }
            // If no clean line found, return first non-label line
            for (const line of lines) {
                if (!line.match(/^[A-Z_]+:\s/)) {
                    return line.trim();
                }
            }
        }
        
        return 'No description available';
    }

    /**
     * Check if an item is a PDF material
     * @param {Object} item - The search result item
     * @returns {boolean} True if item is a PDF material
     */
    isPDFMaterial(item) {
        console.log('🔍 Checking if item is PDF material:', {
            title: item.title,
            source_type: item.source_type,
            content_type: item.content_type,
            type: item.type,
            source: item.source,
            metadata: item.metadata
        });
        
        // Check for material type indicators - includes actual API response fields
        const isMaterial = item.source_type === 'material' || 
                          item.content_type === 'material' ||
                          item.type === 'material' ||
                          (item.metadata && item.metadata.content_type === 'material');
        
        // Check for PDF file indicators in multiple fields based on actual API response
        const hasPdfIndicator = 
            // Source field (actual API field)
            (item.source && item.source.includes('.pdf')) ||
            // Legacy fields for compatibility
            (item.file_path && item.file_path.includes('.pdf')) ||
            (item.document_name && item.document_name.includes('.pdf')) ||
            item.pdf_source ||
            // Origin URL pointing to PDF endpoint
            (item.origin_url && item.origin_url.includes('/pdf/'));
        
        const result = isMaterial && hasPdfIndicator;
        console.log('📊 PDF Material detection result:', { 
            isMaterial, 
            hasPdfIndicator, 
            result,
            checks: {
                type_material: item.type === 'material',
                content_type_material: item.content_type === 'material',
                metadata_content_type: item.metadata && item.metadata.content_type === 'material',
                source_pdf: item.source && item.source.includes('.pdf'),
                file_path_pdf: item.file_path && item.file_path.includes('.pdf'),
                document_name_pdf: item.document_name && item.document_name.includes('.pdf')
            }
        });
        return result;
    }

    /**
     * Get PDF path for a material item
     * @param {Object} item - The search result item
     * @returns {string|null} PDF file path relative to materials folder
     */
    getPDFPath(item) {
        console.log('🔍 Getting PDF path for item:', {
            title: item.title,
            source: item.source,
            document_name: item.document_name,
            file_path: item.file_path,
            origin_url: item.origin_url
        });
        
        // First try the source field (actual API response field)
        if (item.source && item.source.includes('.pdf')) {
            const path = item.source;
            // Extract filename from full path
            const filename = path.replace(/^.*[\/\\]/, '');
            console.log('📄 Found PDF path via source:', filename);
            return filename;
        }
        
        // Try document_name (if available)
        if (item.document_name && item.document_name.includes('.pdf')) {
            const filename = item.document_name;
            console.log('📄 Found PDF path via document_name:', filename);
            return filename;
        }
        
        // Try file_path and extract filename
        if (item.file_path && item.file_path.includes('.pdf')) {
            const path = item.file_path;
            // Extract filename from full path
            const filename = path.replace(/^.*[\/\\]/, '');
            console.log('📄 Found PDF path via file_path:', filename);
            return filename;
        }
        
        // Try extracting from origin_url
        if (item.origin_url && item.origin_url.includes('/pdf/')) {
            const urlParts = item.origin_url.split('/pdf/');
            if (urlParts.length > 1) {
                const filename = urlParts[1];
                console.log('📄 Found PDF path via origin_url:', filename);
                return filename;
            }
        }
        
        // Fallback to pdf_source
        if (item.pdf_source) {
            const filename = item.pdf_source.replace(/^.*[\/\\]/, '');
            console.log('📄 Found PDF path via pdf_source:', filename);
            return filename;
        }
        
        console.log('❌ No PDF path found for item');
        return null;
    }

    /**
     * Open item in appropriate modal (PDF viewer for materials, browser modal for URLs)
     * @param {string} urlOrPath - URL to open or PDF path
     * @param {string} title - Title for the modal
     * @param {Object} item - The original item (optional, for detecting PDF materials)
     */
    openItemInModal(urlOrPath, title, item = null) {
        console.log('🔧 Opening item in modal:', { 
            urlOrPath: urlOrPath.substring(0, 100) + '...', 
            title, 
            item,
            isPDF: item ? this.isPDFMaterial(item) : false 
        });
        
        // If we have the original item and it's a PDF material, use PDF viewer
        if (item && this.isPDFMaterial(item)) {
            const pdfPath = this.getPDFPath(item);
            if (pdfPath) {
                console.log('📚 Opening PDF material in PDF viewer:', { pdfPath, title });
                
                // Extract metadata for PDF viewer
                const metadata = {
                    fields: item.fields || item.material_fields || item.description || '',
                    created_at: item.created_at || item.processed_at || '',
                    file_size: item.file_size || null,
                    description: item.description || item.material_title || item.title || ''
                };
                
                // Use the correct function signature: openPDFViewer(pdfPath, title, metadata)
                if (window.openPDFViewer) {
                    console.log('✅ Calling window.openPDFViewer with:', { pdfPath, title, metadata });
                    window.openPDFViewer(pdfPath, title, metadata);
                } else {
                    console.warn('❌ PDF viewer not available, falling back to browser modal');
                    this.fallbackToBrowserModal(urlOrPath, title);
                }
                return;
            } else {
                console.warn('⚠️ PDF material detected but no PDF path found');
            }
        }
        
        // Default to browser modal for URLs and non-PDF content
        console.log('🌐 Opening in browser modal (not a PDF material)');
        this.fallbackToBrowserModal(urlOrPath, title);
    }

    /**
     * Fallback to browser modal
     * @param {string} url - URL to open
     * @param {string} title - Title for the modal
     */
    fallbackToBrowserModal(url, title) {
        console.log('🌐 Opening in browser modal:', { url: url.substring(0, 100) + '...', title });
        
        if (window.openBrowserModal) {
            window.openBrowserModal(url, title || 'Search Result');
        } else {
            console.warn('Browser modal not available, opening in new tab');
            window.open(url, '_blank');
        }
    }

    /**
     * Categorize search results by type
     * @returns {Object} Categorized data
     */
    getCategorizedData() {
        const categories = {
            methods: [],
            topics: [],
            resources: [],
            definitions: [],
            projects: [],
            other: []
        };

        // Categorize references
        if (this.references) {
            this.references.forEach(ref => {
                const type = (ref.type || '').toLowerCase();
                const sourceCategory = (ref.source_category || '').toLowerCase();
                
                if (type === 'method' || sourceCategory === 'method') {
                    categories.methods.push(ref);
                } else if (type === 'topic_reference' || sourceCategory === 'topic') {
                    categories.topics.push(ref);
                } else if (type === 'definition' || sourceCategory === 'definition') {
                    categories.definitions.push(ref);
                } else if (type === 'project' || sourceCategory === 'project') {
                    categories.projects.push(ref);
                } else {
                    categories.other.push(ref);
                }
            });
        }

        // Categorize content (typically resources)
        if (this.content) {
            this.content.forEach(content => {
                categories.resources.push(content);
            });
        }

        return categories;
    }

    /**
     * Format tags for display
     */
    formatTags(tags) {
        if (!tags || tags.length === 0) return '';
        
        const tagArray = Array.isArray(tags) ? tags : 
                        (typeof tags === 'string' ? JSON.parse(tags) : []);
        
        return tagArray.slice(0, 5).map(tag => 
            `<span class="inline-block px-2 py-1 text-xs bg-gray-700 text-gray-300 rounded-md mr-1 mb-1">${tag}</span>`
        ).join('');
    }

    /**
     * Get relevance indicator
     */
    getRelevanceIndicator(item) {
        const score = item.relevance_score || item.hybrid_score || item.vector_score || 0;
        const percentage = Math.round(score * 100);
        
        let colorClass = 'text-red-400';
        if (percentage >= 80) colorClass = 'text-green-400';
        else if (percentage >= 60) colorClass = 'text-yellow-400';
        else if (percentage >= 40) colorClass = 'text-orange-400';
        
        return `<span class="${colorClass} text-xs font-medium">${percentage}%</span>`;
    }

    /**
     * Create floating item card with relevance-based positioning
     */
    createItemCard(item, category, index, totalItems) {
        const title = item.title || item.term || item.name || 'Untitled';
        const content = item.content || item.description || item.text || '';
        // Use clean content for display (no labels)
        const cleanContent = this.getCleanDisplayContent(item);
        
        const tags = this.formatTags(item.tags);
        const relevance = this.getRelevanceIndicator(item);
        
        // Category-specific icons and styling
        const categoryInfo = item.categoryInfo || this.getCategoryInfo(category, item);
        const typeColors = this.getTypeColors(item.type);
        
        // Check if item has a clickable URL
        const url = this.getClickableLink(item);
        const hasUrl = url && this.isValidUrl(url);
        
        // Debug logging for URL detection
        if (hasUrl) {
            console.log(`🔗 Found URL for item "${title}": ${url}`);
        } else {
            console.log(`📄 No URL found for item "${title}". Checked content for URLs but none found.`);
            // Log the item data to help debug
            console.log('Item data for debugging:', {
                title,
                url_fields: {
                    origin_url: item.origin_url,
                    ORIGIN_URL: item.ORIGIN_URL,
                    source_url: item.source_url,
                    SOURCE_URL: item.SOURCE_URL,
                    url: item.url,
                    URL: item.URL
                },
                content_sample: (item.content || item.description || '').substring(0, 200)
            });
        }
        
        // Calculate relevance-based positioning
        const score = item.relevance_score || item.hybrid_score || item.vector_score || 0;
        const position = this.calculateFloatingPosition(score, index, totalItems);
        
        // Size based on relevance (more relevant = larger)
        const sizeClass = score > 0.8 ? 'w-72 h-44' : 
                         score > 0.6 ? 'w-64 h-40' : 
                         score > 0.4 ? 'w-56 h-36' : 'w-48 h-32';
        
        // Opacity and z-index based on relevance
        const opacity = Math.max(0.7, score);
        const zIndex = Math.round(score * 50) + 10;
        
        // All cards are clickable and show open link button
        const clickableClass = 'cursor-pointer hover:ring-2 hover:ring-blue-400';
        
        // Generate unique ID for this card
        const cardId = `card-${item.id || Math.random().toString(36).substr(2, 9)}`;
        
        // Prepare data for browser modal
        const escapedUrl = hasUrl ? url.replace(/'/g, "\\'").replace(/"/g, '\\"') : '';
        const escapedTitle = title.replace(/'/g, "\\'").replace(/"/g, '\\"');
        const escapedContent = content.replace(/'/g, "\\'").replace(/"/g, '\\"'); // Use full content for modal
        const escapedCleanContent = cleanContent.replace(/'/g, "\\'").replace(/"/g, '\\"'); // Use clean content for card
        
        // Create data URI for content if no URL
        const modalUrl = hasUrl ? escapedUrl : `data:text/html;charset=utf-8,${encodeURIComponent(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>${title}</title>
                <style>
                    body { font-family: system-ui, -apple-system, sans-serif; padding: 2rem; line-height: 1.6; max-width: 800px; margin: 0 auto; }
                    h1 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 0.5rem; }
                    .content { white-space: pre-wrap; background: #f9f9f9; padding: 1rem; border-radius: 8px; border-left: 4px solid #007acc; }
                    .metadata { margin-top: 2rem; padding: 1rem; background: #f0f8ff; border-radius: 8px; font-size: 0.9em; }
                </style>
            </head>
            <body>
                <h1>${title}</h1>
                <div class="content">${content}</div>
                <div class="metadata">
                    <strong>Category:</strong> ${categoryInfo.label}<br>
                    <strong>Type:</strong> ${item.type || 'Unknown'}<br>
                    <strong>Relevance:</strong> ${Math.round(score * 100)}%
                </div>
            </body>
            </html>
        `)}`;
        
        // Store the item data for later retrieval
        const itemKey = `item_${item.id || Math.random()}`;
        this.itemDataCache = this.itemDataCache || {};
        this.itemDataCache[itemKey] = item;

        return `
            <div class="floating-card absolute ${sizeClass} ${typeColors.bg} ${typeColors.border} border rounded-xl p-3 
                        hover:scale-105 transition-all duration-300 ${clickableClass} backdrop-blur-sm
                        hover:shadow-xl hover:bg-opacity-90" 
                 data-item-id="${item.id || Math.random()}"
                 data-card-id="${cardId}"
                 data-item-key="${itemKey}"
                 onclick="this.closest('search-results-display').handleCardClick(this)"
                 style="left: ${position.x}%; top: ${position.y}%; opacity: ${opacity}; z-index: ${zIndex};">
                
                <div class="flex items-start justify-between mb-2">
                    <div class="flex items-center gap-2 flex-1 min-w-0">
                        <span class="text-sm">${categoryInfo.icon}</span>
                        <h3 class="font-medium text-white text-xs truncate">${title}</h3>
                    </div>
                    <div class="flex items-center gap-1 text-xs flex-shrink-0">
                        ${relevance}
                    </div>
                </div>
                
                <div class="text-xs ${typeColors.text} mb-2 leading-relaxed" style="overflow: hidden; word-wrap: break-word; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;">
                    <span class="whitespace-pre-wrap">${cleanContent.substring(0, 120)}${cleanContent.length > 120 ? '...' : ''}</span>
                </div>
                
                <!-- Always show open link button -->
                <button class="open-link-btn text-xs text-blue-400 hover:text-blue-300 mb-2 transition-colors flex items-center gap-1" 
                        onclick="event.stopPropagation(); this.closest('search-results-display').handleButtonClick(this)">
                    <span>${hasUrl ? 'Open Link' : 'View Details'}</span>
                    <span class="text-sm">↗</span>
                </button>
                
                ${hasUrl && (!cleanContent || cleanContent === 'No description available' || cleanContent.length < 20) ? `
                    <div class="text-xs text-blue-400 mb-1 truncate" title="${url}">
                        ${url.length > 30 ? url.substring(0, 30) + '...' : url}
                    </div>
                ` : ''}
                
                <div class="absolute bottom-2 left-3 right-3 flex items-center justify-between text-xs">
                    <span class="${typeColors.accent} font-medium text-xs">${categoryInfo.label}</span>
                    <span class="text-gray-500 text-xs">${item.created_at ? new Date(item.created_at).toLocaleDateString() : ''}</span>
                </div>
            </div>
        `;
    }

    /**
     * Get category-specific information
     */
    getCategoryInfo(category, item) {
        const categoryMap = {
            methods: { icon: '⚙️', label: 'Method', color: 'blue' },
            topics: { icon: '🏷️', label: 'Topic', color: 'purple' },
            resources: { icon: '📚', label: 'Resource', color: 'green' },
            definitions: { icon: '📖', label: 'Definition', color: 'indigo' },
            projects: { icon: '🚀', label: 'Project', color: 'orange' },
            other: { icon: '📄', label: 'Reference', color: 'gray' }
        };
        
        return categoryMap[category] || categoryMap.other;
    }

    /**
     * Create metadata summary
     */
    createMetadataSummary() {
        if (!this.metadata) return '';
        
        const meta = this.metadata;
        const storeInfo = meta.stores_info || {};
        
        return `
            <div class="bg-gray-900/50 border border-gray-700 rounded-lg p-4 mb-6">
                <h3 class="text-lg font-medium text-white mb-3">📊 Search Summary</h3>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div class="text-center">
                        <div class="text-2xl font-bold text-blue-400">${meta.reference_count || 0}</div>
                        <div class="text-gray-400">References</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-green-400">${meta.content_count || 0}</div>
                        <div class="text-gray-400">Content Items</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-purple-400">${storeInfo.topic_stores_loaded || 0}</div>
                        <div class="text-gray-400">Topic Stores</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-yellow-400">${meta.context_words || 0}</div>
                        <div class="text-gray-400">Context Words</div>
                    </div>
                </div>
                
                <div class="mt-4 text-xs text-gray-500">
                    Search completed at ${new Date(meta.timestamp).toLocaleString()}
                    ${meta.k_value ? ` • k=${meta.k_value}` : ''}
                </div>
            </div>
        `;
    }

    /**
     * Create unified floating results section
     */
    createUnifiedFloatingResults() {
        // Combine all items from all categories
        const categories = this.getCategorizedData();
        const allItems = [];
        
        Object.entries(categories).forEach(([category, items]) => {
            items.forEach(item => {
                allItems.push({
                    ...item,
                    category: category,
                    categoryInfo: this.getCategoryInfo(category)
                });
            });
        });
        
        // Sort all items by relevance score (highest first)
        const sortedItems = allItems.sort((a, b) => {
            const scoreA = a.relevance_score || a.hybrid_score || a.vector_score || 0;
            const scoreB = b.relevance_score || b.hybrid_score || b.vector_score || 0;
            return scoreB - scoreA;
        });
        
        return `
            <div class="unified-floating-results relative min-h-screen w-full">
                <!-- All floating items layered together -->
                ${sortedItems.map((item, index) => 
                    this.createItemCard(item, item.category, index, sortedItems.length)
                ).join('')}
            </div>
        `;
    }

    /**
     * Render the component
     */
    render() {
        if (!this.references && !this.content) {
            this.innerHTML = `
                <div class="text-center text-gray-400 py-8">
                    <div class="text-4xl mb-4">🔍</div>
                    <div>No search results to display</div>
                </div>
            `;
            return;
        }

        const categories = this.getCategorizedData();
        const metadataSummary = this.createMetadataSummary();
        
        // Count total items
        const totalItems = Object.values(categories).reduce((sum, items) => sum + items.length, 0);
        
        if (totalItems === 0) {
            this.innerHTML = `
                <div class="text-center text-gray-400 py-8">
                    <div class="text-4xl mb-4">📭</div>
                    <div>No results found for your search</div>
                </div>
            `;
            return;
        }

        // Generate unified floating results
        const floatingResults = this.createUnifiedFloatingResults();

        this.innerHTML = `
            <style>
                .search-results-display {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    pointer-events: none;
                }
                .floating-results-container {
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    pointer-events: none;
                    z-index: 1;
                    background: radial-gradient(ellipse at center, rgba(59, 130, 246, 0.08) 0%, transparent 70%);
                }
                .floating-card {
                    backdrop-filter: blur(8px);
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    pointer-events: all;
                }
                .floating-card:hover {
                    transform: scale(1.05) translateZ(0);
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
                    z-index: 1000 !important;
                }
                .line-clamp-4 {
                    display: -webkit-box;
                    -webkit-line-clamp: 4;
                    -webkit-box-orient: vertical;
                    overflow: hidden;
                }
                .search-results-summary {
                    display: none; /* Hide the large grey summary box that interferes with floating cards */
                    position: absolute;
                    bottom: 120px;
                    left: 50%;
                    transform: translateX(-50%);
                    z-index: 200;
                    pointer-events: all;
                    width: calc(100% - 2rem);
                    max-width: 700px;
                    background: rgba(17, 24, 39, 0.95);
                    backdrop-filter: blur(10px);
                    border-radius: 12px;
                    border: 1px solid rgba(75, 85, 99, 0.3);
                }
                .generate-response-overlay {
                    position: absolute;
                    bottom: -80px;
                    left: 50%;
                    transform: translateX(-50%);
                    z-index: 200;
                    pointer-events: all;
                }
            </style>
            
            <div class="search-results-display">
                <!-- Summary appears above search field -->
                <div class="search-results-summary">
                    ${metadataSummary}
                </div>
                
                <!-- Floating results behind search field -->
                <div class="floating-results-container">
                    ${floatingResults}
                </div>
                
                <!-- Generate response button appears below search field -->
                <div class="generate-response-overlay">
                    <button id="generateResponseBtn" 
                            class="bg-indigo-600/90 hover:bg-indigo-700 text-white px-8 py-4 rounded-full font-medium 
                                   transition-all backdrop-blur-sm border border-indigo-500/50 hover:scale-105 shadow-2xl
                                   flex items-center gap-2">
                        <span class="text-lg">✨</span>
                        <span>Generate Response</span>
                        <span class="text-sm opacity-75">(${totalItems} items)</span>
                    </button>
                </div>
            </div>
        `;

        // Add event listeners
        this.addEventListeners();
    }

    /**
     * Add event listeners for interactions
     */
    addEventListeners() {
        // Generate response button
        const generateBtn = this.querySelector('#generateResponseBtn');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => {
                this.dispatchEvent(new CustomEvent('generate-response', {
                    detail: { 
                        references: this.references,
                        content: this.content,
                        metadata: this.metadata
                    }
                }));
            });
        }

        // Item click handlers - all items now use browser modal
        this.querySelectorAll('[data-item-id]').forEach(card => {
            card.addEventListener('click', (e) => {
                // Don't trigger on button clicks (but open-link-btn is fine since it does the same thing)
                if (e.target.closest('.generate-btn') || 
                    e.target.closest('#generateResponseBtn')) {
                    return;
                }
                
                const itemId = e.currentTarget.dataset.itemId;
                
                // Find the item by ID
                const allItems = [...(this.references || []), ...(this.content || [])];
                const clickedItem = allItems.find(item => 
                    (item.id || Math.random()).toString() === itemId
                );
                
                if (clickedItem) {
                    // Get the actual URL first, if available
                    const url = this.getClickableLink(clickedItem);
                    const title = clickedItem.title || clickedItem.term || clickedItem.name || 'Search Result';
                    
                    // Prioritize real URLs over content display
                    if (url && this.isValidUrl(url)) {
                        console.log('Opening real URL in browser modal via card click:', url);
                        this.openItemInModal(url, title, clickedItem);
                    } else {
                        // Create data URI for content display only if no real URL exists
                        const content = clickedItem.content || clickedItem.description || clickedItem.text || '';
                        const categoryInfo = this.getCategoryInfo('other', clickedItem);
                        const score = clickedItem.relevance_score || clickedItem.hybrid_score || clickedItem.vector_score || 0;
                        
                        const modalUrl = `data:text/html;charset=utf-8,${encodeURIComponent(`
                            <!DOCTYPE html>
                            <html>
                            <head>
                                <title>${title}</title>
                                <style>
                                    body { font-family: system-ui, -apple-system, sans-serif; padding: 2rem; line-height: 1.6; max-width: 800px; margin: 0 auto; }
                                    h1 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 0.5rem; }
                                    .content { white-space: pre-wrap; background: #f9f9f9; padding: 1rem; border-radius: 8px; border-left: 4px solid #007acc; }
                                    .metadata { margin-top: 2rem; padding: 1rem; background: #f0f8ff; border-radius: 8px; font-size: 0.9em; }
                                </style>
                            </head>
                            <body>
                                <h1>${title}</h1>
                                <div class="content">${content}</div>
                                <div class="metadata">
                                    <strong>Category:</strong> ${categoryInfo.label}<br>
                                    <strong>Type:</strong> ${clickedItem.type || 'Unknown'}<br>
                                    <strong>Relevance:</strong> ${Math.round(score * 100)}%
                                </div>
                            </body>
                            </html>
                        `)}`;
                        console.log('Opening content in browser modal via card click for item:', title);
                        this.openItemInModal(modalUrl, title, clickedItem);
                    }
                }
            });
        });

        // Add subtle floating animation
        this.addFloatingAnimation();
    }

    /**
     * Add subtle floating animation to cards
     */
    addFloatingAnimation() {
        const cards = this.querySelectorAll('.floating-card');
        
        cards.forEach((card, index) => {
            // Subtle continuous floating motion
            const duration = 3000 + (index % 1000); // 3-4 second cycles
            const delay = index * 100; // Stagger animations
            
            card.style.animation = `floatGentle ${duration}ms ease-in-out infinite ${delay}ms`;
        });
        
        // Add CSS animation if not already added
        if (!document.getElementById('floating-animations')) {
            const style = document.createElement('style');
            style.id = 'floating-animations';
            style.textContent = `
                @keyframes floatGentle {
                    0%, 100% { transform: translateY(0px) rotate(0deg); }
                    50% { transform: translateY(-3px) rotate(0.5deg); }
                }
            `;
            document.head.appendChild(style);
        }
    }

    /**
     * Export search results
     */
    exportResults() {
        if (!this.references && !this.content) return;
        
        const exportData = {
            timestamp: new Date().toISOString(),
            metadata: this.metadata,
            results: {
                references: this.references || [],
                content: this.content || []
            }
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `search-results-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Calculate floating position based on relevance score
     * @param {number} score - Relevance score (0-1)
     * @param {number} index - Item index
     * @param {number} total - Total items
     * @returns {Object} Position with x, y percentages
     */
    calculateFloatingPosition(score, index, total) {
        // Add some randomness but maintain relevance-based zones
        const randomOffset = () => (Math.random() - 0.5) * 15; // ±7.5% random variation
        
        // Most relevant items go in center area (avoid search field)
        if (score > 0.8) {
            return {
                x: 30 + (Math.random() * 40) + randomOffset(), // Center area (30-70%)
                y: 60 + (Math.random() * 30) + randomOffset()  // Lower area to avoid search field
            };
        }
        
        // Medium relevance items in middle zones
        if (score > 0.6) {
            const side = index % 2; // Alternate sides
            return {
                x: side === 0 ? 15 + (Math.random() * 30) : 55 + (Math.random() * 30), // Left or right
                y: 40 + (Math.random() * 50) + randomOffset()  // Middle-bottom area
            };
        }
        
        // Lower relevance items on the edges and corners
        if (score > 0.4) {
            const position = index % 6;
            switch(position) {
                case 0: return { x: 5 + (Math.random() * 20), y: 20 + (Math.random() * 30) }; // Top-left
                case 1: return { x: 75 + (Math.random() * 20), y: 20 + (Math.random() * 30) }; // Top-right
                case 2: return { x: 5 + (Math.random() * 20), y: 60 + (Math.random() * 30) }; // Bottom-left
                case 3: return { x: 75 + (Math.random() * 20), y: 60 + (Math.random() * 30) }; // Bottom-right
                case 4: return { x: 30 + (Math.random() * 40), y: 15 + (Math.random() * 20) }; // Top center (sparse)
                case 5: return { x: 10 + (Math.random() * 80), y: 85 + (Math.random() * 10) }; // Very bottom
            }
        }
        
        // Lowest relevance items scattered everywhere except center
        const angle = (index / total) * 2 * Math.PI;
        const radius = 40 + (Math.random() * 25); // 40-65% from center
        const centerX = 50;
        const centerY = 50;
        
        return {
            x: Math.max(5, Math.min(90, centerX + Math.cos(angle) * radius)),
            y: Math.max(10, Math.min(85, centerY + Math.sin(angle) * radius * 0.8))
        };
    }

    /**
     * Handle card click events with proper item data retrieval
     * @param {HTMLElement} cardElement - The clicked card element
     */
    handleCardClick(cardElement) {
        const itemKey = cardElement.dataset.itemKey;
        const item = this.itemDataCache[itemKey];
        
        if (!item) {
            console.warn('Item data not found for key:', itemKey);
            return;
        }

        console.log('🖱️ Card clicked for item:', {
            title: item.title,
            source_type: item.source_type,
            content_type: item.content_type,
            type: item.type,
            document_name: item.document_name,
            origin_url: item.origin_url
        });
        
        const title = item.title || item.term || item.name || 'Search Result';
        
        // Check if this is a PDF material first
        if (this.isPDFMaterial(item)) {
            console.log('📚 Item detected as PDF material, opening PDF viewer');
            const pdfPath = this.getPDFPath(item);
            if (pdfPath) {
                // Create metadata for PDF viewer
                const metadata = {
                    fields: item.fields || item.material_fields || item.description || '',
                    created_at: item.created_at || item.processed_at || '',
                    file_size: item.file_size || null,
                    description: item.description || item.material_title || item.title || ''
                };
                
                console.log('✅ Opening PDF viewer directly with:', { pdfPath, title, metadata });
                
                if (window.openPDFViewer) {
                    window.openPDFViewer(pdfPath, title, metadata);
                    return;
                } else {
                    console.warn('❌ PDF viewer not available');
                }
            } else {
                console.warn('⚠️ PDF material detected but no PDF path found');
            }
        }
        
        // Fallback to URL-based opening
        const url = this.getClickableLink(item);
        
        if (url && this.isValidUrl(url)) {
            console.log('🔗 Opening real URL in browser modal via card click:', url);
            this.openItemInModal(url, title, item);
        } else {
            // Create data URI for content display only if no real URL exists
            const content = item.content || item.description || item.text || '';
            const cleanContent = this.getCleanDisplayContent(item);
            const categoryInfo = this.getCategoryInfo('other', item);
            const score = item.relevance_score || item.hybrid_score || item.vector_score || 0;
            
            const modalUrl = `data:text/html;charset=utf-8,${encodeURIComponent(`
                <!DOCTYPE html>
                <html>
                <head>
                    <title>${title}</title>
                    <style>
                        body { font-family: system-ui, -apple-system, sans-serif; padding: 2rem; line-height: 1.6; max-width: 800px; margin: 0 auto; }
                        h1 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 0.5rem; }
                        .content { white-space: pre-wrap; background: #f9f9f9; padding: 1rem; border-radius: 8px; border-left: 4px solid #007acc; }
                        .metadata { margin-top: 2rem; padding: 1rem; background: #f0f8ff; border-radius: 8px; font-size: 0.9em; }
                    </style>
                </head>
                <body>
                    <h1>${title}</h1>
                    <div class="content">${cleanContent || 'No content available'}</div>
                    <div class="metadata">
                        <strong>Category:</strong> ${categoryInfo.name}<br>
                        <strong>Type:</strong> ${item.source_type || item.type || 'Unknown'}<br>
                        <strong>Relevance:</strong> ${Math.round(score * 100)}%
                    </div>
                </body>
                </html>
            `)}`;
            
            console.log('📄 Opening content in browser modal via card click for item:', title);
            this.openItemInModal(modalUrl, title, item);
        }
    }

    /**
     * Handle button click events with proper item data retrieval
     * @param {HTMLElement} buttonElement - The clicked button element
     */
    handleButtonClick(buttonElement) {
        const cardElement = buttonElement.closest('[data-item-key]');
        if (cardElement) {
            this.handleCardClick(cardElement);
        }
    }

    // ...existing code...
}

// Register the custom element
customElements.define('search-results-display', SearchResultsDisplay);

console.log('✅ SearchResultsDisplay component registered');
