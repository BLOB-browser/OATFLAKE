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
    }

    connectedCallback() {
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
     * Get the best clickable link from an item
     * @param {Object} item - The search result item
     * @returns {string|null} URL or null
     */
    getClickableLink(item) {
        const linkCandidates = ['origin_url', 'resource_url', 'source_url', 'url', 'link'];
        for (const candidate of linkCandidates) {
            if (item[candidate]) {
                return item[candidate];
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
     * Get display content from an item
     * @param {Object} item - The search result item
     * @returns {string} Display content
     */
    getDisplayContent(item) {
        const contentCandidates = ['content', 'description', 'text', 'definition'];
        for (const candidate of contentCandidates) {
            if (item[candidate]) {
                return item[candidate];
            }
        }
        return 'No content available';
    }

    /**
     * Toggle expansion of an item
     * @param {string} itemId - Unique item identifier
     */
    toggleExpansion(itemId) {
        if (this.expandedItems.has(itemId)) {
            this.expandedItems.delete(itemId);
        } else {
            this.expandedItems.add(itemId);
        }
        this.render();
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
        const truncatedContent = content.length > 120 ? content.substring(0, 120) + '...' : content;
        
        const tags = this.formatTags(item.tags);
        const relevance = this.getRelevanceIndicator(item);
        
        // Category-specific icons and styling
        const categoryInfo = item.categoryInfo || this.getCategoryInfo(category, item);
        const typeColors = this.getTypeColors(item.type);
        
        // Calculate relevance-based positioning
        const score = item.relevance_score || item.hybrid_score || item.vector_score || 0;
        const position = this.calculateFloatingPosition(score, index, totalItems);
        
        // Size based on relevance (more relevant = larger)
        const sizeClass = score > 0.8 ? 'w-72 h-40' : 
                         score > 0.6 ? 'w-64 h-36' : 
                         score > 0.4 ? 'w-56 h-32' : 'w-48 h-28';
        
        // Opacity and z-index based on relevance
        const opacity = Math.max(0.7, score);
        const zIndex = Math.round(score * 50) + 10; // Lower z-index range to stay behind content
        
        return `
            <div class="floating-card absolute ${sizeClass} ${typeColors.bg} ${typeColors.border} border rounded-xl p-3 
                        hover:scale-105 transition-all duration-300 cursor-pointer backdrop-blur-sm
                        hover:shadow-xl" 
                 data-item-id="${item.id || Math.random()}"
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
                
                <div class="text-xs ${typeColors.text} mb-2 leading-relaxed line-clamp-3">
                    ${truncatedContent}
                </div>
                
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

        // Item click handlers
        this.querySelectorAll('[data-item-id]').forEach(card => {
            card.addEventListener('click', (e) => {
                const itemId = e.currentTarget.dataset.itemId;
                this.dispatchEvent(new CustomEvent('item-selected', {
                    detail: { 
                        itemId,
                        references: this.references,
                        content: this.content,
                        metadata: this.metadata
                    }
                }));
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
}

// Register the custom element
customElements.define('search-results-display', SearchResultsDisplay);

console.log('✅ SearchResultsDisplay component registered');
