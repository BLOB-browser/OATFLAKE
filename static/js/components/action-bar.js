/**
 * Action Bar Component
 * A flexible action bar that displays a switch on the left and settings on the right
 * Now includes view switching functionality
 */
class ActionBar extends HTMLElement {
    constructor() {
        super();
        this.switchEnabled = false;
        this.switchLabel = 'Enable';
        this.onSwitchToggle = null;
        
        // View switching functionality
        this.activeView = 'search'; // Default view
        this.previousView = 'search'; // Track previous view for returning from modal
        this.views = {
            'search': {
                id: 'searchView',
                icon: 'search',
                label: 'Search',
                script: '/static/js/slides/search.js'
            },
            'manage_data': {
                id: 'dataView',
                icon: 'database',
                label: 'Data',
                script: '/static/js/slides/data.js'
            }
        };
        
        // Settings button configuration (but not a real view anymore)
        this.settingsView = {
            id: 'settingsButton',
            icon: 'settings',
            label: 'Settings'
        };
        
        // All scripts are already loaded in index.html
        this.loadedScripts = {
            'search': true,
            'manage_data': true
        };
    }

    connectedCallback() {
        this.render();
        this.setupEventListeners();
    }

    render() {
        this.innerHTML = `
            <div class="flex w-full items-center justify-between py-2">
                <!-- View switching section (left) -->
                <div id="viewSwitchSection" class="flex">
                    <div class="flex bg-neutral-900 rounded-lg p-1 shadow-md">
                        ${Object.keys(this.views).map(key => {
                            const view = this.views[key];
                            const isActive = this.activeView === key;
                            
                            return `
                                <button id="${view.id}Button" 
                                    class="view-switch-btn flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                                        isActive 
                                            ? 'bg-indigo-600 text-white' 
                                            : 'text-gray-400 hover:text-white hover:bg-neutral-800'
                                    }" 
                                    data-view="${key}">
                                    <span class="icon mr-2">
                                        ${this.getIconSvg(view.icon)}
                                    </span>
                                    ${view.label}
                                </button>
                            `;
                        }).join('')}
                    </div>
                </div>
                
                <!-- Settings button (right) -->
                <div>
                    <button id="settingsButton" 
                        class="view-switch-btn flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors text-gray-400 hover:text-white hover:bg-neutral-800" 
                        data-view="settings">
                        <span class="icon mr-2">
                            ${this.getIconSvg('settings')}
                        </span>
                        ${this.settingsView.label}
                    </button>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        const switchElement = this.querySelector('#action-switch');
        const settingsButton = this.querySelector('#settingsButton');
        const actionButton = this.querySelector('#action-button');
        
        if (switchElement) {
            switchElement.checked = this.switchEnabled;
            switchElement.addEventListener('change', (e) => {
                this.switchEnabled = e.target.checked;
                if (typeof this.onSwitchToggle === 'function') {
                    this.onSwitchToggle(this.switchEnabled);
                }
                // Dispatch a custom event that other components can listen for
                this.dispatchEvent(new CustomEvent('switchToggle', {
                    detail: { enabled: this.switchEnabled },
                    bubbles: true
                }));
            });
        }
        
        if (settingsButton) {
            settingsButton.addEventListener('click', () => {
                // Store the current view before opening settings
                this.previousView = this.activeView;
                
                // Open settings modal directly
                const settingsModal = document.getElementById('settingsModal');
                if (settingsModal) {
                    // Don't switch view, just open the modal over the current view
                    settingsModal.classList.remove('hidden');
                    // Apply animation
                    settingsModal.style.opacity = '0';
                    setTimeout(() => {
                        settingsModal.style.opacity = '1';
                        settingsModal.style.transition = 'opacity 0.2s ease-in-out';
                    }, 10);
                    
                    // Set up listener for when settings modal closes
                    const closeButtons = settingsModal.querySelectorAll('#closeSettingsModal, #settingsDoneButton');
                    closeButtons.forEach(button => {
                        // Use once: true to ensure this only happens once per click
                        button.addEventListener('click', () => {
                            console.log('Settings closed, returning to:', this.activeView);
                            // Hide modal with animation
                            settingsModal.style.opacity = '0';
                            setTimeout(() => {
                                settingsModal.classList.add('hidden');
                                
                                // Make sure the view container matches the active view in the action bar
                                const allViewContainers = document.querySelectorAll('.view-container');
                                allViewContainers.forEach(container => {
                                    container.classList.add('hidden');
                                });
                                
                                const currentViewContainer = document.getElementById(`${this.activeView}ViewContainer`);
                                if (currentViewContainer) {
                                    console.log(`Ensuring visibility of ${this.activeView}ViewContainer`);
                                    currentViewContainer.classList.remove('hidden');
                                }
                            }, 200);
                        }, { once: true });
                    });
                }
                
                // Dispatch a custom event for settings click
                this.dispatchEvent(new CustomEvent('settingsClick', {
                    bubbles: true
                }));
            });
        }

        if (actionButton) {
            actionButton.addEventListener('click', () => {
                // Dispatch a custom event for action button click
                this.dispatchEvent(new CustomEvent('actionClick', {
                    bubbles: true
                }));
            });
        }

        // Add view switch event listeners for the main view buttons (search and data)
        const viewButtons = this.querySelectorAll('#viewSwitchSection .view-switch-btn');
        viewButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const view = e.currentTarget.dataset.view;
                this.switchView(view);
            });
        });
    }
    
    // Public API methods
    setSwitchState(enabled) {
        this.switchEnabled = enabled;
        const switchElement = this.querySelector('#action-switch');
        if (switchElement) {
            switchElement.checked = enabled;
        }
        return this;
    }
    
    setSwitchLabel(label) {
        this.switchLabel = label;
        const labelElement = this.querySelector('#action-label');
        if (labelElement) {
            labelElement.textContent = label;
        }
        return this;
    }
    
    onToggle(callback) {
        this.onSwitchToggle = callback;
        return this;
    }

    // View switching methods from ViewSwitch
    getIconSvg(iconName) {
        const icons = {
            'search': '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>',
            'database': '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7c-2 0-3 1-3 3z M9 17v-6 M9 7v0 M15 17v-2 M15 11v-4" /></svg>',
            'settings': '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>'
        };
        
        return icons[iconName] || '';
    }

    /**
     * Switch to the selected view
     * @param {string} view - The view to switch to
     */
    switchView(view) {
        // Don't do anything if already on selected view
        if (view === this.activeView) return;
        
        console.log(`Switching to view: ${view}`);
        
        // Hide all view containers
        const allViewContainers = document.querySelectorAll('.view-container');
        allViewContainers.forEach(container => {
            container.classList.add('hidden');
        });
        
        // Show selected view container
        const selectedViewContainer = document.getElementById(`${view}ViewContainer`);
        if (selectedViewContainer) {
            console.log(`Found container: ${view}ViewContainer`);
            selectedViewContainer.classList.remove('hidden');
        } else {
            console.error(`Container not found: ${view}ViewContainer`);
        }
        
        // Load script if not already loaded (though all are already loaded in our setup)
        this.loadViewScript(view);
        
        // Update active view state
        this.activeView = view;
        
        // Always ensure settings button appears as a regular button, not an active tab
        const settingsButton = this.querySelector('#settingsButton');
        if (settingsButton) {
            settingsButton.classList.remove('bg-indigo-600', 'text-white');
            settingsButton.classList.add('text-gray-400', 'hover:text-white', 'hover:bg-neutral-800');
        }
        
        // Re-render the view switching buttons and reattach event listeners
        this.render();
        this.setupEventListeners();
        
        // Dispatch event for other components to listen to
        const event = new CustomEvent('viewChanged', { detail: { view } });
        document.dispatchEvent(event);
    }

    /**
     * Load the JavaScript file for a view if not already loaded
     * @param {string} view - The view whose script needs to be loaded
     */
    loadViewScript(view) {
        if (this.loadedScripts[view]) return;
        
        // Get the script source from the view object
        let scriptSrc;
        if (this.views[view]) {
            scriptSrc = this.views[view].script;
        } else {
            console.error(`Unknown view: ${view}`);
            return;
        }
        
        const script = document.createElement('script');
        script.src = scriptSrc;
        script.onload = () => {
            console.log(`Loaded ${view} script`);
            this.loadedScripts[view] = true;
        };
        document.body.appendChild(script);
    }

    /**
     * Initialize view with default or specified view
     * @param {string} initialView - Optional initial view to display
     */
    initView(initialView) {
        if (initialView && this.views[initialView]) {
            this.activeView = initialView;
        }
        
        // Give the other components a moment to initialize before switching
        setTimeout(() => {
            this.switchView(this.activeView);
        }, 100);
    }
}

customElements.define('action-bar', ActionBar);

// Initialize the view functionality when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing ActionBar with view switching');
    const actionBar = document.getElementById('globalActionBar');
    if (actionBar) {
        // Initialize with default view
        actionBar.initView();
    }
});
