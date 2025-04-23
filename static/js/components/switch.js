/**
 * View Switch Component
 * Handles switching between different application views
 */
class ViewSwitch {
    constructor() {
        this.activeView = 'search'; // Default view
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
            },
            'settings': {
                id: 'settingsView',
                icon: 'settings',
                label: 'Settings',
                script: '/static/js/slides/settings.js'
            }
        };
        
        // All scripts are already loaded in index.html
        this.loadedScripts = {
            'search': true,
            'manage_data': true,
            'settings': true
        };
    }

    /**
     * Initialize the component in the DOM
     */
    init() {
        this.render();
        this.attachEventListeners();
    }

    /**
     * Render the switch component in the DOM
     */
    render() {
        const container = document.getElementById('viewSwitchContainer');
        if (!container) return;

        // Create the switch component
        let html = '<div class="flex bg-neutral-900 rounded-lg p-1 shadow-md">';
        
        Object.keys(this.views).forEach(key => {
            const view = this.views[key];
            const isActive = this.activeView === key;
            
            html += `
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
        });
        
        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * Get SVG icon markup for the view buttons
     */
    getIconSvg(iconName) {
        const icons = {
            'search': '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>',
            'database': '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7c-2 0-3 1-3 3z M9 17v-6 M9 7v0 M15 17v-2 M15 11v-4" /></svg>',
            'settings': '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>'
        };
        
        return icons[iconName] || '';
    }

    /**
     * Attach event listeners to switch buttons
     */
    attachEventListeners() {
        const buttons = document.querySelectorAll('.view-switch-btn');
        buttons.forEach(button => {
            button.addEventListener('click', (e) => {
                const view = e.currentTarget.dataset.view;
                this.switchView(view);
            });
        });
    }

    /**
     * Switch to the selected view
     * @param {string} view - The view to switch to
     */
    switchView(view) {
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
        
        // Update active state
        this.activeView = view;
        this.render();
        
        // Reattach event listeners to the newly rendered buttons
        this.attachEventListeners();
        
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
        
        const scriptSrc = this.views[view].script;
        const script = document.createElement('script');
        script.src = scriptSrc;
        script.onload = () => {
            console.log(`Loaded ${view} script`);
            this.loadedScripts[view] = true;
        };
        document.body.appendChild(script);
    }
}

// Initialize the component when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing ViewSwitch component');
    window.viewSwitch = new ViewSwitch();
    window.viewSwitch.init();
    
    // Give the other components a moment to initialize
    setTimeout(() => {
        // Set the initial view (the default is already 'search' in the constructor)
        window.viewSwitch.switchView(window.viewSwitch.activeView);
    }, 100);
});
