/**
 * SettingsSlide - Handles the generation of the settings modal overlay
 */
const SettingsSlide = (() => {
    // HTML template for the settings modal with tabs
    const template = `
        <!-- Settings Modal Overlay -->
        <div id="settingsModal" class="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 hidden">
            <div class="bg-black rounded-3xl border border-neutral-700 shadow-2xl w-full max-w-6xl max-h-[90vh] flex flex-col">
                <!-- Modal Header with Tabs -->
                <div class="px-6 pt-6 pb-0">
                    <div class="flex items-center justify-between mb-6">
                        <h2 class="text-2xl font-semibold flex items-center">
                            <svg class="w-6 h-6 mr-2 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                            </svg>
                            Settings
                        </h2>
                        <button id="closeSettingsModal" class="text-neutral-400 hover:text-white transition p-2">
                            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </button>
                    </div>
                    
                    <!-- Tabs -->
                    <div class="flex border-b border-neutral-700">
                        <button id="localSettingsTab" 
                                class="px-6 py-3 text-indigo-400 border-b-2 border-indigo-500 font-medium"
                                onclick="switchToTab('local')">
                            AI Integration
                        </button>
                        <button id="connectionSettingsTab" 
                                class="px-6 py-3 text-neutral-400 hover:text-neutral-200 font-medium"
                                onclick="switchToTab('connection')">
                            Connections
                        </button>
                    </div>
                    
                    <script>
                        // Update the existing global switchToTab function with a better implementation
                        (function() {
                            // Store the original function as fallback
                            const originalSwitchToTab = window.switchToTab;
                            
                            // Replace with our improved version
                            window.switchToTab = function(tab) {
                                console.log('Enhanced switchToTab from settings.js called for', tab);
                                const localTab = document.getElementById('localSettingsTab');
                                const connectionTab = document.getElementById('connectionSettingsTab');
                                const localPanel = document.getElementById('localSettingsPanel');
                                const connectionPanel = document.getElementById('connectionSettingsPanel');
                                
                                console.log('Tab elements found:', {
                                    localTab: !!localTab, 
                                    connectionTab: !!connectionTab, 
                                    localPanel: !!localPanel,
                                    connectionPanel: !!connectionPanel
                                });
                                
                                if (!localTab || !connectionTab || !localPanel || !connectionPanel) {
                                    console.error('Tab elements not found in settings.js implementation');
                                    if (originalSwitchToTab) {
                                        return originalSwitchToTab(tab);
                                    }
                                    return;
                                }
                                
                                if (tab === 'local') {
                                    // Update styles
                                    localTab.classList.add('text-indigo-400', 'border-b-2', 'border-indigo-500');
                                    localTab.classList.remove('text-neutral-400');
                                    connectionTab.classList.remove('text-indigo-400', 'border-b-2', 'border-indigo-500');
                                    connectionTab.classList.add('text-neutral-400');
                                    
                                    // Show/hide panels
                                    localPanel.classList.remove('hidden');
                                    connectionPanel.classList.add('hidden');
                                    console.log('Switched to local tab');
                                } else {
                                    // Update styles
                                    connectionTab.classList.add('text-indigo-400', 'border-b-2', 'border-indigo-500');
                                    connectionTab.classList.remove('text-neutral-400');
                                    localTab.classList.remove('text-indigo-400', 'border-b-2', 'border-indigo-500');
                                    localTab.classList.add('text-neutral-400');
                                    
                                    // Show/hide panels
                                    connectionPanel.classList.remove('hidden');
                                    localPanel.classList.add('hidden');
                                    console.log('Switched to connection tab');
                                }
                            };
                        })();
                    </script>
                </div>
                
                <!-- Modal Content Area with Tab Panels -->
                <div class="overflow-y-auto p-6 flex-1">
                    <!-- Local Settings Tab Panel -->
                    <div id="localSettingsPanel" class="tab-panel">
                        <!-- System Settings Component (includes embedded model widgets) -->
                        <system-settings id="globalSystemSettings"></system-settings>
                        
                        <!-- Hidden original widgets - for backward compatibility -->
                        <div class="hidden">
                            <div id="ollamaWidget">
                                <div id="ollamaStatusIcon"></div>
                                <div id="ollamaStatus"></div>
                                <button id="ollamaRefreshBtn"></button>
                                <select id="ollamaModelSelect">
                                    <option value="loading">Loading models...</option>
                                </select>
                                <div id="ollamaModelsContainer">
                                    <div id="ollamaModelList"></div>
                                </div>
                            </div>
                            
                            <div id="openrouterWidget">
                                <div id="openrouterStatusIcon"></div>
                                <div id="openrouterStatus"></div>
                                <button id="openrouterRefreshBtn"></button>
                                <input id="openrouterTokenInput" type="password">
                                <select id="openrouterModelSelect">
                                    <option value="loading">Loading models...</option>
                                </select>
                                <div id="openrouterModelsContainer">
                                    <div id="openrouterModelList"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Connection Settings Tab Panel -->
                    <div id="connectionSettingsPanel" class="tab-panel hidden">
                        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            <!-- Tunnel Widget -->
                            <div id="tunnelWidget" class="bg-black rounded-xl p-6 shadow-md border border-neutral-700 hover:border-indigo-500 transition-colors">
                                <div class="flex items-center justify-between mb-4">
                                    <div class="flex items-center">
                                        <div class="w-10 h-10 mr-3">
                                            <img src="/static/icons/NGROKLOGO.png" class="w-full h-full" alt="Ngrok Logo" />
                                        </div>
                                        <div>
                                            <h3 class="text-lg font-medium">Tunnel</h3>
                                            <p id="tunnelStatusText" class="text-sm text-neutral-400">Checking status...</p>
                                        </div>
                                    </div>
                                    <div id="tunnelStatusIcon" class="w-4 h-4 rounded-full bg-yellow-500"></div>
                                </div>
                                
                                <!-- Token input field -->
                                <div class="mb-4">
                                    <label class="block text-sm font-medium text-neutral-400 mb-2">Authentication Token</label>
                                    <div class="flex space-x-2">
                                        <input id="ngrokTokenInput" type="text" placeholder="Enter your Ngrok Auth Token" 
                                               class="flex-1 bg-neutral-700 px-3 py-2 text-sm rounded border border-neutral-600 focus:border-indigo-500 focus:outline-none">
                                        <button onclick="TunnelWidget.saveToken()" 
                                                class="bg-indigo-600 hover:bg-indigo-700 px-3 py-2 rounded flex items-center">
                                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                                            </svg>
                                        </button>
                                    </div>
                                    <p class="text-xs text-neutral-400 mt-1"><a href="https://dashboard.ngrok.com/get-started/your-authtoken" target="_blank" class="text-indigo-400 hover:underline">Get your token here</a></p>
                                </div>
                                
                                <!-- Domain display -->
                                <div id="tunnelDomainDisplay" class="mt-3 hidden">
                                    <label class="block text-sm font-medium text-neutral-400 mb-2">Domain</label>
                                    <div class="flex space-x-2">
                                        <input id="widgetTunnelUrl" readonly type="text" 
                                               class="flex-1 bg-neutral-900 px-3 py-2 text-sm rounded border border-neutral-700 focus:outline-none">
                                        <button onclick="TunnelWidget.copyTunnelUrl()" 
                                                class="bg-neutral-700 hover:bg-neutral-600 px-3 py-2 rounded flex items-center">
                                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                                      d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                            </div>

                            <!-- Group Widget -->
                            <div id="groupWidget" class="bg-black rounded-xl p-6 shadow-md border border-neutral-700 hover:border-indigo-500 transition-colors">
                                <div class="flex items-center justify-between">
                                    <div class="flex items-center">
                                        <div class="w-10 h-10 mr-3 relative overflow-hidden">
                                            <img src="/static/icons/GROUPLOGO.png" 
                                                class="w-full h-full object-cover transition-opacity duration-300"
                                                onerror="this.onerror=null; this.src='/static/icons/GROUPLOGO.png';"
                                                style="opacity: 0.7;"
                                                alt="Group" />
                                        </div>
                                        <div>
                                            <h3 class="text-lg font-medium">Connect to Blob Browser</h3>
                                            <p id="groupStatusText" class="text-sm text-neutral-400">Not connected</p>
                                        </div>
                                    </div>
                                    <div id="groupStatusIcon" class="w-4 h-4 rounded-full bg-neutral-500"></div>
                                </div>
                                
                                <!-- Instructions for connecting -->
                                <div class="my-4 p-3 bg-neutral-900 rounded text-sm">
                                    <h4 class="font-medium text-indigo-400 mb-2">How to connect:</h4>
                                    <ol class="list-decimal pl-5 space-y-2 text-neutral-300">
                                        <li>Copy the Domain from your Tunnel 
                                            <button id="copyTunnelForGroupBtn" 
                                                    class="ml-1 text-xs bg-indigo-600 hover:bg-indigo-700 px-1.5 py-0.5 rounded inline-flex items-center">
                                                <svg class="w-3 h-3 mr-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                                                </svg>
                                                Copy
                                            </button>
                                        </li>
                                        <li>Open a group on Blob Browser</li>
                                        <li>Enter the Domain in the group settings</li>
                                    </ol>
                                </div>
                                
                                <div class="mt-4 pt-4 border-t border-neutral-800">
                                    <button id="groupConnectButton" 
                                            class="w-full px-3 py-2 bg-indigo-600 hover:bg-indigo-700 rounded text-sm flex items-center justify-center"
                                            onclick="window.open('https://blob-browser.net/login', '_blank')">
                                        <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                                        </svg>
                                        Open Blob Browser
                                    </button>
                                </div>
                                <!-- Group actions menu will be dynamically added here when needed -->
                            </div>

                            <!-- Slack Widget -->
                            <div id="slackWidget" class="bg-black rounded-xl p-6 shadow-md border border-neutral-700 hover:border-indigo-500 transition-colors">
                                <div class="flex items-center justify-between mb-4">
                                    <div class="flex items-center">
                                        <div class="w-10 h-10 mr-3">
                                            <img src="/static/icons/SLACKLOGO.png" class="w-full h-full" alt="Slack Logo" />
                                        </div>
                                        <div>
                                            <h3 class="text-lg font-medium">Slack Integration</h3>
                                            <div class="flex items-center mt-1">
                                                <div id="slackStatusIcon" class="w-2 h-2 rounded-full bg-yellow-500 mr-2"></div>
                                                <p id="slackStatusText" class="text-sm text-neutral-400">Not connected</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Documentation link -->
                                <div class="mb-4 p-3 bg-neutral-900 rounded text-sm">
                                    <p class="text-neutral-300">
                                        Connect OATFLAKE to your Slack workspace to share insights directly with your team.
                                        <a href="https://blob-browser.net/documentation" 
                                           target="_blank" 
                                           class="text-indigo-400 hover:text-indigo-300 hover:underline flex items-center mt-2">
                                            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                                            </svg>
                                            View Slack Integration Documentation
                                        </a>
                                    </p>
                                </div>
                                
                                <!-- Token input field -->
                                <div class="mb-4">
                                    <label class="block text-sm font-medium text-neutral-400 mb-2">API Token</label>
                                    <input id="slackTokenInput" type="password" placeholder="Enter your Slack API Token" 
                                           class="w-full bg-neutral-700 px-3 py-2 text-sm rounded border border-neutral-600 focus:border-indigo-500 focus:outline-none">
                                </div>

                                <!-- Signing Secret input field -->
                                <div class="mb-4">
                                    <label class="block text-sm font-medium text-neutral-400 mb-2">Signing Secret</label>
                                    <input id="slackSigningSecretInput" type="password" placeholder="Enter your Slack Signing Secret" 
                                           class="w-full bg-neutral-700 px-3 py-2 text-sm rounded border border-neutral-600 focus:border-indigo-500 focus:outline-none">
                                </div>
                                
                                <!-- Bot User ID input field -->
                                <div class="mb-4">
                                    <label class="block text-sm font-medium text-neutral-400 mb-2">Bot User ID</label>
                                    <input id="slackBotUserIdInput" type="text" placeholder="Enter your Slack Bot User ID (e.g. A089BE80EP6)" 
                                           class="w-full bg-neutral-700 px-3 py-2 text-sm rounded border border-neutral-600 focus:border-indigo-500 focus:outline-none">
                                    <p class="text-xs text-neutral-400 mt-1">Your Bot User ID starts with 'U' or 'A' followed by alphanumeric characters (found in your Slack App settings)</p>
                                </div>

                                <button onclick="SlackWidget.saveSlackConfig()" 
                                        class="w-full bg-indigo-600 hover:bg-indigo-700 px-4 py-2 rounded text-white">
                                    Save Configuration
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    /**
     * Render the settings button and modal in the container
     * @param {HTMLElement} container - The container element to render the settings interface in
     */
    function render(container) {
        if (!container) return;
        
        // Insert template into container
        container.innerHTML = template;
        
        // Initialize UI
        initializeUI();
        
        // Initialize widgets
        initializeWidgets();
        
        // Add a safety check for the global function
        if (typeof window.switchToTab !== 'function') {
            console.log('Making switchToTab globally available from render');
            window.switchToTab = function(tab) {
                console.log('Global fallback switchToTab called for', tab);
                const localTab = document.getElementById('localSettingsTab');
                const connectionTab = document.getElementById('connectionSettingsTab');
                const localPanel = document.getElementById('localSettingsPanel');
                const connectionPanel = document.getElementById('connectionSettingsPanel');
                
                if (!localTab || !connectionTab || !localPanel || !connectionPanel) {
                    console.error('Missing required elements for tab switching');
                    return;
                }
                
                if (tab === 'local') {
                    localPanel.classList.remove('hidden');
                    connectionPanel.classList.add('hidden');
                    
                    localTab.classList.add('text-indigo-400', 'border-b-2', 'border-indigo-500');
                    localTab.classList.remove('text-neutral-400');
                    connectionTab.classList.remove('text-indigo-400', 'border-b-2', 'border-indigo-500');
                    connectionTab.classList.add('text-neutral-400');
                } else {
                    connectionPanel.classList.remove('hidden');
                    localPanel.classList.add('hidden');
                    
                    connectionTab.classList.add('text-indigo-400', 'border-b-2', 'border-indigo-500');
                    connectionTab.classList.remove('text-neutral-400');
                    localTab.classList.remove('text-indigo-400', 'border-b-2', 'border-indigo-500');
                    localTab.classList.add('text-neutral-400');
                }
            };
        }
    }
    
    /**
     * Initialize the UI elements (modal, tabs, buttons)
     */
    function initializeUI() {
        // Get elements
        const modal = document.getElementById('settingsModal');
        const openButton = document.getElementById('settingsButton'); // Using the button from the action bar
        const closeButton = document.getElementById('closeSettingsModal');
        const doneButton = document.getElementById('settingsDoneButton');
        
        // Check if there are multiple elements with these IDs
        const localTabs = document.querySelectorAll('#localSettingsTab');
        const connectionTabs = document.querySelectorAll('#connectionSettingsTab');
        const localPanels = document.querySelectorAll('#localSettingsPanel');
        const connectionPanels = document.querySelectorAll('#connectionSettingsPanel');
        
        console.log('Multiple elements check:', {
            localTabs: localTabs.length,
            connectionTabs: connectionTabs.length,
            localPanels: localPanels.length, 
            connectionPanels: connectionPanels.length
        });
        
        const localTab = document.getElementById('localSettingsTab');
        const connectionTab = document.getElementById('connectionSettingsTab');
        const localPanel = document.getElementById('localSettingsPanel');
        const connectionPanel = document.getElementById('connectionSettingsPanel');
        
        // Open modal 
        openButton.addEventListener('click', () => {
            modal.classList.remove('hidden');
            // Apply animation
            modal.style.opacity = '0';
            setTimeout(() => {
                modal.style.opacity = '1';
                modal.style.transition = 'opacity 0.2s ease-in-out';
            }, 10);
        });
        
        // Close modal functions
        const closeModal = () => {
            modal.style.opacity = '0';
            setTimeout(() => {
                modal.classList.add('hidden');
                
                // Dispatch event that modal has been closed
                document.dispatchEvent(new CustomEvent('settingsModalClosed'));
            }, 200);
        };
        
        closeButton.addEventListener('click', closeModal);
        doneButton.addEventListener('click', closeModal);
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeModal();
            }
        });
        
        console.log('Setting up tab switching listeners');
        console.log('Tabs found:', {
            localTab: !!localTab, 
            connectionTab: !!connectionTab, 
            localPanel: !!localPanel,
            connectionPanel: !!connectionPanel
        });
        
        // Set up additional tab switching for direct clicks (as a backup to the inline handlers)
        localTab.addEventListener('click', (e) => {
            // Don't interfere with the inline onclick handler
            if (e.handled) return;
            e.handled = true;
            
            console.log('Local tab clicked through event listener');
            // Use the global function for consistency
            if (window.switchToTab) {
                window.switchToTab('local');
            } else {
                // Fallback if global function is not available
                localPanel.classList.remove('hidden');
                connectionPanel.classList.add('hidden');
                
                // Update styles
                localTab.classList.add('text-indigo-400', 'border-b-2', 'border-indigo-500');
                localTab.classList.remove('text-neutral-400');
                connectionTab.classList.remove('text-indigo-400', 'border-b-2', 'border-indigo-500');
                connectionTab.classList.add('text-neutral-400');
            }
        });
        
        connectionTab.addEventListener('click', (e) => {
            // Don't interfere with the inline onclick handler
            if (e.handled) return;
            e.handled = true;
            
            console.log('Connection tab clicked through event listener');
            // Use the global function for consistency
            if (window.switchToTab) {
                window.switchToTab('connection');
            } else {
                // Fallback if global function is not available
                connectionPanel.classList.remove('hidden');
                localPanel.classList.add('hidden');
                
                // Update styles
                connectionTab.classList.add('text-indigo-400', 'border-b-2', 'border-indigo-500');
                connectionTab.classList.remove('text-neutral-400');
                localTab.classList.remove('text-indigo-400', 'border-b-2', 'border-indigo-500');
                localTab.classList.add('text-neutral-400');
            }
        });
    }
    
    /**
     * Initialize all the widgets
     */
    function initializeWidgets() {
        // We still initialize the original widget code for backward compatibility
        // (though the UI elements are now hidden)
        if (typeof OllamaWidget !== 'undefined' && OllamaWidget.initialize) {
            OllamaWidget.initialize();
        }
        
        if (typeof OpenRouterWidget !== 'undefined' && OpenRouterWidget.initialize) {
            OpenRouterWidget.initialize();
        }
        
        // Initialize the System Settings component which has embedded widgets
        const systemSettings = document.getElementById('globalSystemSettings');
        if (systemSettings) {
            console.log('System Settings component found and initialized');
            
            // Set up our custom event listener to handle provider changes
            document.addEventListener('embeddedModelChanged', (event) => {
                if (!event.detail) return;
                
                const { provider, modelName } = event.detail;
                
                // Update the hidden original widgets for backwards compatibility
                // This is needed in case any other code depends on these
                if (provider === 'ollama') {
                    const mainSelect = document.getElementById('ollamaModelSelect');
                    if (mainSelect && modelName && mainSelect.value !== modelName) {
                        // Find and select the option
                        for (let i = 0; i < mainSelect.options.length; i++) {
                            if (mainSelect.options[i].value === modelName) {
                                mainSelect.selectedIndex = i;
                                // Dispatch change event
                                mainSelect.dispatchEvent(new Event('change'));
                                break;
                            }
                        }
                    }
                } else if (provider === 'openrouter') {
                    const mainSelect = document.getElementById('openrouterModelSelect');
                    if (mainSelect && modelName && mainSelect.value !== modelName) {
                        // Find and select the option
                        for (let i = 0; i < mainSelect.options.length; i++) {
                            if (mainSelect.options[i].value === modelName) {
                                mainSelect.selectedIndex = i;
                                // Dispatch change event
                                mainSelect.dispatchEvent(new Event('change'));
                                break;
                            }
                        }
                    }
                }
            });
            
            // Make sure models are properly displayed after loading
            setTimeout(() => {
                if (typeof systemSettings.updateSelectedModelDisplay === 'function') {
                    systemSettings.updateSelectedModelDisplay();
                }
            }, 1500);
        }
    }

    // Return public API
    return {
        render,
        initializeWidgets
    };
})();

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    const settingsContainer = document.getElementById('settingsContainer');
    if (settingsContainer) {
        SettingsSlide.render(settingsContainer);
        
        // Initialize widgets after a short delay to ensure the DOM is ready
        setTimeout(() => {
            SettingsSlide.initializeWidgets();
        }, 100);
        
        // Additional safety check for tab switching
        setTimeout(() => {
            console.log('Adding direct tab switching handler');
            const connectionTab = document.getElementById('connectionSettingsTab');
            const localTab = document.getElementById('localSettingsTab');
            const connectionPanel = document.getElementById('connectionSettingsPanel');
            const localPanel = document.getElementById('localSettingsPanel');
            
            if (connectionTab && localTab && connectionPanel && localPanel) {
                console.log('Found all required tab elements');
                
                // Ensure proper tab switching with direct event handler
                connectionTab.onclick = function() {
                    console.log('Direct connection tab click handler fired');
                    // Update styles
                    connectionTab.classList.add('text-indigo-400', 'border-b-2', 'border-indigo-500');
                    connectionTab.classList.remove('text-neutral-400');
                    localTab.classList.remove('text-indigo-400', 'border-b-2', 'border-indigo-500');
                    localTab.classList.add('text-neutral-400');
                    
                    // Show/hide panels
                    connectionPanel.classList.remove('hidden');
                    localPanel.classList.add('hidden');
                    
                    console.log('Connection tab panel hidden?', connectionPanel.classList.contains('hidden'));
                    console.log('Local tab panel hidden?', localPanel.classList.contains('hidden'));
                };
            } else {
                console.error('Could not find all required tab elements');
            }
        }, 500);
    }
});