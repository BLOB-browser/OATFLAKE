/**
 * SettingsSlide - Handles the generation of the settings interface containing all widgets
 */
const SettingsSlide = (() => {
    // HTML template for the settings interface with widgets
    const template = `
        <!-- Service Widgets Section -->
        <div class="mb-10 p-2">
            <h2 class="text-2xl font-semibold mb-6">Services</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <!-- Ollama Widget -->
                <div id="ollamaWidget" class="bg-black rounded-3xl p-6 shadow-lg border border-neutral-700 hover:border-indigo-500 transition-colors">
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center">
                            <div class="w-12 h-12 mr-4">
                                <img src="/static/icons/OLLAMALOGO.png" class="w-full h-full" alt="Ollama Logo" />
                            </div>
                            <div>
                                <h3 class="text-lg font-medium">Ollama</h3>
                                <div class="flex items-center mt-1">
                                    <div id="ollamaStatusIcon" class="w-2 h-2 rounded-full bg-yellow-500 mr-2"></div>
                                    <p id="ollamaStatus" class="text-sm text-neutral-400">Checking status...</p>
                                </div>
                            </div>
                        </div>
                        <!-- Fix refresh button position -->
                        <button id="ollamaRefreshBtn" class="text-neutral-400 hover:text-white transition-colors">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                            </svg>
                        </button>
                    </div>
                    
                    <!-- Fix model selector display -->
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-neutral-400 mb-2">Available Models</label>
                        <select id="ollamaModelSelect" class="w-full bg-neutral-700 text-white p-2 rounded border border-neutral-600 focus:border-indigo-500 focus:outline-none">
                            <option value="loading">Loading models...</option>
                        </select>
                    </div>
                    
                    <div id="ollamaModelsContainer" class="mt-4">
                        <div id="ollamaModelList" class="text-sm text-neutral-400">
                            <p>No models found</p>
                        </div>
                    </div>
                </div>

                <!-- OpenRouter Widget -->
                <div id="openrouterWidget" class="bg-black rounded-3xl p-6 shadow-lg border border-neutral-700 hover:border-indigo-500 transition-colors">
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center">
                            <div class="w-12 h-12 mr-4">
                                <img src="/static/icons/OPENROUTERLOGO.png" class="w-full h-full" alt="OpenRouter Logo" 
                                     onerror="this.src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBhdGggZD0iTTEyIDJMMiA3bDEwIDVNMTIgMmwxMCA1LTEwIDVNMiAxN2wxMCA1IDEwLTUiLz48L3N2Zz4='" />
                            </div>
                            <div>
                                <h3 class="text-lg font-medium">OpenRouter</h3>
                                <div class="flex items-center mt-1">
                                    <div id="openrouterStatusIcon" class="w-2 h-2 rounded-full bg-yellow-500 mr-2"></div>
                                    <p id="openrouterStatus" class="text-sm text-neutral-400">Checking status...</p>
                                </div>
                            </div>
                        </div>
                        <button id="openrouterRefreshBtn" class="text-neutral-400 hover:text-white transition-colors">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                            </svg>
                        </button>
                    </div>
                    
                    <!-- API Token input field -->
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-neutral-400 mb-2">API Token</label>
                        <div class="flex space-x-2">
                            <input id="openrouterTokenInput" type="password" placeholder="Enter your OpenRouter API Token" 
                                   class="flex-1 bg-neutral-700 px-3 py-2 text-sm rounded border border-neutral-600 focus:border-indigo-500 focus:outline-none">
                            <button onclick="OpenRouterWidget.saveToken()" 
                                    class="bg-indigo-600 hover:bg-indigo-700 px-3 py-2 rounded flex items-center">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                                </svg>
                            </button>
                        </div>
                        <p class="text-xs text-neutral-400 mt-1"><a href="https://openrouter.ai/keys" target="_blank" class="text-indigo-400 hover:underline">Get your API key here</a></p>
                    </div>
                    
                    <!-- Model selection -->
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-neutral-400 mb-2">Available Models</label>
                        <select id="openrouterModelSelect" class="w-full bg-neutral-700 text-white p-2 rounded border border-neutral-600 focus:border-indigo-500 focus:outline-none">
                            <option value="loading">Loading models...</option>
                        </select>
                    </div>
                    
                    <div id="openrouterModelsContainer" class="mt-4">
                        <div id="openrouterModelList" class="text-sm text-neutral-400">
                            <p>No models found</p>
                        </div>
                    </div>
                </div>

                <!-- Tunnel Widget -->
                <div id="tunnelWidget" class="bg-black rounded-3xl p-6 shadow-lg border border-neutral-700 hover:border-indigo-500 transition-colors">
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center">
                            <div class="w-12 h-12 mr-4">
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
                <div id="groupWidget" class="bg-black rounded-3xl p-6 shadow-lg border border-neutral-700">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center">
                            <div class="w-12 h-12 mr-4 relative overflow-hidden">
                                <img src="/static/icons/GROUPLOGO.png" 
                                    class="w-full h-full object-cover transition-opacity duration-300"
                                    onerror="this.onerror=null; this.src='/static/icons/GROUPLOGO.png';"
                                    style="opacity: 0.7;"
                                    alt="Group" />
                            </div>
                            <div>
                                <h3 class="text-lg font-medium">Group</h3>
                                <p id="groupStatusText" class="text-sm text-neutral-400">Not connected</p>
                            </div>
                        </div>
                        <div id="groupStatusIcon" class="w-4 h-4 rounded-full bg-neutral-500"></div>
                    </div>
                    <!-- Group actions menu will be dynamically added here when needed -->
                </div>
                
                <!-- Data Storage Widget moved to data.js slide -->

                <!-- Task Management Widget -->
                <div id="taskWidget" class="bg-black rounded-3xl p-6 shadow-lg border border-neutral-700 hover:border-indigo-500 transition-colors">
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center">
                            <div class="w-12 h-12 mr-4 flex items-center justify-center">
                                <svg xmlns="http://www.w3.org/2000/svg" class="w-10 h-10 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                                </svg>
                            </div>
                            <div>
                                <h3 class="text-lg font-medium">Task Management</h3>
                                <div class="flex items-center mt-1">
                                    <div id="taskStatusIcon" class="w-2 h-2 rounded-full bg-yellow-500 mr-2"></div>
                                    <p id="taskStatusText" class="text-sm text-neutral-400">Loading tasks...</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Task list container -->
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-neutral-400 mb-2">Available Tasks</label>
                        <div id="tasksList" class="bg-neutral-900 rounded border border-neutral-700 max-h-40 overflow-y-auto">
                            <div class="text-neutral-400 text-sm p-2">Loading tasks...</div>
                        </div>
                    </div>
                    
                    <!-- Task actions -->
                    <div class="flex space-x-2">
                        <button id="createTaskButton" 
                                class="flex-1 px-3 py-2 bg-indigo-600 hover:bg-indigo-700 rounded text-sm flex items-center justify-center">
                            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
                            </svg>
                            Create Task
                        </button>
                        <button id="editTaskButton" disabled
                                class="flex-1 px-3 py-2 bg-neutral-700 hover:bg-neutral-600 disabled:opacity-50 rounded text-sm">
                            Edit
                        </button>
                        <button id="runTaskButton" disabled
                                class="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 disabled:opacity-50 rounded text-sm">
                            Run
                        </button>
                    </div>
                </div>

                <!-- Slack Widget -->
                <div id="slackWidget" class="bg-black rounded-3xl p-6 shadow-lg border border-neutral-700 hover:border-indigo-500 transition-colors">
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center">
                            <div class="w-12 h-12 mr-4">
                                <img src="/static/icons/SLACKLOGO.png" class="w-full h-full" alt="Slack Logo" />
                            </div>
                            <div>
                                <h3 class="text-lg font-medium">Slack</h3>
                                <div class="flex items-center mt-1">
                                    <div id="slackStatusIcon" class="w-2 h-2 rounded-full bg-yellow-500 mr-2"></div>
                                    <p id="slackStatusText" class="text-sm text-neutral-400">Not connected</p>
                                </div>
                            </div>
                        </div>
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
                        <input id="slackBotUserIdInput" type="text" placeholder="Enter your Slack Bot User ID" 
                               class="w-full bg-neutral-700 px-3 py-2 text-sm rounded border border-neutral-600 focus:border-indigo-500 focus:outline-none">
                    </div>

                    <button onclick="SlackWidget.saveSlackConfig()" 
                            class="bg-indigo-600 hover:bg-indigo-700 px-4 py-2 rounded text-white">
                        Save Configuration
                    </button>
                </div>
            </div>
        </div>
    `;

    /**
     * Render the settings interface in the container
     * @param {HTMLElement} container - The container element to render the settings interface in
     */
    function render(container) {
        if (!container) return;
        
        // Insert template into container
        container.innerHTML = template;
        
        // Initialize widgets after rendering
        initializeWidgets();
    }
    
    /**
     * Initialize all the widgets
     */
    function initializeWidgets() {
        // Each widget has its own initialize function that will be called from their respective files
        // The logic is kept in the individual widget files
    }

    // Return public API
    return {
        render
    };
})();

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    const settingsContainer = document.getElementById('settingsContainer');
    if (settingsContainer) {
        SettingsSlide.render(settingsContainer);
    }
});