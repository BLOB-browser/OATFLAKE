/**
 * DataSlide - Handles the generation of the data interface
 */
const DataSlide = (() => {
    // HTML template for the data interface
    const template = `
        <!-- Data Management Section -->
        <div class="mb-10 p-6">
            <h2 class="text-2xl font-semibold mb-6">Data Management</h2>
            
            <!-- Data Storage Widget -->
            <div class="mb-8">
                <div id="storageWidget" class="bg-black rounded-xl p-6 shadow-md border border-neutral-700 hover:border-indigo-500 transition-colors mb-6">
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center">
                            <div class="w-10 h-10 mr-3 flex items-center justify-center">
                                <svg xmlns="http://www.w3.org/2000/svg" class="w-8 h-8 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                                </svg>
                            </div>
                            <div>
                                <h3 class="text-lg font-medium">Data Storage</h3>
                                <p id="storageStatusText" class="text-sm text-neutral-400">Select a folder</p>
                            </div>
                        </div>
                        <div id="storageStatusIcon" class="w-4 h-4 rounded-full bg-yellow-500"></div>
                    </div>
                    
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-neutral-400 mb-2">Storage Location</label>
                        <div class="flex space-x-2">
                            <input id="dataPath" type="text" readonly 
                                  class="flex-1 bg-neutral-700 px-3 py-2 text-sm rounded border border-neutral-600 focus:outline-none">
                            <input type="file" id="folderPicker" webkitdirectory directory 
                                  class="hidden">
                            <button onclick="StorageWidget.selectFolder()" 
                                   class="bg-indigo-600 hover:bg-indigo-700 px-3 py-2 rounded flex items-center">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                          d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"></path>
                                </svg>
                            </button>
                        </div>
                    </div>
                    
                    <div class="text-xs text-neutral-400">
                        <p>Select a folder where data will be stored locally</p>
                        <p class="mt-1" id="storageSpaceInfo"></p>
                    </div>
                </div>
            </div>
            
            <!-- Task Management Section -->
            <h2 class="text-2xl font-semibold my-6">Task Management</h2>
            <div class="bg-neutral-900 rounded-3xl p-6 border border-neutral-800 shadow-lg">
                <div class="flex items-center mb-4">
                    <svg class="w-6 h-6 mr-2 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                    <h3 class="text-xl font-medium">Task Scheduler</h3>
                </div>
                
                <div id="taskWidget" class="bg-black rounded-xl p-6 shadow-md border border-neutral-700 hover:border-indigo-500 transition-colors">
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center">
                            <div class="w-10 h-10 mr-3 flex items-center justify-center">
                                <svg xmlns="http://www.w3.org/2000/svg" class="w-8 h-8 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                                </svg>
                            </div>
                            <div>
                                <h3 class="text-lg font-medium">Knowledge Processing</h3>
                                <div class="flex items-center mt-1">
                                    <div id="taskStatusIcon" class="w-2 h-2 rounded-full bg-yellow-500 mr-2"></div>
                                    <p id="taskStatusText" class="text-sm text-neutral-400">Loading tasks...</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Task list container -->
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-neutral-400 mb-2">Scheduled Tasks</label>
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
            </div>
        </div>
    `;

    /**
     * Render the data interface in the container
     * @param {HTMLElement} container - The container element to render the data interface in
     */
    function render(container) {
        if (!container) return;
        
        // Insert template into container
        container.innerHTML = template;
        
        // Initialize functionality after rendering
        initialize(container);
    }
    
    /**
     * Initialize the data functionality
     * @param {HTMLElement} container - The container with the data interface
     */
    function initialize(container) {
        // Data source selection
        const dataSourceSelect = container.querySelector('#dataSourceSelect');
        const urlInputContainer = container.querySelector('#urlInputContainer');
        const fileInputContainer = container.querySelector('#fileInputContainer');
        const textInputContainer = container.querySelector('#textInputContainer');
        
        if (dataSourceSelect) {
            dataSourceSelect.addEventListener('change', function() {
                // Hide all input containers
                urlInputContainer.classList.add('hidden');
                fileInputContainer.classList.add('hidden');
                textInputContainer.classList.add('hidden');
                
                // Show the selected input container
                const selectedValue = this.value;
                if (selectedValue === 'url') {
                    urlInputContainer.classList.remove('hidden');
                } else if (selectedValue === 'file') {
                    fileInputContainer.classList.remove('hidden');
                } else if (selectedValue === 'text') {
                    textInputContainer.classList.remove('hidden');
                }
            });
        }
        
        // File input handling
        const fileInput = container.querySelector('#fileInput');
        const selectedFileName = container.querySelector('#selectedFileName');
        
        if (fileInput && selectedFileName) {
            fileInput.addEventListener('change', function() {
                if (this.files.length > 0) {
                    selectedFileName.textContent = this.files[0].name;
                } else {
                    selectedFileName.textContent = 'No file selected';
                }
            });
        }
        
        // Process button handling
        const processDataButton = container.querySelector('#processDataButton');
        
        if (processDataButton) {
            processDataButton.addEventListener('click', function() {
                // This would handle processing based on the selected data source
                // For now, just show a basic progress indication
                const dataProcessingStatus = container.querySelector('#dataProcessingStatus');
                const processingProgressBar = container.querySelector('#processingProgressBar');
                const processingProgressText = container.querySelector('#processingProgressText');
                const processingStatusText = container.querySelector('#processingStatusText');
                
                // Show processing status
                dataProcessingStatus.classList.remove('hidden');
                
                // Simulate progress with a simple animation
                let progress = 0;
                const interval = setInterval(() => {
                    progress += 5;
                    if (progress > 100) {
                        clearInterval(interval);
                        processingStatusText.textContent = 'Processing complete!';
                        return;
                    }
                    
                    processingProgressBar.style.width = `${progress}%`;
                    processingProgressText.textContent = `${progress}%`;
                    
                    if (progress < 30) {
                        processingStatusText.textContent = 'Downloading content...';
                    } else if (progress < 60) {
                        processingStatusText.textContent = 'Extracting information...';
                    } else if (progress < 90) {
                        processingStatusText.textContent = 'Generating embeddings...';
                    } else {
                        processingStatusText.textContent = 'Finalizing...';
                    }
                }, 200);
            });
        }
        
        // Refresh data button
        const refreshDataButton = container.querySelector('#refreshDataButton');
        
        if (refreshDataButton) {
            refreshDataButton.addEventListener('click', function() {
                // This would fetch and display existing data
                const dataBrowser = container.querySelector('#dataBrowser');
                
                // Simulate loading state
                dataBrowser.innerHTML = `
                    <div class="flex items-center justify-center h-full">
                        <div class="animate-spin h-5 w-5 border-2 border-indigo-500 rounded-full border-t-transparent"></div>
                        <span class="ml-2 text-neutral-400">Loading data...</span>
                    </div>
                `;
                
                // Simulate data loading with a timeout
                setTimeout(() => {
                    // Example data items
                    const dataItems = [
                        { type: 'web', name: 'example.com/page1', date: '2023-04-01' },
                        { type: 'file', name: 'document.pdf', date: '2023-04-02' },
                        { type: 'text', name: 'Manual Entry 1', date: '2023-04-03' }
                    ];
                    
                    if (dataItems.length === 0) {
                        dataBrowser.innerHTML = `<p class="text-neutral-400 text-sm">No data available</p>`;
                        return;
                    }
                    
                    // Render data items
                    let html = '<ul class="space-y-2">';
                    dataItems.forEach(item => {
                        let icon = '';
                        if (item.type === 'web') {
                            icon = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"></path></svg>';
                        } else if (item.type === 'file') {
                            icon = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>';
                        } else {
                            icon = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path></svg>';
                        }
                        
                        html += `
                            <li class="flex items-center justify-between p-2 hover:bg-neutral-800 rounded">
                                <div class="flex items-center">
                                    <span class="mr-2 text-indigo-400">${icon}</span>
                                    <span class="text-sm text-white">${item.name}</span>
                                </div>
                                <div class="flex items-center">
                                    <span class="text-xs text-neutral-400 mr-2">${item.date}</span>
                                    <input type="checkbox" class="data-item-checkbox w-4 h-4 bg-neutral-700 border-neutral-600 rounded focus:ring-indigo-500">
                                </div>
                            </li>
                        `;
                    });
                    html += '</ul>';
                    
                    dataBrowser.innerHTML = html;
                    
                    // Add checkbox handlers
                    const checkboxes = container.querySelectorAll('.data-item-checkbox');
                    const deleteDataButton = container.querySelector('#deleteDataButton');
                    
                    checkboxes.forEach(checkbox => {
                        checkbox.addEventListener('change', function() {
                            const anyChecked = Array.from(checkboxes).some(cb => cb.checked);
                            deleteDataButton.disabled = !anyChecked;
                        });
                    });
                }, 1000);
            });
        }
    }

    // Return public API
    return {
        render
    };
})();

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    const dataContainer = document.getElementById('dataContainer');
    if (dataContainer) {
        DataSlide.render(dataContainer);
    }
});