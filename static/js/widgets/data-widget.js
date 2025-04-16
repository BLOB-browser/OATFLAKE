// filepath: /Users/mars/Documents/GitHub/blob/BLOB/OATMEAL/OATFLAKE/static/js/widgets/data-widget.js
/**
 * Data Widget Module
 * Handles all data storage-related functionality
 */

const StorageWidget = (() => {
    // Cache DOM elements
    const elements = {
        statusText: () => document.getElementById('storageStatusText'),
        statusIcon: () => document.getElementById('storageStatusIcon'),
        dataPath: () => document.getElementById('dataPath'),
        folderPicker: () => document.getElementById('folderPicker'),
        spaceInfo: () => document.getElementById('storageSpaceInfo')
    };

    /**
     * Initialize the Storage widget
     */
    function initialize() {
        // Set up handlers for the folder selection
        const folderPickerButton = document.querySelector('button[onclick="selectFolder()"]');
        if (folderPickerButton) {
            folderPickerButton.onclick = selectFolder;
        }
        
        // Replace inline handler with our module function
        const folderInput = elements.folderPicker();
        if (folderInput) {
            folderInput.onchange = handleFolderSelect;
        }
        
        // Load the stored path from localStorage or config
        loadStoredPath();
    }

    /**
     * Load the stored data path from config or localStorage
     */
    async function loadStoredPath() {
        try {
            // First try to get from API
            const response = await fetch('/api/status');
            if (response.ok) {
                const data = await response.json();
                if (data.data_path) {
                    updateDataPath(data.data_path);
                    updateStatus(true, data.data_path);
                    return;
                }
            }
            
            // Fallback to localStorage
            const savedPath = localStorage.getItem('dataPath');
            if (savedPath) {
                updateDataPath(savedPath);
                // Verify this path with the server
                saveDataPath(savedPath);
            } else {
                updateStatus(false);
            }
        } catch (error) {
            console.error('Error loading stored path:', error);
            updateStatus(false);
        }
    }

    /**
     * Open folder picker dialog
     */
    function selectFolder() {
        const folderPicker = elements.folderPicker();
        if (folderPicker) {
            folderPicker.click();
        }
    }

    /**
     * Handle folder selection from the file input
     * @param {Event} event - The change event
     */
    function handleFolderSelect(event) {
        const files = event.target.files;
        if (!files || files.length === 0) return;
        
        // Get the selected folder path
        // For web security, we can only get the name, not the full path
        // In native apps, like Electron, we might get more info
        
        // For now, let's just use the first file's path as a reference
        const firstFile = files[0];
        
        // Extract folder path (this needs to be handled differently in production)
        // In a real electron app, you'd likely have access to the actual folder path
        let path = firstFile.webkitRelativePath;
        if (path) {
            // Extract just the first directory
            path = path.split('/')[0];
            // Save this path
            saveDataPath(path);
        }
        
        // In a real app, you'd want to get the actual folder path using Electron or backend APIs
    }

    /**
     * Save the data path to the server
     * @param {string} path - The data storage path
     */
    async function saveDataPath(path) {
        if (!path) return;
        
        try {
            const response = await fetch('/api/data/storage/set', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ path })
            });
            
            if (!response.ok) {
                throw new Error('Failed to save data path');
            }
            
            const data = await response.json();
            console.log('Data path saved:', data);
            
            // Update UI
            updateDataPath(path);
            updateStatus(true, path);
            
            // Save to localStorage as a backup
            localStorage.setItem('dataPath', path);
            
            return data;
        } catch (error) {
            console.error('Error saving data path:', error);
            updateStatus(false);
            alert('Failed to save data path: ' + error.message);
        }
    }

    /**
     * Update the data path in the UI
     * @param {string} path - The data storage path
     */
    function updateDataPath(path) {
        const dataPathInput = elements.dataPath();
        if (dataPathInput && path) {
            dataPathInput.value = path;
        }
    }

    /**
     * Fetch storage space information
     * @param {string} path - The path to check
     * @returns {Promise<Object>} - Storage space info
     */
    async function fetchStorageInfo(path) {
        try {
            const response = await fetch(`/api/storage/space?path=${encodeURIComponent(path)}`);
            if (!response.ok) {
                throw new Error('Failed to fetch storage info');
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching storage space info:', error);
            return null;
        }
    }

    /**
     * Update the storage status display
     * @param {boolean} isConnected - Whether storage is configured
     * @param {string} path - The configured data path
     */
    async function updateStatus(isConnected, path = null) {
        const statusText = elements.statusText();
        const statusIcon = elements.statusIcon();
        const spaceInfo = elements.spaceInfo();
        
        if (!statusText || !statusIcon) return;
        
        if (isConnected && path) {
            statusText.textContent = 'Connected';
            statusText.classList.remove('text-neutral-400', 'text-red-500');
            statusText.classList.add('text-green-500');
            
            statusIcon.classList.remove('bg-yellow-500', 'bg-red-500');
            statusIcon.classList.add('bg-green-500');
            
            // Show storage info if available
            if (spaceInfo) {
                try {
                    // Fetch actual disk space information from the server
                    const storageData = await fetchStorageInfo(path);
                    if (storageData && storageData.status === 'success') {
                        spaceInfo.textContent = `Free: ${storageData.free_formatted} of ${storageData.total_formatted} (${storageData.usage_percent}% used)`;
                    } else {
                        spaceInfo.textContent = `Storage location: ${path}`;
                    }
                } catch (error) {
                    console.error('Error updating storage info:', error);
                    spaceInfo.textContent = `Storage location: ${path}`;
                }
            }
        } else {
            statusText.textContent = 'Select a folder';
            statusText.classList.remove('text-green-500', 'text-red-500');
            statusText.classList.add('text-neutral-400');
            
            statusIcon.classList.remove('bg-green-500', 'bg-red-500');
            statusIcon.classList.add('bg-yellow-500');
            
            if (spaceInfo) {
                spaceInfo.textContent = '';
            }
        }
    }

    // Public API
    return {
        initialize,
        updateStatus,
        saveDataPath,
        selectFolder,
        handleFolderSelect
    };
})();

// Initialize the widget when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    StorageWidget.initialize();
});
