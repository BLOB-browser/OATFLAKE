console.log('Loading app.js...');

// Global variables
let authToken = null;

// Function to open the connection iframe via the settings modal
function showConnectionModal() {
    console.log('Opening settings modal with connection tab');
    
    // Get the settings modal and connection tab
    const settingsModal = document.getElementById('settingsModal');
    const connectionTab = document.getElementById('connectionSettingsTab');
    
    // If we have both elements, show the modal and switch to connection tab
    if (settingsModal && connectionTab) {
        // Show the settings modal
        settingsModal.classList.remove('hidden');
        
        // Apply animation
        settingsModal.style.opacity = '0';
        setTimeout(() => {
            settingsModal.style.opacity = '1';
            settingsModal.style.transition = 'opacity 0.2s ease-in-out';
        }, 10);
        
        // Switch to connection tab using the global function
        if (window.switchToTab) {
            console.log('Using global switchToTab function');
            window.switchToTab('connection');
        } else {
            console.log('Fallback to direct click');
            // Click the connection tab to switch to it
            connectionTab.click();
        }
        
        return;
    }
    
    // Fall back to original iframe method if settings modal isn't available
    console.log('Settings modal not found, falling back to original connection modal');
    
    const modal = document.getElementById('loginSection');
    const iframe = document.getElementById('connectIframe');
    const iframeMessage = document.getElementById('iframeMessage');
    const directConnectForm = document.getElementById('directConnectForm');
    const modalNgrokUrl = document.getElementById('modalNgrokUrl');
    
    // Show the modal
    if (modal) modal.classList.remove('hidden');
    
    // Update the ngrok URL display
    if (modalNgrokUrl) {
        const ngrokUrl = document.getElementById('domainInput')?.value || 'Not available';
        modalNgrokUrl.textContent = ngrokUrl;
    }
    
    // Show loading message, hide iframe and direct form
    if (iframeMessage) iframeMessage.classList.remove('hidden');
    if (iframe) iframe.classList.add('hidden');
    if (directConnectForm) directConnectForm.classList.add('hidden');
    
    // Set the iframe source to the connection page on the frontend
    const FRONTEND_URL = 'https://blob.oatflake.ai'; // Update this to your actual frontend URL
    const connectionPage = `${FRONTEND_URL}/connect`;
    
    // Try to load the iframe
    if (iframe) {
        iframe.src = connectionPage;
        
        // Set up iframe load event
        iframe.onload = function() {
            console.log('Iframe loaded successfully');
            iframeMessage.classList.add('hidden');
            iframe.classList.remove('hidden');
            
            // Try to pass the ngrok URL to the iframe
            try {
                const ngrokUrl = document.getElementById('domainInput')?.value;
                if (ngrokUrl) {
                    setTimeout(() => {
                        iframe.contentWindow.postMessage({ 
                            type: 'OATFLAKE_CONNECT',
                            ngrokUrl: ngrokUrl
                        }, FRONTEND_URL);
                        console.log('Sent ngrok URL to iframe:', ngrokUrl);
                    }, 1000); // Give the iframe a moment to initialize
                }
            } catch (e) {
                console.error('Failed to communicate with iframe:', e);
            }
        };
        
        // Set up iframe error event
        iframe.onerror = function() {
            console.error('Failed to load the connection iframe');
            showDirectConnectForm();
        };
        
        // Fallback if iframe doesn't load within 5 seconds
        setTimeout(() => {
            if (iframeMessage.classList.contains('hidden') === false) {
                console.warn('Iframe taking too long to load, showing direct form');
                showDirectConnectForm();
            }
        }, 5000);
    } else {
        console.error('Iframe element not found');
        showDirectConnectForm();
    }
    
    // Listen for messages from the iframe
    window.addEventListener('message', receiveConnectMessage);
}

// Function to show direct connect form if iframe fails
function showDirectConnectForm() {
    const iframeMessage = document.getElementById('iframeMessage');
    const iframe = document.getElementById('connectIframe');
    const directConnectForm = document.getElementById('directConnectForm');
    
    if (iframeMessage) iframeMessage.classList.add('hidden');
    if (iframe) iframe.classList.add('hidden');
    if (directConnectForm) directConnectForm.classList.remove('hidden');
}

// Function to handle direct connection
async function handleDirectConnect() {
    const groupId = document.getElementById('directGroupId')?.value;
    const groupName = document.getElementById('directGroupName')?.value;
    
    if (!groupId) {
        alert('Please enter a Group ID');
        return;
    }
    
    try {
        await connectToGroup(groupId, {
            name: groupName || `Group ${groupId.substring(0, 6)}`,
            description: 'Direct connection'
        });
    } catch (error) {
        console.error('Direct connection error:', error);
        alert(`Connection failed: ${error.message}`);
    }
}

// Function to receive messages from the iframe
function receiveConnectMessage(event) {
    // Verify origin for security
    const FRONTEND_URL = 'https://blob.oatflake.ai'; // Must match the iframe source
    if (event.origin !== FRONTEND_URL) {
        console.warn('Received message from untrusted origin:', event.origin);
        return;
    }
    
    console.log('Received message from iframe:', event.data);
    
    // Process connection data
    if (event.data.type === 'OATFLAKE_CONNECT_DATA') {
        const { groupId, groupInfo } = event.data;
        
        if (groupId) {
            connectToGroup(groupId, groupInfo);
        }
    }
}

// Function to connect to a group
async function connectToGroup(groupId, groupInfo) {
    console.log('Connecting to group:', groupId, groupInfo);
    
    try {
        const response = await fetch('/api/auth/connect', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                group_id: groupId,
                client_version: '0.1.0',
                group_info: groupInfo || undefined
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Connection failed');
        }
        
        console.log('Connection successful:', data);
        
        // Close the modal
        document.getElementById('loginSection').classList.add('hidden');
        
        // Update status
        updateStatus();
        
        // Show success message
        alert(`Successfully connected to ${data.group_name || 'group'}`);
        
    } catch (error) {
        console.error('Connection error:', error);
        alert(`Failed to connect: ${error.message}`);
    }
}

// Initialize the slide content
function initializeSlides() {
    console.log('Initializing application slides');
    
    // Initialize the search view
    const searchContainer = document.getElementById('searchContainer');
    if (searchContainer && typeof SearchSlide !== 'undefined') {
        console.log('Initializing search slide');
        SearchSlide.render(searchContainer);
    } else {
        console.error('Could not initialize search slide');
    }
    
    // Initialize the data view
    const dataContainer = document.getElementById('dataContainer');
    if (dataContainer && typeof DataSlide !== 'undefined') {
        console.log('Initializing data slide');
        DataSlide.render(dataContainer);
    } else {
        console.error('Could not initialize data slide');
    }
    
    // The settings view is now initialized in settings.js
    // No need to initialize it here to avoid duplicates
    const settingsContainer = document.getElementById('settingsContainer');
    if (settingsContainer && typeof SettingsSlide !== 'undefined') {
        console.log('Settings slide is handled by settings.js');
        // We don't call SettingsSlide.render() here to avoid duplication
    } else {
        console.error('Settings container or script not found');
    }
    
    // Initialize view switch after slides are ready
    setTimeout(() => {
        if (window.viewSwitch) {
            console.log('Refreshing view switch after slide initialization');
            window.viewSwitch.switchView(window.viewSwitch.activeView);
        }
    }, 200);
}

// Update group connection status in UI
function updateGroupStatus() {
    const statusText = document.getElementById('statusText');
    const connectButton = document.getElementById('connectButton');
    
    // Get connection info from config
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            console.log('Group status:', data);
            
            if (data.group_id && data.group_name) {
                // Connected to a group
                if (statusText) {
                    statusText.innerHTML = `<span class="inline-block px-2 py-1 text-xs font-semibold rounded-full bg-green-500/10 text-green-500">Connected to ${data.group_name}</span>`;
                }
                
                // Change connect button to disconnect
                if (connectButton) {
                    connectButton.innerHTML = `
                        <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                        Disconnect
                    `;
                    // Change onclick to disconnect
                    connectButton.onclick = disconnectFromGroup;
                }
            } else {
                // Not connected to a group
                if (statusText) {
                    statusText.innerHTML = `<span class="inline-block px-2 py-1 text-xs font-semibold rounded-full bg-gray-500/10 text-gray-300">Local Mode</span>`;
                }
                
                // Reset connect button
                if (connectButton) {
                    connectButton.innerHTML = `
                        <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                        </svg>
                        Connect
                    `;
                    // Reset onclick
                    connectButton.onclick = showConnectionModal;
                }
            }
        })
        .catch(error => {
            console.error('Error updating group status:', error);
            // Show disconnected status
            if (statusText) {
                statusText.innerHTML = `<span class="inline-block px-2 py-1 text-xs font-semibold rounded-full bg-red-500/10 text-red-500">Connection Error</span>`;
            }
        });
    
    // Always keep main content visible
    const mainContent = document.getElementById('mainContent');
    if (mainContent) mainContent.classList.remove('hidden');
}

// Function to disconnect from current group
async function disconnectFromGroup() {
    if (confirm('Disconnect from current group?')) {
        try {
            // Clear group connection info
            const response = await fetch('/api/auth/logout', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to disconnect');
            }
            
            // Update UI
            updateGroupStatus();
            updateStatus();
            
            // Show success message
            alert('Successfully disconnected from group');
        } catch (error) {
            console.error('Disconnect error:', error);
            alert(`Failed to disconnect: ${error.message}`);
        }
    }
}

// Add handleLogout function
async function handleLogout() {
    try {
        // Clear stored credentials
        localStorage.removeItem('authToken');
        localStorage.removeItem('userEmail');
        authToken = null;
        
        // Update UI
        updateLoginStatus();
        
        // Clear any existing status update interval
        if (window.statusInterval) {
            clearInterval(window.statusInterval);
        }
        
        // Reset status displays
        updateStatusElement('server', 'disconnected');
        updateStatusElement('ollama', 'disconnected');
        updateStatusElement('tunnel', 'disconnected');
        updateStatusElement('group', 'Not Connected');
        
        console.log('Logout successful');
    } catch (error) {
        console.error('Logout error:', error);
    }
}

// Add immediate initialization
console.log('Setting up event listeners...');
document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOM loaded');
    
    // Initialize data path
    const savedPath = localStorage.getItem('dataPath') || './data';
    const dataPathElement = document.getElementById('dataPath');
    if (dataPathElement) {
        dataPathElement.value = savedPath;
    }
    
    // Always start with the main content visible - no login required
    const mainContent = document.getElementById('mainContent');
    if (mainContent) {
        mainContent.classList.remove('hidden');
    }
    
    // Initialize slides right away
    initializeSlides();
    
    // Update group status immediately
    updateGroupStatus();
    
    // Start status updates
    updateStatus();
    setInterval(updateStatus, 2000);
    
    // Set up window message listener for iframe communication
    window.addEventListener('message', receiveConnectMessage);
});

// Status update function
async function updateStatus() {
    try {
        // Add better error handling for fetch
        const response = await fetch('/api/status', {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
            credentials: 'same-origin'  // Include cookies if any
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Status response:', data);  // Debug log
        
        // Update group connection status
        updateGroupStatus();
        
        // Update UI elements for services
        updateStatusElement('server', 'running');
        updateStatusElement('ollama', data.ollama);
        updateStatusElement('tunnel', data.tunnel);

        if (data.ngrok_url) {
            const domainInput = document.getElementById('domainInput');
            if (domainInput) {
                domainInput.value = data.ngrok_url;
            }
        }
        if (data.data_path) {
            const dataPathInput = document.getElementById('dataPath');
            if (dataPathInput) {
                dataPathInput.value = data.data_path;
            }
        }
    } catch (error) {
        console.error('Status update error:', error);
        
        // Handle specific error cases
        if (error.message.includes('Failed to fetch')) {
            console.log('Server might be starting up...');
            // Maybe show a "connecting..." message
        }
        
        // Update service statuses to disconnected
        updateStatusElement('server', 'disconnected');
        updateStatusElement('ollama', 'disconnected');
        updateStatusElement('tunnel', 'disconnected');
        
        // Update group status with error
        const statusText = document.getElementById('statusText');
        if (statusText) {
            statusText.innerHTML = `<span class="inline-block px-2 py-1 text-xs font-semibold rounded-full bg-red-500/10 text-red-500">Connection Error</span>`;
        }
    }
}

function updateStatusElement(type, status, extraData = {}) {
    const element = document.getElementById(`${type}Status`);
    if (!element) return;

    const icon = element.querySelector('img');
    const text = element.querySelector('span');
    if (!text) return; // Don't proceed if there's no text element
    
    const isActive = status === 'running' || status === 'connected';
    
    // Special handling for group status
    if (type === 'group') {
        if (status !== 'disconnected' && status !== 'Not Connected') {
            if (extraData.group_image && icon) {
                const imageUrl = extraData.group_image;
                console.log('Loading group image:', imageUrl);
                
                icon.onload = () => {
                    console.log('Successfully loaded image');
                    icon.style.opacity = '1';
                };
                
                icon.onerror = () => {
                    console.error('Failed to load image, using default');
                    icon.src = '/static/icons/GROUPLOGO.png';
                    icon.style.opacity = '1';
                };
                
                icon.src = imageUrl;
                if (icon.style) icon.style.filter = 'none';  // Remove any filters
            }
            text.innerHTML = `Group: <span class="inline-block px-2 py-1 text-xs font-semibold rounded-full bg-green-500/10 text-green-500">${extraData.group_name || status}</span>`;
        } else {
            if (icon) {
                icon.src = '/static/icons/GROUPLOGO.png';
                if (icon.style) icon.style.filter = 'none';  // Remove any filters
            }
            text.innerHTML = `Group: <span class="inline-block px-2 py-1 text-xs font-semibold rounded-full bg-red-500/10 text-red-500">Not Connected</span>`;
        }
    } else {
        // Reset any existing filters
        if (icon && icon.style) {
            icon.style.filter = 'none';
        }
        
        // Create status badge
        const statusBadge = `<span class="inline-block px-2 py-1 text-xs font-semibold rounded-full ${
            isActive ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'
        }">${status.charAt(0).toUpperCase() + status.slice(1)}</span>`;
        
        text.innerHTML = `${type.charAt(0).toUpperCase() + type.slice(1)}: ${statusBadge}`;
    }
}

function copyDomain() {
    const input = document.getElementById('domainInput');
    if (!input || !input.value) return;

    // Copy to clipboard
    input.select();
    document.execCommand('copy');

    // Visual feedback
    const originalBg = input.style.backgroundColor;
    input.style.backgroundColor = '#374151';
    setTimeout(() => {
        input.style.backgroundColor = originalBg;
        // Deselect the input
        window.getSelection().removeAllRanges();
    }, 200);
}

// Remove or comment out the existing selectFolder function
// async function selectFolder() { ... }

async function checkForUpdates() {
    const button = document.querySelector('button[onclick="checkForUpdates()"]');
    const statusEl = document.getElementById('updateStatus');
    
    try {
        button.disabled = true;
        button.innerHTML = `
            <svg class="w-4 h-4 animate-spin" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
            </svg>
            Checking...
        `;
        
        const response = await fetch('/api/check-update');
        const data = await response.json();
        
        if (data.update_available) {
            statusEl.innerHTML = `
                <span class="text-yellow-400">
                    Update available! Version ${data.latest_version}
                </span>`;
        } else {
            statusEl.innerHTML = `
                <span class="text-green-400">
                    You're up to date (v${data.current_version})
                </span>`;
        }
    } catch (error) {
        console.error('Error checking updates:', error);
        statusEl.innerHTML = `<span class="text-red-400">Error checking for updates</span>`;
    } finally {
        button.disabled = false;
        button.innerHTML = `
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Check for Updates
        `;
    }
}
