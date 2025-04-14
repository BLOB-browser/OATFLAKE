/**
 * Tunnel Widget Module
 * Handles all Ngrok tunnel-related functionality
 */

const TunnelWidget = (() => {
    // Cache DOM elements
    const elements = {
        statusText: () => document.getElementById('tunnelStatusText'),
        statusIcon: () => document.getElementById('tunnelStatusIcon'),
        tokenInput: () => document.getElementById('ngrokTokenInput'),
        tunnelUrl: () => document.getElementById('widgetTunnelUrl'),
        domainDisplay: () => document.getElementById('tunnelDomainDisplay')
    };

    /**
     * Initialize the Tunnel widget
     */
    function initialize() {
        // Set up handlers for the copy button and token save
        const copyButton = document.querySelector('button[onclick="copyTunnelUrlFromWidget()"]');
        if (copyButton) {
            copyButton.onclick = copyTunnelUrl;
        }

        // Replace the inline handler with our module function
        const saveButton = document.querySelector('button[onclick="saveNgrokToken()"]');
        if (saveButton) {
            saveButton.onclick = saveToken;
        }
        
        // Check token status on initialization
        checkTokenStatus();
    }

    /**
     * Update the Tunnel status display
     * @param {boolean} isConnected - Whether the tunnel is connected
     * @param {string} tunnelUrl - The URL of the active tunnel (if any)
     */
    function updateStatus(isConnected, tunnelUrl = null) {
        const statusText = elements.statusText();
        const statusIcon = elements.statusIcon();
        const domainDisplay = elements.domainDisplay();
        
        if (!statusText || !statusIcon) return;
        
        if (isConnected && tunnelUrl) {
            statusText.textContent = 'Connected';
            statusText.classList.remove('text-neutral-400', 'text-red-500');
            statusText.classList.add('text-green-500');
            
            statusIcon.classList.remove('bg-yellow-500', 'bg-red-500');
            statusIcon.classList.add('bg-green-500');
            
            // Show and set the tunnel URL
            if (domainDisplay) {
                domainDisplay.classList.remove('hidden');
                
                const urlInput = elements.tunnelUrl();
                if (urlInput) {
                    urlInput.value = tunnelUrl;
                }
            }
        } else {
            statusText.textContent = isConnected ? 'No active domain' : 'Disconnected';
            statusText.classList.remove('text-green-500');
            statusText.classList.add('text-neutral-400');
            
            statusIcon.classList.remove('bg-green-500', 'bg-red-500');
            statusIcon.classList.add(isConnected ? 'bg-yellow-500' : 'bg-red-500');
            
            // Hide the domain display
            if (domainDisplay) {
                domainDisplay.classList.add('hidden');
            }
        }
    }

    /**
     * Check if a Ngrok token is set
     */
    async function checkTokenStatus() {
        try {
            const response = await fetch('/api/system/get-ngrok-status');
            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }
            
            const data = await response.json();
            const tokenInput = elements.tokenInput();
            
            if (tokenInput && data.hasToken) {
                tokenInput.placeholder = "Ngrok token is set";
            }
            
            // Update status based on tunnel presence
            if (data.hasTunnel && data.tunnelUrl) {
                updateStatus(true, data.tunnelUrl);
            } else {
                updateStatus(data.hasToken, null);
            }
        } catch (error) {
            console.error('Error checking token status:', error);
            updateStatus(false);
        }
    }

    /**
     * Save the Ngrok token
     */
    async function saveToken() {
        const tokenInput = elements.tokenInput();
        
        if (!tokenInput) return;
        
        const token = tokenInput.value.trim();
        if (!token) {
            alert('Please enter a valid Ngrok Auth Token');
            return;
        }

        try {
            // Save the token to the server
            const response = await fetch('/api/system/set-ngrok-token', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ token: token })
            });

            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }

            const result = await response.json();
            
            // Show success or failure feedback
            if (result.success) {
                // Show success feedback
                tokenInput.value = '';
                tokenInput.placeholder = "Ngrok token saved successfully";
                tokenInput.classList.add('border-green-500');
                
                setTimeout(() => {
                    tokenInput.placeholder = "Ngrok token is set";
                    tokenInput.classList.remove('border-green-500');
                }, 3000);
                
                // Check status after token is saved
                await checkTokenStatus();
                
                // Request a status update from the main app
                if (typeof updateAllStatus === 'function') {
                    updateAllStatus();
                }
            } else {
                // Show error feedback
                alert(`Failed to save token: ${result.message || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error saving token:', error);
            alert('Error saving token: ' + error.message);
        }
    }

    /**
     * Copy tunnel URL to clipboard
     */
    function copyTunnelUrl() {
        const urlInput = elements.tunnelUrl();
        if (!urlInput || !urlInput.value) return;
        
        urlInput.select();
        document.execCommand('copy');
        
        // Show brief visual feedback
        const copyButton = document.querySelector('button[onclick="copyTunnelUrlFromWidget()"]');
        if (copyButton) {
            copyButton.classList.add('bg-indigo-600');
            
            setTimeout(() => {
                copyButton.classList.remove('bg-indigo-600');
            }, 1000);
        }
    }

    // Public API
    return {
        initialize,
        updateStatus,
        checkTokenStatus,
        saveToken,
        copyTunnelUrl
    };
})();

// Initialize on script load
document.addEventListener('DOMContentLoaded', TunnelWidget.initialize);
