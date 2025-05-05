/**
 * Group Widget Module
 * Handles all Group-related functionality and integration with Blob Browser
 */

const GroupWidget = (() => {
    // Cache DOM elements
    const elements = {
        statusText: () => document.getElementById('groupStatusText'),
        statusIcon: () => document.getElementById('groupStatusIcon'),
        widgetContainer: () => document.getElementById('groupWidget'),
        groupImage: () => document.querySelector('#groupWidget img'),
        connectButton: () => document.getElementById('groupConnectButton'),
        copyTunnelButton: () => document.getElementById('copyTunnelForGroupBtn'),
        tunnelUrlInput: () => document.getElementById('widgetTunnelUrl')
    };

    /**
     * Initialize the Group widget
     */
    function initialize() {
        // Set initial state
        updateStatus(false);
        
        // Add event listeners
        const widgetContainer = elements.widgetContainer();
        if (widgetContainer) {
            widgetContainer.addEventListener('click', handleWidgetClick);
        }
        
        // Add connect button event listener
        const connectButton = elements.connectButton();
        if (connectButton) {
            connectButton.addEventListener('click', function(e) {
                e.preventDefault();
                console.log('Opening Blob Browser login page');
                window.open('https://blob-browser.net/login', '_blank');
            });
        }
        
        // Add copy tunnel button event listener
        const copyTunnelButton = elements.copyTunnelButton();
        if (copyTunnelButton) {
            copyTunnelButton.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation(); // Prevent widget click handler
                copyTunnelUrl();
            });
        }
    }

    /**
     * Update the Group status display
     * @param {boolean} isConnected - Whether the group is connected
     * @param {object} data - Additional group data if available
     */
    function updateStatus(isConnected, data = {}) {
        const statusText = elements.statusText();
        const statusIcon = elements.statusIcon();
        const groupImage = elements.groupImage();
        
        if (!statusText || !statusIcon) return;
        
        // Debug log for group data inspection
        console.log("Group widget raw data:", data);
        
        if (isConnected && data.group_id) {
            // Extract group details from data
            const groupId = data.group_id || 'Unknown';
            
            // Find the group name - STRICTLY only use name fields, never institution_type
            // Log all fields to help with debugging
            const nameFields = {
                group_name: data.group_name,
                name: data.name
            };
            console.log("Group name fields available:", nameFields);
            
            let groupName = null;
            
            // ONLY use actual name fields (never institution_type or type)
            if (data.group_name && typeof data.group_name === 'string' && data.group_name !== 'null' && data.group_name !== 'undefined') {
                groupName = data.group_name;
            } else if (data.name && typeof data.name === 'string' && data.name !== 'null' && data.name !== 'undefined') {
                groupName = data.name;
            } else {
                // If no name is found, use a shortened ID as fallback
                groupName = `Group ${groupId.substring(0, 6)}...`;
            }
            
            console.log(`Using group name: "${groupName}"`);
            
            // Get member count if available
            const groupMembers = data.group_members || data.members || [];
            const memberCount = Array.isArray(groupMembers) ? groupMembers.length : 
                               (typeof groupMembers === 'number' ? groupMembers : 0);
            
            // Format the status text
            let formattedStatus = groupName;
            if (memberCount > 0) {
                formattedStatus += ` (${memberCount} ${memberCount === 1 ? 'member' : 'members'})`;
            }
            
            // Update the UI
            statusText.textContent = formattedStatus;
            statusText.classList.remove('text-neutral-400', 'text-red-500');
            statusText.classList.add('text-green-500');
            
            statusIcon.classList.remove('bg-neutral-500', 'bg-red-500');
            statusIcon.classList.add('bg-green-500');
            
            // Update group image if available
            if (groupImage) {
                if (data.group_image) {
                    groupImage.src = data.group_image;
                    groupImage.style.opacity = '1';
                } else {
                    // Use default image with full opacity to indicate connected state
                    groupImage.src = '/static/icons/GROUPLOGO.png';
                    groupImage.style.opacity = '1';
                }
            }
            
            // Add hover effect to show it's interactive
            const widgetContainer = elements.widgetContainer();
            if (widgetContainer) {
                widgetContainer.classList.add('hover:border-indigo-500', 'transition-colors', 'cursor-pointer');
            }
        } else {
            statusText.textContent = 'Not connected';
            statusText.classList.remove('text-green-500', 'text-red-500');
            statusText.classList.add('text-neutral-400');
            
            statusIcon.classList.remove('bg-green-500', 'bg-red-500');
            statusIcon.classList.add('bg-neutral-500');
            
            // Reset group image to default with reduced opacity
            if (groupImage) {
                groupImage.src = '/static/icons/GROUPLOGO.png';
                groupImage.style.opacity = '0.7';
            }
            
            // Remove interactive styling
            const widgetContainer = elements.widgetContainer();
            if (widgetContainer) {
                widgetContainer.classList.remove('hover:border-indigo-500', 'transition-colors', 'cursor-pointer');
            }
        }
    }

    /**
     * Handle click on the group widget
     * @param {Event} e - Click event
     */
    function handleWidgetClick(e) {
        // If clicking on the connect button or copy button, let their handlers deal with it
        if (e.target.closest('#groupConnectButton') || e.target.closest('#copyTunnelForGroupBtn')) {
            return;
        }
        
        // Only show management options when connected to a group
        const statusIcon = elements.statusIcon();
        if (!statusIcon || !statusIcon.classList.contains('bg-green-500')) {
            console.log('No active group to manage');
            // If not connected, open the Blob Browser login page
            window.open('https://blob-browser.net/login', '_blank');
            return;
        }
        
        console.log('Group widget clicked, showing group management options');
        // Show group management UI for connected groups
        // Future implementation: Add more management options here
    }
    
    /**
     * Copy the tunnel URL to clipboard for easy pasting into Blob Browser
     */
    function copyTunnelUrl() {
        const tunnelUrlInput = elements.tunnelUrlInput();
        
        if (tunnelUrlInput && tunnelUrlInput.value) {
            // Select and copy the tunnel URL
            tunnelUrlInput.select();
            document.execCommand('copy');
            
            // Visual feedback
            const copyButton = elements.copyTunnelButton();
            if (copyButton) {
                // Store original text
                const originalText = copyButton.innerHTML;
                
                // Change to confirmation
                copyButton.innerHTML = `
                    <svg class="w-3 h-3 mr-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                    Copied!
                `;
                
                // Restore original text after a delay
                setTimeout(() => {
                    copyButton.innerHTML = originalText;
                }, 2000);
            }
            
            console.log('Tunnel URL copied to clipboard');
        } else {
            console.error('No tunnel URL available to copy');
            alert('Please set up a tunnel first to get a domain to share');
        }
    }

    // Public API
    return {
        initialize,
        updateStatus,
        copyTunnelUrl
    };
})();

// Initialize on script load
document.addEventListener('DOMContentLoaded', GroupWidget.initialize);
