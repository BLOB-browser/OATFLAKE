/**
 * Group Widget Module
 * Handles all Group-related functionality
 */

const GroupWidget = (() => {
    // Cache DOM elements
    const elements = {
        statusText: () => document.getElementById('groupStatusText'),
        statusIcon: () => document.getElementById('groupStatusIcon'),
        widgetContainer: () => document.getElementById('groupWidget'),
        groupImage: () => document.querySelector('#groupWidget img')
    };

    /**
     * Initialize the Group widget
     */
    function initialize() {
        // Set initial state
        updateStatus(false);
        
        // Add event listeners if needed
        const widgetContainer = elements.widgetContainer();
        if (widgetContainer) {
            widgetContainer.addEventListener('click', handleWidgetClick);
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
        // Only respond to clicks when connected to a group
        const statusIcon = elements.statusIcon();
        if (!statusIcon || !statusIcon.classList.contains('bg-green-500')) {
            console.log('No active group to manage');
            return;
        }
        
        console.log('Group widget clicked, showing group management options');
        // Future implementation: Show group management UI
    }

    // Public API
    return {
        initialize,
        updateStatus
    };
})();

// Initialize on script load
document.addEventListener('DOMContentLoaded', GroupWidget.initialize);
