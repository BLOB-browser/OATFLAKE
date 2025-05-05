// Slack Widget Module
// Handles all Slack-related functionality

const SlackWidget = (() => {
    // Cache DOM elements
    const elements = {
        statusText: () => document.getElementById('slackStatusText'),
        statusIcon: () => document.getElementById('slackStatusIcon'),
        tokenInput: () => document.getElementById('slackTokenInput'),
        signingSecretInput: () => document.getElementById('slackSigningSecretInput'),
        botUserIdInput: () => document.getElementById('slackBotUserIdInput'),
        widgetContainer: () => document.getElementById('slackWidget'),
        saveButton: () => document.querySelector('button[onclick="SlackWidget.saveSlackConfig()"]')
    };

    /**
     * Initialize the Slack widget
     */
    function initialize() {
        console.log('Initializing Slack widget');
        // Load the stored token from localStorage
        loadStoredToken();

        // Set up event listeners for configuration
        const saveButton = elements.saveButton();
        if (saveButton) {
            console.log('Save button found for Slack widget');
            // The onclick attribute handles the click
        } else {
            console.warn('Save button not found for Slack widget');
        }
        
        // Check for widget container click for documentation
        const widgetContainer = elements.widgetContainer();
        if (widgetContainer) {
            widgetContainer.addEventListener('click', handleDocLinkClick);
        }
        
        // Check if we already have values in .env
        checkExistingConfiguration();
    }
    
    /**
     * Handle clicks on the documentation link
     */
    function handleDocLinkClick(e) {
        // Check if clicking on the documentation link
        const docLink = e.target.closest('a[href*="blob-browser.net/documentation"]');
        if (docLink) {
            console.log('Opening Slack documentation');
            // Let the link's default behavior handle it (opens in new tab)
        }
    }

    /**
     * Update the signing secret input with the provided value
     * @param {string} signingSecret - The signing secret to display
     */
    function updateSigningSecretInput(signingSecret) {
        const input = elements.signingSecretInput();
        if (input) {
            input.value = signingSecret;
            // Mask the value for display
            const maskedValue = '*'.repeat(Math.min(signingSecret.length, 10));
            console.log(`Updated signing secret input: ${maskedValue}...`);
        }
    }
    
    /**
     * Update the bot user ID input with the provided value
     * @param {string} botUserId - The bot user ID to display
     */
    function updateBotUserIdInput(botUserId) {
        const input = elements.botUserIdInput();
        if (input) {
            input.value = botUserId;
            console.log(`Updated bot user ID input: ${botUserId}`);
        }
    }

    /**
     * Load the stored Slack token from localStorage
     */
    function loadStoredToken() {
        const savedToken = localStorage.getItem('slackToken');
        const savedSigningSecret = localStorage.getItem('slackSigningSecret');
        const savedBotUserId = localStorage.getItem('slackBotUserId');
        
        if (savedToken) {
            updateTokenInput(savedToken);
            if (savedSigningSecret) {
                updateSigningSecretInput(savedSigningSecret);
            }
            if (savedBotUserId) {
                updateBotUserIdInput(savedBotUserId);
            }
            updateStatus(true);
        } else {
            updateStatus(false);
        }
    }
    
    /**
     * Check for existing Slack configuration in the .env file
     */
    async function checkExistingConfiguration() {
        try {
            // Use the system route to get environment variables
            const response = await fetch('/api/system/env-variables', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    variables: ['SLACK_BOT_TOKEN', 'SLACK_SIGNING_SECRET', 'SLACK_BOT_USER_ID']
                })
            });
            
            const data = await response.json();
            
            if (data && data.values) {
                console.log('Checking Slack configuration in .env');
                
                const token = data.values.SLACK_BOT_TOKEN;
                const signingSecret = data.values.SLACK_SIGNING_SECRET;
                const botUserId = data.values.SLACK_BOT_USER_ID;
                
                let isConfigured = false;
                
                if (token) {
                    updateTokenInput(token);
                    localStorage.setItem('slackToken', token);
                    isConfigured = true;
                }
                
                if (signingSecret) {
                    updateSigningSecretInput(signingSecret);
                    localStorage.setItem('slackSigningSecret', signingSecret);
                    isConfigured = true;
                }
                
                if (botUserId) {
                    updateBotUserIdInput(botUserId);
                    localStorage.setItem('slackBotUserId', botUserId);
                }
                
                updateStatus(isConfigured);
            } else {
                console.log('No existing Slack configuration found in .env');
                updateStatus(false);
            }
        } catch (error) {
            console.error('Error checking existing Slack configuration:', error);
            
            // Fall back to localStorage if API fails
            loadStoredToken();
        }
    }

    /**
     * Save the Slack configuration to the .env file via API
     */
    async function saveSlackConfigToEnv(token, signingSecret, botUserId) {
        try {
            // Create an object with all the env variables to update
            const envUpdates = {
                'SLACK_BOT_TOKEN': token,
                'SLACK_SIGNING_SECRET': signingSecret
            };
            
            // Add Bot User ID if provided
            if (botUserId) {
                envUpdates['SLACK_BOT_USER_ID'] = botUserId;
            }
            
            // Save all values at once
            const response = await fetch('/api/system/update-env', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ variables: envUpdates })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || 'Failed to save Slack configuration to .env');
            }
            
            // Fall back to individual saves if the batch update endpoint isn't available
            if (response.status === 404) {
                await saveSlackConfigToEnvIndividually(token, signingSecret, botUserId);
            } else {
                console.log('Slack configuration saved to .env successfully');
            }
            
            return true;
        } catch (error) {
            console.error('Error saving Slack configuration to .env:', error);
            
            // Try individual saves as fallback
            try {
                await saveSlackConfigToEnvIndividually(token, signingSecret, botUserId);
                return true;
            } catch (fallbackError) {
                console.error('Fallback save failed:', fallbackError);
                throw error; // Throw original error
            }
        }
    }
    
    /**
     * Save Slack config individually as a fallback
     */
    async function saveSlackConfigToEnvIndividually(token, signingSecret, botUserId) {
        // Save SLACK_BOT_TOKEN
        const response = await fetch('/api/system/update-env-var', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                key: 'SLACK_BOT_TOKEN',
                value: token
            })
        });

        if (!response.ok) {
            throw new Error('Failed to save SLACK_BOT_TOKEN to .env');
        }

        // Save SLACK_SIGNING_SECRET
        const response2 = await fetch('/api/system/update-env-var', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                key: 'SLACK_SIGNING_SECRET',
                value: signingSecret
            })
        });

        if (!response2.ok) {
            throw new Error('Failed to save SLACK_SIGNING_SECRET to .env');
        }
        
        // Save SLACK_BOT_USER_ID if provided
        if (botUserId) {
            const response3 = await fetch('/api/system/update-env-var', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    key: 'SLACK_BOT_USER_ID',
                    value: botUserId
                })
            });

            if (!response3.ok) {
                throw new Error('Failed to save SLACK_BOT_USER_ID to .env');
            }
        }

        console.log('Slack configuration saved to .env successfully (individually)');
    }

    /**
     * Save the Slack configuration to localStorage and .env
     */
    async function saveSlackConfig() {
        const tokenInput = elements.tokenInput();
        const signingSecretInput = elements.signingSecretInput();
        const botUserIdInput = elements.botUserIdInput();
        const saveButton = elements.saveButton();
        
        if (!tokenInput || !signingSecretInput || !botUserIdInput) {
            console.error('Missing input elements');
            alert('Error: Could not find input elements. Please try refreshing the page.');
            return;
        }
        
        const token = tokenInput.value.trim();
        const signingSecret = signingSecretInput.value.trim();
        let botUserId = botUserIdInput.value.trim();
        
        if (!token || !signingSecret) {
            alert('Please enter both Slack API Token and Signing Secret.');
            return;
        }
        
        // Validate Bot User ID format if provided
        // Note: May start with U or A depending on Slack's implementation
        if (botUserId && !botUserId.match(/^[UA][A-Z0-9]+$/i)) {
            // Check if it looks like the user entered the display name (with @) instead of ID
            if (botUserId.startsWith('@')) {
                alert('Please enter the Bot User ID (alphanumeric code), not the display name. You can find this in your Slack workspace settings.');
                return;
            }
            
            // If not empty and not valid format, show warning but allow proceeding
            if (!confirm('Bot User ID format looks incorrect. It should typically start with U or A followed by alphanumeric characters (e.g., U12345678 or A089BE80EP6).\n\nDo you want to proceed anyway?')) {
                return;
            }
        }
        
        try {
            // Show saving state
            if (saveButton) {
                const originalText = saveButton.innerHTML;
                saveButton.innerHTML = 'Saving...';
                saveButton.disabled = true;
                
                // Reset after timeout if something goes wrong
                setTimeout(() => {
                    if (saveButton.innerHTML === 'Saving...') {
                        saveButton.innerHTML = originalText;
                        saveButton.disabled = false;
                    }
                }, 10000);
            }
            
            // If botUserId is empty, check if there's an existing one in .env before saving
            if (!botUserId) {
                try {
                    const response = await fetch('/api/system/env-variables', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            variables: ['SLACK_BOT_USER_ID']
                        })
                    });
                    
                    const data = await response.json();
                    if (data && data.values && data.values.SLACK_BOT_USER_ID) {
                        // Keep existing Bot User ID if present in .env and not overwritten
                        botUserId = data.values.SLACK_BOT_USER_ID;
                        console.log('Preserving existing Bot User ID from .env:', botUserId);
                    }
                } catch (e) {
                    console.warn('Failed to check existing Bot User ID:', e);
                }
            }
            
            // Save to .env file
            await saveSlackConfigToEnv(token, signingSecret, botUserId);
            
            // Save to localStorage
            localStorage.setItem('slackToken', token);
            localStorage.setItem('slackSigningSecret', signingSecret);
            if (botUserId) {
                localStorage.setItem('slackBotUserId', botUserId);
                // Update the field to show the value if it was pulled from .env
                updateBotUserIdInput(botUserId);
            }
            
            // Update status
            updateStatus(true);
            
            // Reset button
            if (saveButton) {
                saveButton.innerHTML = 'Save Configuration';
                saveButton.disabled = false;
            }
            
            alert('Slack configuration saved successfully!');
        } catch (error) {
            console.error('Error saving Slack config:', error);
            alert(`Failed to save Slack configuration: ${error.message}`);
            
            // Reset button
            if (saveButton) {
                saveButton.innerHTML = 'Save Configuration';
                saveButton.disabled = false;
            }
        }
    }

    /**
     * Update the token input field
     * @param {string} token - The Slack token
     */
    function updateTokenInput(token) {
        const tokenInput = elements.tokenInput();
        if (tokenInput) {
            tokenInput.value = token;
            // Mask the real value in logs
            const maskedValue = '*'.repeat(Math.min(token.length, 10));
            console.log(`Updated token input: ${maskedValue}...`);
        }
    }

    /**
     * Update the Slack status display
     * @param {boolean} isConnected - Whether a token is configured
     */
    function updateStatus(isConnected) {
        const statusText = elements.statusText();
        const statusIcon = elements.statusIcon();

        if (!statusText || !statusIcon) return;

        if (isConnected) {
            statusText.textContent = 'Connected';
            statusText.classList.remove('text-neutral-400', 'text-red-500');
            statusText.classList.add('text-green-500');

            statusIcon.classList.remove('bg-yellow-500', 'bg-red-500');
            statusIcon.classList.add('bg-green-500');
        } else {
            statusText.textContent = 'Not connected';
            statusText.classList.remove('text-green-500', 'text-red-500');
            statusText.classList.add('text-neutral-400');

            statusIcon.classList.remove('bg-green-500', 'bg-red-500');
            statusIcon.classList.add('bg-yellow-500');
        }
    }
    
    /**
     * Open the Slack documentation
     */
    function openDocumentation() {
        window.open('https://blob-browser.net/documentation', '_blank');
        console.log('Opening Slack documentation');
    }

    // Public API
    return {
        initialize,
        saveSlackConfig,
        openDocumentation
    };
})();

// Initialize the widget when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    SlackWidget.initialize();
});
