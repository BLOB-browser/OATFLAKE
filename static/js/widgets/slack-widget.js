// Slack Widget Module
// Handles all Slack-related functionality

const SlackWidget = (() => {
    // Cache DOM elements
    const elements = {
        statusText: () => document.getElementById('slackStatusText'),
        statusIcon: () => document.getElementById('slackStatusIcon'),
        tokenInput: () => document.getElementById('slackTokenInput')
    };

    /**
     * Initialize the Slack widget
     */
    function initialize() {
        // Load the stored token from localStorage
        loadStoredToken();

        // Set up event listeners
        const saveButton = document.querySelector('button[onclick="SlackWidget.saveToken()"]');
        if (saveButton) {
            saveButton.onclick = saveToken;
        }
    }

    /**
     * Load the stored Slack token from localStorage
     */
    function loadStoredToken() {
        const savedToken = localStorage.getItem('slackToken');
        if (savedToken) {
            updateTokenInput(savedToken);
            updateStatus(true);
        } else {
            updateStatus(false);
        }
    }

    /**
     * Save the Slack configuration to the .env file via API
     */
    async function saveSlackConfigToEnv(token, signingSecret, botUserId) {
        try {
            const response = await fetch('/api/save-to-env', {
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

            const response2 = await fetch('/api/save-to-env', {
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

            const response3 = await fetch('/api/save-to-env', {
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

            console.log('Slack configuration saved to .env successfully');
        } catch (error) {
            console.error('Error saving Slack configuration to .env:', error);
        }
    }

    /**
     * Save the Slack configuration to localStorage and .env
     */
    function saveSlackConfig() {
        const tokenInput = elements.tokenInput();
        const signingSecretInput = document.getElementById('slackSigningSecretInput');
        const botUserIdInput = document.getElementById('slackBotUserIdInput');

        if (tokenInput && signingSecretInput && botUserIdInput && tokenInput.value && signingSecretInput.value && botUserIdInput.value) {
            const token = tokenInput.value;
            const signingSecret = signingSecretInput.value;
            const botUserId = botUserIdInput.value;

            localStorage.setItem('slackToken', token);
            localStorage.setItem('slackSigningSecret', signingSecret);
            localStorage.setItem('slackBotUserId', botUserId);

            updateStatus(true);
            saveSlackConfigToEnv(token, signingSecret, botUserId); // Save to .env
            alert('Slack configuration saved successfully!');
        } else {
            alert('Please enter valid Slack configuration values.');
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

    // Public API
    return {
        initialize,
        saveToken
    };
})();

// Initialize the widget when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    SlackWidget.initialize();
});
