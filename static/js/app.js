console.log('Loading app.js...');

// Global variables
let authToken = null;

// Explicitly define the login function
async function handleLoginClick() {
    console.log('Login button clicked');
    
    const emailInput = document.getElementById('emailInput');
    const passwordInput = document.getElementById('passwordInput');
    
    if (!emailInput || !passwordInput) {
        console.error('Login inputs not found');
        return;
    }
    
    const email = emailInput.value;
    const password = passwordInput.value;
    
    console.log('Attempting login with email:', email);
    
    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password })
        });
        
        console.log('Server response status:', response.status);
        
        const data = await response.json();
        console.log('Server response:', data);
        
        if (!response.ok) {
            throw new Error(data.detail || 'Login failed');
        }
        
        // Store auth token and email
        localStorage.setItem('authToken', data.token);
        localStorage.setItem('userEmail', email);
        authToken = data.token;
        
        // Update UI
        document.getElementById('loginSection').classList.add('hidden');
        document.getElementById('mainContent').classList.remove('hidden');
        updateLoginStatus();
        
        // Initialize slides
        initializeSlides();
        
        // Start status updates
        updateStatus();
        setInterval(updateStatus, 2000);
        
    } catch (error) {
        console.error('Login error:', error);
        alert('Login failed: ' + error.message);
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
    
    // Initialize the settings view
    const settingsContainer = document.getElementById('settingsContainer');
    if (settingsContainer && typeof SettingsSlide !== 'undefined') {
        console.log('Initializing settings slide');
        SettingsSlide.render(settingsContainer);
    } else {
        console.error('Could not initialize settings slide');
    }
    
    // Initialize view switch after slides are ready
    setTimeout(() => {
        if (window.viewSwitch) {
            console.log('Refreshing view switch after slide initialization');
            window.viewSwitch.switchView(window.viewSwitch.activeView);
        }
    }, 200);
}

// Add updateLoginStatus function
function updateLoginStatus() {
    const loginStatus = document.getElementById('loginStatusText');
    const logoutButton = document.getElementById('logoutButton');
    const token = localStorage.getItem('authToken');
    const email = localStorage.getItem('userEmail');
    const loginSection = document.getElementById('loginSection');
    const mainContent = document.getElementById('mainContent');

    console.log('Updating login status:', { token: !!token, email });

    // Add null checks before trying to modify elements
    if (token && email) {
        if (loginStatus) loginStatus.innerHTML = `<span class="inline-block px-2 py-1 text-xs font-semibold rounded-full bg-green-500/10 text-green-500">Logged in as ${email}</span>`;
        if (logoutButton) logoutButton.classList.remove('hidden');
        if (loginSection) loginSection.classList.add('hidden');
        if (mainContent) mainContent.classList.remove('hidden');
    } else {
        if (loginStatus) loginStatus.innerHTML = `<span class="inline-block px-2 py-1 text-xs font-semibold rounded-full bg-red-500/10 text-red-500">Not logged in</span>`;
        if (logoutButton) logoutButton.classList.add('hidden');
        if (loginSection) loginSection.classList.remove('hidden');
        if (mainContent) mainContent.classList.add('hidden');
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
    document.getElementById('dataPath').value = savedPath;
    
    // Setup login button
    const loginButton = document.getElementById('loginButton');
    if (loginButton) {
        loginButton.addEventListener('click', handleLoginClick);
    }
    
    // Check existing auth and update UI immediately
    const token = localStorage.getItem('authToken');
    const email = localStorage.getItem('userEmail');
    
    if (!token || !email) {
        document.getElementById('loginSection').classList.remove('hidden');
        document.getElementById('mainContent').classList.add('hidden');
    } else {
        authToken = token;
        document.getElementById('loginSection').classList.add('hidden');
        document.getElementById('mainContent').classList.remove('hidden');
        
        // Initialize slides for returning users
        initializeSlides();
    }
    
    // Update login status immediately
    updateLoginStatus();
    
    // Start status updates
    updateStatus();
    setInterval(updateStatus, 2000);
});

// Status update function
async function updateStatus() {
    try {
        const token = localStorage.getItem('authToken');
        const email = localStorage.getItem('userEmail');
        
        // Always update login status first
        updateLoginStatus();
        
        // Add better error handling for fetch
        const response = await fetch('/api/status', {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                ...(token && { 'Authorization': `Bearer ${token}` })
            },
            credentials: 'same-origin'  // Include cookies if any
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Status response:', data);  // Debug log
        
        // Update UI elements
        updateStatusElement('server', 'running');
        updateStatusElement('ollama', data.ollama);
        updateStatusElement('tunnel', data.tunnel);
        updateStatusElement('group', data.group_id || 'Not Connected', {
            group_image: data.group_image,
            group_name: data.group_name
        });

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
        // Also update login status on error
        updateLoginStatus();
        // Handle specific error cases
        if (error.message.includes('Failed to fetch')) {
            console.log('Server might be starting up...');
            // Maybe show a "connecting..." message
        }
        updateStatusElement('server', 'disconnected');
        updateStatusElement('ollama', 'disconnected');
        updateStatusElement('tunnel', 'disconnected');
        updateStatusElement('group', 'Not Connected');
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
