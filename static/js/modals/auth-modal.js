class AuthModal extends HTMLElement {
    constructor() {
        super();
        this.isOpen = false;
        this.currentView = 'login'; // Default view (login, register, reset)
        this.isStandalone = false;  // Whether this is in a standalone page or embedded
    }

    connectedCallback() {
        // Check if this is a standalone auth modal on login page
        this.isStandalone = this.getAttribute('id') === 'mainAuthModal' ||
            this.getAttribute('id') === 'resetPasswordModal';

        // First render the initial state
        this.render();

        // Check URL parameters, but only after Supabase is ready
        if (!window.supabaseClient) {
            window.addEventListener('supabaseReady', () => this.checkUrlParams());
        } else {
            this.checkUrlParams();
        }
    }

    checkUrlParams() {
        // Don't check URL params for embedded modals on pages other than login/reset
        if (!this.isStandalone &&
            window.location.pathname.indexOf('login.html') === -1 &&
            window.location.pathname.indexOf('reset-password.html') === -1) {
            return;
        }

        const urlParams = new URLSearchParams(window.location.search);
        console.log('Checking URL parameters for auth flow:', Object.fromEntries(urlParams.entries()));

        // Check if this is a password reset link - multiple possible parameters
        const hasRecoveryType = urlParams.has('type') && urlParams.get('type') === 'recovery';
        const hasToken = urlParams.has('token');
        const hasAccessToken = urlParams.has('access_token');
        const hasRefreshToken = urlParams.has('refresh_token');
        
        // Also check for hash parameters which Supabase sometimes uses
        let hasHashToken = false;
        if (window.location.hash) {
            const hashParams = new URLSearchParams(window.location.hash.replace('#', ''));
            hasHashToken = hashParams.has('access_token');
        }

        // If any token parameter is present, show the password reset form
        if (hasRecoveryType || hasToken || hasAccessToken || hasRefreshToken || hasHashToken) {
            console.log('Recovery token detected in URL, showing password reset form');
            this.currentView = 'new-password';
            this.show('new-password');
            return;
        }

        // Check for specific auth view request
        if (urlParams.has('view')) {
            const view = urlParams.get('view');
            if (['login', 'register', 'reset', 'new-password'].includes(view)) {
                this.show(view);
                return;
            }
        }

        // For standalone pages without parameters, show the default view
        if (this.isStandalone && !urlParams.has('view') &&
            !urlParams.has('auth') && !urlParams.has('type')) {
            this.show('login');
        }
    }

    show(view = 'login') {
        this.currentView = view;
        this.isOpen = true;
        this.render();

        // Don't mess with body overflow if in standalone mode
        if (!this.isStandalone) {
            document.body.classList.add('overflow-hidden');
        }

        // Focus the first input field
        setTimeout(() => {
            const firstInput = this.querySelector('input');
            if (firstInput) firstInput.focus();
        }, 100);
    }

    hide() {
        // Don't allow hiding in standalone mode
        if (this.isStandalone) {
            // Redirect to homepage instead
            window.location.href = '/';
            return;
        }

        this.isOpen = false;
        this.render();
        document.body.classList.remove('overflow-hidden');
    }

    async handleLogin(e) {
        e.preventDefault();

        const email = this.querySelector('#auth-email').value;
        const password = this.querySelector('#auth-password').value;
        const errorEl = this.querySelector('#auth-error');
        const submitBtn = this.querySelector('button[type="submit"]');

        errorEl.classList.add('hidden');
        errorEl.classList.remove('text-green-500');

        if (!email || !password) {
            errorEl.textContent = 'Email and password are required';
            errorEl.classList.remove('hidden');
            return;
        }

        // Show loading state
        const originalBtnText = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
            <svg class="animate-spin -ml-1 mr-3 h-5 w-5 inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Signing in...
        `;

        try {
            const { data, error } = await window.supabaseClient.auth.signInWithPassword({
                email,
                password
            });

            if (error) throw error;

            // Success - close modal and reload or redirect
            if (this.isStandalone) {
                // Redirect to homepage if on login page
                window.location.href = '/';
            } else {
                // Just reload the current page to update user state
                this.hide();
                window.location.reload();
            }

        } catch (error) {
            console.error('Login error:', error);
            errorEl.textContent = error.message || 'Failed to sign in. Please check your credentials.';
            errorEl.classList.remove('hidden');

            // Reset button
            submitBtn.disabled = false;
            submitBtn.textContent = originalBtnText;
        }
    }

    async handleRegistration(e) {
        e.preventDefault();

        const email = this.querySelector('#auth-email').value;
        const password = this.querySelector('#auth-password').value;
        const confirmPassword = this.querySelector('#auth-confirm-password').value;
        const errorEl = this.querySelector('#auth-error');
        const submitBtn = this.querySelector('button[type="submit"]');

        errorEl.classList.add('hidden');
        errorEl.classList.remove('text-green-500');

        if (!email || !password) {
            errorEl.textContent = 'Email and password are required';
            errorEl.classList.remove('hidden');
            return;
        }

        if (password !== confirmPassword) {
            errorEl.textContent = 'Passwords do not match';
            errorEl.classList.remove('hidden');
            return;
        }

        if (password.length < 6) {
            errorEl.textContent = 'Password must be at least 6 characters';
            errorEl.classList.remove('hidden');
            return;
        }

        // Show loading state
        const originalBtnText = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
            <svg class="animate-spin -ml-1 mr-3 h-5 w-5 inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Creating account...
        `;

        try {
            const { data, error } = await window.supabaseClient.auth.signUp({
                email,
                password
            });

            if (error) throw error;

            // Show success message
            errorEl.textContent = 'Registration successful! Please check your email to confirm your account.';
            errorEl.classList.remove('hidden');
            errorEl.classList.add('text-green-500');

            // Switch back to login after delay
            setTimeout(() => {
                this.currentView = 'login';
                this.render();
            }, 3000);

        } catch (error) {
            console.error('Registration error:', error);
            errorEl.textContent = error.message || 'Failed to create account. Please try again.';
            errorEl.classList.remove('hidden');

            // Reset button
            submitBtn.disabled = false;
            submitBtn.textContent = originalBtnText;
        }
    }

    async handlePasswordReset(e) {
        e.preventDefault();

        const email = this.querySelector('#auth-email').value;
        const errorEl = this.querySelector('#auth-error');
        const submitBtn = this.querySelector('button[type="submit"]');

        errorEl.classList.add('hidden');
        errorEl.classList.remove('text-green-500');

        if (!email) {
            errorEl.textContent = 'Email is required';
            errorEl.classList.remove('hidden');
            return;
        }

        // Show loading state
        const originalBtnText = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
            <svg class="animate-spin -ml-1 mr-3 h-5 w-5 inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Sending reset link...
        `;

        try {
            // Get the absolute URL for the reset page - use full URL path to avoid auth errors
            const origin = window.location.origin;
            const resetUrl = `${origin}/reset-password.html`;

            console.log(`Sending password reset with redirectTo: ${resetUrl}`);

            // Check if we're using the supabaseClient properly
            if (!window.supabaseClient) {
                throw new Error('Supabase client not initialized');
            }

            // Ensure the auth API is properly initialized with the site URL
            const { error } = await window.supabaseClient.auth.resetPasswordForEmail(email, {
                redirectTo: resetUrl,
            });

            if (error) throw error;

            // Show success message and redirect to instructions page
            errorEl.textContent = 'Password reset link sent! Please check your email.';
            errorEl.classList.remove('hidden');
            errorEl.classList.add('text-green-500');

            // Always redirect to reset-password.html for instructions
            setTimeout(() => {
                window.location.href = resetUrl;
            }, 2000);

        } catch (error) {
            console.error('Password reset error:', error);
            errorEl.textContent = error.message || 'Failed to send reset link. Please try again.';
            errorEl.classList.remove('hidden');

            // Reset button
            submitBtn.disabled = false;
            submitBtn.textContent = originalBtnText;
        }
    }

    async handleNewPassword(e) {
        e.preventDefault();

        const password = this.querySelector('#auth-new-password').value;
        const confirmPassword = this.querySelector('#auth-confirm-new-password').value;
        const errorEl = this.querySelector('#auth-error');
        const submitBtn = this.querySelector('button[type="submit"]');

        errorEl.classList.add('hidden');
        errorEl.classList.remove('text-green-500');

        if (!password) {
            errorEl.textContent = 'Password is required';
            errorEl.classList.remove('hidden');
            return;
        }

        if (password !== confirmPassword) {
            errorEl.textContent = 'Passwords do not match';
            errorEl.classList.remove('hidden');
            return;
        }

        if (password.length < 6) {
            errorEl.textContent = 'Password must be at least 6 characters';
            errorEl.classList.remove('hidden');
            return;
        }

        // Show loading state
        const originalBtnText = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
            <svg class="animate-spin -ml-1 mr-3 h-5 w-5 inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Setting new password...
        `;

        try {
            // First, try to get the session from URL if it's a reset link
            if (window.location.hash || window.location.search.includes('token') || 
                window.location.search.includes('access_token') || 
                window.location.search.includes('type=recovery')) {
                
                console.log('Attempting to get session from URL for password reset');
                
                try {
                    // This will set the session based on the URL parameters
                    await window.supabaseClient.auth.getSessionFromUrl();
                } catch (sessionError) {
                    console.warn('Could not get session from URL, trying direct update:', sessionError);
                }
            }

            // Update password for the user
            const { data, error } = await window.supabaseClient.auth.updateUser({
                password: password
            });

            if (error) throw error;

            console.log('Password updated successfully:', data);

            // Show success message
            errorEl.textContent = 'Password updated successfully! You can now log in with your new password.';
            errorEl.classList.remove('hidden');
            errorEl.classList.add('text-green-500');

            // Always redirect to login page after successful password reset
            setTimeout(() => {
                window.location.href = `${window.location.origin}/login.html`;
            }, 2000);

        } catch (error) {
            console.error('Password update error:', error);
            errorEl.textContent = error.message || 'Failed to update password. The reset link may have expired. Please request a new one.';
            errorEl.classList.remove('hidden');

            // Reset button
            submitBtn.disabled = false;
            submitBtn.textContent = originalBtnText;

            // Add a button to request a new reset link
            errorEl.innerHTML += `<br><br><a href="${window.location.origin}/login.html?view=reset" class="text-indigo-400 hover:underline">Request a new reset link</a>`;
        }
    }

    setupEventListeners() {
        // Close button - only shown in popup mode
        const closeBtn = this.querySelector('.auth-close-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hide());
        }

        // Background overlay click to close - only in popup mode
        const overlay = this.querySelector('.auth-overlay');
        if (overlay && !this.isStandalone) {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) this.hide();
            });
        }

        // Form submission
        const form = this.querySelector('form');
        if (form) {
            form.addEventListener('submit', (e) => {
                if (this.currentView === 'login') {
                    this.handleLogin(e);
                } else if (this.currentView === 'register') {
                    this.handleRegistration(e);
                } else if (this.currentView === 'reset') {
                    this.handlePasswordReset(e);
                } else if (this.currentView === 'new-password') {
                    this.handleNewPassword(e);
                }
            });
        }

        // View switchers
        const switchToRegister = this.querySelector('.switch-to-register');
        if (switchToRegister) {
            switchToRegister.addEventListener('click', (e) => {
                e.preventDefault();
                this.currentView = 'register';
                this.render();
            });
        }

        const switchToLogin = this.querySelector('.switch-to-login');
        if (switchToLogin) {
            switchToLogin.addEventListener('click', (e) => {
                e.preventDefault();
                this.currentView = 'login';
                this.render();
            });
        }

        const switchToReset = this.querySelector('.switch-to-reset');
        if (switchToReset) {
            switchToReset.addEventListener('click', (e) => {
                e.preventDefault();
                this.currentView = 'reset';
                this.render();
            });
        }
    }

    render() {
        // Prepare the modal content
        let modalContent = '';

        if (this.isStandalone) {
            // Full page auth form
            modalContent = `
                <div class="max-w-md w-full bg-black border border-2 border-stone-800 p-8 rounded-3xl shadow-lg">
                    <div class="flex flex-col items-center mb-6">
                        <img src="icons/24/logo.png" alt="Logo" class="h-16 mb-6 mr-3">
                        ${this.getHeaderForView()}
                    </div>
                    
                    <form id="authForm">
                        ${this.getFormFieldsForView()}
                        
                        <button type="submit"
                            class="w-full py-3 bg-indigo-600 hover:bg-indigo-700 rounded-lg text-white font-medium">
                            ${this.getButtonTextForView()}
                        </button>
                    </form>
                    
                    <div class="mt-4 text-center text-sm text-gray-400">
                        ${this.getLinkOptionsForView()}
                    </div>
                    
                    <div id="auth-error" class="mt-4 text-red-500 text-center hidden"></div>
                </div>
            `;
        } else {
            // Modal overlay for regular pages
            modalContent = `
                <div class="auth-overlay fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75 ${!this.isOpen ? 'hidden' : ''}" tabindex="-1">
                    <div class="auth-modal-content max-w-md w-full bg-black border border-2 border-stone-800 p-8 rounded-3xl shadow-lg relative">
                        <div class="flex flex-col items-center mb-6">
                            <img src="icons/24/logo.png" alt="Logo" class="h-16 mb-6 mr-3">
                            ${this.getHeaderForView()}
                        </div>
                        
                        <form id="authForm">
                            ${this.getFormFieldsForView()}
                            
                            <button type="submit"
                                class="w-full py-3 bg-indigo-600 hover:bg-indigo-700 rounded-lg text-white font-medium">
                                ${this.getButtonTextForView()}
                            </button>
                        </form>
                        
                        <div class="mt-4 text-center text-sm text-gray-400">
                            ${this.getLinkOptionsForView()}
                        </div>
                        
                        <div id="auth-error" class="mt-4 text-red-500 text-center hidden"></div>
                        
                        <button class="auth-close-btn absolute top-4 right-4 text-gray-400 hover:text-white">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                </div>
            `;
        }

        this.innerHTML = modalContent;

        // Set up event listeners if modal is visible
        if (this.isOpen || this.isStandalone) {
            this.setupEventListeners();
        }
    }

    getHeaderForView() {
        switch (this.currentView) {
            case 'register':
                return `
                    <h1 class="text-3xl uppercase italic font-bold text-start mb-2">Join BLOB</h1>
                    <h2 class="text-xl text-stone-400 text-start">Create an account</h2>
                `;
            case 'reset':
                return `
                    <h1 class="text-3xl uppercase italic font-bold text-start mb-2">Reset Password</h1>
                    <h2 class="text-xl text-stone-400 text-start">We'll send you a reset link</h2>
                `;
            case 'new-password':
                return `
                    <h1 class="text-3xl uppercase italic font-bold text-start mb-2">Set New Password</h1>
                    <h2 class="text-xl text-stone-400 text-start">Enter your new password</h2>
                `;
            default: // login
                return `
                    <h1 class="text-3xl uppercase italic font-bold text-start mb-2">Welcome back</h1>
                    <h2 class="text-xl text-stone-400 text-start">Sign In</h2>
                `;
        }
    }

    getFormFieldsForView() {
        let fields = '';

        if (this.currentView === 'new-password') {
            // New password form after reset
            fields = `
                <div class="mb-6">
                    <label for="auth-new-password" class="block text-sm font-medium mb-2">New Password</label>
                    <input type="password" id="auth-new-password"
                        class="w-full p-3 bg-black border border-stone-800 rounded-lg text-white">
                </div>
                <div class="mb-6">
                    <label for="auth-confirm-new-password" class="block text-sm font-medium mb-2">Confirm New Password</label>
                    <input type="password" id="auth-confirm-new-password"
                        class="w-full p-3 bg-black border border-stone-800 rounded-lg text-white">
                </div>
            `;
        } else {
            // Email field for login, register, and reset views
            fields = `
                <div class="mb-4">
                    <label for="auth-email" class="block text-sm font-medium mb-2">Email</label>
                    <input type="email" id="auth-email"
                        class="w-full p-3 bg-black border border-stone-800 rounded-lg text-white">
                </div>
            `;

            // Password fields for login and register (not for reset)
            if (this.currentView !== 'reset') {
                fields += `
                    <div class="mb-6">
                        <label for="auth-password" class="block text-sm font-medium mb-2">Password</label>
                        <input type="password" id="auth-password"
                            class="w-full p-3 bg-black border border-stone-800 rounded-lg text-white">
                    </div>
                `;
            }

            // Add confirm password for registration
            if (this.currentView === 'register') {
                fields += `
                    <div class="mb-6">
                        <label for="auth-confirm-password" class="block text-sm font-medium mb-2">Confirm Password</label>
                        <input type="password" id="auth-confirm-password"
                            class="w-full p-3 bg-black border border-stone-800 rounded-lg text-white">
                    </div>
                `;
            }
        }

        return fields;
    }

    getButtonTextForView() {
        switch (this.currentView) {
            case 'register':
                return 'Create Account';
            case 'reset':
                return 'Send Reset Link';
            case 'new-password':
                return 'Update Password';
            default: // login
                return 'Sign In';
        }
    }

    getLinkOptionsForView() {
        switch (this.currentView) {
            case 'register':
                return `Already have an account? <a href="#" class="switch-to-login text-indigo-400 hover:underline">Sign in</a>`;
            case 'reset':
                return `Remember your password? <a href="#" class="switch-to-login text-indigo-400 hover:underline">Sign in</a>`;
            case 'new-password':
                return ''; // No options needed for new password screen
            default: // login
                return `
                    Don't have an account? <a href="#" class="switch-to-register text-indigo-400 hover:underline">Create an account</a>
                    <div class="mt-2">
                        <a href="#" class="switch-to-reset text-sm text-indigo-400 hover:underline">Forgot password?</a>
                    </div>
                `;
        }
    }
}

customElements.define('auth-modal', AuthModal);
