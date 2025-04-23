// Check if SearchContainer is already defined
if (!customElements.get('search-container')) {
    class SearchContainer extends HTMLElement {
        constructor() {
            super();
            this.innerHTML = `
                <div class="search-box">
                    <h1 class="text-3xl font-bold text-center mb-8 text-white">BLOB</h1>
                    <input type="text" 
                        class="w-full p-3 rounded-lg bg-opacity-50"
                        placeholder="${this.getAttribute('placeholder') || 'Search...'}"
                    >
                    <button class="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg mt-2 w-full">
                        ${this.getAttribute('button-text') || 'Search'}
                    </button>
                </div>
            `;
        }
    }

    customElements.define('search-container', SearchContainer);
}
