export class BaseModal extends HTMLElement {
    constructor() {
        super();
        this.isVisible = false;
    }

    connectedCallback() {
        this.render();
        this.setupBaseListeners();
    }

    show(type) {
        this.currentType = type;
        this.isVisible = true;
        this.style.display = 'block';
        this.render();
        this.updateVisibility();
    }

    show() {
        this.isVisible = true;
        this.style.display = 'block';
        const container = this.querySelector('.modal-container');
        if (container) {
            container.classList.remove('hidden');
            container.classList.add('flex');
        }
    }

    hide() {
        this.isVisible = false;
        this.style.display = 'none';
        const container = this.querySelector('.modal-container');
        if (container) {
            container.classList.add('hidden');
            container.classList.remove('flex');
        }
    }

    updateVisibility() {
        const container = this.querySelector('.modal-container');
        if (container) {
            container.style.display = this.isVisible ? 'flex' : 'none';
        }
    }

    setupBaseListeners() {
        // Close on backdrop click
        this.querySelector('.modal-container')?.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-container')) {
                this.hide();
            }
        });

        // Prevent propagation from modal content
        this.querySelector('.modal-content')?.addEventListener('click', (e) => {
            e.stopPropagation();
        });

        // Close button handler
        this.querySelector('.modal-close')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.hide();
        });
    }

    getFormFields() {
        switch (this.currentType) {
            case 'definition':
                return `
                    <div class="space-y-4">
                        <input type="text" placeholder="Term" class="w-full p-2 bg-gray-800 rounded">
                        <textarea placeholder="Definition" class="w-full p-2 bg-gray-800 rounded h-32"></textarea>
                        <input type="text" placeholder="Tags (comma separated)" class="w-full p-2 bg-gray-800 rounded">
                    </div>
                `;
            case 'method':
                return `
                    <div class="space-y-4">
                        <input type="text" placeholder="Method Name" class="w-full p-2 bg-gray-800 rounded">
                        <textarea placeholder="Description" class="w-full p-2 bg-gray-800 rounded h-32"></textarea>
                        <textarea placeholder="Steps (one per line)" class="w-full p-2 bg-gray-800 rounded h-32"></textarea>
                        <input type="text" placeholder="Tags (comma separated)" class="w-full p-2 bg-gray-800 rounded">
                    </div>
                `;
            // ...add other form types as needed...
        }
    }

    renderModal(content) {
        return `
            <div class="modal-container fixed inset-0 bg-black bg-opacity-50 hidden items-center justify-center z-50">
                <div class="modal-content bg-gray-900 p-8 rounded-lg w-full max-w-[70vw] max-h-[70vh] overflow-y-auto relative">
                    <button class="modal-close absolute top-4 right-4 text-gray-400 hover:text-white">
                        ✕
                    </button>
                    ${content || '<slot></slot>'}
                </div>
            </div>
        `;
    }

    render() {
        const title = this.currentType ? `Add ${this.currentType.charAt(0).toUpperCase() + this.currentType.slice(1)}` : '';
        const content = this.currentType ? `
            <h2 class="text-2xl font-bold mb-6">${title}</h2>
            <form class="space-y-6" onsubmit="return false;">
                ${this.getFormFields()}
                <div class="flex justify-end gap-4">
                    <button class="px-4 py-2 rounded bg-gray-700 hover:bg-gray-600" onclick="this.closest('base-modal').hide()">Cancel</button>
                    <button class="px-4 py-2 rounded bg-purple-600 hover:bg-purple-500">Save</button>
                </div>
            </form>
        ` : '';

        this.innerHTML = this.renderModal(content);
    }
}

// Also define the element
customElements.define('base-modal', BaseModal);
