# OATFLAKE Frontend

This directory contains the frontend assets for the OATFLAKE project, including styles, scripts, and icons.

## Structure

```
static/
├── css/               # Stylesheet files
├── js/                # JavaScript files and modules
│   ├── components/    # Reusable UI components
│   ├── modals/        # Modal dialog implementations
│   ├── slides/        # Slide-based UI components
│   └── widgets/       # Interactive widget implementations
└── icons/             # Icon assets
```

## Frontend Setup

### Tailwind CSS

This project uses [Tailwind CSS](https://tailwindcss.com/) for styling. Tailwind is a utility-first CSS framework that allows for rapid UI development through composable utility classes.

- The main styling is applied through Tailwind utility classes in the HTML
- Custom styles beyond Tailwind are defined in `css/main.css`
- Additional specialized styles are in separate CSS files like `search-styles.css`

### JavaScript Architecture

The JavaScript code follows a modular architecture:
- `app.js`: Main application entry point
- Component-based structure for better organization and reusability
- Event-driven communication between components

## Widget Functionality

The application provides several interactive widgets located in the `js/widgets/` directory:

### Data Widget (`data-widget.js`)
Handles data visualization and manipulation, offering users interactive ways to explore and work with datasets.

### Group Widget (`group-widget.js`)
Manages group-based functionality, allowing users to create, join, and interact within group contexts.

### Ollama Widget (`ollama-widget.js`)
Integrates with Ollama, providing an interface for local LLM model interactions.

### OpenRouter Widget (`openrouter-widget.js`)
Connects to OpenRouter API, enabling access to various AI models through a unified interface.

### Search Widget (`search-widget.js`)
Provides advanced search capabilities throughout the application.

### Slack Widget (`slack-widget.js`)
Enables integration with Slack, allowing for notifications and data sharing.

### Task Widget (`task-widget.js`)
Manages task creation, assignment, tracking, and completion workflows.

### Tunnel Widget (`tunnel-widget.js`)
Handles networking tunnel functionality, likely for exposing local services.

## Getting Started

To work with the frontend:

1. Ensure you have Node.js and npm installed
2. Install dependencies (if using npm):
   ```bash
   npm install
   ```
3. For Tailwind CSS development, run:
   ```bash
   npm run build:css
   ```
   or for watch mode during development:
   ```bash
   npm run watch:css
   ```

## Development Guidelines

When modifying or creating new frontend components:

1. Follow the existing component pattern
2. Utilize Tailwind CSS utility classes where possible
3. Add custom CSS only when necessary
4. Keep widget functionality modular and focused on a single responsibility
5. Ensure responsive design for all screen sizes

## Widget Development

To create a new widget:

1. Create a new file in the `js/widgets/` directory
2. Follow the widget pattern with initialization, event handling, and DOM manipulation
3. Register the widget in the main application flow
4. Add any specific styles to the appropriate CSS file
