/**
 * Example frontend code for handling the two-phase response mechanism
 * This allows the frontend to show that it's still waiting for the LLM response
 * even after the initial request returns
 */

// Function to send a query to the API and handle the two-phase response
async function sendQueryWithPolling(query) {
  // Show loading state
  setLoading(true);
  updateLoadingMessage("Starting query processing...");
  
  try {
    // Phase 1: Initial request
    const initialResponse = await fetch('/api/web', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: query
      }),
    }).then(res => res.json());
    
    // Check for errors in the initial response
    if (initialResponse.status === "error") {
      console.error("Error from server:", initialResponse.error);
      setLoading(false);
      return {
        text: `Error: ${initialResponse.error || initialResponse.message || "Unknown error occurred"}`,
        metadata: null
      };
    }
    
    // Check if we got a phase 1 response with request_id
    if (initialResponse.status === "processing" && initialResponse.request_id) {
      // We need to poll for the complete response
      const requestId = initialResponse.request_id;
      updateLoadingMessage("Retrieving context and generating response...");
      
      // Start polling
      return await pollForCompletion(requestId, query);
    } else {
      // We somehow got an immediate complete response
      setLoading(false);
      return {
        text: initialResponse.response || "No response received from server",
        metadata: null
      };
    }
  } catch (error) {
    console.error('Error sending query:', error);
    setLoading(false);
    return {
      text: `Error: ${error.message}`,
      metadata: null
    };
  }
}

// Function to poll for completion of a response
async function pollForCompletion(requestId, originalQuery, maxAttempts = 30) {
  // Poll with delay between attempts
  let attempts = 0;
  const startTime = new Date();
  
  while (attempts < maxAttempts) {
    attempts++;
    
    try {
      // Wait before polling again (start with shorter interval and gradually increase)
      await new Promise(resolve => setTimeout(resolve, 
        Math.min(1000 + (attempts * 500), 5000))); // Start with 1.5s, max 5s
      
      // Update loading message with attempt count
      updateLoadingMessage(`Generating response... (${attempts}s)`);
      
      // Phase 2: Check if response is ready
      const response = await fetch('/api/web', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          phase: 2,
          request_id: requestId,
          prompt: originalQuery // Include original query for convenience
        }),
      }).then(res => res.json());
      
      // Check for errors
      if (response.status === "error") {
        console.error("Error from server during polling:", response.error);
        setLoading(false);
        return {
          text: `Error: ${response.error || "Unknown error occurred"}`,
          metadata: null
        };
      }
      
      // Check if we have a complete response
      if (response.complete === true) {
        // We got the final response!
        setLoading(false);
        const endTime = new Date();
        
        // Extract metadata about model and timing
        const metadata = {
          model: response.model_info || { provider: 'unknown', model_name: 'unknown' },
          timing: response.timing || { total_seconds: 0, retrieval_seconds: 0, generation_seconds: 0 },
          word_count: response.word_count || 0,
          timestamps: {
            start: startTime,
            end: endTime,
            duration_seconds: (endTime - startTime) / 1000
          }
        };
        
        console.log("Response received with metadata:", metadata);
        
        return {
          text: response.response || "Empty response received from server",
          metadata: metadata
        };
      }
      
      // If response indicates still processing, continue polling
      if (response.status === "processing") {
        // Keep polling
        console.log(`Still processing response... (attempt ${attempts})`);
        continue;
      }
      
    } catch (error) {
      console.error(`Polling error (attempt ${attempts}):`, error);
      // Continue polling despite errors, but log them
    }
  }
  
  // If we reach max attempts, give up
  setLoading(false);
  return {
    text: "Sorry, response generation is taking too long. Please try again later.",
    metadata: null
  };
}

// Optional function to update the loading message shown to the user
function updateLoadingMessage(message) {
  const loadingEl = document.getElementById('loading-message');
  if (loadingEl) {
    loadingEl.textContent = message;
  }
}

// Usage example with improved response handling:
async function handleUserQuery(query) {
  // Update UI to show loading state with initial message
  setLoading(true);
  updateLoadingMessage("Processing your query...");
  
  const result = await sendQueryWithPolling(query);
  
  // Update UI with response text and hide loading indicator
  setLoading(false);
  
  // Display the text response
  displayResponse(result.text);
  
  // If we have metadata, display it as well
  if (result.metadata) {
    displayResponseMetadata(result.metadata);
  }
}

// Function to set loading state in the UI
function setLoading(isLoading) {
  const loadingEl = document.getElementById('loading-indicator');
  if (loadingEl) {
    loadingEl.style.display = isLoading ? 'block' : 'none';
  }
}

// Function to display the response in the UI
function displayResponse(text) {
  const responseEl = document.getElementById('response-container');
  if (responseEl) {
    responseEl.textContent = text;
  }
}

// New function to display metadata about the response
function displayResponseMetadata(metadata) {
  // Create or get the metadata container
  let metadataEl = document.getElementById('response-metadata');
  if (!metadataEl) {
    metadataEl = document.createElement('div');
    metadataEl.id = 'response-metadata';
    metadataEl.className = 'response-metadata';
    
    // Find appropriate place to insert it (after the response container)
    const responseEl = document.getElementById('response-container');
    if (responseEl && responseEl.parentNode) {
      responseEl.parentNode.insertBefore(metadataEl, responseEl.nextSibling);
    } else {
      // Fallback to appending to body
      document.body.appendChild(metadataEl);
    }
  }
  
  // Format the timing information
  const totalTime = metadata.timing.total_seconds.toFixed(2);
  const retrievalTime = metadata.timing.retrieval_seconds.toFixed(2);
  const generationTime = metadata.timing.generation_seconds.toFixed(2);
  const clientTime = metadata.timestamps.duration_seconds.toFixed(2);
  
  // Format the provider/model information
  const provider = metadata.model.provider.charAt(0).toUpperCase() + metadata.model.provider.slice(1);
  const modelName = metadata.model.model_name;
  
  // Format timestamps
  const startTime = metadata.timestamps.start.toLocaleTimeString();
  const endTime = metadata.timestamps.end.toLocaleTimeString();
  
  // Create the HTML content
  metadataEl.innerHTML = `
    <div class="metadata-item">
      <span class="metadata-label">Model:</span>
      <span class="metadata-value">${provider} / ${modelName}</span>
    </div>
    <div class="metadata-item">
      <span class="metadata-label">Server time:</span>
      <span class="metadata-value">${totalTime}s</span>
    </div>
    <div class="metadata-item">
      <span class="metadata-label">Client time:</span>
      <span class="metadata-value">${clientTime}s</span>
    </div>
    <div class="metadata-item">
      <span class="metadata-label">Retrieval:</span>
      <span class="metadata-value">${retrievalTime}s</span>
    </div>
    <div class="metadata-item">
      <span class="metadata-label">Generation:</span>
      <span class="metadata-value">${generationTime}s</span>
    </div>
    <div class="metadata-item">
      <span class="metadata-label">Words:</span>
      <span class="metadata-value">${metadata.word_count}</span>
    </div>
    <div class="metadata-item timestamps">
      <span class="metadata-label">Start:</span>
      <span class="metadata-value">${startTime}</span>
      <span class="metadata-label">End:</span>
      <span class="metadata-value">${endTime}</span>
    </div>
  `;
}

// CSS for styling the metadata display (can be added to your stylesheet)
function addMetadataStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .response-metadata {
      margin-top: 10px;
      border-top: 1px solid #eee;
      padding-top: 8px;
      font-size: 12px;
      color: #666;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    
    .metadata-item {
      display: inline-block;
      margin-right: 15px;
    }
    
    .metadata-label {
      font-weight: bold;
      margin-right: 5px;
    }
    
    .metadata-value {
      font-family: monospace;
    }
    
    .timestamps {
      display: flex;
      gap: 10px;
      width: 100%;
      margin-top: 5px;
      padding-top: 5px;
      border-top: 1px dotted #eee;
    }
  `;
  document.head.appendChild(style);
}

// Call this when the page loads to add the necessary styles
document.addEventListener('DOMContentLoaded', function() {
  addMetadataStyles();
});
