"""
Interruptible LLM processing that can be safely cancelled during API calls.
This module provides utilities to make LLM requests interruptible with Ctrl+C.
"""

import time
import logging
import threading
import signal
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError, CancelledError
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

# Global flags for tracking interrupts
_interrupt_requested = threading.Event()
_active_requests = []
_original_sigint_handler = None

def _handle_keyboard_interrupt(sig, frame):
    """Handle keyboard interrupts gracefully by setting a flag"""
    logger.warning("\nKeyboard interrupt detected during LLM processing")
    _interrupt_requested.set()
    
    # Attempt to cancel any active requests
    for request in _active_requests:
        if hasattr(request, 'close'):
            try:
                request.close()
            except:
                pass
    
    # Log information but don't exit - let the request finish gracefully
    logger.warning("Attempting to cancel LLM processing safely...")
    
    # If we get multiple interrupts, use the original handler
    if _original_sigint_handler:
        if _interrupt_requested.is_set():
            logger.warning("Multiple interrupts detected, forcing exit...")
            _original_sigint_handler(sig, frame)

def setup_interrupt_handling():
    """Set up interrupt handling for the process"""
    global _original_sigint_handler
    # Store the original handler so we can restore it later or delegate to it
    _original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_keyboard_interrupt)
    _interrupt_requested.clear()
    logger.debug("Installed interruptible LLM signal handler")

def restore_interrupt_handling():
    """Restore the original interrupt handler"""
    global _original_sigint_handler
    if _original_sigint_handler:
        signal.signal(signal.SIGINT, _original_sigint_handler)
        logger.debug("Restored original signal handler")

def is_interrupt_requested():
    """Check if an interrupt has been requested"""
    return _interrupt_requested.is_set()

def clear_interrupt():
    """Clear the interrupt flag"""
    _interrupt_requested.clear()

def interruptible_request(method, url, **kwargs):
    """Make a request that can be interrupted with Ctrl+C"""
    # Set up custom interrupt handling
    prior_handler_installed = False
    if signal.getsignal(signal.SIGINT) != _handle_keyboard_interrupt:
        setup_interrupt_handling()
        prior_handler_installed = True
    
    _interrupt_requested.clear()
    
    # Extract timeout from kwargs and save it for the session.send() call
    timeout_val = kwargs.pop('timeout', 60.0) if 'timeout' in kwargs else 60.0
    
    try:
        # Create the request session
        session = requests.Session()
        
        # Prepare the request without timeout (timeout goes to send(), not Request.__init__)
        req = requests.Request(method, url, **kwargs).prepare()
        
        # Send the request but keep track of it
        response = None
        start_time = time.time()
        
        # Keep checking for interrupts during the request
        try:
            with session as s:
                # Add to active requests
                _active_requests.append(s)
                
                # Send the request with timeout here
                response = s.send(req, timeout=timeout_val)
                
                # Remove from active requests
                if s in _active_requests:
                    _active_requests.remove(s)
        
        finally:
            duration = time.time() - start_time
            logger.info(f"Request to {url} took {duration:.2f} seconds")
            
            # Check if interrupt was requested
            if _interrupt_requested.is_set():
                logger.warning("Request was interrupted by user")
                if response:
                    response.close()
                    
                # This will raise an exception to be caught by the caller
                raise KeyboardInterrupt("Request interrupted by user")
        
        return response
        
    finally:
        # Restore original handler if we installed ours
        if prior_handler_installed:
            restore_interrupt_handling()
        
        # Clean up active requests
        for r in _active_requests[:]:
            if r in _active_requests:
                _active_requests.remove(r)

async def async_interruptible_request(method, url, **kwargs):
    """Async version of interruptible_request"""
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            return await loop.run_in_executor(
                executor,
                lambda: interruptible_request(method, url, **kwargs)
            )
        except KeyboardInterrupt:
            logger.warning("Async request canceled due to keyboard interrupt")
            # Propagate the interrupt for proper async handling
            raise

# Convenient wrappers for common methods
def interruptible_post(url, **kwargs):
    """Make an interruptible POST request"""
    return interruptible_request('POST', url, **kwargs)

def interruptible_get(url, **kwargs):
    """Make an interruptible GET request"""
    return interruptible_request('GET', url, **kwargs)

async def async_interruptible_post(url, **kwargs):
    """Make an async interruptible POST request"""
    return await async_interruptible_request('POST', url, **kwargs)

async def async_interruptible_get(url, **kwargs):
    """Make an async interruptible GET request"""
    return await async_interruptible_request('GET', url, **kwargs)
