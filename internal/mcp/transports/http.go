package transports

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/mcp"
)

// HTTPTransport handles communication over HTTP
type HTTPTransport struct {
	baseURL   string
	client    *http.Client
	connected bool
	mu        sync.RWMutex
	headers   map[string]string
}

// NewHTTPTransport creates a new HTTP transport
func NewHTTPTransport() *HTTPTransport {
	return &HTTPTransport{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		headers: make(map[string]string),
	}
}

// Connect establishes HTTP connection
func (t *HTTPTransport) Connect(ctx context.Context, config mcp.TransportConfig) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.connected {
		return fmt.Errorf("already connected")
	}

	// Build base URL
	if config.Address == "" {
		return fmt.Errorf("address is required for HTTP transport")
	}

	port := config.Port
	if port == 0 {
		port = 8080 // Default HTTP port
	}

	t.baseURL = fmt.Sprintf("http://%s:%d", config.Address, port)

	// Set default headers
	t.headers["Content-Type"] = "application/json"
	t.headers["Accept"] = "application/json"

	// Add custom headers from options
	if headers, ok := config.Options["headers"].(map[string]interface{}); ok {
		for key, value := range headers {
			if strValue, ok := value.(string); ok {
				t.headers[key] = strValue
			}
		}
	}

	// Set timeout from config
	if config.Timeout > 0 {
		t.client.Timeout = config.Timeout
	}

	// Test connection with a ping
	if err := t.ping(ctx); err != nil {
		return fmt.Errorf("connection test failed: %v", err)
	}

	t.connected = true
	return nil
}

// Send sends a message via HTTP POST
func (t *HTTPTransport) Send(ctx context.Context, message *mcp.MCPMessage) error {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if !t.connected {
		return fmt.Errorf("not connected")
	}

	data, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %v", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", t.baseURL+"/mcp", bytes.NewBuffer(data))
	if err != nil {
		return fmt.Errorf("failed to create request: %v", err)
	}

	// Add headers
	for key, value := range t.headers {
		req.Header.Set(key, value)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return fmt.Errorf("HTTP request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP request failed with status: %s", resp.Status)
	}

	return nil
}

// Receive receives a message via HTTP (for request-response pattern)
func (t *HTTPTransport) Receive(ctx context.Context) (*mcp.MCPMessage, error) {
	// For HTTP transport, we implement a polling mechanism or server-sent events
	// This is a simplified version that polls for messages
	return t.pollForMessage(ctx)
}

// Close closes the HTTP transport
func (t *HTTPTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.connected = false
	return nil
}

// IsConnected returns whether the transport is connected
func (t *HTTPTransport) IsConnected() bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.connected
}

// ping tests the HTTP connection
func (t *HTTPTransport) ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", t.baseURL+"/health", nil)
	if err != nil {
		return err
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %s", resp.Status)
	}

	return nil
}

// pollForMessage polls for messages (simplified implementation)
func (t *HTTPTransport) pollForMessage(ctx context.Context) (*mcp.MCPMessage, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", t.baseURL+"/mcp/poll", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create poll request: %v", err)
	}

	// Add headers
	for key, value := range t.headers {
		req.Header.Set(key, value)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("poll request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNoContent {
		// No messages available
		return nil, fmt.Errorf("no messages available")
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("poll request failed with status: %s", resp.Status)
	}

	var message mcp.MCPMessage
	if err := json.NewDecoder(resp.Body).Decode(&message); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	return &message, nil
}
