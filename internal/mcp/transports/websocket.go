package transports

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/yourusername/ai-code-assistant/internal/mcp"
)

// WebSocketTransport handles communication over WebSocket
type WebSocketTransport struct {
	conn        *websocket.Conn
	connected   bool
	mu          sync.RWMutex
	messageChan chan *mcp.MCPMessage
	errorChan   chan error
	stopChan    chan struct{}
	url         string
}

// NewWebSocketTransport creates a new WebSocket transport
func NewWebSocketTransport() *WebSocketTransport {
	return &WebSocketTransport{
		messageChan: make(chan *mcp.MCPMessage, 100),
		errorChan:   make(chan error, 10),
		stopChan:    make(chan struct{}),
	}
}

// Connect establishes WebSocket connection
func (t *WebSocketTransport) Connect(ctx context.Context, config mcp.TransportConfig) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.connected {
		return fmt.Errorf("already connected")
	}

	// Build WebSocket URL
	if config.Address == "" {
		return fmt.Errorf("address is required for WebSocket transport")
	}

	port := config.Port
	if port == 0 {
		port = 8080 // Default WebSocket port
	}

	wsURL := fmt.Sprintf("ws://%s:%d/mcp", config.Address, port)
	t.url = wsURL

	// Parse URL
	u, err := url.Parse(wsURL)
	if err != nil {
		return fmt.Errorf("invalid WebSocket URL: %v", err)
	}

	// Setup dialer with timeout
	dialer := websocket.DefaultDialer
	if config.Timeout > 0 {
		dialer.HandshakeTimeout = config.Timeout
	}

	// Add custom headers if specified
	headers := make(map[string][]string)
	if headerMap, ok := config.Options["headers"].(map[string]interface{}); ok {
		for key, value := range headerMap {
			if strValue, ok := value.(string); ok {
				headers[key] = []string{strValue}
			}
		}
	}

	// Connect to WebSocket
	conn, _, err := dialer.DialContext(ctx, u.String(), headers)
	if err != nil {
		return fmt.Errorf("WebSocket connection failed: %v", err)
	}

	t.conn = conn
	t.connected = true

	// Start reading goroutine
	go t.readMessages()

	// Setup ping/pong for connection health
	go t.pingPong()

	return nil
}

// Send sends a message via WebSocket
func (t *WebSocketTransport) Send(ctx context.Context, message *mcp.MCPMessage) error {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if !t.connected {
		return fmt.Errorf("not connected")
	}

	data, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %v", err)
	}

	// Set write deadline
	if deadline, ok := ctx.Deadline(); ok {
		t.conn.SetWriteDeadline(deadline)
	}

	err = t.conn.WriteMessage(websocket.TextMessage, data)
	if err != nil {
		return fmt.Errorf("failed to write WebSocket message: %v", err)
	}

	return nil
}

// Receive receives a message via WebSocket
func (t *WebSocketTransport) Receive(ctx context.Context) (*mcp.MCPMessage, error) {
	select {
	case message := <-t.messageChan:
		return message, nil
	case err := <-t.errorChan:
		return nil, err
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-t.stopChan:
		return nil, fmt.Errorf("transport stopped")
	}
}

// Close closes the WebSocket transport
func (t *WebSocketTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.connected {
		return nil
	}

	t.connected = false
	close(t.stopChan)

	if t.conn != nil {
		// Send close frame
		t.conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))

		// Close connection
		t.conn.Close()
	}

	return nil
}

// IsConnected returns whether the transport is connected
func (t *WebSocketTransport) IsConnected() bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.connected
}

// readMessages reads messages from WebSocket
func (t *WebSocketTransport) readMessages() {
	defer func() {
		if r := recover(); r != nil {
			t.errorChan <- fmt.Errorf("panic in readMessages: %v", r)
		}
	}()

	for {
		select {
		case <-t.stopChan:
			return
		default:
		}

		messageType, data, err := t.conn.ReadMessage()
		if err != nil {
			t.errorChan <- fmt.Errorf("WebSocket read error: %v", err)
			return
		}

		if messageType == websocket.CloseMessage {
			return
		}

		if messageType != websocket.TextMessage {
			continue // Skip non-text messages
		}

		var message mcp.MCPMessage
		if err := json.Unmarshal(data, &message); err != nil {
			t.errorChan <- fmt.Errorf("failed to unmarshal message: %v", err)
			continue
		}

		select {
		case t.messageChan <- &message:
		case <-t.stopChan:
			return
		default:
			// Channel is full, skip message
		}
	}
}

// pingPong handles WebSocket ping/pong for connection health
func (t *WebSocketTransport) pingPong() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	t.conn.SetPongHandler(func(appData string) error {
		return nil
	})

	for {
		select {
		case <-ticker.C:
			t.mu.RLock()
			connected := t.connected
			conn := t.conn
			t.mu.RUnlock()

			if !connected || conn == nil {
				return
			}

			if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				t.errorChan <- fmt.Errorf("ping failed: %v", err)
				return
			}

		case <-t.stopChan:
			return
		}
	}
}
