package mcp

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/yourusername/ai-code-assistant/models"
)

// Client represents an MCP client that communicates with MCP servers
type Client struct {
	registry   *Registry
	transports map[string]Transport
	mu         sync.RWMutex
}

// Transport defines the interface for MCP communication transports
type Transport interface {
	Connect(ctx context.Context, config TransportConfig) error
	Send(ctx context.Context, message *MCPMessage) error
	Receive(ctx context.Context) (*MCPMessage, error)
	Close() error
	IsConnected() bool
}

// TransportConfig contains transport-specific configuration
type TransportConfig struct {
	Type       string                 `json:"type"`
	Address    string                 `json:"address,omitempty"`
	Port       int                    `json:"port,omitempty"`
	Command    string                 `json:"command,omitempty"`
	Args       []string               `json:"args,omitempty"`
	Env        []string               `json:"env,omitempty"`
	WorkingDir string                 `json:"working_dir,omitempty"`
	Timeout    time.Duration          `json:"timeout"`
	Options    map[string]interface{} `json:"options,omitempty"`
}

// MCPMessage represents a message in the MCP protocol
type MCPMessage struct {
	ID     string                 `json:"id"`
	Type   string                 `json:"type"`
	Method string                 `json:"method,omitempty"`
	Params map[string]interface{} `json:"params,omitempty"`
	Result interface{}            `json:"result,omitempty"`
	Error  *MCPError              `json:"error,omitempty"`
}

// MCPError represents an error in the MCP protocol
type MCPError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// NewClient creates a new MCP client
func NewClient(registry *Registry) *Client {
	return &Client{
		registry:   registry,
		transports: make(map[string]Transport),
	}
}

// ConnectToServer establishes connection to an MCP server
func (c *Client) ConnectToServer(ctx context.Context, serverID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	server, err := c.registry.GetServer(serverID)
	if err != nil {
		return fmt.Errorf("failed to get server %s: %v", serverID, err)
	}

	if !server.Enabled {
		return fmt.Errorf("server %s is not enabled", serverID)
	}

	transport, err := c.createTransport(server.Transport.Type)
	if err != nil {
		return fmt.Errorf("failed to create transport: %v", err)
	}

	config := TransportConfig{
		Type:       server.Transport.Type,
		Command:    server.Command,
		Args:       server.Args,
		Env:        server.Env,
		WorkingDir: server.WorkingDir,
		Timeout:    parseDuration(server.Transport.Timeout),
	}

	if err := transport.Connect(ctx, config); err != nil {
		return fmt.Errorf("failed to connect transport: %v", err)
	}

	c.transports[serverID] = transport

	// Perform handshake
	if err := c.performHandshake(ctx, serverID); err != nil {
		transport.Close()
		delete(c.transports, serverID)
		return fmt.Errorf("handshake failed: %v", err)
	}

	// Update server status
	server.Status = "connected"
	server.LastStarted = timePtr(time.Now())
	c.registry.UpdateServer(server)

	return nil
}

// DisconnectFromServer disconnects from an MCP server
func (c *Client) DisconnectFromServer(serverID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	transport, exists := c.transports[serverID]
	if !exists {
		return fmt.Errorf("not connected to server %s", serverID)
	}

	if err := transport.Close(); err != nil {
		return fmt.Errorf("failed to close transport: %v", err)
	}

	delete(c.transports, serverID)

	// Update server status
	if server, err := c.registry.GetServer(serverID); err == nil {
		server.Status = "disconnected"
		c.registry.UpdateServer(server)
	}

	return nil
}

// ExecuteTool executes a tool on the specified MCP server
func (c *Client) ExecuteTool(ctx context.Context, serverID string, toolName string, input map[string]interface{}) (*models.MCPToolExecution, error) {
	c.mu.RLock()
	transport, exists := c.transports[serverID]
	c.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("not connected to server %s", serverID)
	}

	start := time.Now()
	executionID := uuid.New().String()

	// Create tool execution request
	request := &MCPMessage{
		ID:     uuid.New().String(),
		Type:   "request",
		Method: "tools/call",
		Params: map[string]interface{}{
			"name":      toolName,
			"arguments": input,
		},
	}

	// Send request
	if err := transport.Send(ctx, request); err != nil {
		return &models.MCPToolExecution{
			ID:        executionID,
			ToolName:  toolName,
			ServerID:  serverID,
			Input:     input,
			Success:   false,
			Error:     fmt.Sprintf("failed to send request: %v", err),
			Duration:  time.Since(start),
			Timestamp: time.Now(),
		}, err
	}

	// Receive response
	response, err := transport.Receive(ctx)
	if err != nil {
		return &models.MCPToolExecution{
			ID:        executionID,
			ToolName:  toolName,
			ServerID:  serverID,
			Input:     input,
			Success:   false,
			Error:     fmt.Sprintf("failed to receive response: %v", err),
			Duration:  time.Since(start),
			Timestamp: time.Now(),
		}, err
	}

	execution := &models.MCPToolExecution{
		ID:        executionID,
		ToolName:  toolName,
		ServerID:  serverID,
		Input:     input,
		Duration:  time.Since(start),
		Timestamp: time.Now(),
	}

	if response.Error != nil {
		execution.Success = false
		execution.Error = response.Error.Message
	} else {
		execution.Success = true
		execution.Output = response.Result
	}

	return execution, nil
}

// ListTools returns all available tools from connected servers
func (c *Client) ListTools(ctx context.Context) ([]models.MCPTool, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var allTools []models.MCPTool

	for serverID, transport := range c.transports {
		if !transport.IsConnected() {
			continue
		}

		request := &MCPMessage{
			ID:     uuid.New().String(),
			Type:   "request",
			Method: "tools/list",
		}

		if err := transport.Send(ctx, request); err != nil {
			continue // Skip this server on error
		}

		response, err := transport.Receive(ctx)
		if err != nil || response.Error != nil {
			continue // Skip this server on error
		}

		// Parse tools from response
		if tools, ok := response.Result.(map[string]interface{})["tools"].([]interface{}); ok {
			for _, toolData := range tools {
				if toolMap, ok := toolData.(map[string]interface{}); ok {
					tool := models.MCPTool{
						ServerID:    serverID,
						Name:        getString(toolMap, "name"),
						Description: getString(toolMap, "description"),
					}

					if schema, ok := toolMap["inputSchema"].(map[string]interface{}); ok {
						tool.InputSchema = schema
					}

					allTools = append(allTools, tool)
				}
			}
		}
	}

	return allTools, nil
}

// GetResources returns available resources from connected servers
func (c *Client) GetResources(ctx context.Context) ([]models.MCPResource, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var allResources []models.MCPResource

	for serverID, transport := range c.transports {
		if !transport.IsConnected() {
			continue
		}

		request := &MCPMessage{
			ID:     uuid.New().String(),
			Type:   "request",
			Method: "resources/list",
		}

		if err := transport.Send(ctx, request); err != nil {
			continue
		}

		response, err := transport.Receive(ctx)
		if err != nil || response.Error != nil {
			continue
		}

		// Parse resources from response
		if resources, ok := response.Result.(map[string]interface{})["resources"].([]interface{}); ok {
			for _, resourceData := range resources {
				if resourceMap, ok := resourceData.(map[string]interface{}); ok {
					resource := models.MCPResource{
						ServerID:    serverID,
						URI:         getString(resourceMap, "uri"),
						Name:        getString(resourceMap, "name"),
						Description: getString(resourceMap, "description"),
						MimeType:    getString(resourceMap, "mimeType"),
					}

					if metadata, ok := resourceMap["metadata"].(map[string]interface{}); ok {
						resource.Metadata = metadata
					}

					allResources = append(allResources, resource)
				}
			}
		}
	}

	return allResources, nil
}

// IsConnected checks if client is connected to a server
func (c *Client) IsConnected(serverID string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	transport, exists := c.transports[serverID]
	return exists && transport.IsConnected()
}

// GetConnectedServers returns list of connected server IDs
func (c *Client) GetConnectedServers() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var connected []string
	for serverID, transport := range c.transports {
		if transport.IsConnected() {
			connected = append(connected, serverID)
		}
	}

	return connected
}

// performHandshake performs the MCP handshake with a server
func (c *Client) performHandshake(ctx context.Context, serverID string) error {
	transport := c.transports[serverID]

	// Send initialize request
	request := &MCPMessage{
		ID:     uuid.New().String(),
		Type:   "request",
		Method: "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities": map[string]interface{}{
				"roots": map[string]interface{}{
					"listChanged": true,
				},
				"sampling": map[string]interface{}{},
			},
			"clientInfo": map[string]interface{}{
				"name":    "AI Code Assistant",
				"version": "1.0.0",
			},
		},
	}

	if err := transport.Send(ctx, request); err != nil {
		return err
	}

	response, err := transport.Receive(ctx)
	if err != nil {
		return err
	}

	if response.Error != nil {
		return fmt.Errorf("initialization failed: %s", response.Error.Message)
	}

	// Send initialized notification
	notification := &MCPMessage{
		ID:     uuid.New().String(),
		Type:   "notification",
		Method: "notifications/initialized",
	}

	return transport.Send(ctx, notification)
}

// createTransport creates a transport based on type
func (c *Client) createTransport(transportType string) (Transport, error) {
	switch transportType {
	case "stdio":
		return NewStdioTransport(), nil
	case "http":
		return NewHTTPTransport(), nil
	case "websocket":
		return NewWebSocketTransport(), nil
	default:
		return nil, fmt.Errorf("unsupported transport type: %s", transportType)
	}
}

// Helper functions
func parseDuration(s string) time.Duration {
	if d, err := time.ParseDuration(s); err == nil {
		return d
	}
	return 30 * time.Second
}

func timePtr(t time.Time) *time.Time {
	return &t
}

func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}
