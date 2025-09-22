package mcp

import (
	"fmt"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/models"
)

// Registry manages MCP servers and their configurations
type Registry struct {
	servers map[string]*models.MCPServer
	mu      sync.RWMutex
}

// NewRegistry creates a new MCP server registry
func NewRegistry() *Registry {
	return &Registry{
		servers: make(map[string]*models.MCPServer),
	}
}

// RegisterServer registers a new MCP server
func (r *Registry) RegisterServer(server *models.MCPServer) error {
	if server == nil {
		return fmt.Errorf("server cannot be nil")
	}

	if server.ID == "" {
		return fmt.Errorf("server ID is required")
	}

	if server.Name == "" {
		return fmt.Errorf("server name is required")
	}

	if server.Command == "" {
		return fmt.Errorf("server command is required")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	// Set defaults
	if server.Transport.Type == "" {
		server.Transport.Type = "stdio"
	}

	if server.Transport.Timeout == "" {
		server.Transport.Timeout = "30s"
	}

	if server.Status == "" {
		server.Status = "registered"
	}

	r.servers[server.ID] = server
	return nil
}

// UnregisterServer removes a server from the registry
func (r *Registry) UnregisterServer(serverID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.servers[serverID]; !exists {
		return fmt.Errorf("server %s not found", serverID)
	}

	delete(r.servers, serverID)
	return nil
}

// GetServer returns a server by ID
func (r *Registry) GetServer(serverID string) (*models.MCPServer, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	server, exists := r.servers[serverID]
	if !exists {
		return nil, fmt.Errorf("server %s not found", serverID)
	}

	// Return a copy to prevent external modification
	serverCopy := *server
	return &serverCopy, nil
}

// UpdateServer updates an existing server
func (r *Registry) UpdateServer(server *models.MCPServer) error {
	if server == nil || server.ID == "" {
		return fmt.Errorf("invalid server")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.servers[server.ID]; !exists {
		return fmt.Errorf("server %s not found", server.ID)
	}

	r.servers[server.ID] = server
	return nil
}

// ListServers returns all registered servers
func (r *Registry) ListServers() []*models.MCPServer {
	r.mu.RLock()
	defer r.mu.RUnlock()

	servers := make([]*models.MCPServer, 0, len(r.servers))
	for _, server := range r.servers {
		serverCopy := *server
		servers = append(servers, &serverCopy)
	}

	return servers
}

// GetEnabledServers returns only enabled servers
func (r *Registry) GetEnabledServers() []*models.MCPServer {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var enabled []*models.MCPServer
	for _, server := range r.servers {
		if server.Enabled {
			serverCopy := *server
			enabled = append(enabled, &serverCopy)
		}
	}

	return enabled
}

// GetServersByCapability returns servers that support a specific capability
func (r *Registry) GetServersByCapability(capability string) []*models.MCPServer {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var matching []*models.MCPServer
	for _, server := range r.servers {
		if server.Enabled && r.hasCapability(server, capability) {
			serverCopy := *server
			matching = append(matching, &serverCopy)
		}
	}

	return matching
}

// EnableServer enables a server
func (r *Registry) EnableServer(serverID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	server, exists := r.servers[serverID]
	if !exists {
		return fmt.Errorf("server %s not found", serverID)
	}

	server.Enabled = true
	return nil
}

// DisableServer disables a server
func (r *Registry) DisableServer(serverID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	server, exists := r.servers[serverID]
	if !exists {
		return fmt.Errorf("server %s not found", serverID)
	}

	server.Enabled = false
	server.Status = "disabled"
	return nil
}

// UpdateServerStatus updates the status of a server
func (r *Registry) UpdateServerStatus(serverID, status string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	server, exists := r.servers[serverID]
	if !exists {
		return fmt.Errorf("server %s not found", serverID)
	}

	server.Status = status
	if status == "connected" {
		now := time.Now()
		server.LastStarted = &now
	}

	return nil
}

// UpdateHealthCheck updates the last health check time
func (r *Registry) UpdateHealthCheck(serverID string, healthy bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	server, exists := r.servers[serverID]
	if !exists {
		return fmt.Errorf("server %s not found", serverID)
	}

	now := time.Now()
	server.LastHealthCheck = &now

	if !healthy {
		server.Status = "unhealthy"
	} else if server.Status == "unhealthy" {
		server.Status = "connected"
	}

	return nil
}

// GetServerCount returns the total number of registered servers
func (r *Registry) GetServerCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return len(r.servers)
}

// GetEnabledServerCount returns the number of enabled servers
func (r *Registry) GetEnabledServerCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	count := 0
	for _, server := range r.servers {
		if server.Enabled {
			count++
		}
	}

	return count
}

// GetConnectedServerCount returns the number of connected servers
func (r *Registry) GetConnectedServerCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	count := 0
	for _, server := range r.servers {
		if server.Status == "connected" {
			count++
		}
	}

	return count
}

// LoadFromConfig loads servers from configuration
func (r *Registry) LoadFromConfig(config map[string]interface{}) error {
	servers, ok := config["servers"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid servers configuration")
	}

	for serverID, serverConfig := range servers {
		serverMap, ok := serverConfig.(map[string]interface{})
		if !ok {
			continue
		}

		server := &models.MCPServer{
			ID:         serverID,
			Name:       getStringFromMap(serverMap, "name"),
			Command:    getStringFromMap(serverMap, "command"),
			Args:       getStringSliceFromMap(serverMap, "args"),
			Env:        getStringSliceFromMap(serverMap, "env"),
			WorkingDir: getStringFromMap(serverMap, "working_directory"),
			Enabled:    getBoolFromMap(serverMap, "enabled"),
			Transport: models.TransportConfig{
				Type:    getStringFromMap(serverMap, "transport.type"),
				Timeout: getStringFromMap(serverMap, "transport.timeout"),
			},
		}

		if server.Transport.Type == "" {
			server.Transport.Type = "stdio"
		}

		if err := r.RegisterServer(server); err != nil {
			return fmt.Errorf("failed to register server %s: %v", serverID, err)
		}
	}

	return nil
}

// GetRegistryStats returns statistics about the registry
func (r *Registry) GetRegistryStats() RegistryStats {
	r.mu.RLock()
	defer r.mu.RUnlock()

	stats := RegistryStats{
		TotalServers:     len(r.servers),
		EnabledServers:   0,
		ConnectedServers: 0,
		ServersByType:    make(map[string]int),
		ServersByStatus:  make(map[string]int),
	}

	for _, server := range r.servers {
		if server.Enabled {
			stats.EnabledServers++
		}

		if server.Status == "connected" {
			stats.ConnectedServers++
		}

		stats.ServersByType[server.Transport.Type]++
		stats.ServersByStatus[server.Status]++
	}

	return stats
}

// hasCapability checks if a server has a specific capability
func (r *Registry) hasCapability(server *models.MCPServer, capability string) bool {
	for _, cap := range server.Capabilities {
		if cap == capability {
			return true
		}
	}
	return false
}

// Helper functions for config parsing
func getStringFromMap(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func getStringSliceFromMap(m map[string]interface{}, key string) []string {
	if v, ok := m[key].([]interface{}); ok {
		result := make([]string, len(v))
		for i, item := range v {
			if s, ok := item.(string); ok {
				result[i] = s
			}
		}
		return result
	}
	return nil
}

func getBoolFromMap(m map[string]interface{}, key string) bool {
	if v, ok := m[key].(bool); ok {
		return v
	}
	return false
}

// RegistryStats represents statistics about the registry
type RegistryStats struct {
	TotalServers     int            `json:"total_servers"`
	EnabledServers   int            `json:"enabled_servers"`
	ConnectedServers int            `json:"connected_servers"`
	ServersByType    map[string]int `json:"servers_by_type"`
	ServersByStatus  map[string]int `json:"servers_by_status"`
}
