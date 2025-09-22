package models

import (
	"time"
)

// MCPServer represents an MCP server configuration
type MCPServer struct {
	ID          string                 `json:"id" db:"id"`
	Name        string                 `json:"name" db:"name"`
	Description string                 `json:"description" db:"description"`
	Version     string                 `json:"version" db:"version"`
	Command     []string               `json:"command" db:"command"`
	Args        []string               `json:"args" db:"args"`
	Transport   string                 `json:"transport" db:"transport"` // stdio, http, websocket
	Enabled     bool                   `json:"enabled" db:"enabled"`
	AutoInstall bool                   `json:"auto_install" db:"auto_install"`
	AutoRestart bool                   `json:"auto_restart" db:"auto_restart"`
	Timeout     int                    `json:"timeout" db:"timeout"` // seconds
	Environment map[string]string      `json:"environment" db:"environment"`
	Status      string                 `json:"status" db:"status"` // stopped, starting, running, error
	LastPing    *time.Time             `json:"last_ping,omitempty" db:"last_ping"`
	ErrorCount  int                    `json:"error_count" db:"error_count"`
	Metadata    map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt   time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" db:"updated_at"`
}

// MCPTool represents an available MCP tool
type MCPTool struct {
	ID           string                 `json:"id" db:"id"`
	ServerID     string                 `json:"server_id" db:"server_id"`
	Name         string                 `json:"name" db:"name"`
	Description  string                 `json:"description" db:"description"`
	InputSchema  map[string]interface{} `json:"input_schema" db:"input_schema"`
	OutputSchema map[string]interface{} `json:"output_schema,omitempty" db:"output_schema"`
	Category     string                 `json:"category" db:"category"` // filesystem, git, database, etc.
	UsageCount   int64                  `json:"usage_count" db:"usage_count"`
	LastUsed     *time.Time             `json:"last_used,omitempty" db:"last_used"`
	Enabled      bool                   `json:"enabled" db:"enabled"`
	Metadata     map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt    time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at" db:"updated_at"`
}

// MCPResource represents an available MCP resource
type MCPResource struct {
	ID           string                 `json:"id" db:"id"`
	ServerID     string                 `json:"server_id" db:"server_id"`
	URI          string                 `json:"uri" db:"uri"`
	Name         string                 `json:"name" db:"name"`
	Description  string                 `json:"description" db:"description"`
	MimeType     string                 `json:"mime_type" db:"mime_type"`
	Category     string                 `json:"category" db:"category"`
	AccessCount  int64                  `json:"access_count" db:"access_count"`
	LastAccessed *time.Time             `json:"last_accessed,omitempty" db:"last_accessed"`
	Enabled      bool                   `json:"enabled" db:"enabled"`
	Metadata     map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt    time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at" db:"updated_at"`
}

// MCPMessage represents an MCP protocol message
type MCPMessage struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      interface{} `json:"id,omitempty"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
	Result  interface{} `json:"result,omitempty"`
	Error   *MCPError   `json:"error,omitempty"`
}

// MCPError represents an MCP protocol error
type MCPError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// MCPToolCall represents a call to an MCP tool
type MCPToolCall struct {
	ID          string                 `json:"id" db:"id"`
	SessionID   string                 `json:"session_id" db:"session_id"`
	QueryID     string                 `json:"query_id,omitempty" db:"query_id"`
	ServerID    string                 `json:"server_id" db:"server_id"`
	ToolName    string                 `json:"tool_name" db:"tool_name"`
	Arguments   map[string]interface{} `json:"arguments" db:"arguments"`
	Result      interface{}            `json:"result,omitempty" db:"result"`
	Status      string                 `json:"status" db:"status"` // pending, running, completed, failed
	StartedAt   time.Time              `json:"started_at" db:"started_at"`
	CompletedAt *time.Time             `json:"completed_at,omitempty" db:"completed_at"`
	Duration    time.Duration          `json:"duration" db:"duration"`
	Error       string                 `json:"error,omitempty" db:"error"`
	Metadata    map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt   time.Time              `json:"created_at" db:"created_at"`
}

// MCPConnection represents a connection to an MCP server
type MCPConnection struct {
	ID               string                 `json:"id"`
	ServerID         string                 `json:"server_id"`
	Transport        string                 `json:"transport"`
	Status           string                 `json:"status"` // connecting, connected, disconnected, error
	ConnectedAt      *time.Time             `json:"connected_at,omitempty"`
	DisconnectedAt   *time.Time             `json:"disconnected_at,omitempty"`
	LastActivity     time.Time              `json:"last_activity"`
	MessagesSent     int64                  `json:"messages_sent"`
	MessagesReceived int64                  `json:"messages_received"`
	Errors           int64                  `json:"errors"`
	Capabilities     []string               `json:"capabilities"`
	ProtocolVersion  string                 `json:"protocol_version"`
	ClientInfo       map[string]interface{} `json:"client_info"`
	ServerInfo       map[string]interface{} `json:"server_info"`
}

// MCPHealthCheck represents health check information for MCP servers
type MCPHealthCheck struct {
	ServerID           string                 `json:"server_id"`
	Status             string                 `json:"status"` // healthy, unhealthy, unknown
	LastChecked        time.Time              `json:"last_checked"`
	ResponseTime       time.Duration          `json:"response_time"`
	AvailableTools     int                    `json:"available_tools"`
	AvailableResources int                    `json:"available_resources"`
	ErrorCount         int                    `json:"error_count"`
	Uptime             time.Duration          `json:"uptime"`
	Version            string                 `json:"version"`
	Metadata           map[string]interface{} `json:"metadata"`
}

// MCPServerStats represents statistics for an MCP server
type MCPServerStats struct {
	ServerID        string           `json:"server_id"`
	Period          string           `json:"period"` // hour, day, week, month
	StartTime       time.Time        `json:"start_time"`
	EndTime         time.Time        `json:"end_time"`
	TotalCalls      int64            `json:"total_calls"`
	SuccessfulCalls int64            `json:"successful_calls"`
	FailedCalls     int64            `json:"failed_calls"`
	AvgResponseTime time.Duration    `json:"avg_response_time"`
	MaxResponseTime time.Duration    `json:"max_response_time"`
	MinResponseTime time.Duration    `json:"min_response_time"`
	TopTools        []ToolUsageStats `json:"top_tools"`
	ErrorBreakdown  map[string]int64 `json:"error_breakdown"`
}

// ToolUsageStats represents usage statistics for a specific tool
type ToolUsageStats struct {
	ToolName    string        `json:"tool_name"`
	CallCount   int64         `json:"call_count"`
	SuccessRate float64       `json:"success_rate"`
	AvgDuration time.Duration `json:"avg_duration"`
	LastUsed    time.Time     `json:"last_used"`
}

// MCPIntegration represents integration settings between query types and MCP tools
type MCPIntegration struct {
	ID         string                 `json:"id" db:"id"`
	QueryType  string                 `json:"query_type" db:"query_type"` // search_files, analyze_code, etc.
	MCPTools   []string               `json:"mcp_tools" db:"mcp_tools"`   // List of MCP tools to use
	Priority   int                    `json:"priority" db:"priority"`     // Execution priority
	Enabled    bool                   `json:"enabled" db:"enabled"`
	Conditions map[string]interface{} `json:"conditions" db:"conditions"` // When to trigger
	Parameters map[string]interface{} `json:"parameters" db:"parameters"` // Default parameters
	Timeout    int                    `json:"timeout" db:"timeout"`       // seconds
	RetryCount int                    `json:"retry_count" db:"retry_count"`
	Metadata   map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt  time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at" db:"updated_at"`
}
