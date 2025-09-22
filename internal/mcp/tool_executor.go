package mcp

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/models"
)

// ToolExecutor orchestrates the execution of MCP tools
type ToolExecutor struct {
	client     *Client
	registry   *Registry
	executions map[string]*models.MCPToolExecution
	mu         sync.RWMutex
}

// ExecutionOptions contains options for tool execution
type ExecutionOptions struct {
	Timeout    time.Duration          `json:"timeout"`
	ServerHint string                 `json:"server_hint,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// NewToolExecutor creates a new tool executor
func NewToolExecutor(client *Client, registry *Registry) *ToolExecutor {
	return &ToolExecutor{
		client:     client,
		registry:   registry,
		executions: make(map[string]*models.MCPToolExecution),
	}
}

// ExecuteTool executes a tool with the given input
func (e *ToolExecutor) ExecuteTool(ctx context.Context, toolName string, input map[string]interface{}, options *ExecutionOptions) (*models.MCPToolExecution, error) {
	if options == nil {
		options = &ExecutionOptions{
			Timeout: 30 * time.Second,
		}
	}

	// Apply timeout to context
	if options.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, options.Timeout)
		defer cancel()
	}

	// Find servers that provide this tool
	servers, err := e.findServersForTool(ctx, toolName, options.ServerHint)
	if err != nil {
		return nil, fmt.Errorf("failed to find servers for tool %s: %v", toolName, err)
	}

	if len(servers) == 0 {
		return nil, fmt.Errorf("no servers found that provide tool %s", toolName)
	}

	// Try executing on each server until one succeeds
	var lastExecution *models.MCPToolExecution
	var lastError error

	for _, serverID := range servers {
		execution, err := e.client.ExecuteTool(ctx, serverID, toolName, input)
		if err == nil && execution.Success {
			e.mu.Lock()
			e.executions[execution.ID] = execution
			e.mu.Unlock()
			return execution, nil
		}

		lastExecution = execution
		lastError = err

		// If this was the preferred server (server hint), try others
		if serverID == options.ServerHint {
			continue
		}
	}

	// All executions failed
	if lastExecution != nil {
		e.mu.Lock()
		e.executions[lastExecution.ID] = lastExecution
		e.mu.Unlock()
		return lastExecution, lastError
	}

	return nil, fmt.Errorf("tool execution failed on all available servers")
}

// ExecuteToolOnServer executes a tool on a specific server
func (e *ToolExecutor) ExecuteToolOnServer(ctx context.Context, serverID, toolName string, input map[string]interface{}) (*models.MCPToolExecution, error) {
	if !e.client.IsConnected(serverID) {
		return nil, fmt.Errorf("not connected to server %s", serverID)
	}

	execution, err := e.client.ExecuteTool(ctx, serverID, toolName, input)
	if execution != nil {
		e.mu.Lock()
		e.executions[execution.ID] = execution
		e.mu.Unlock()
	}

	return execution, err
}

// BatchExecuteTools executes multiple tools concurrently
func (e *ToolExecutor) BatchExecuteTools(ctx context.Context, requests []ToolRequest) ([]*models.MCPToolExecution, error) {
	if len(requests) == 0 {
		return nil, fmt.Errorf("no tool requests provided")
	}

	resultChan := make(chan ExecutionResult, len(requests))
	var wg sync.WaitGroup

	// Execute all tools concurrently
	for i, req := range requests {
		wg.Add(1)
		go func(index int, request ToolRequest) {
			defer wg.Done()

			execution, err := e.ExecuteTool(ctx, request.ToolName, request.Input, request.Options)
			resultChan <- ExecutionResult{
				Index:     index,
				Execution: execution,
				Error:     err,
			}
		}(i, req)
	}

	wg.Wait()
	close(resultChan)

	// Collect results
	results := make([]*models.MCPToolExecution, len(requests))
	var errors []error

	for result := range resultChan {
		results[result.Index] = result.Execution
		if result.Error != nil {
			errors = append(errors, result.Error)
		}
	}

	if len(errors) > 0 {
		return results, fmt.Errorf("batch execution had %d errors", len(errors))
	}

	return results, nil
}

// GetExecution returns a tool execution by ID
func (e *ToolExecutor) GetExecution(executionID string) (*models.MCPToolExecution, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	execution, exists := e.executions[executionID]
	if !exists {
		return nil, fmt.Errorf("execution %s not found", executionID)
	}

	return execution, nil
}

// ListExecutions returns all tool executions
func (e *ToolExecutor) ListExecutions() []*models.MCPToolExecution {
	e.mu.RLock()
	defer e.mu.RUnlock()

	executions := make([]*models.MCPToolExecution, 0, len(e.executions))
	for _, execution := range e.executions {
		executions = append(executions, execution)
	}

	return executions
}

// GetExecutionsByServer returns executions for a specific server
func (e *ToolExecutor) GetExecutionsByServer(serverID string) []*models.MCPToolExecution {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var serverExecutions []*models.MCPToolExecution
	for _, execution := range e.executions {
		if execution.ServerID == serverID {
			serverExecutions = append(serverExecutions, execution)
		}
	}

	return serverExecutions
}

// GetExecutionStats returns execution statistics
func (e *ToolExecutor) GetExecutionStats() ExecutionStats {
	e.mu.RLock()
	defer e.mu.RUnlock()

	stats := ExecutionStats{
		TotalExecutions:      len(e.executions),
		SuccessfulExecutions: 0,
		FailedExecutions:     0,
		ToolCounts:           make(map[string]int),
		ServerCounts:         make(map[string]int),
		AverageDuration:      0,
	}

	var totalDuration time.Duration

	for _, execution := range e.executions {
		if execution.Success {
			stats.SuccessfulExecutions++
		} else {
			stats.FailedExecutions++
		}

		stats.ToolCounts[execution.ToolName]++
		stats.ServerCounts[execution.ServerID]++
		totalDuration += execution.Duration
	}

	if len(e.executions) > 0 {
		stats.AverageDuration = totalDuration / time.Duration(len(e.executions))
	}

	return stats
}

// ClearExecutions clears all stored executions
func (e *ToolExecutor) ClearExecutions() {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.executions = make(map[string]*models.MCPToolExecution)
}

// findServersForTool finds servers that provide a specific tool
func (e *ToolExecutor) findServersForTool(ctx context.Context, toolName string, serverHint string) ([]string, error) {
	// If server hint is provided and connected, try it first
	var servers []string
	if serverHint != "" && e.client.IsConnected(serverHint) {
		servers = append(servers, serverHint)
	}

	// Get all available tools from connected servers
	tools, err := e.client.ListTools(ctx)
	if err != nil {
		return servers, err
	}

	// Find servers that provide this tool
	for _, tool := range tools {
		if tool.Name == toolName {
			// Avoid duplicates (in case serverHint was already added)
			found := false
			for _, existing := range servers {
				if existing == tool.ServerID {
					found = true
					break
				}
			}
			if !found {
				servers = append(servers, tool.ServerID)
			}
		}
	}

	return servers, nil
}

// ToolRequest represents a tool execution request
type ToolRequest struct {
	ToolName string                 `json:"tool_name"`
	Input    map[string]interface{} `json:"input"`
	Options  *ExecutionOptions      `json:"options,omitempty"`
}

// ExecutionResult represents the result of a tool execution
type ExecutionResult struct {
	Index     int
	Execution *models.MCPToolExecution
	Error     error
}

// ExecutionStats represents statistics about tool executions
type ExecutionStats struct {
	TotalExecutions      int            `json:"total_executions"`
	SuccessfulExecutions int            `json:"successful_executions"`
	FailedExecutions     int            `json:"failed_executions"`
	ToolCounts           map[string]int `json:"tool_counts"`
	ServerCounts         map[string]int `json:"server_counts"`
	AverageDuration      time.Duration  `json:"average_duration"`
}
