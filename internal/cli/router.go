package cli

import (
	"fmt"
	"strings"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/logger"
)

// CommandHandler interface for command implementations
type CommandHandler interface {
	Execute(session *Session, args []string) *CommandResult
}

// CommandResult represents the result of a command execution
type CommandResult struct {
	Success   bool                   `json:"success"`
	Output    string                 `json:"output,omitempty"`
	Error     error                  `json:"error,omitempty"`
	Data      interface{}            `json:"data,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Duration  time.Duration          `json:"duration"`
	Timestamp time.Time              `json:"timestamp"`
}

// NewCommandResult creates a new command result
func NewCommandResult(output string, err error) *CommandResult {
	return &CommandResult{
		Success:   err == nil,
		Output:    output,
		Error:     err,
		Metadata:  make(map[string]interface{}),
		Timestamp: time.Now(),
	}
}

// Router manages command routing and execution
type Router struct {
	commands map[string]CommandHandler
	session  *Session
	logger   *logger.Logger
}

// NewRouter creates a new command router
func NewRouter(session *Session, logger *logger.Logger) *Router {
	return &Router{
		commands: make(map[string]CommandHandler),
		session:  session,
		logger:   logger,
	}
}

// Register registers a command handler
func (r *Router) Register(name string, handler CommandHandler) {
	r.commands[strings.ToLower(name)] = handler
	r.logger.Debug("Registered command handler", map[string]interface{}{
		"command": name,
	})
}

// Execute executes a parsed command
func (r *Router) Execute(cmd *Command) *CommandResult {
	startTime := time.Now()

	// Add to session history
	r.session.AddToHistory(cmd.Raw)
	r.session.IncrementCommandCount()

	// Find command handler
	handler, exists := r.commands[strings.ToLower(cmd.Name)]
	if !exists {
		return &CommandResult{
			Success:   false,
			Error:     fmt.Errorf("unknown command: %s", cmd.Name),
			Duration:  time.Since(startTime),
			Timestamp: time.Now(),
		}
	}

	// Execute command with error handling
	var result *CommandResult

	func() {
		defer func() {
			if r := recover(); r != nil {
				result = &CommandResult{
					Success:   false,
					Error:     fmt.Errorf("command panic: %v", r),
					Duration:  time.Since(startTime),
					Timestamp: time.Now(),
				}
			}
		}()

		result = handler.Execute(r.session, cmd.Args)
		if result == nil {
			result = NewCommandResult("", fmt.Errorf("command returned nil result"))
		}
	}()

	// Set duration and timestamp
	result.Duration = time.Since(startTime)
	result.Timestamp = time.Now()

	// Add metadata
	if result.Metadata == nil {
		result.Metadata = make(map[string]interface{})
	}
	result.Metadata["command"] = cmd.Name
	result.Metadata["args"] = cmd.Args
	result.Metadata["flags"] = cmd.Flags
	result.Metadata["session_id"] = r.session.ID

	// Log command execution
	logData := map[string]interface{}{
		"command":    cmd.Name,
		"success":    result.Success,
		"duration":   result.Duration.String(),
		"session_id": r.session.ID,
	}

	if result.Error != nil {
		logData["error"] = result.Error.Error()
		r.logger.Error("Command execution failed", result.Error, logData)
	} else {
		r.logger.Info("Command executed successfully", logData)
	}

	return result
}

// GetAvailableCommands returns a list of registered commands
func (r *Router) GetAvailableCommands() []string {
	commands := make([]string, 0, len(r.commands))
	for name := range r.commands {
		commands = append(commands, name)
	}
	return commands
}

// HasCommand checks if a command is registered
func (r *Router) HasCommand(name string) bool {
	_, exists := r.commands[strings.ToLower(name)]
	return exists
}

// ExecuteString executes a command from a string
func (r *Router) ExecuteString(input string) *CommandResult {
	parser := NewParser()

	cmd, err := parser.Parse(input)
	if err != nil {
		return &CommandResult{
			Success:   false,
			Error:     fmt.Errorf("parse error: %v", err),
			Timestamp: time.Now(),
		}
	}

	// Validate command
	if err := parser.ValidateCommand(cmd); err != nil {
		return &CommandResult{
			Success:   false,
			Error:     err,
			Timestamp: time.Now(),
		}
	}

	return r.Execute(cmd)
}

// Middleware function type
type Middleware func(cmd *Command, next func(*Command) *CommandResult) *CommandResult

// middleware stack
type middlewareStack []Middleware

// Apply applies middleware to command execution
func (ms middlewareStack) Apply(cmd *Command, handler func(*Command) *CommandResult) *CommandResult {
	if len(ms) == 0 {
		return handler(cmd)
	}

	// Build middleware chain
	var chain func(*Command) *CommandResult
	chain = func(c *Command) *CommandResult {
		if len(ms) == 0 {
			return handler(c)
		}

		middleware := ms[0]
		remainingStack := ms[1:]

		return middleware(c, func(innerCmd *Command) *CommandResult {
			return remainingStack.Apply(innerCmd, handler)
		})
	}

	return chain(cmd)
}

// AddMiddleware adds middleware to the router
func (r *Router) AddMiddleware(middleware Middleware) {
	// Implementation would add middleware to execution chain
	// For now, we'll keep it simple without middleware
}

// Built-in middleware examples

// LoggingMiddleware logs command execution
func LoggingMiddleware(logger *logger.Logger) Middleware {
	return func(cmd *Command, next func(*Command) *CommandResult) *CommandResult {
		start := time.Now()

		logger.Info("Command started", map[string]interface{}{
			"command": cmd.Name,
			"args":    cmd.Args,
		})

		result := next(cmd)

		logger.Info("Command completed", map[string]interface{}{
			"command":  cmd.Name,
			"success":  result.Success,
			"duration": time.Since(start).String(),
		})

		return result
	}
}

// TimingMiddleware adds timing information
func TimingMiddleware() Middleware {
	return func(cmd *Command, next func(*Command) *CommandResult) *CommandResult {
		start := time.Now()
		result := next(cmd)
		result.Duration = time.Since(start)
		return result
	}
}

// ValidationMiddleware validates commands before execution
func ValidationMiddleware(parser *Parser) Middleware {
	return func(cmd *Command, next func(*Command) *CommandResult) *CommandResult {
		if err := parser.ValidateCommand(cmd); err != nil {
			return &CommandResult{
				Success:   false,
				Error:     err,
				Timestamp: time.Now(),
			}
		}
		return next(cmd)
	}
}
