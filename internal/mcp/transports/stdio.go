package transports

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/mcp"
)

// StdioTransport handles communication over stdin/stdout
type StdioTransport struct {
	cmd         *exec.Cmd
	stdin       io.WriteCloser
	stdout      io.ReadCloser
	stderr      io.ReadCloser
	scanner     *bufio.Scanner
	connected   bool
	mu          sync.RWMutex
	messageChan chan *mcp.MCPMessage
	errorChan   chan error
	stopChan    chan struct{}
}

// NewStdioTransport creates a new stdio transport
func NewStdioTransport() *StdioTransport {
	return &StdioTransport{
		messageChan: make(chan *mcp.MCPMessage, 100),
		errorChan:   make(chan error, 10),
		stopChan:    make(chan struct{}),
	}
}

// Connect establishes connection to MCP server via stdio
func (t *StdioTransport) Connect(ctx context.Context, config mcp.TransportConfig) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.connected {
		return fmt.Errorf("already connected")
	}

	// Create command
	cmd := exec.CommandContext(ctx, config.Command, config.Args...)

	// Set environment variables
	if len(config.Env) > 0 {
		cmd.Env = append(os.Environ(), config.Env...)
	}

	// Set working directory
	if config.WorkingDir != "" {
		cmd.Dir = config.WorkingDir
	}

	// Setup pipes
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdin pipe: %v", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return fmt.Errorf("failed to create stdout pipe: %v", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		stdin.Close()
		stdout.Close()
		return fmt.Errorf("failed to create stderr pipe: %v", err)
	}

	// Start the process
	if err := cmd.Start(); err != nil {
		stdin.Close()
		stdout.Close()
		stderr.Close()
		return fmt.Errorf("failed to start process: %v", err)
	}

	t.cmd = cmd
	t.stdin = stdin
	t.stdout = stdout
	t.stderr = stderr
	t.scanner = bufio.NewScanner(stdout)
	t.connected = true

	// Start reading goroutines
	go t.readMessages()
	go t.readErrors()

	return nil
}

// Send sends a message to the MCP server
func (t *StdioTransport) Send(ctx context.Context, message *mcp.MCPMessage) error {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if !t.connected {
		return fmt.Errorf("not connected")
	}

	data, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %v", err)
	}

	// Write JSON-RPC message with newline
	_, err = fmt.Fprintf(t.stdin, "%s\n", data)
	if err != nil {
		return fmt.Errorf("failed to write message: %v", err)
	}

	return nil
}

// Receive receives a message from the MCP server
func (t *StdioTransport) Receive(ctx context.Context) (*mcp.MCPMessage, error) {
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

// Close closes the transport connection
func (t *StdioTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.connected {
		return nil
	}

	t.connected = false
	close(t.stopChan)

	// Close pipes
	if t.stdin != nil {
		t.stdin.Close()
	}
	if t.stdout != nil {
		t.stdout.Close()
	}
	if t.stderr != nil {
		t.stderr.Close()
	}

	// Terminate process
	if t.cmd != nil && t.cmd.Process != nil {
		// Give process time to cleanup
		done := make(chan error, 1)
		go func() {
			done <- t.cmd.Wait()
		}()

		select {
		case <-done:
			// Process exited cleanly
		case <-time.After(5 * time.Second):
			// Force kill after timeout
			t.cmd.Process.Kill()
			<-done
		}
	}

	return nil
}

// IsConnected returns whether the transport is connected
func (t *StdioTransport) IsConnected() bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.connected
}

// readMessages reads messages from stdout
func (t *StdioTransport) readMessages() {
	defer func() {
		if r := recover(); r != nil {
			t.errorChan <- fmt.Errorf("panic in readMessages: %v", r)
		}
	}()

	for t.scanner.Scan() {
		line := t.scanner.Text()
		if line == "" {
			continue
		}

		var message mcp.MCPMessage
		if err := json.Unmarshal([]byte(line), &message); err != nil {
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

	if err := t.scanner.Err(); err != nil {
		t.errorChan <- fmt.Errorf("scanner error: %v", err)
	}
}

// readErrors reads errors from stderr
func (t *StdioTransport) readErrors() {
	defer func() {
		if r := recover(); r != nil {
			t.errorChan <- fmt.Errorf("panic in readErrors: %v", r)
		}
	}()

	scanner := bufio.NewScanner(t.stderr)
	for scanner.Scan() {
		line := scanner.Text()
		if line != "" {
			// Log stderr output (could be debug info or errors)
			// For now, we'll ignore stderr unless it's critical
			continue
		}
	}
}
