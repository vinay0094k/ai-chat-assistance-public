package logger

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/fatih/color"
)

// LogLevel represents the severity level of a log entry
type LogLevel int

const (
	TRACE LogLevel = iota
	DEBUG
	INFO
	WARN
	ERROR
	FATAL
)

// String returns the string representation of a log level
func (l LogLevel) String() string {
	switch l {
	case TRACE:
		return "TRACE"
	case DEBUG:
		return "DEBUG"
	case INFO:
		return "INFO"
	case WARN:
		return "WARN"
	case ERROR:
		return "ERROR"
	case FATAL:
		return "FATAL"
	default:
		return "UNKNOWN"
	}
}

// Logger represents a logger instance
type Logger struct {
	level     LogLevel
	outputs   []LogOutput
	context   map[string]interface{}
	sessionID string
	component string
	mutex     sync.RWMutex
}

// LogEntry represents a single log entry
type LogEntry struct {
	Timestamp  time.Time              `json:"timestamp"`
	Level      LogLevel               `json:"level"`
	Component  string                 `json:"component"`
	SessionID  string                 `json:"session_id"`
	RequestID  string                 `json:"request_id,omitempty"`
	Message    string                 `json:"message"`
	Data       map[string]interface{} `json:"data,omitempty"`
	Duration   time.Duration          `json:"duration,omitempty"`
	Error      string                 `json:"error,omitempty"`
	StackTrace string                 `json:"stack_trace,omitempty"`
	File       string                 `json:"file,omitempty"`
	Line       int                    `json:"line,omitempty"`
	Function   string                 `json:"function,omitempty"`
}

// LogOutput represents a log output destination
type LogOutput interface {
	Write(entry *LogEntry) error
	Close() error
}

// ConsoleOutput writes logs to console with color coding
type ConsoleOutput struct {
	colors  map[LogLevel]*color.Color
	noColor bool
	mutex   sync.Mutex
}

// FileOutput writes logs to a file with rotation
type FileOutput struct {
	filePath    string
	maxSize     int64
	maxFiles    int
	currentFile *os.File
	currentSize int64
	mutex       sync.Mutex
}

// JSONOutput writes structured JSON logs
type JSONOutput struct {
	filePath string
	file     *os.File
	encoder  *json.Encoder
	mutex    sync.Mutex
}

// NewLogger creates a new logger instance
func NewLogger(component string, sessionID string) *Logger {
	logger := &Logger{
		level:     INFO,
		component: component,
		sessionID: sessionID,
		context:   make(map[string]interface{}),
		outputs:   make([]LogOutput, 0),
	}

	// Add default console output
	logger.AddOutput(NewConsoleOutput())

	// Add file output if LOG_FILE is set
	if logFile := os.Getenv("LOG_FILE"); logFile != "" {
		if fileOutput, err := NewFileOutput(logFile, 100*1024*1024, 10); err == nil {
			logger.AddOutput(fileOutput)
		}
	}

	// Set log level from environment
	if levelStr := os.Getenv("LOG_LEVEL"); levelStr != "" {
		if level := parseLogLevel(levelStr); level != -1 {
			logger.level = level
		}
	}

	return logger
}

// NewDefaultLogger creates a default logger instance
func NewDefaultLogger(component string) *Logger {
	sessionID := fmt.Sprintf("session_%d", time.Now().UnixNano())
	return NewLogger(component, sessionID)
}

// NewConsoleOutput creates a new console output
func NewConsoleOutput() *ConsoleOutput {
	return &ConsoleOutput{
		colors: map[LogLevel]*color.Color{
			TRACE: color.New(color.FgWhite),
			DEBUG: color.New(color.FgBlue),
			INFO:  color.New(color.FgGreen),
			WARN:  color.New(color.FgYellow),
			ERROR: color.New(color.FgRed),
			FATAL: color.New(color.FgRed, color.Bold),
		},
		noColor: os.Getenv("NO_COLOR") != "" || os.Getenv("ENABLE_COLORS") == "false",
	}
}

// NewFileOutput creates a new file output with rotation
func NewFileOutput(filePath string, maxSize int64, maxFiles int) (*FileOutput, error) {
	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		return nil, fmt.Errorf("failed to create log directory: %v", err)
	}

	file, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %v", err)
	}

	stat, _ := file.Stat()
	currentSize := int64(0)
	if stat != nil {
		currentSize = stat.Size()
	}

	return &FileOutput{
		filePath:    filePath,
		maxSize:     maxSize,
		maxFiles:    maxFiles,
		currentFile: file,
		currentSize: currentSize,
	}, nil
}

// NewJSONOutput creates a new JSON output
func NewJSONOutput(filePath string) (*JSONOutput, error) {
	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		return nil, fmt.Errorf("failed to create log directory: %v", err)
	}

	file, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open JSON log file: %v", err)
	}

	return &JSONOutput{
		filePath: filePath,
		file:     file,
		encoder:  json.NewEncoder(file),
	}, nil
}

// AddOutput adds a log output destination
func (l *Logger) AddOutput(output LogOutput) {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	l.outputs = append(l.outputs, output)
}

// SetLevel sets the minimum log level
func (l *Logger) SetLevel(level LogLevel) {
	if level < TRACE || level > FATAL {
		level = INFO // default fallback
	}
	l.mutex.Lock()
	defer l.mutex.Unlock()
	l.level = level
}

// SetContext sets context data for all log entries
func (l *Logger) SetContext(key string, value interface{}) {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	l.context[key] = value
}

// WithContext creates a new logger with additional context
func (l *Logger) WithContext(context map[string]interface{}) *Logger {
	l.mutex.RLock()
	newContext := make(map[string]interface{})
	for k, v := range l.context {
		newContext[k] = v
	}
	for k, v := range context {
		newContext[k] = v
	}
	l.mutex.RUnlock()

	return &Logger{
		level:     l.level,
		outputs:   l.outputs,
		context:   newContext,
		sessionID: l.sessionID,
		component: l.component,
	}
}

// log writes a log entry
func (l *Logger) log(level LogLevel, message string, data map[string]interface{}, err error) {
	if level < l.level {
		return
	}

	// Get caller information
	file, line, function := getCallerInfo(3)

	l.mutex.RLock()

	// Merge context with data
	mergedData := make(map[string]interface{})
	for k, v := range l.context {
		mergedData[k] = v
	}
	for k, v := range data {
		mergedData[k] = v
	}

	entry := &LogEntry{
		Timestamp: time.Now(),
		Level:     level,
		Component: l.component,
		SessionID: l.sessionID,
		Message:   message,
		Data:      mergedData,
		File:      file,
		Line:      line,
		Function:  function,
	}

	if err != nil {
		entry.Error = err.Error()
		if level >= ERROR {
			entry.StackTrace = getStackTrace()
		}
	}

	// Write to all outputs
	for _, output := range l.outputs {
		if writeErr := output.Write(entry); writeErr != nil {
			// Fallback to stderr if output fails
			fmt.Fprintf(os.Stderr, "Failed to write log: %v\n", writeErr)
		}
	}

	l.mutex.RUnlock()
}

// Trace logs a trace level message
func (l *Logger) Trace(message string, data ...map[string]interface{}) {
	var d map[string]interface{}
	if len(data) > 0 {
		d = data[0]
	}
	l.log(TRACE, message, d, nil)
}

// Debug logs a debug level message
func (l *Logger) Debug(message string, data ...map[string]interface{}) {
	var d map[string]interface{}
	if len(data) > 0 {
		d = data[0]
	}
	l.log(DEBUG, message, d, nil)
}

// Info logs an info level message
func (l *Logger) Info(message string, data ...map[string]interface{}) {
	var d map[string]interface{}
	if len(data) > 0 {
		d = data[0]
	}
	l.log(INFO, message, d, nil)
}

// Warn logs a warning level message
func (l *Logger) Warn(message string, data ...map[string]interface{}) {
	var d map[string]interface{}
	if len(data) > 0 {
		d = data[0]
	}
	l.log(WARN, message, d, nil)
}

// Error logs an error level message
func (l *Logger) Error(message string, err error, data ...map[string]interface{}) {
	var d map[string]interface{}
	if len(data) > 0 {
		d = data[0]
	}
	l.log(ERROR, message, d, err)
}

// Fatal logs a fatal level message and exits
func (l *Logger) Fatal(message string, err error, data ...map[string]interface{}) {
	var d map[string]interface{}
	if len(data) > 0 {
		d = data[0]
	}
	l.log(FATAL, message, d, err)
	os.Exit(1)
}

// Write implements LogOutput interface for ConsoleOutput
func (co *ConsoleOutput) Write(entry *LogEntry) error {
	co.mutex.Lock()
	defer co.mutex.Unlock()

	timestamp := entry.Timestamp.Format("15:04:05")
	level := strings.ToUpper(entry.Level.String())

	if co.noColor {
		fmt.Printf("[%s] %s [%s] %s", timestamp, level, entry.Component, entry.Message)
	} else {
		levelColor := co.colors[entry.Level]
		fmt.Printf("[%s] ", timestamp)
		levelColor.Printf("%-5s", level)
		fmt.Printf(" [%s] %s", entry.Component, entry.Message)
	}

	// Add data if present
	if len(entry.Data) > 0 {
		fmt.Printf(" | %v", entry.Data)
	}

	// Add error if present
	if entry.Error != "" {
		if co.noColor {
			fmt.Printf(" | ERROR: %s", entry.Error)
		} else {
			color.Red(" | ERROR: %s", entry.Error)
		}
	}

	fmt.Println()
	return nil
}

// Close implements LogOutput interface for ConsoleOutput
func (co *ConsoleOutput) Close() error {
	return nil
}

// Continue from internal/logger/logger.go

// Write implements LogOutput interface for FileOutput
func (fo *FileOutput) Write(entry *LogEntry) error {
	fo.mutex.Lock()
	defer fo.mutex.Unlock()

	// Check if rotation is needed
	if fo.currentSize >= fo.maxSize {
		if err := fo.rotate(); err != nil {
			return fmt.Errorf("failed to rotate log file: %v", err)
		}
	}

	// Format log entry
	timestamp := entry.Timestamp.Format("2006-01-02 15:04:05.000")
	logLine := fmt.Sprintf("[%s] %-5s [%s] %s",
		timestamp,
		entry.Level.String(),
		entry.Component,
		entry.Message)

	// Add data if present
	if len(entry.Data) > 0 {
		if dataBytes, err := json.Marshal(entry.Data); err == nil {
			logLine += fmt.Sprintf(" | %s", string(dataBytes))
		}
	}

	// Add error if present
	if entry.Error != "" {
		logLine += fmt.Sprintf(" | ERROR: %s", entry.Error)
	}

	// Add caller info if available
	if entry.File != "" && entry.Line > 0 {
		logLine += fmt.Sprintf(" | %s:%d", entry.File, entry.Line)
	}

	logLine += "\n"

	// Write to file
	n, err := fo.currentFile.WriteString(logLine)
	if err != nil {
		return err
	}

	fo.currentSize += int64(n)
	return fo.currentFile.Sync()
}

// rotate rotates the log file
func (fo *FileOutput) rotate() error {
	// Close current file
	if fo.currentFile != nil {
		fo.currentFile.Close()
	}

	// Rotate existing files
	for i := fo.maxFiles - 1; i > 0; i-- {
		oldName := fmt.Sprintf("%s.%d", fo.filePath, i)
		newName := fmt.Sprintf("%s.%d", fo.filePath, i+1)

		if i == fo.maxFiles-1 {
			// Remove oldest file
			os.Remove(newName)
		}

		if _, err := os.Stat(oldName); err == nil {
			os.Rename(oldName, newName)
		}
	}

	// Move current file to .1
	if _, err := os.Stat(fo.filePath); err == nil {
		os.Rename(fo.filePath, fo.filePath+".1")
	}

	// Create new file
	file, err := os.OpenFile(fo.filePath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}

	fo.currentFile = file
	fo.currentSize = 0
	return nil
}

// Close implements LogOutput interface for FileOutput
func (fo *FileOutput) Close() error {
	fo.mutex.Lock()
	defer fo.mutex.Unlock()

	if fo.currentFile != nil {
		return fo.currentFile.Close()
	}
	return nil
}

// Write implements LogOutput interface for JSONOutput
func (jo *JSONOutput) Write(entry *LogEntry) error {
	jo.mutex.Lock()
	defer jo.mutex.Unlock()

	return jo.encoder.Encode(entry)
}

// Close implements LogOutput interface for JSONOutput
func (jo *JSONOutput) Close() error {
	jo.mutex.Lock()
	defer jo.mutex.Unlock()

	if jo.file != nil {
		return jo.file.Close()
	}
	return nil
}

// Utility functions

// parseLogLevel parses a log level string
func parseLogLevel(levelStr string) LogLevel {
	switch strings.ToUpper(levelStr) {
	case "TRACE":
		return TRACE
	case "DEBUG":
		return DEBUG
	case "INFO":
		return INFO
	case "WARN", "WARNING":
		return WARN
	case "ERROR":
		return ERROR
	case "FATAL":
		return FATAL
	default:
		return -1
	}
}

// getCallerInfo gets information about the caller
func getCallerInfo(skip int) (file string, line int, function string) {
	pc, file, line, ok := runtime.Caller(skip)
	if !ok {
		return "", 0, ""
	}

	// Get just the filename, not the full path
	if idx := strings.LastIndex(file, "/"); idx >= 0 {
		file = file[idx+1:]
	}

	// Get function name
	if fn := runtime.FuncForPC(pc); fn != nil {
		function = fn.Name()
		if idx := strings.LastIndex(function, "."); idx >= 0 {
			function = function[idx+1:]
		}
	}

	return file, line, function
}

// getStackTrace gets the current stack trace
func getStackTrace() string {
	buf := make([]byte, 4096)
	n := runtime.Stack(buf, false)
	return string(buf[:n])
}
