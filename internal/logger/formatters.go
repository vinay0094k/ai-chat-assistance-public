package logger

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/fatih/color"
)

// Formatter interface for different log formats
type Formatter interface {
	Format(entry *LogEntry) ([]byte, error)
}

// TextFormatter formats logs as human-readable text
type TextFormatter struct {
	TimestampFormat string
	ShowCaller      bool
	ShowLevel       bool
	ShowComponent   bool
	ShowSessionID   bool
	ShowDuration    bool
	ColorOutput     bool
	IndentSize      int
}

// JSONFormatter formats logs as JSON
type JSONFormatter struct {
	PrettyPrint     bool
	TimestampFormat string
	IncludeLevel    bool
	IncludeCaller   bool
}

// StructuredFormatter formats logs with consistent structure
type StructuredFormatter struct {
	Template        string
	TimestampFormat string
	ShowMetadata    bool
	MaxLineLength   int
}

// CompactFormatter formats logs in a compact single-line format
type CompactFormatter struct {
	TimestampFormat  string
	ShowLevel        bool
	ShowComponent    bool
	MaxMessageLength int
}

// NewTextFormatter creates a new text formatter with defaults
func NewTextFormatter() *TextFormatter {
	return &TextFormatter{
		TimestampFormat: "15:04:05.000",
		ShowCaller:      false,
		ShowLevel:       true,
		ShowComponent:   true,
		ShowSessionID:   false,
		ShowDuration:    false,
		ColorOutput:     true,
		IndentSize:      2,
	}
}

// Format implements the Formatter interface for TextFormatter
func (f *TextFormatter) Format(entry *LogEntry) ([]byte, error) {
	var buf bytes.Buffer

	// Timestamp
	timestamp := entry.Timestamp.Format(f.TimestampFormat)
	buf.WriteString(fmt.Sprintf("[%s]", timestamp))

	// Level
	if f.ShowLevel {
		levelStr := strings.ToUpper(entry.Level.String())
		if f.ColorOutput {
			switch entry.Level {
			case TRACE:
				levelStr = color.New(color.FgWhite).Sprint(levelStr)
			case DEBUG:
				levelStr = color.New(color.FgBlue).Sprint(levelStr)
			case INFO:
				levelStr = color.New(color.FgGreen).Sprint(levelStr)
			case WARN:
				levelStr = color.New(color.FgYellow).Sprint(levelStr)
			case ERROR:
				levelStr = color.New(color.FgRed).Sprint(levelStr)
			case FATAL:
				levelStr = color.New(color.FgRed, color.Bold).Sprint(levelStr)
			}
		}
		buf.WriteString(fmt.Sprintf(" %-5s", levelStr))
	}

	// Component
	if f.ShowComponent && entry.Component != "" {
		componentStr := entry.Component
		if f.ColorOutput {
			componentStr = color.New(color.FgCyan).Sprint(componentStr)
		}
		buf.WriteString(fmt.Sprintf(" [%s]", componentStr))
	}

	// Session ID
	if f.ShowSessionID && entry.SessionID != "" {
		sessionStr := entry.SessionID[:8] // Show first 8 chars
		if f.ColorOutput {
			sessionStr = color.New(color.FgMagenta).Sprint(sessionStr)
		}
		buf.WriteString(fmt.Sprintf(" {%s}", sessionStr))
	}

	// Message
	buf.WriteString(fmt.Sprintf(" %s", entry.Message))

	// Duration
	if f.ShowDuration && entry.Duration > 0 {
		durationStr := entry.Duration.String()
		if f.ColorOutput {
			durationStr = color.New(color.FgYellow).Sprint(durationStr)
		}
		buf.WriteString(fmt.Sprintf(" (%s)", durationStr))
	}

	// Data
	if len(entry.Data) > 0 {
		buf.WriteString("\n")
		f.formatData(&buf, entry.Data, 1)
	}

	// Error
	if entry.Error != "" {
		errorStr := entry.Error
		if f.ColorOutput {
			errorStr = color.New(color.FgRed).Sprint(errorStr)
		}
		buf.WriteString(fmt.Sprintf("\n%sError: %s", f.getIndent(1), errorStr))
	}

	// Stack trace for errors
	if entry.StackTrace != "" && entry.Level >= ERROR {
		buf.WriteString(fmt.Sprintf("\n%sStack Trace:\n%s", f.getIndent(1), entry.StackTrace))
	}

	// Caller info
	if f.ShowCaller && entry.File != "" {
		callerStr := fmt.Sprintf("%s:%d", entry.File, entry.Line)
		if entry.Function != "" {
			callerStr = fmt.Sprintf("%s (%s)", callerStr, entry.Function)
		}
		if f.ColorOutput {
			callerStr = color.New(color.FgHiBlack).Sprint(callerStr)
		}
		buf.WriteString(fmt.Sprintf("\n%sCaller: %s", f.getIndent(1), callerStr))
	}

	buf.WriteString("\n")
	return buf.Bytes(), nil
}

// formatData formats the data section
func (f *TextFormatter) formatData(buf *bytes.Buffer, data map[string]interface{}, level int) {
	indent := f.getIndent(level)

	for key, value := range data {
		keyStr := key
		if f.ColorOutput {
			keyStr = color.New(color.FgCyan).Sprint(keyStr)
		}

		switch v := value.(type) {
		case map[string]interface{}:
			buf.WriteString(fmt.Sprintf("%s%s:\n", indent, keyStr))
			f.formatData(buf, v, level+1)
		case []interface{}:
			buf.WriteString(fmt.Sprintf("%s%s: [", indent, keyStr))
			for i, item := range v {
				if i > 0 {
					buf.WriteString(", ")
				}
				buf.WriteString(fmt.Sprintf("%v", item))
			}
			buf.WriteString("]\n")
		default:
			valueStr := fmt.Sprintf("%v", v)
			if f.ColorOutput {
				valueStr = color.New(color.FgWhite).Sprint(valueStr)
			}
			buf.WriteString(fmt.Sprintf("%s%s: %s\n", indent, keyStr, valueStr))
		}
	}
}

// getIndent returns indentation string
func (f *TextFormatter) getIndent(level int) string {
	return strings.Repeat(" ", level*f.IndentSize)
}

// NewJSONFormatter creates a new JSON formatter
func NewJSONFormatter() *JSONFormatter {
	return &JSONFormatter{
		PrettyPrint:     false,
		TimestampFormat: time.RFC3339Nano,
		IncludeLevel:    true,
		IncludeCaller:   false,
	}
}

// Format implements the Formatter interface for JSONFormatter
func (f *JSONFormatter) Format(entry *LogEntry) ([]byte, error) {
	// Create a map for JSON serialization
	logData := make(map[string]interface{})

	// Basic fields
	logData["timestamp"] = entry.Timestamp.Format(f.TimestampFormat)
	logData["message"] = entry.Message

	if f.IncludeLevel {
		logData["level"] = entry.Level.String()
	}

	if entry.Component != "" {
		logData["component"] = entry.Component
	}

	if entry.SessionID != "" {
		logData["session_id"] = entry.SessionID
	}

	if entry.RequestID != "" {
		logData["request_id"] = entry.RequestID
	}

	if entry.Duration > 0 {
		logData["duration_ms"] = entry.Duration.Milliseconds()
	}

	if entry.Error != "" {
		logData["error"] = entry.Error
	}

	if f.IncludeCaller && entry.File != "" {
		caller := map[string]interface{}{
			"file": entry.File,
			"line": entry.Line,
		}
		if entry.Function != "" {
			caller["function"] = entry.Function
		}
		logData["caller"] = caller
	}

	if entry.StackTrace != "" {
		logData["stack_trace"] = entry.StackTrace
	}

	// Include data fields
	if len(entry.Data) > 0 {
		for key, value := range entry.Data {
			logData[key] = value
		}
	}

	// Serialize to JSON
	if f.PrettyPrint {
		return json.MarshalIndent(logData, "", "  ")
	}

	return json.Marshal(logData)
}

// NewStructuredFormatter creates a new structured formatter
func NewStructuredFormatter() *StructuredFormatter {
	return &StructuredFormatter{
		Template:        "[{{.timestamp}}] {{.level}} [{{.component}}] {{.message}}",
		TimestampFormat: "15:04:05",
		ShowMetadata:    true,
		MaxLineLength:   120,
	}
}

// Format implements the Formatter interface for StructuredFormatter
func (f *StructuredFormatter) Format(entry *LogEntry) ([]byte, error) {
	var buf bytes.Buffer

	// Basic log line using template
	timestamp := entry.Timestamp.Format(f.TimestampFormat)
	level := strings.ToUpper(entry.Level.String())
	component := entry.Component
	if component == "" {
		component = "APP"
	}

	// Replace template variables
	line := f.Template
	line = strings.ReplaceAll(line, "{{.timestamp}}", timestamp)
	line = strings.ReplaceAll(line, "{{.level}}", level)
	line = strings.ReplaceAll(line, "{{.component}}", component)
	line = strings.ReplaceAll(line, "{{.message}}", entry.Message)

	buf.WriteString(line)

	// Add metadata if enabled
	if f.ShowMetadata {
		metadata := make([]string, 0)

		if entry.SessionID != "" {
			metadata = append(metadata, fmt.Sprintf("session=%s", entry.SessionID[:8]))
		}

		if entry.RequestID != "" {
			metadata = append(metadata, fmt.Sprintf("request=%s", entry.RequestID[:8]))
		}

		if entry.Duration > 0 {
			metadata = append(metadata, fmt.Sprintf("duration=%s", entry.Duration))
		}

		// Add selected data fields
		for key, value := range entry.Data {
			if f.isImportantField(key) {
				metadata = append(metadata, fmt.Sprintf("%s=%v", key, value))
			}
		}

		if len(metadata) > 0 {
			buf.WriteString(fmt.Sprintf(" [%s]", strings.Join(metadata, " ")))
		}
	}

	// Add error if present
	if entry.Error != "" {
		buf.WriteString(fmt.Sprintf(" ERROR: %s", entry.Error))
	}

	// Wrap long lines
	if f.MaxLineLength > 0 {
		result := buf.String()
		if len(result) > f.MaxLineLength {
			wrapped := f.wrapLine(result, f.MaxLineLength)
			buf.Reset()
			buf.WriteString(wrapped)
		}
	}

	buf.WriteString("\n")
	return buf.Bytes(), nil
}

// isImportantField determines if a data field should be included in structured output
func (f *StructuredFormatter) isImportantField(key string) bool {
	importantFields := map[string]bool{
		"user_id":    true,
		"query_id":   true,
		"file_path":  true,
		"function":   true,
		"error_code": true,
		"status":     true,
		"action":     true,
		"provider":   true,
		"model":      true,
		"tokens":     true,
		"cost":       true,
	}

	return importantFields[key]
}

// wrapLine wraps a line to fit within the specified length
func (f *StructuredFormatter) wrapLine(line string, maxLength int) string {
	if len(line) <= maxLength {
		return line
	}

	// Find a good break point
	breakPoint := maxLength
	for i := maxLength - 1; i >= maxLength/2; i-- {
		if line[i] == ' ' || line[i] == '\t' {
			breakPoint = i
			break
		}
	}

	return line[:breakPoint] + "\n    " + f.wrapLine(line[breakPoint+1:], maxLength-4)
}

// NewCompactFormatter creates a new compact formatter
func NewCompactFormatter() *CompactFormatter {
	return &CompactFormatter{
		TimestampFormat:  "15:04:05",
		ShowLevel:        true,
		ShowComponent:    true,
		MaxMessageLength: 80,
	}
}

// Format implements the Formatter interface for CompactFormatter
func (f *CompactFormatter) Format(entry *LogEntry) ([]byte, error) {
	var parts []string

	// Timestamp
	parts = append(parts, entry.Timestamp.Format(f.TimestampFormat))

	// Level
	if f.ShowLevel {
		level := strings.ToUpper(entry.Level.String())
		parts = append(parts, fmt.Sprintf("%-5s", level))
	}

	// Component
	if f.ShowComponent && entry.Component != "" {
		parts = append(parts, fmt.Sprintf("[%s]", entry.Component))
	}

	// Message (potentially truncated)
	message := entry.Message
	if f.MaxMessageLength > 0 && len(message) > f.MaxMessageLength {
		message = message[:f.MaxMessageLength-3] + "..."
	}
	parts = append(parts, message)

	// Key data points
	var keyData []string
	for key, value := range entry.Data {
		if f.isKeyField(key) {
			keyData = append(keyData, fmt.Sprintf("%s=%v", key, value))
		}
	}

	if len(keyData) > 0 {
		parts = append(parts, fmt.Sprintf("(%s)", strings.Join(keyData, " ")))
	}

	// Error indicator
	if entry.Error != "" {
		parts = append(parts, "ERR")
	}

	// Duration
	if entry.Duration > 0 {
		parts = append(parts, fmt.Sprintf("%s", entry.Duration))
	}

	line := strings.Join(parts, " ") + "\n"
	return []byte(line), nil
}

// isKeyField determines if a field is important enough for compact display
func (f *CompactFormatter) isKeyField(key string) bool {
	keyFields := map[string]bool{
		"status":   true,
		"code":     true,
		"provider": true,
		"tokens":   true,
		"action":   true,
	}

	return keyFields[key]
}

// ConditionalFormatter chooses formatter based on conditions
type ConditionalFormatter struct {
	DefaultFormatter Formatter
	Conditions       []FormatterCondition
}

// FormatterCondition represents a condition for choosing a formatter
type FormatterCondition struct {
	Condition func(*LogEntry) bool
	Formatter Formatter
}

// NewConditionalFormatter creates a conditional formatter
func NewConditionalFormatter(defaultFormatter Formatter) *ConditionalFormatter {
	return &ConditionalFormatter{
		DefaultFormatter: defaultFormatter,
		Conditions:       make([]FormatterCondition, 0),
	}
}

// AddCondition adds a condition for formatter selection
func (cf *ConditionalFormatter) AddCondition(condition func(*LogEntry) bool, formatter Formatter) {
	cf.Conditions = append(cf.Conditions, FormatterCondition{
		Condition: condition,
		Formatter: formatter,
	})
}

// Format implements the Formatter interface for ConditionalFormatter
func (cf *ConditionalFormatter) Format(entry *LogEntry) ([]byte, error) {
	// Check conditions in order
	for _, condition := range cf.Conditions {
		if condition.Condition(entry) {
			return condition.Formatter.Format(entry)
		}
	}

	// Use default formatter
	return cf.DefaultFormatter.Format(entry)
}

// Common condition functions

// IsErrorLevel returns true if log level is ERROR or FATAL
func IsErrorLevel(entry *LogEntry) bool {
	return entry.Level >= ERROR
}

// IsPerformanceLog returns true if entry contains performance data
func IsPerformanceLog(entry *LogEntry) bool {
	_, hasDuration := entry.Data["duration"]
	_, hasResponseTime := entry.Data["response_time"]
	_, hasMemory := entry.Data["memory_used"]
	return entry.Duration > 0 || hasDuration || hasResponseTime || hasMemory
}

// IsAPILog returns true if entry is related to API calls
func IsAPILog(entry *LogEntry) bool {
	_, hasProvider := entry.Data["provider"]
	_, hasTokens := entry.Data["tokens"]
	_, hasModel := entry.Data["model"]
	return hasProvider || hasTokens || hasModel
}

// IsUserAction returns true if entry represents a user action
func IsUserAction(entry *LogEntry) bool {
	action, hasAction := entry.Data["action"]
	if !hasAction {
		return false
	}

	userActions := map[string]bool{
		"query":    true,
		"search":   true,
		"generate": true,
		"explain":  true,
		"refactor": true,
		"feedback": true,
	}

	if actionStr, ok := action.(string); ok {
		return userActions[actionStr]
	}

	return false
}
