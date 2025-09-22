package display

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/fatih/color"
)

// Config holds display configuration
type Config struct {
	ColorsEnabled      bool
	Theme              string
	ShowLineNumbers    bool
	LineNumberWidth    int
	SyntaxHighlighting bool
	ProgressBarWidth   int
	StreamingDelay     time.Duration
	MaxLineLength      int
}

// Manager manages display output and formatting
type Manager struct {
	// config          *app.CLIDisplayConfig
	config          *Config
	colors          *ColorScheme
	streaming       bool
	streamingBuffer strings.Builder
	progressBars    map[string]*cli.ProgressBar
	mu              sync.RWMutex
}

// ColorScheme defines colors for different types of output
type ColorScheme struct {
	Primary   *color.Color
	Secondary *color.Color
	Success   *color.Color
	Warning   *color.Color
	Error     *color.Color
	Info      *color.Color
	Debug     *color.Color
	Muted     *color.Color
	Header    *color.Color
}

// Spinner represents a spinning indicator
type Spinner struct {
	title   string
	chars   []string
	current int
	active  bool
	stopCh  chan bool
	mu      sync.Mutex
}

// NewManager creates a new display manager
// func NewManager(config *app.CLIDisplayConfig) *Manager {
// 	manager := &Manager{
// 		config:       config,
// 		progressBars: make(map[string]*cli.ProgressBar),
// 	}

// 	manager.initializeColors()
// 	return manager
// }

// NewDisplayManager creates a new display manager
func NewDisplayManager(config *Config) *Manager {
	if config == nil {
		config = &Config{
			ColorsEnabled:    true,
			Theme:            "default",
			ShowLineNumbers:  false,
			LineNumberWidth:  4,
			ProgressBarWidth: 50,
			MaxLineLength:    80,
		}
	}

	manager := &Manager{
		config: config,
	}

	manager.initializeColors()
	return manager
}

// initializeColors sets up the color scheme
func (m *Manager) initializeColors() {
	if !m.config.ColorsEnabled {
		// Disable colors
		color.NoColor = true
		m.colors = &ColorScheme{
			Primary:   color.New(),
			Secondary: color.New(),
			Success:   color.New(),
			Warning:   color.New(),
			Error:     color.New(),
			Info:      color.New(),
			Debug:     color.New(),
			Muted:     color.New(),
			Header:    color.New(),
		}
		return
	}

	// Set up color scheme based on theme
	switch m.config.Theme {
	case "dark":
		m.colors = &ColorScheme{
			Primary:   color.New(color.FgCyan, color.Bold),
			Secondary: color.New(color.FgWhite),
			Success:   color.New(color.FgGreen, color.Bold),
			Warning:   color.New(color.FgYellow, color.Bold),
			Error:     color.New(color.FgRed, color.Bold),
			Info:      color.New(color.FgBlue),
			Debug:     color.New(color.FgMagenta),
			Muted:     color.New(color.FgHiBlack),
			Header:    color.New(color.FgCyan, color.Bold),
		}
	case "light":
		m.colors = &ColorScheme{
			Primary:   color.New(color.FgBlue, color.Bold),
			Secondary: color.New(color.FgBlack),
			Success:   color.New(color.FgGreen, color.Bold),
			Warning:   color.New(color.FgYellow, color.Bold),
			Error:     color.New(color.FgRed, color.Bold),
			Info:      color.New(color.FgBlue),
			Debug:     color.New(color.FgMagenta),
			Muted:     color.New(color.FgHiBlack),
			Header:    color.New(color.FgBlue, color.Bold),
		}
	default: // default theme
		m.colors = &ColorScheme{
			Primary:   color.New(color.FgCyan, color.Bold),
			Secondary: color.New(color.FgWhite),
			Success:   color.New(color.FgGreen, color.Bold),
			Warning:   color.New(color.FgYellow, color.Bold),
			Error:     color.New(color.FgRed, color.Bold),
			Info:      color.New(color.FgBlue),
			Debug:     color.New(color.FgMagenta),
			Muted:     color.New(color.FgHiBlack),
			Header:    color.New(color.FgCyan, color.Bold),
		}
	}
}

// Clear clears the screen
func (m *Manager) Clear() {
	fmt.Print("\033[2J\033[H")
}

// PrintHeader prints a header with formatting
func (m *Manager) PrintHeader(text string) {
	m.colors.Primary.Println(text)
}

// PrintSuccess prints a success message
func (m *Manager) PrintSuccess(text string) {
	m.colors.Success.Println(text)
}

// PrintError prints an error message
func (m *Manager) PrintError(text string) {
	m.colors.Error.Println(text)
}

// PrintWarning prints a warning message
func (m *Manager) PrintWarning(text string) {
	m.colors.Warning.Println(text)
}

// PrintInfo prints an info message
func (m *Manager) PrintInfo(text string) {
	m.colors.Info.Println(text)
}

// PrintDebug prints a debug message
func (m *Manager) PrintDebug(text string) {
	m.colors.Debug.Println(text)
}

// PrintMuted prints muted text
func (m *Manager) PrintMuted(text string) {
	m.colors.Muted.Println(text)
}

// PrintHeadered prints a headered section
func (m *Manager) PrintHeadered(text string) {
	m.colors.Header.Println(text)
}

// PrintSeparator prints a separator line
func (m *Manager) PrintSeparator() {
	separator := strings.Repeat("─", 60)
	m.colors.Muted.Println(separator)
}

// PrintResult prints a command result with appropriate formatting
func (m *Manager) PrintResult(result *cli.CommandResult) {
	if result.Success {
		if result.Output != "" {
			fmt.Print(result.Output)
		}
	} else {
		if result.Error != nil {
			m.PrintError(result.Error.Error())
		}
	}

	// Show metadata if verbose mode is enabled
	if m.config.ShowLineNumbers && result.Metadata != nil {
		if duration, ok := result.Metadata["duration"]; ok {
			m.PrintMuted(fmt.Sprintf("Duration: %v", duration))
		}
	}
}

// NewProgressBar creates a new progress bar
func (m *Manager) NewProgressBar(title string, total int) *cli.ProgressBar {
	pb := cli.NewProgressBar(title, total)
	pb.SetWidth(m.config.ProgressBarWidth)

	// Set colors
	colors := &cli.ProgressColors{
		Complete:   m.colors.Success,
		Incomplete: m.colors.Muted,
		Percentage: m.colors.Primary,
		Title:      m.colors.Secondary,
		ETA:        m.colors.Info,
	}
	pb.SetColors(colors)

	return pb
}

// StartStreaming starts streaming mode for real-time output
func (m *Manager) StartStreaming(title string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.streaming = true
	m.streamingBuffer.Reset()

	if title != "" {
		m.colors.Info.Printf("🔄 %s\n", title)
	}
}

// StreamLine adds a line to the streaming output
func (m *Manager) StreamLine(line string) {
	if !m.streaming {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Add line with optional syntax highlighting
	if m.config.SyntaxHighlighting {
		line = m.highlightSyntax(line)
	}

	fmt.Println(line)
	m.streamingBuffer.WriteString(line + "\n")

	// Add artificial delay for streaming effect
	if m.config.StreamingDelay > 0 {
		time.Sleep(time.Duration(m.config.StreamingDelay) * time.Millisecond)
	}
}

// StopStreaming stops streaming mode
func (m *Manager) StopStreaming() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.streaming = false
}

// GetStreamedContent returns the streamed content
func (m *Manager) GetStreamedContent() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.streamingBuffer.String()
}

// highlightSyntax applies syntax highlighting to a line
func (m *Manager) highlightSyntax(line string) string {
	// Simple syntax highlighting for common patterns

	// Go keywords
	goKeywords := []string{"func", "var", "const", "if", "else", "for", "range", "return", "package", "import", "struct", "type", "interface", "switch", "case", "default", "break", "continue", "go", "defer", "map", "chan", "select"}
	for _, keyword := range goKeywords {
		if strings.Contains(line, keyword+" ") {
			line = strings.ReplaceAll(line, keyword+" ", m.colors.Primary.Sprint(keyword)+" ")
		}
	}

	// String literals
	if strings.Contains(line, "\"") {
		// This is a simplified version - a full implementation would need proper parsing
		parts := strings.Split(line, "\"")
		for i := 1; i < len(parts); i += 2 {
			if i < len(parts) {
				parts[i] = m.colors.Success.Sprint("\"" + parts[i] + "\"")
			}
		}
		line = strings.Join(parts, "")
	}

	// Comments
	if strings.Contains(line, "//") {
		commentIndex := strings.Index(line, "//")
		if commentIndex >= 0 {
			before := line[:commentIndex]
			comment := line[commentIndex:]
			line = before + m.colors.Muted.Sprint(comment)
		}
	}

	return line
}

// PrintCode prints code with syntax highlighting and line numbers
func (m *Manager) PrintCode(code, language string) {
	lines := strings.Split(code, "\n")

	for i, line := range lines {
		lineNum := ""
		if m.config.ShowLineNumbers {
			lineNum = m.colors.Muted.Sprintf("%*d │ ", m.config.LineNumberWidth, i+1)
		}

		// Apply syntax highlighting
		if m.config.SyntaxHighlighting.Enabled {
			line = m.highlightSyntax(line)
		}

		fmt.Printf("%s%s\n", lineNum, line)
	}
}

// PrintTable prints data in a table format
func (m *Manager) PrintTable(headers []string, rows [][]string) {
	if len(headers) == 0 || len(rows) == 0 {
		return
	}

	// Calculate column widths
	colWidths := make([]int, len(headers))
	for i, header := range headers {
		colWidths[i] = len(header)
	}

	for _, row := range rows {
		for i, cell := range row {
			if i < len(colWidths) && len(cell) > colWidths[i] {
				colWidths[i] = len(cell)
			}
		}
	}

	// Print header
	m.colors.Primary.Print("┌")
	for i, width := range colWidths {
		m.colors.Primary.Print(strings.Repeat("─", width+2))
		if i < len(colWidths)-1 {
			m.colors.Primary.Print("┬")
		}
	}
	m.colors.Primary.Print("┐\n")

	// Print header row
	m.colors.Primary.Print("│")
	for i, header := range headers {
		fmt.Printf(" %-*s ", colWidths[i], header)
		m.colors.Primary.Print("│")
	}
	fmt.Println()

	// Print separator
	m.colors.Primary.Print("├")
	for i, width := range colWidths {
		m.colors.Primary.Print(strings.Repeat("─", width+2))
		if i < len(colWidths)-1 {
			m.colors.Primary.Print("┼")
		}
	}
	m.colors.Primary.Print("┤\n")

	// Print data rows
	for _, row := range rows {
		m.colors.Primary.Print("│")
		for i, cell := range row {
			if i < len(colWidths) {
				fmt.Printf(" %-*s ", colWidths[i], cell)
			}
			m.colors.Primary.Print("│")
		}
		fmt.Println()
	}

	// Print bottom border
	m.colors.Primary.Print("└")
	for i, width := range colWidths {
		m.colors.Primary.Print(strings.Repeat("─", width+2))
		if i < len(colWidths)-1 {
			m.colors.Primary.Print("┴")
		}
	}
	m.colors.Primary.Print("┘\n")
}

// PrintKeyValue prints key-value pairs with formatting
func (m *Manager) PrintKeyValue(pairs map[string]interface{}) {
	maxKeyLength := 0
	for key := range pairs {
		if len(key) > maxKeyLength {
			maxKeyLength = len(key)
		}
	}

	for key, value := range pairs {
		m.colors.Primary.Printf("%-*s: ", maxKeyLength, key)
		fmt.Printf("%v\n", value)
	}
}

// Confirm prompts for user confirmation
func (m *Manager) Confirm(message string) bool {
	m.colors.Warning.Printf("%s (y/N): ", message)

	scanner := bufio.NewScanner(os.Stdin)
	if scanner.Scan() {
		response := strings.ToLower(strings.TrimSpace(scanner.Text()))
		return response == "y" || response == "yes"
	}

	return false
}

// ReadInput reads user input with a prompt
func (m *Manager) ReadInput(prompt string) (string, error) {
	m.colors.Primary.Print(prompt)

	scanner := bufio.NewScanner(os.Stdin)
	if scanner.Scan() {
		return scanner.Text(), nil
	}

	return "", scanner.Err()
}

// PrintBox prints text in a box
func (m *Manager) PrintBox(title, content string) {
	lines := strings.Split(content, "\n")
	width := len(title)

	// Find maximum line length
	for _, line := range lines {
		if len(line) > width {
			width = len(line)
		}
	}

	width += 4 // Add padding

	// Top border
	m.colors.Primary.Print("╔")
	m.colors.Primary.Print(strings.Repeat("═", width))
	m.colors.Primary.Print("╗\n")

	// Title
	if title != "" {
		padding := (width - len(title)) / 2
		m.colors.Primary.Print("║")
		fmt.Print(strings.Repeat(" ", padding))
		m.colors.Primary.Print(title)
		fmt.Print(strings.Repeat(" ", width-len(title)-padding))
		m.colors.Primary.Print("║\n")

		// Title separator
		m.colors.Primary.Print("╠")
		m.colors.Primary.Print(strings.Repeat("═", width))
		m.colors.Primary.Print("╣\n")
	}

	// Content
	for _, line := range lines {
		m.colors.Primary.Print("║ ")
		fmt.Print(line)
		fmt.Print(strings.Repeat(" ", width-len(line)-2))
		m.colors.Primary.Print(" ║\n")
	}

	// Bottom border
	m.colors.Primary.Print("╚")
	m.colors.Primary.Print(strings.Repeat("═", width))
	m.colors.Primary.Print("╝\n")
}

// PrintList prints a list with bullets
func (m *Manager) PrintList(items []string, numbered bool) {
	for i, item := range items {
		if numbered {
			m.colors.Primary.Printf("%d. ", i+1)
		} else {
			m.colors.Primary.Print("• ")
		}
		fmt.Println(item)
	}
}

// WrapText wraps text to fit within the specified width
func (m *Manager) WrapText(text string, width int) []string {
	if width <= 0 {
		width = 80
	}

	words := strings.Fields(text)
	if len(words) == 0 {
		return []string{""}
	}

	var lines []string
	var currentLine strings.Builder

	for _, word := range words {
		// Check if adding this word would exceed the width
		if currentLine.Len() > 0 && currentLine.Len()+1+len(word) > width {
			lines = append(lines, currentLine.String())
			currentLine.Reset()
		}

		if currentLine.Len() > 0 {
			currentLine.WriteString(" ")
		}
		currentLine.WriteString(word)
	}

	if currentLine.Len() > 0 {
		lines = append(lines, currentLine.String())
	}

	return lines
}

// PrintWrappedText prints text wrapped to fit the terminal width
func (m *Manager) PrintWrappedText(text string) {
	width := 80 // Default width, could be detected from terminal
	if m.config.MaxLineLength > 0 {
		width = m.config.MaxLineLength
	}

	lines := m.WrapText(text, width)
	for _, line := range lines {
		fmt.Println(line)
	}
}

// ShowTypingEffect shows a typing effect for text
func (m *Manager) ShowTypingEffect(text string, speed time.Duration) {
	for _, char := range text {
		fmt.Print(string(char))
		time.Sleep(speed)
	}
	fmt.Println()
}

// GetColors returns the color scheme
func (m *Manager) GetColors() *ColorScheme {
	return m.colors
}

// SetTheme changes the color theme
func (m *Manager) SetTheme(theme string) {
	m.config.Colors.Theme = theme
	m.initializeColors()
}

// IsColorsEnabled returns whether colors are enabled
func (m *Manager) IsColorsEnabled() bool {
	return m.config.Colors.Enabled
}
