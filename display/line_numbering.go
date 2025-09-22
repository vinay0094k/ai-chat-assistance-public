// display/line_numbering.go
package display

import (
	"fmt"
	"strings"
)

type LineNumberStyle int

const (
	LineNumberNone LineNumberStyle = iota
	LineNumberSimple
	LineNumberWithSeparator
	LineNumberHighlight
	LineNumberGutter
	LineNumberDiff
)

type LineNumberConfig struct {
	Style           LineNumberStyle
	Width           int
	StartFrom       int
	ShowOnlyContext bool
	ContextLines    int
	HighlightLines  []int
	DiffLines       map[int]DiffType
	Padding         int
	ShowRelative    bool
	CurrentLine     int
}

type DiffType int

const (
	DiffNone DiffType = iota
	DiffAdded
	DiffRemoved
	DiffModified
	DiffContext
)

type LineNumberer struct {
	config *LineNumberConfig
	theme  *ThemeManager
}

type LineDisplay struct {
	Number      int
	Content     string
	IsHighlight bool
	DiffType    DiffType
	IsContext   bool
	IsRelative  bool
	Distance    int
}

// NewLineNumberer creates a new line numberer
func NewLineNumberer(config *LineNumberConfig, theme *ThemeManager) *LineNumberer {
	if config == nil {
		config = DefaultLineNumberConfig()
	}
	if theme == nil {
		theme = GlobalTheme
	}

	return &LineNumberer{
		config: config,
		theme:  theme,
	}
}

// DefaultLineNumberConfig returns default configuration
func DefaultLineNumberConfig() *LineNumberConfig {
	return &LineNumberConfig{
		Style:           LineNumberWithSeparator,
		Width:           4,
		StartFrom:       1,
		ShowOnlyContext: false,
		ContextLines:    3,
		HighlightLines:  []int{},
		DiffLines:       make(map[int]DiffType),
		Padding:         1,
		ShowRelative:    false,
		CurrentLine:     -1,
	}
}

// FormatLines formats multiple lines with line numbers
func (ln *LineNumberer) FormatLines(lines []string) []string {
	displays := ln.createLineDisplays(lines)
	return ln.renderLineDisplays(displays)
}

// FormatLine formats a single line with line number
func (ln *LineNumberer) FormatLine(lineNum int, content string) string {
	display := LineDisplay{
		Number:      lineNum,
		Content:     content,
		IsHighlight: ln.isHighlightLine(lineNum),
		DiffType:    ln.getDiffType(lineNum),
		IsContext:   ln.isContextLine(lineNum),
		IsRelative:  ln.config.ShowRelative && ln.config.CurrentLine > 0,
		Distance:    ln.getDistance(lineNum),
	}

	return ln.renderLineDisplay(display)
}

// FormatCodeBlock formats a code block with line numbers
func (ln *LineNumberer) FormatCodeBlock(content string, startLine int) string {
	lines := strings.Split(content, "\n")

	// Adjust start line
	oldStart := ln.config.StartFrom
	ln.config.StartFrom = startLine

	formatted := ln.FormatLines(lines)

	// Restore original start line
	ln.config.StartFrom = oldStart

	return strings.Join(formatted, "\n")
}

// FormatDiff formats a diff view with appropriate line numbers
func (ln *LineNumberer) FormatDiff(oldLines, newLines []string, changes map[int]DiffType) string {
	// Store original config
	originalStyle := ln.config.Style
	originalDiff := ln.config.DiffLines

	// Set diff mode
	ln.config.Style = LineNumberDiff
	ln.config.DiffLines = changes

	var result []string

	// Process diff lines
	maxLines := len(oldLines)
	if len(newLines) > maxLines {
		maxLines = len(newLines)
	}

	for i := 0; i < maxLines; i++ {
		lineNum := i + ln.config.StartFrom

		// Get content based on diff type
		var content string
		diffType := ln.getDiffType(lineNum)

		switch diffType {
		case DiffAdded:
			if i < len(newLines) {
				content = newLines[i]
			}
		case DiffRemoved:
			if i < len(oldLines) {
				content = oldLines[i]
			}
		case DiffModified:
			if i < len(newLines) {
				content = newLines[i]
			}
		default:
			if i < len(oldLines) {
				content = oldLines[i]
			} else if i < len(newLines) {
				content = newLines[i]
			}
		}

		formatted := ln.FormatLine(lineNum, content)
		result = append(result, formatted)
	}

	// Restore original config
	ln.config.Style = originalStyle
	ln.config.DiffLines = originalDiff

	return strings.Join(result, "\n")
}

// createLineDisplays creates line display structures
func (ln *LineNumberer) createLineDisplays(lines []string) []LineDisplay {
	var displays []LineDisplay

	for i, content := range lines {
		lineNum := i + ln.config.StartFrom

		display := LineDisplay{
			Number:      lineNum,
			Content:     content,
			IsHighlight: ln.isHighlightLine(lineNum),
			DiffType:    ln.getDiffType(lineNum),
			IsContext:   ln.isContextLine(lineNum),
			IsRelative:  ln.config.ShowRelative && ln.config.CurrentLine > 0,
			Distance:    ln.getDistance(lineNum),
		}

		// Skip non-context lines if configured
		if ln.config.ShowOnlyContext && !display.IsContext && !display.IsHighlight {
			continue
		}

		displays = append(displays, display)
	}

	return displays
}

// renderLineDisplays renders multiple line displays
func (ln *LineNumberer) renderLineDisplays(displays []LineDisplay) []string {
	var result []string

	for _, display := range displays {
		rendered := ln.renderLineDisplay(display)
		result = append(result, rendered)
	}

	return result
}

// renderLineDisplay renders a single line display
func (ln *LineNumberer) renderLineDisplay(display LineDisplay) string {
	if ln.config.Style == LineNumberNone {
		return display.Content
	}

	// Format line number
	var lineNumStr string
	if display.IsRelative && ln.config.CurrentLine > 0 {
		if display.Number == ln.config.CurrentLine {
			lineNumStr = fmt.Sprintf("%*d", ln.config.Width, display.Number)
		} else {
			lineNumStr = fmt.Sprintf("%*d", ln.config.Width, display.Distance)
		}
	} else {
		lineNumStr = fmt.Sprintf("%*d", ln.config.Width, display.Number)
	}

	// Apply padding
	padding := strings.Repeat(" ", ln.config.Padding)

	// Choose colors based on state
	var lineNumColor, separatorColor, contentColor string

	if display.IsHighlight {
		lineNumColor = "highlight"
		separatorColor = "highlight"
		contentColor = "highlight"
	} else if display.Number == ln.config.CurrentLine {
		lineNumColor = "primary"
		separatorColor = "primary"
		contentColor = "primary"
	} else {
		lineNumColor = "line_num"
		separatorColor = "border"
		contentColor = "code"
	}

	// Handle diff coloring
	switch display.DiffType {
	case DiffAdded:
		lineNumColor = "success"
		contentColor = "success"
		separatorColor = "success"
	case DiffRemoved:
		lineNumColor = "error"
		contentColor = "error"
		separatorColor = "error"
	case DiffModified:
		lineNumColor = "warning"
		contentColor = "warning"
		separatorColor = "warning"
	}

	// Build the formatted line
	var parts []string

	// Add line number
	parts = append(parts, ln.theme.Sprint(lineNumColor, lineNumStr))

	// Add separator based on style
	switch ln.config.Style {
	case LineNumberSimple:
		parts = append(parts, padding)
	case LineNumberWithSeparator:
		parts = append(parts, ln.theme.Sprint(separatorColor, "│"))
		parts = append(parts, padding)
	case LineNumberHighlight:
		if display.IsHighlight {
			parts = append(parts, ln.theme.Sprint(separatorColor, "►"))
		} else {
			parts = append(parts, ln.theme.Sprint(separatorColor, "│"))
		}
		parts = append(parts, padding)
	case LineNumberGutter:
		parts = append(parts, ln.theme.Sprint(separatorColor, "┃"))
		parts = append(parts, padding)
	case LineNumberDiff:
		switch display.DiffType {
		case DiffAdded:
			parts = append(parts, ln.theme.Sprint(separatorColor, "+"))
		case DiffRemoved:
			parts = append(parts, ln.theme.Sprint(separatorColor, "-"))
		case DiffModified:
			parts = append(parts, ln.theme.Sprint(separatorColor, "~"))
		default:
			parts = append(parts, ln.theme.Sprint(separatorColor, " "))
		}
		parts = append(parts, padding)
	}

	// Add content
	parts = append(parts, ln.theme.Sprint(contentColor, display.Content))

	return strings.Join(parts, "")
}

// Helper methods

func (ln *LineNumberer) isHighlightLine(lineNum int) bool {
	for _, highlightLine := range ln.config.HighlightLines {
		if lineNum == highlightLine {
			return true
		}
	}
	return false
}

func (ln *LineNumberer) getDiffType(lineNum int) DiffType {
	if diffType, exists := ln.config.DiffLines[lineNum]; exists {
		return diffType
	}
	return DiffNone
}

func (ln *LineNumberer) isContextLine(lineNum int) bool {
	if ln.config.CurrentLine <= 0 {
		return true // Show all lines if no current line set
	}

	distance := ln.getDistance(lineNum)
	return distance <= ln.config.ContextLines
}

func (ln *LineNumberer) getDistance(lineNum int) int {
	if ln.config.CurrentLine <= 0 {
		return 0
	}

	distance := lineNum - ln.config.CurrentLine
	if distance < 0 {
		distance = -distance
	}
	return distance
}

// Configuration methods

func (ln *LineNumberer) SetStyle(style LineNumberStyle) {
	ln.config.Style = style
}

func (ln *LineNumberer) SetWidth(width int) {
	ln.config.Width = width
}

func (ln *LineNumberer) SetCurrentLine(lineNum int) {
	ln.config.CurrentLine = lineNum
}

func (ln *LineNumberer) SetHighlightLines(lines []int) {
	ln.config.HighlightLines = lines
}

func (ln *LineNumberer) AddHighlightLine(lineNum int) {
	ln.config.HighlightLines = append(ln.config.HighlightLines, lineNum)
}

func (ln *LineNumberer) SetDiffLines(diffLines map[int]DiffType) {
	ln.config.DiffLines = diffLines
}

func (ln *LineNumberer) SetContextLines(contextLines int) {
	ln.config.ContextLines = contextLines
}

func (ln *LineNumberer) SetShowOnlyContext(showOnly bool) {
	ln.config.ShowOnlyContext = showOnly
}

func (ln *LineNumberer) SetShowRelative(showRelative bool) {
	ln.config.ShowRelative = showRelative
}

// Utility functions

// CreateContextWindow creates a context window around specific lines
func CreateContextWindow(lines []string, targetLines []int, contextSize int) ([]string, *LineNumberConfig) {
	if len(lines) == 0 || len(targetLines) == 0 {
		return lines, DefaultLineNumberConfig()
	}

	// Find the range of lines to include
	minLine := targetLines[0]
	maxLine := targetLines[0]

	for _, line := range targetLines {
		if line < minLine {
			minLine = line
		}
		if line > maxLine {
			maxLine = line
		}
	}

	// Expand the range to include context
	start := minLine - contextSize
	if start < 1 {
		start = 1
	}

	end := maxLine + contextSize
	if end > len(lines) {
		end = len(lines)
	}

	// Extract the context window
	contextLines := lines[start-1 : end]

	// Create configuration
	config := DefaultLineNumberConfig()
	config.StartFrom = start
	config.HighlightLines = targetLines
	config.ShowOnlyContext = false
	config.ContextLines = contextSize

	return contextLines, config
}
