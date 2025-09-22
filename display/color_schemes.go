// display/color_schemes.go
package display

import (
	"fmt"
	"os"
	"strings"

	"github.com/fatih/color"
)

type ColorScheme struct {
	Name        string
	Description string
	Colors      map[string]*color.Color
	Background  string
	Foreground  string
}

type ThemeManager struct {
	schemes       map[string]*ColorScheme
	currentScheme *ColorScheme
	supportsColor bool
}

// Predefined color schemes
var (
	// Dark themes
	DarkScheme = &ColorScheme{
		Name:        "dark",
		Description: "Dark theme with vibrant colors",
		Background:  "black",
		Foreground:  "white",
		Colors: map[string]*color.Color{
			"primary":     color.New(color.FgCyan, color.Bold),
			"secondary":   color.New(color.FgBlue),
			"success":     color.New(color.FgGreen, color.Bold),
			"warning":     color.New(color.FgYellow, color.Bold),
			"error":       color.New(color.FgRed, color.Bold),
			"info":        color.New(color.FgBlue),
			"muted":       color.New(color.FgHiBlack),
			"highlight":   color.New(color.FgMagenta, color.Bold),
			"code":        color.New(color.FgHiCyan),
			"keyword":     color.New(color.FgMagenta),
			"string":      color.New(color.FgGreen),
			"comment":     color.New(color.FgHiBlack),
			"number":      color.New(color.FgYellow),
			"function":    color.New(color.FgHiBlue),
			"variable":    color.New(color.FgWhite),
			"type":        color.New(color.FgHiGreen),
			"line_num":    color.New(color.FgHiBlack),
			"border":      color.New(color.FgHiBlack),
			"prompt":      color.New(color.FgCyan, color.Bold),
			"ai_response": color.New(color.FgHiGreen),
		},
	}

	LightScheme = &ColorScheme{
		Name:        "light",
		Description: "Light theme with subdued colors",
		Background:  "white",
		Foreground:  "black",
		Colors: map[string]*color.Color{
			"primary":     color.New(color.FgBlue, color.Bold),
			"secondary":   color.New(color.FgHiBlue),
			"success":     color.New(color.FgHiGreen, color.Bold),
			"warning":     color.New(color.FgHiYellow, color.Bold),
			"error":       color.New(color.FgHiRed, color.Bold),
			"info":        color.New(color.FgBlue),
			"muted":       color.New(color.FgHiBlack),
			"highlight":   color.New(color.FgMagenta, color.Bold),
			"code":        color.New(color.FgBlue),
			"keyword":     color.New(color.FgHiMagenta),
			"string":      color.New(color.FgHiGreen),
			"comment":     color.New(color.FgHiBlack),
			"number":      color.New(color.FgHiYellow),
			"function":    color.New(color.FgHiBlue),
			"variable":    color.New(color.FgBlack),
			"type":        color.New(color.FgHiGreen),
			"line_num":    color.New(color.FgHiBlack),
			"border":      color.New(color.FgHiBlack),
			"prompt":      color.New(color.FgBlue, color.Bold),
			"ai_response": color.New(color.FgHiGreen),
		},
	}

	HackerScheme = &ColorScheme{
		Name:        "hacker",
		Description: "Matrix-style green on black",
		Background:  "black",
		Foreground:  "green",
		Colors: map[string]*color.Color{
			"primary":     color.New(color.FgHiGreen, color.Bold),
			"secondary":   color.New(color.FgGreen),
			"success":     color.New(color.FgHiGreen, color.Bold),
			"warning":     color.New(color.FgYellow),
			"error":       color.New(color.FgRed, color.Bold),
			"info":        color.New(color.FgGreen),
			"muted":       color.New(color.FgHiBlack),
			"highlight":   color.New(color.FgHiGreen, color.Bold, color.Reverse),
			"code":        color.New(color.FgHiGreen),
			"keyword":     color.New(color.FgGreen, color.Bold),
			"string":      color.New(color.FgGreen),
			"comment":     color.New(color.FgHiBlack),
			"number":      color.New(color.FgHiGreen),
			"function":    color.New(color.FgGreen, color.Bold),
			"variable":    color.New(color.FgGreen),
			"type":        color.New(color.FgHiGreen),
			"line_num":    color.New(color.FgHiBlack),
			"border":      color.New(color.FgGreen),
			"prompt":      color.New(color.FgHiGreen, color.Bold),
			"ai_response": color.New(color.FgGreen),
		},
	}

	SolarizedScheme = &ColorScheme{
		Name:        "solarized",
		Description: "Solarized color palette",
		Background:  "#002b36",
		Foreground:  "#839496",
		Colors: map[string]*color.Color{
			"primary":     color.New(color.FgHiCyan, color.Bold),
			"secondary":   color.New(color.FgCyan),
			"success":     color.New(color.FgHiGreen, color.Bold),
			"warning":     color.New(color.FgHiYellow, color.Bold),
			"error":       color.New(color.FgHiRed, color.Bold),
			"info":        color.New(color.FgCyan),
			"muted":       color.New(color.FgHiBlack),
			"highlight":   color.New(color.FgMagenta, color.Bold),
			"code":        color.New(color.FgHiCyan),
			"keyword":     color.New(color.FgMagenta),
			"string":      color.New(color.FgGreen),
			"comment":     color.New(color.FgHiBlack),
			"number":      color.New(color.FgYellow),
			"function":    color.New(color.FgHiBlue),
			"variable":    color.New(color.FgWhite),
			"type":        color.New(color.FgHiGreen),
			"line_num":    color.New(color.FgHiBlack),
			"border":      color.New(color.FgHiBlack),
			"prompt":      color.New(color.FgHiCyan, color.Bold),
			"ai_response": color.New(color.FgHiGreen),
		},
	}
)

// NewThemeManager creates a new theme manager
func NewThemeManager() *ThemeManager {
	tm := &ThemeManager{
		schemes:       make(map[string]*ColorScheme),
		supportsColor: checkColorSupport(),
	}

	// Register built-in schemes
	tm.RegisterScheme(DarkScheme)
	tm.RegisterScheme(LightScheme)
	tm.RegisterScheme(HackerScheme)
	tm.RegisterScheme(SolarizedScheme)

	// Set default theme
	tm.SetScheme("dark")

	return tm
}

// RegisterScheme adds a new color scheme
func (tm *ThemeManager) RegisterScheme(scheme *ColorScheme) {
	tm.schemes[scheme.Name] = scheme
}

// SetScheme changes the current color scheme
func (tm *ThemeManager) SetScheme(name string) error {
	scheme, exists := tm.schemes[name]
	if !exists {
		return fmt.Errorf("color scheme '%s' not found", name)
	}
	tm.currentScheme = scheme
	return nil
}

// GetScheme returns the current color scheme
func (tm *ThemeManager) GetScheme() *ColorScheme {
	return tm.currentScheme
}

// ListSchemes returns all available color schemes
func (tm *ThemeManager) ListSchemes() []string {
	var names []string
	for name := range tm.schemes {
		names = append(names, name)
	}
	return names
}

// Color returns a color for the given type
func (tm *ThemeManager) Color(colorType string) *color.Color {
	if !tm.supportsColor {
		return color.New() // No color
	}

	if tm.currentScheme == nil {
		return color.New(color.FgWhite)
	}

	if c, exists := tm.currentScheme.Colors[colorType]; exists {
		return c
	}

	// Return default color if type not found
	return color.New(color.FgWhite)
}

// Sprint applies color to text
func (tm *ThemeManager) Sprint(colorType, text string) string {
	return tm.Color(colorType).Sprint(text)
}

// Sprintf applies color with formatting
func (tm *ThemeManager) Sprintf(colorType, format string, args ...interface{}) string {
	return tm.Color(colorType).Sprintf(format, args...)
}

// Print outputs colored text
func (tm *ThemeManager) Print(colorType, text string) {
	tm.Color(colorType).Print(text)
}

// Printf outputs formatted colored text
func (tm *ThemeManager) Printf(colorType, format string, args ...interface{}) {
	tm.Color(colorType).Printf(format, args...)
}

// Println outputs colored text with newline
func (tm *ThemeManager) Println(colorType, text string) {
	tm.Color(colorType).Println(text)
}

// PrintBorder prints a styled border
func (tm *ThemeManager) PrintBorder(width int, char rune) {
	border := strings.Repeat(string(char), width)
	tm.Println("border", border)
}

// PrintHeader prints a styled header
func (tm *ThemeManager) PrintHeader(title string) {
	width := 60
	padding := (width - len(title) - 2) / 2

	tm.PrintBorder(width, '═')
	headerLine := fmt.Sprintf("║%s %s %s║",
		strings.Repeat(" ", padding),
		title,
		strings.Repeat(" ", width-len(title)-padding-4))
	tm.Println("primary", headerLine)
	tm.PrintBorder(width, '═')
}

// PrintSection prints a styled section divider
func (tm *ThemeManager) PrintSection(title string) {
	tm.Printf("secondary", "\n▶ %s\n", title)
	tm.PrintBorder(40, '─')
}

// PrintSuccess prints a success message
func (tm *ThemeManager) PrintSuccess(message string) {
	tm.Printf("success", "✓ %s\n", message)
}

// PrintWarning prints a warning message
func (tm *ThemeManager) PrintWarning(message string) {
	tm.Printf("warning", "⚠ %s\n", message)
}

// PrintError prints an error message
func (tm *ThemeManager) PrintError(message string) {
	tm.Printf("error", "✗ %s\n", message)
}

// PrintInfo prints an info message
func (tm *ThemeManager) PrintInfo(message string) {
	tm.Printf("info", "ℹ %s\n", message)
}

// PrintCodeBlock prints a syntax-highlighted code block
func (tm *ThemeManager) PrintCodeBlock(language, code string) {
	tm.PrintSection(fmt.Sprintf("Code (%s)", language))
	lines := strings.Split(code, "\n")

	for i, line := range lines {
		lineNum := fmt.Sprintf("%3d", i+1)
		tm.Printf("line_num", "%s ", lineNum)
		tm.Printf("border", "│ ")
		tm.Println("code", line)
	}
}

// PrintProgress prints a progress bar
func (tm *ThemeManager) PrintProgress(current, total int, title string) {
	width := 40
	progress := float64(current) / float64(total)
	filled := int(progress * float64(width))

	bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
	percentage := int(progress * 100)

	tm.Printf("info", "%s [", title)
	tm.Printf("primary", "%s", bar)
	tm.Printf("info", "] %d%%\n", percentage)
}

// checkColorSupport checks if terminal supports colors
func checkColorSupport() bool {
	term := os.Getenv("TERM")
	if term == "" {
		return false
	}

	// Check for common terminals that support color
	colorTerms := []string{"xterm", "screen", "tmux", "rxvt", "ansi"}
	for _, colorTerm := range colorTerms {
		if strings.Contains(term, colorTerm) {
			return true
		}
	}

	// Check for color capability environment variables
	if os.Getenv("COLORTERM") != "" {
		return true
	}

	return term != "dumb"
}

// Global theme manager instance
var GlobalTheme = NewThemeManager()
