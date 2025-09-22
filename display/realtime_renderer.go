// display/realtime_renderer.go
package display

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

type RenderBuffer struct {
	Width     int
	Height    int
	Cells     [][]Cell
	DirtyRows map[int]bool
	mu        sync.RWMutex
}

type Cell struct {
	Char  rune
	Style CellStyle
	Dirty bool
}

type CellStyle struct {
	ForegroundColor string
	BackgroundColor string
	Bold            bool
	Italic          bool
	Underline       bool
	Blink           bool
	Reverse         bool
}

type CursorPosition struct {
	X int
	Y int
}

type RealtimeRenderer struct {
	buffer          *RenderBuffer
	previousBuffer  *RenderBuffer
	theme           *ThemeManager
	cursor          CursorPosition
	cursorVisible   bool
	optimizeUpdates bool
	doubleBuffering bool
	frameRate       time.Duration

	// Terminal capabilities
	terminalWidth  int
	terminalHeight int
	supportsColor  bool
	supportsUTF8   bool

	// Performance metrics
	frameCount  int64
	renderTimes []time.Duration

	// Control channels
	ctx        context.Context
	cancel     context.CancelFunc
	renderChan chan RenderRequest
	running    bool
	mu         sync.RWMutex
}

type RenderRequest struct {
	X, Y     int
	Width    int
	Height   int
	Content  []string
	Style    CellStyle
	Priority int
}

// NewRealtimeRenderer creates a new realtime renderer
func NewRealtimeRenderer(width, height int, theme *ThemeManager) *RealtimeRenderer {
	ctx, cancel := context.WithCancel(context.Background())

	if theme == nil {
		theme = GlobalTheme
	}

	renderer := &RealtimeRenderer{
		buffer:          NewRenderBuffer(width, height),
		previousBuffer:  NewRenderBuffer(width, height),
		theme:           theme,
		cursor:          CursorPosition{X: 0, Y: 0},
		cursorVisible:   true,
		optimizeUpdates: true,
		doubleBuffering: true,
		frameRate:       16 * time.Millisecond, // 60 FPS
		terminalWidth:   width,
		terminalHeight:  height,
		supportsColor:   checkColorSupport(),
		supportsUTF8:    checkUTF8Support(),
		ctx:             ctx,
		cancel:          cancel,
		renderChan:      make(chan RenderRequest, 1000),
		renderTimes:     make([]time.Duration, 0, 100),
	}

	return renderer
}

// NewRenderBuffer creates a new render buffer
func NewRenderBuffer(width, height int) *RenderBuffer {
	buffer := &RenderBuffer{
		Width:     width,
		Height:    height,
		Cells:     make([][]Cell, height),
		DirtyRows: make(map[int]bool),
	}

	// Initialize cells
	for y := 0; y < height; y++ {
		buffer.Cells[y] = make([]Cell, width)
		for x := 0; x < width; x++ {
			buffer.Cells[y][x] = Cell{
				Char:  ' ',
				Style: CellStyle{},
				Dirty: true,
			}
		}
	}

	return buffer
}

// Start begins the renderer
func (rr *RealtimeRenderer) Start() error {
	rr.mu.Lock()
	if rr.running {
		rr.mu.Unlock()
		return fmt.Errorf("renderer already running")
	}
	rr.running = true
	rr.mu.Unlock()

	// Initialize terminal
	if err := rr.initializeTerminal(); err != nil {
		return fmt.Errorf("failed to initialize terminal: %w", err)
	}

	// Start render loop
	go rr.renderLoop()

	return nil
}

// Stop stops the renderer
func (rr *RealtimeRenderer) Stop() {
	rr.mu.Lock()
	if !rr.running {
		rr.mu.Unlock()
		return
	}
	rr.running = false
	rr.mu.Unlock()

	rr.cancel()
	rr.restoreTerminal()
}

// renderLoop is the main rendering loop
func (rr *RealtimeRenderer) renderLoop() {
	ticker := time.NewTicker(rr.frameRate)
	defer ticker.Stop()

	for {
		select {
		case <-rr.ctx.Done():
			return
		case <-ticker.C:
			rr.renderFrame()
		case request := <-rr.renderChan:
			rr.processRenderRequest(request)
		}
	}
}

// renderFrame renders a single frame
func (rr *RealtimeRenderer) renderFrame() {
	startTime := time.Now()

	rr.buffer.mu.RLock()
	defer rr.buffer.mu.RUnlock()

	if rr.optimizeUpdates {
		rr.renderDifferential()
	} else {
		rr.renderFull()
	}

	// Update performance metrics
	renderTime := time.Since(startTime)
	rr.updatePerformanceMetrics(renderTime)

	// Swap buffers if double buffering is enabled
	if rr.doubleBuffering {
		rr.swapBuffers()
	}
}

// renderDifferential renders only changed parts
func (rr *RealtimeRenderer) renderDifferential() {
	for y := range rr.buffer.DirtyRows {
		if y >= 0 && y < rr.buffer.Height {
			rr.renderRow(y)
			delete(rr.buffer.DirtyRows, y)
		}
	}
}

// renderFull renders the entire buffer
func (rr *RealtimeRenderer) renderFull() {
	// Move cursor to top-left
	fmt.Print("\033[H")

	for y := 0; y < rr.buffer.Height; y++ {
		rr.renderRow(y)
	}
}

// renderRow renders a single row
func (rr *RealtimeRenderer) renderRow(y int) {
	if y < 0 || y >= rr.buffer.Height {
		return
	}

	// Move cursor to row
	fmt.Printf("\033[%d;1H", y+1)

	var output strings.Builder
	var lastStyle CellStyle
	var styleChanged bool

	for x := 0; x < rr.buffer.Width; x++ {
		cell := rr.buffer.Cells[y][x]

		// Check if style changed
		if x == 0 || cell.Style != lastStyle {
			styleChanged = true
			lastStyle = cell.Style
		}

		// Apply style if changed and color is supported
		if styleChanged && rr.supportsColor {
			output.WriteString(rr.cellStyleToANSI(cell.Style))
			styleChanged = false
		}

		// Write character
		if rr.supportsUTF8 || cell.Char <= 127 {
			output.WriteRune(cell.Char)
		} else {
			output.WriteRune('?') // Fallback for non-UTF8 terminals
		}

		// Mark cell as clean
		rr.buffer.Cells[y][x].Dirty = false
	}

	// Reset style at end of row
	if rr.supportsColor {
		output.WriteString("\033[0m")
	}

	fmt.Print(output.String())
}

// RenderText renders text at specified position
func (rr *RealtimeRenderer) RenderText(x, y int, text string, style CellStyle) {
	request := RenderRequest{
		X:        x,
		Y:        y,
		Width:    utf8.RuneCountInString(text),
		Height:   1,
		Content:  []string{text},
		Style:    style,
		Priority: 0,
	}

	select {
	case rr.renderChan <- request:
	default:
		// Channel full, process immediately
		rr.processRenderRequest(request)
	}
}

// RenderBlock renders a block of text
func (rr *RealtimeRenderer) RenderBlock(x, y int, lines []string, style CellStyle) {
	request := RenderRequest{
		X:        x,
		Y:        y,
		Width:    rr.calculateMaxWidth(lines),
		Height:   len(lines),
		Content:  lines,
		Style:    style,
		Priority: 0,
	}

	select {
	case rr.renderChan <- request:
	default:
		rr.processRenderRequest(request)
	}
}

// RenderBox renders a styled box
func (rr *RealtimeRenderer) RenderBox(x, y, width, height int, title string, style CellStyle) {
	lines := make([]string, height)

	// Top border
	if title != "" {
		titlePadding := (width - len(title) - 4) / 2
		lines[0] = "┌" + strings.Repeat("─", titlePadding) + "│ " + title + " │" + strings.Repeat("─", width-titlePadding-len(title)-4) + "┐"
	} else {
		lines[0] = "┌" + strings.Repeat("─", width-2) + "┐"
	}

	// Side borders
	for i := 1; i < height-1; i++ {
		lines[i] = "│" + strings.Repeat(" ", width-2) + "│"
	}

	// Bottom border
	lines[height-1] = "└" + strings.Repeat("─", width-2) + "┘"

	rr.RenderBlock(x, y, lines, style)
}

// processRenderRequest processes a render request
func (rr *RealtimeRenderer) processRenderRequest(request RenderRequest) {
	rr.buffer.mu.Lock()
	defer rr.buffer.mu.Unlock()

	for lineIndex, line := range request.Content {
		targetY := request.Y + lineIndex
		if targetY < 0 || targetY >= rr.buffer.Height {
			continue
		}

		runes := []rune(line)
		for i, char := range runes {
			targetX := request.X + i
			if targetX < 0 || targetX >= rr.buffer.Width {
				continue
			}

			// Update cell
			cell := &rr.buffer.Cells[targetY][targetX]
			if cell.Char != char || cell.Style != request.Style {
				cell.Char = char
				cell.Style = request.Style
				cell.Dirty = true
				rr.buffer.DirtyRows[targetY] = true
			}
		}
	}
}

// Clear clears the entire buffer
func (rr *RealtimeRenderer) Clear() {
	rr.buffer.mu.Lock()
	defer rr.buffer.mu.Unlock()

	for y := 0; y < rr.buffer.Height; y++ {
		for x := 0; x < rr.buffer.Width; x++ {
			cell := &rr.buffer.Cells[y][x]
			if cell.Char != ' ' {
				cell.Char = ' '
				cell.Style = CellStyle{}
				cell.Dirty = true
				rr.buffer.DirtyRows[y] = true
			}
		}
	}

	if !rr.optimizeUpdates {
		fmt.Print("\033[2J\033[H")
	}
}

// ClearRegion clears a specific region
func (rr *RealtimeRenderer) ClearRegion(x, y, width, height int) {
	lines := make([]string, height)
	for i := range lines {
		lines[i] = strings.Repeat(" ", width)
	}

	rr.RenderBlock(x, y, lines, CellStyle{})
}

// SetCursor sets the cursor position
func (rr *RealtimeRenderer) SetCursor(x, y int) {
	rr.cursor.X = x
	rr.cursor.Y = y

	if rr.cursorVisible {
		fmt.Printf("\033[%d;%dH", y+1, x+1)
	}
}

// ShowCursor shows the cursor
func (rr *RealtimeRenderer) ShowCursor() {
	rr.cursorVisible = true
	fmt.Print("\033[?25h")
}

// HideCursor hides the cursor
func (rr *RealtimeRenderer) HideCursor() {
	rr.cursorVisible = false
	fmt.Print("\033[?25l")
}

// Helper methods

// cellStyleToANSI converts cell style to ANSI escape codes
func (rr *RealtimeRenderer) cellStyleToANSI(style CellStyle) string {
	var codes []string

	// Reset
	codes = append(codes, "0")

	// Foreground color
	if style.ForegroundColor != "" {
		if color := rr.theme.Color(style.ForegroundColor); color != nil {
			// This is simplified - in a real implementation, you'd extract ANSI codes from color
			codes = append(codes, "37") // Default to white
		}
	}

	// Background color
	if style.BackgroundColor != "" {
		codes = append(codes, "40") // Default to black background
	}

	// Text attributes
	if style.Bold {
		codes = append(codes, "1")
	}
	if style.Italic {
		codes = append(codes, "3")
	}
	if style.Underline {
		codes = append(codes, "4")
	}
	if style.Blink {
		codes = append(codes, "5")
	}
	if style.Reverse {
		codes = append(codes, "7")
	}

	return "\033[" + strings.Join(codes, ";") + "m"
}

// calculateMaxWidth calculates the maximum width of lines
func (rr *RealtimeRenderer) calculateMaxWidth(lines []string) int {
	maxWidth := 0
	for _, line := range lines {
		width := utf8.RuneCountInString(line)
		if width > maxWidth {
			maxWidth = width
		}
	}
	return maxWidth
}

// swapBuffers swaps the current and previous buffers
func (rr *RealtimeRenderer) swapBuffers() {
	rr.buffer, rr.previousBuffer = rr.previousBuffer, rr.buffer

	// Clear dirty flags in new current buffer
	rr.buffer.DirtyRows = make(map[int]bool)
}

// updatePerformanceMetrics updates rendering performance metrics
func (rr *RealtimeRenderer) updatePerformanceMetrics(renderTime time.Duration) {
	rr.frameCount++

	// Keep last 100 render times
	if len(rr.renderTimes) >= 100 {
		rr.renderTimes = rr.renderTimes[1:]
	}
	rr.renderTimes = append(rr.renderTimes, renderTime)
}

// GetPerformanceStats returns performance statistics
func (rr *RealtimeRenderer) GetPerformanceStats() (avgRenderTime time.Duration, fps float64) {
	if len(rr.renderTimes) == 0 {
		return 0, 0
	}

	var total time.Duration
	for _, t := range rr.renderTimes {
		total += t
	}

	avgRenderTime = total / time.Duration(len(rr.renderTimes))
	fps = 1.0 / avgRenderTime.Seconds()

	return avgRenderTime, fps
}

// Terminal management

// initializeTerminal initializes the terminal for rendering
func (rr *RealtimeRenderer) initializeTerminal() error {
	// Switch to alternate screen buffer
	fmt.Print("\033[?1049h")

	// Clear screen
	fmt.Print("\033[2J\033[H")

	// Hide cursor initially
	rr.HideCursor()

	return nil
}

// restoreTerminal restores terminal to original state
func (rr *RealtimeRenderer) restoreTerminal() {
	// Show cursor
	rr.ShowCursor()

	// Reset all attributes
	fmt.Print("\033[0m")

	// Switch back to main screen buffer
	fmt.Print("\033[?1049l")
}

// checkUTF8Support checks if terminal supports UTF-8
func checkUTF8Support() bool {
	lang := os.Getenv("LANG")
	return strings.Contains(strings.ToLower(lang), "utf-8") || strings.Contains(strings.ToLower(lang), "utf8")
}

// Configuration methods

// SetFrameRate sets the rendering frame rate
func (rr *RealtimeRenderer) SetFrameRate(fps int) {
	rr.frameRate = time.Second / time.Duration(fps)
}

// SetOptimizeUpdates enables or disables update optimization
func (rr *RealtimeRenderer) SetOptimizeUpdates(enabled bool) {
	rr.optimizeUpdates = enabled
}

// SetDoubleBuffering enables or disables double buffering
func (rr *RealtimeRenderer) SetDoubleBuffering(enabled bool) {
	rr.doubleBuffering = enabled
}

// Resize resizes the render buffer
func (rr *RealtimeRenderer) Resize(width, height int) {
	rr.buffer.mu.Lock()
	defer rr.buffer.mu.Unlock()

	rr.buffer = NewRenderBuffer(width, height)
	rr.previousBuffer = NewRenderBuffer(width, height)
	rr.terminalWidth = width
	rr.terminalHeight = height
}
