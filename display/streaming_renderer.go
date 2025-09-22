// display/streaming_renderer.go
package display

import (
	"context"
	"fmt"
	"io"
	"math"
	"strings"
	"sync"
	"time"
)

type StreamingEffect int

const (
	EffectTypewriter StreamingEffect = iota
	EffectWave
	EffectFade
	EffectMatrix
	EffectGlitch
	EffectRainbow
)

type StreamingConfig struct {
	Effect        StreamingEffect
	Speed         time.Duration
	WordByWord    bool
	ShowCursor    bool
	WrapText      bool
	MaxWidth      int
	Prefix        string
	Suffix        string
	HighlightCode bool
	AnimateEmojis bool
}

type StreamingRenderer struct {
	config      *StreamingConfig
	theme       *ThemeManager
	renderer    *RealtimeRenderer
	position    CursorPosition
	buffer      strings.Builder
	currentText string
	isActive    bool
	ctx         context.Context
	cancel      context.CancelFunc
	mu          sync.RWMutex

	// Animation state
	animationFrame int
	lastUpdate     time.Time
	charIndex      int
	wordIndex      int
	words          []string

	// Effect-specific state
	waveOffset  float64
	glitchChars map[int]rune
	rainbowHue  float64
}

type StreamChunk struct {
	Text       string
	IsComplete bool
	Metadata   map[string]interface{}
}

// NewStreamingRenderer creates a new streaming renderer
func NewStreamingRenderer(config *StreamingConfig, theme *ThemeManager, renderer *RealtimeRenderer) *StreamingRenderer {
	ctx, cancel := context.WithCancel(context.Background())

	if config == nil {
		config = DefaultStreamingConfig()
	}
	if theme == nil {
		theme = GlobalTheme
	}

	sr := &StreamingRenderer{
		config:      config,
		theme:       theme,
		renderer:    renderer,
		ctx:         ctx,
		cancel:      cancel,
		glitchChars: make(map[int]rune),
	}

	return sr
}

// DefaultStreamingConfig returns default streaming configuration
func DefaultStreamingConfig() *StreamingConfig {
	return &StreamingConfig{
		Effect:        EffectTypewriter,
		Speed:         50 * time.Millisecond,
		WordByWord:    false,
		ShowCursor:    true,
		WrapText:      true,
		MaxWidth:      80,
		Prefix:        "",
		Suffix:        "",
		HighlightCode: true,
		AnimateEmojis: true,
	}
}

// Start begins streaming at the specified position
func (sr *StreamingRenderer) Start(x, y int) {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	sr.position = CursorPosition{X: x, Y: y}
	sr.isActive = true
	sr.buffer.Reset()
	sr.currentText = ""
	sr.charIndex = 0
	sr.wordIndex = 0
	sr.animationFrame = 0
	sr.lastUpdate = time.Now()

	// Start animation loop
	go sr.animationLoop()
}

// Stop stops the streaming
func (sr *StreamingRenderer) Stop() {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	sr.isActive = false
	sr.cancel()
}

// StreamText streams text with the configured effect
func (sr *StreamingRenderer) StreamText(text string) error {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	if !sr.isActive {
		return fmt.Errorf("streaming renderer not active")
	}

	sr.buffer.WriteString(text)
	sr.currentText = sr.buffer.String()

	// Prepare words if word-by-word mode
	if sr.config.WordByWord {
		sr.words = strings.Fields(sr.currentText)
	}

	return nil
}

// StreamFromReader streams text from a reader
func (sr *StreamingRenderer) StreamFromReader(reader io.Reader) error {
	buffer := make([]byte, 1024)

	for {
		n, err := reader.Read(buffer)
		if n > 0 {
			text := string(buffer[:n])
			if streamErr := sr.StreamText(text); streamErr != nil {
				return streamErr
			}
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		// Small delay to allow for rendering
		time.Sleep(sr.config.Speed / 4)
	}

	return nil
}

// StreamFromChannel streams text from a channel
func (sr *StreamingRenderer) StreamFromChannel(textChan <-chan StreamChunk) error {
	for {
		select {
		case chunk, ok := <-textChan:
			if !ok {
				return nil // Channel closed
			}

			if err := sr.StreamText(chunk.Text); err != nil {
				return err
			}

			if chunk.IsComplete {
				return nil
			}

		case <-sr.ctx.Done():
			return sr.ctx.Err()
		}
	}
}

// animationLoop runs the main animation loop
func (sr *StreamingRenderer) animationLoop() {
	ticker := time.NewTicker(sr.config.Speed)
	defer ticker.Stop()

	for {
		select {
		case <-sr.ctx.Done():
			return
		case <-ticker.C:
			sr.updateAnimation()
		}
	}
}

// updateAnimation updates the animation frame
func (sr *StreamingRenderer) updateAnimation() {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	if !sr.isActive || sr.currentText == "" {
		return
	}

	now := time.Now()
	deltaTime := now.Sub(sr.lastUpdate)
	sr.lastUpdate = now
	sr.animationFrame++

	switch sr.config.Effect {
	case EffectTypewriter:
		sr.updateTypewriter(deltaTime)
	case EffectWave:
		sr.updateWave(deltaTime)
	case EffectFade:
		sr.updateFade(deltaTime)
	case EffectMatrix:
		sr.updateMatrix(deltaTime)
	case EffectGlitch:
		sr.updateGlitch(deltaTime)
	case EffectRainbow:
		sr.updateRainbow(deltaTime)
	}
}

// updateTypewriter updates typewriter effect
func (sr *StreamingRenderer) updateTypewriter(deltaTime time.Duration) {
	var visibleText string

	if sr.config.WordByWord {
		// Show words one by one
		if sr.wordIndex < len(sr.words) {
			visibleText = strings.Join(sr.words[:sr.wordIndex+1], " ")
			if sr.animationFrame%3 == 0 { // Advance word every 3 frames
				sr.wordIndex++
			}
		} else {
			visibleText = sr.currentText
		}
	} else {
		// Show characters one by one
		runes := []rune(sr.currentText)
		if sr.charIndex < len(runes) {
			visibleText = string(runes[:sr.charIndex+1])
			sr.charIndex++
		} else {
			visibleText = sr.currentText
		}
	}

	// Add blinking cursor
	if sr.config.ShowCursor && sr.animationFrame%2 == 0 {
		visibleText += "█"
	} else if sr.config.ShowCursor {
		visibleText += " "
	}

	sr.renderText(visibleText, "ai_response")
}

// updateWave updates wave effect
func (sr *StreamingRenderer) updateWave(deltaTime time.Duration) {
	sr.waveOffset += deltaTime.Seconds() * 2.0

	lines := sr.wrapText(sr.currentText)

	for lineIndex, line := range lines {
		runes := []rune(line)
		for i, char := range runes {
			// Calculate wave offset for each character
			phase := sr.waveOffset + float64(i)*0.3 + float64(lineIndex)*0.5
			yOffset := int(math.Sin(phase) * 2)

			// Render character with vertical offset
			x := sr.position.X + i
			y := sr.position.Y + lineIndex + yOffset

			if sr.renderer != nil {
				style := CellStyle{ForegroundColor: "ai_response"}
				sr.renderer.RenderText(x, y, string(char), style)
			}
		}
	}
}

// updateFade updates fade effect
func (sr *StreamingRenderer) updateFade(deltaTime time.Duration) {
	runes := []rune(sr.currentText)
	visibleLength := int(float64(len(runes)) * (float64(sr.animationFrame) / 100.0))

	if visibleLength > len(runes) {
		visibleLength = len(runes)
	}

	var output strings.Builder
	for i, char := range runes {
		if i < visibleLength {
			// Calculate fade intensity
			intensity := 1.0 - float64(visibleLength-i)/float64(visibleLength)
			if intensity > 0.3 {
				output.WriteRune(char)
			} else {
				output.WriteRune('░') // Faded character
			}
		}
	}

	sr.renderText(output.String(), "ai_response")
}

// updateMatrix updates matrix effect
func (sr *StreamingRenderer) updateMatrix(deltaTime time.Duration) {
	runes := []rune(sr.currentText)

	// Gradually reveal characters from random positions
	revealRate := len(runes) / 50
	if revealRate < 1 {
		revealRate = 1
	}

	var output strings.Builder
	for i, char := range runes {
		if sr.animationFrame > i/revealRate {
			output.WriteRune(char)
		} else {
			// Show random matrix characters
			matrixChars := []rune{'ｦ', 'ｱ', 'ｳ', 'ｴ', 'ｵ', 'ｶ', 'ｷ', 'ｹ', 'ｺ', 'ｻ', 'ｼ', 'ｽ', 'ﾀ', 'ﾃ', 'ﾄ', 'ﾅ', 'ﾆ', 'ﾇ', 'ﾈ', 'ﾊ', 'ﾋ', 'ﾍ', 'ﾎ', 'ﾏ', 'ﾐ', 'ﾑ', 'ﾒ', 'ﾓ', 'ﾔ', 'ﾕ', 'ﾖ', 'ﾗ', 'ﾘ', 'ﾙ', 'ﾚ', 'ﾛ', 'ﾜ', 'ﾝ'}
			randomChar := matrixChars[sr.animationFrame%len(matrixChars)]
			output.WriteRune(randomChar)
		}
	}

	sr.renderText(output.String(), "success") // Green for matrix effect
}

// updateGlitch updates glitch effect
func (sr *StreamingRenderer) updateGlitch(deltaTime time.Duration) {
	runes := []rune(sr.currentText)

	// Randomly glitch some characters
	if sr.animationFrame%5 == 0 {
		sr.glitchChars = make(map[int]rune)
		for i := 0; i < len(runes)/10; i++ {
			pos := sr.animationFrame % len(runes)
			glitchChars := []rune{'█', '▓', '▒', '░', '▄', '▀', '▐', '▌'}
			sr.glitchChars[pos] = glitchChars[sr.animationFrame%len(glitchChars)]
		}
	}

	var output strings.Builder
	for i, char := range runes {
		if glitchChar, exists := sr.glitchChars[i]; exists && sr.animationFrame%3 == 0 {
			output.WriteRune(glitchChar)
		} else {
			output.WriteRune(char)
		}
	}

	sr.renderText(output.String(), "error") // Red for glitch effect
}

// updateRainbow updates rainbow effect
func (sr *StreamingRenderer) updateRainbow(deltaTime time.Duration) {
	sr.rainbowHue += deltaTime.Seconds() * 60.0 // Degrees per second
	if sr.rainbowHue > 360 {
		sr.rainbowHue -= 360
	}

	lines := sr.wrapText(sr.currentText)

	for lineIndex, line := range lines {
		runes := []rune(line)
		for i, char := range runes {
			// Calculate hue for each character
			hue := sr.rainbowHue + float64(i*10)
			if hue > 360 {
				hue -= 360
			}

			// Simple HSV to RGB conversion for terminal colors
			var colorName string
			switch {
			case hue < 60:
				colorName = "error" // Red
			case hue < 120:
				colorName = "warning" // Yellow
			case hue < 180:
				colorName = "success" // Green
			case hue < 240:
				colorName = "info" // Cyan
			case hue < 300:
				colorName = "primary" // Blue
			default:
				colorName = "highlight" // Magenta
			}

			x := sr.position.X + i
			y := sr.position.Y + lineIndex

			if sr.renderer != nil {
				style := CellStyle{ForegroundColor: colorName}
				sr.renderer.RenderText(x, y, string(char), style)
			}
		}
	}
}

// renderText renders text with color
func (sr *StreamingRenderer) renderText(text, colorType string) {
	if sr.renderer == nil {
		// Fallback to direct console output
		formatted := sr.theme.Sprint(colorType, text)
		fmt.Printf("\033[%d;%dH%s", sr.position.Y+1, sr.position.X+1, formatted)
		return
	}

	lines := sr.wrapText(text)
	for i, line := range lines {
		style := CellStyle{ForegroundColor: colorType}
		sr.renderer.RenderText(sr.position.X, sr.position.Y+i, line, style)
	}
}

// wrapText wraps text to fit within max width
func (sr *StreamingRenderer) wrapText(text string) []string {
	if !sr.config.WrapText || sr.config.MaxWidth <= 0 {
		return []string{text}
	}

	words := strings.Fields(text)
	if len(words) == 0 {
		return []string{text}
	}

	var lines []string
	var currentLine strings.Builder

	for _, word := range words {
		// Check if adding this word would exceed max width
		if currentLine.Len() > 0 && currentLine.Len()+1+len(word) > sr.config.MaxWidth {
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

// Utility methods

// SetEffect changes the streaming effect
func (sr *StreamingRenderer) SetEffect(effect StreamingEffect) {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	sr.config.Effect = effect
	sr.resetAnimation()
}

// SetSpeed changes the streaming speed
func (sr *StreamingRenderer) SetSpeed(speed time.Duration) {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	sr.config.Speed = speed
}

// SetWordByWord enables or disables word-by-word mode
func (sr *StreamingRenderer) SetWordByWord(enabled bool) {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	sr.config.WordByWord = enabled
	sr.resetAnimation()
}

// SetMaxWidth sets the maximum text width
func (sr *StreamingRenderer) SetMaxWidth(width int) {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	sr.config.MaxWidth = width
}

// resetAnimation resets animation state
func (sr *StreamingRenderer) resetAnimation() {
	sr.animationFrame = 0
	sr.charIndex = 0
	sr.wordIndex = 0
	sr.waveOffset = 0
	sr.rainbowHue = 0
	sr.glitchChars = make(map[int]rune)
}

// IsComplete returns true if streaming is complete
func (sr *StreamingRenderer) IsComplete() bool {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	if sr.config.WordByWord {
		return sr.wordIndex >= len(sr.words)
	}

	runes := []rune(sr.currentText)
	return sr.charIndex >= len(runes)
}

// GetProgress returns streaming progress (0.0 to 1.0)
func (sr *StreamingRenderer) GetProgress() float64 {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	if sr.currentText == "" {
		return 0.0
	}

	if sr.config.WordByWord {
		if len(sr.words) == 0 {
			return 1.0
		}
		return float64(sr.wordIndex) / float64(len(sr.words))
	}

	runes := []rune(sr.currentText)
	if len(runes) == 0 {
		return 1.0
	}
	return float64(sr.charIndex) / float64(len(runes))
}

// Clear clears the current streaming text
func (sr *StreamingRenderer) Clear() {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	sr.buffer.Reset()
	sr.currentText = ""
	sr.resetAnimation()

	if sr.renderer != nil {
		// Clear the display area
		lines := sr.wrapText(strings.Repeat(" ", sr.config.MaxWidth))
		for i := range lines {
			sr.renderer.ClearRegion(sr.position.X, sr.position.Y+i, sr.config.MaxWidth, 1)
		}
	}
}
