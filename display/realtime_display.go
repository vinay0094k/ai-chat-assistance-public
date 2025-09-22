// display/realtime_display.go
package display

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

type DisplayRegion struct {
	ID         string
	X, Y       int
	Width      int
	Height     int
	Content    []string
	Visible    bool
	Priority   int
	LastUpdate time.Time
	mu         sync.RWMutex
}

type NotificationLevel int

const (
	NotificationInfo NotificationLevel = iota
	NotificationWarning
	NotificationError
	NotificationSuccess
)

type Notification struct {
	ID       string
	Level    NotificationLevel
	Message  string
	Duration time.Duration
	Created  time.Time
	Seen     bool
}

type ProgressBar struct {
	ID             string
	Title          string
	Current        int
	Total          int
	Width          int
	ShowPercentage bool
	ShowETA        bool
	StartTime      time.Time
	Visible        bool
}

type StatusIndicator struct {
	ID         string
	Label      string
	Status     string
	Color      string
	Blinking   bool
	LastUpdate time.Time
}

type RealtimeDisplay struct {
	regions          map[string]*DisplayRegion
	notifications    []*Notification
	progressBars     map[string]*ProgressBar
	statusIndicators map[string]*StatusIndicator
	theme            *ThemeManager
	width            int
	height           int
	refreshRate      time.Duration
	ctx              context.Context
	cancel           context.CancelFunc
	mu               sync.RWMutex
	updateChan       chan bool
	running          bool
}

// NewRealtimeDisplay creates a new realtime display
func NewRealtimeDisplay(width, height int, theme *ThemeManager) *RealtimeDisplay {
	ctx, cancel := context.WithCancel(context.Background())

	if theme == nil {
		theme = GlobalTheme
	}

	rd := &RealtimeDisplay{
		regions:          make(map[string]*DisplayRegion),
		notifications:    make([]*Notification, 0),
		progressBars:     make(map[string]*ProgressBar),
		statusIndicators: make(map[string]*StatusIndicator),
		theme:            theme,
		width:            width,
		height:           height,
		refreshRate:      100 * time.Millisecond,
		ctx:              ctx,
		cancel:           cancel,
		updateChan:       make(chan bool, 100),
	}

	return rd
}

// Start begins the realtime display loop
func (rd *RealtimeDisplay) Start() {
	rd.mu.Lock()
	if rd.running {
		rd.mu.Unlock()
		return
	}
	rd.running = true
	rd.mu.Unlock()

	go rd.displayLoop()
	go rd.cleanupLoop()
}

// Stop stops the realtime display
func (rd *RealtimeDisplay) Stop() {
	rd.mu.Lock()
	if !rd.running {
		rd.mu.Unlock()
		return
	}
	rd.running = false
	rd.mu.Unlock()

	rd.cancel()
}

// displayLoop is the main display refresh loop
func (rd *RealtimeDisplay) displayLoop() {
	ticker := time.NewTicker(rd.refreshRate)
	defer ticker.Stop()

	for {
		select {
		case <-rd.ctx.Done():
			return
		case <-ticker.C:
			rd.render()
		case <-rd.updateChan:
			rd.render()
		}
	}
}

// cleanupLoop removes expired notifications and updates
func (rd *RealtimeDisplay) cleanupLoop() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-rd.ctx.Done():
			return
		case <-ticker.C:
			rd.cleanupExpiredNotifications()
		}
	}
}

// Region Management

// CreateRegion creates a new display region
func (rd *RealtimeDisplay) CreateRegion(id string, x, y, width, height int) *DisplayRegion {
	rd.mu.Lock()
	defer rd.mu.Unlock()

	region := &DisplayRegion{
		ID:         id,
		X:          x,
		Y:          y,
		Width:      width,
		Height:     height,
		Content:    make([]string, 0),
		Visible:    true,
		Priority:   0,
		LastUpdate: time.Now(),
	}

	rd.regions[id] = region
	rd.triggerUpdate()

	return region
}

// UpdateRegion updates the content of a region
func (rd *RealtimeDisplay) UpdateRegion(id string, content []string) error {
	rd.mu.RLock()
	region, exists := rd.regions[id]
	rd.mu.RUnlock()

	if !exists {
		return fmt.Errorf("region %s not found", id)
	}

	region.mu.Lock()
	defer region.mu.Unlock()

	// Trim content to fit region height
	if len(content) > region.Height {
		content = content[:region.Height]
	}

	// Ensure each line fits within width
	for i, line := range content {
		if len(line) > region.Width {
			content[i] = line[:region.Width-3] + "..."
		}
	}

	region.Content = content
	region.LastUpdate = time.Now()

	rd.triggerUpdate()
	return nil
}

// SetRegionVisibility sets the visibility of a region
func (rd *RealtimeDisplay) SetRegionVisibility(id string, visible bool) error {
	rd.mu.RLock()
	region, exists := rd.regions[id]
	rd.mu.RUnlock()

	if !exists {
		return fmt.Errorf("region %s not found", id)
	}

	region.mu.Lock()
	region.Visible = visible
	region.mu.Unlock()

	rd.triggerUpdate()
	return nil
}

// Notification Management

// ShowNotification displays a notification
func (rd *RealtimeDisplay) ShowNotification(level NotificationLevel, message string, duration time.Duration) string {
	id := fmt.Sprintf("notif_%d", time.Now().UnixNano())

	notification := &Notification{
		ID:       id,
		Level:    level,
		Message:  message,
		Duration: duration,
		Created:  time.Now(),
		Seen:     false,
	}

	rd.mu.Lock()
	rd.notifications = append(rd.notifications, notification)
	rd.mu.Unlock()

	rd.triggerUpdate()
	return id
}

// Progress Bar Management

// CreateProgressBar creates a new progress bar
func (rd *RealtimeDisplay) CreateProgressBar(id, title string, total int) *ProgressBar {
	progressBar := &ProgressBar{
		ID:             id,
		Title:          title,
		Current:        0,
		Total:          total,
		Width:          40,
		ShowPercentage: true,
		ShowETA:        true,
		StartTime:      time.Now(),
		Visible:        true,
	}

	rd.mu.Lock()
	rd.progressBars[id] = progressBar
	rd.mu.Unlock()

	rd.triggerUpdate()
	return progressBar
}

// UpdateProgress updates a progress bar
func (rd *RealtimeDisplay) UpdateProgress(id string, current int) error {
	rd.mu.RLock()
	progressBar, exists := rd.progressBars[id]
	rd.mu.RUnlock()

	if !exists {
		return fmt.Errorf("progress bar %s not found", id)
	}

	progressBar.Current = current
	if current >= progressBar.Total {
		progressBar.Visible = false
	}

	rd.triggerUpdate()
	return nil
}

// Status Indicator Management

// CreateStatusIndicator creates a status indicator
func (rd *RealtimeDisplay) CreateStatusIndicator(id, label, status, color string) *StatusIndicator {
	indicator := &StatusIndicator{
		ID:         id,
		Label:      label,
		Status:     status,
		Color:      color,
		Blinking:   false,
		LastUpdate: time.Now(),
	}

	rd.mu.Lock()
	rd.statusIndicators[id] = indicator
	rd.mu.Unlock()

	rd.triggerUpdate()
	return indicator
}

// UpdateStatus updates a status indicator
func (rd *RealtimeDisplay) UpdateStatus(id, status, color string) error {
	rd.mu.RLock()
	indicator, exists := rd.statusIndicators[id]
	rd.mu.RUnlock()

	if !exists {
		return fmt.Errorf("status indicator %s not found", id)
	}

	indicator.Status = status
	indicator.Color = color
	indicator.LastUpdate = time.Now()

	rd.triggerUpdate()
	return nil
}

// Rendering

// render renders the entire display
func (rd *RealtimeDisplay) render() {
	rd.mu.RLock()
	defer rd.mu.RUnlock()

	// Clear screen (in a real implementation, you might use a terminal library)
	// fmt.Print("\033[2J\033[H")

	// Create display buffer
	buffer := make([][]string, rd.height)
	for i := range buffer {
		buffer[i] = make([]string, rd.width)
		for j := range buffer[i] {
			buffer[i][j] = " "
		}
	}

	// Render regions
	rd.renderRegions(buffer)

	// Render notifications
	rd.renderNotifications(buffer)

	// Render progress bars
	rd.renderProgressBars(buffer)

	// Render status indicators
	rd.renderStatusIndicators(buffer)

	// Output buffer (simplified - in real implementation, optimize screen updates)
	rd.outputBuffer(buffer)
}

// renderRegions renders all visible regions
func (rd *RealtimeDisplay) renderRegions(buffer [][]string) {
	for _, region := range rd.regions {
		if !region.Visible {
			continue
		}

		region.mu.RLock()
		for i, line := range region.Content {
			if region.Y+i >= len(buffer) {
				break
			}
			for j, char := range line {
				if region.X+j >= len(buffer[region.Y+i]) {
					break
				}
				buffer[region.Y+i][region.X+j] = string(char)
			}
		}
		region.mu.RUnlock()
	}
}

// renderNotifications renders notifications at the top
func (rd *RealtimeDisplay) renderNotifications(buffer [][]string) {
	y := 0
	for _, notif := range rd.notifications {
		if y >= len(buffer) {
			break
		}

		// Format notification
		var prefix string
		var colorType string

		switch notif.Level {
		case NotificationInfo:
			prefix = "ℹ"
			colorType = "info"
		case NotificationWarning:
			prefix = "⚠"
			colorType = "warning"
		case NotificationError:
			prefix = "✗"
			colorType = "error"
		case NotificationSuccess:
			prefix = "✓"
			colorType = "success"
		}

		message := fmt.Sprintf("%s %s", prefix, notif.Message)
		if len(message) > rd.width {
			message = message[:rd.width-3] + "..."
		}

		// Place in buffer
		for i, char := range message {
			if i >= len(buffer[y]) {
				break
			}
			buffer[y][i] = string(char)
		}

		y++
	}
}

// renderProgressBars renders progress bars
func (rd *RealtimeDisplay) renderProgressBars(buffer [][]string) {
	y := rd.height - len(rd.progressBars)
	if y < 0 {
		y = 0
	}

	for _, pb := range rd.progressBars {
		if !pb.Visible || y >= len(buffer) {
			continue
		}

		// Create progress bar visualization
		percentage := float64(pb.Current) / float64(pb.Total)
		filled := int(percentage * float64(pb.Width))

		var barStr strings.Builder
		barStr.WriteString(pb.Title)
		barStr.WriteString(" [")
		barStr.WriteString(strings.Repeat("█", filled))
		barStr.WriteString(strings.Repeat("░", pb.Width-filled))
		barStr.WriteString("]")

		if pb.ShowPercentage {
			barStr.WriteString(fmt.Sprintf(" %.1f%%", percentage*100))
		}

		if pb.ShowETA && pb.Current > 0 {
			elapsed := time.Since(pb.StartTime)
			rate := float64(pb.Current) / elapsed.Seconds()
			remaining := time.Duration(float64(pb.Total-pb.Current)/rate) * time.Second
			barStr.WriteString(fmt.Sprintf(" ETA: %s", remaining.Truncate(time.Second)))
		}

		line := barStr.String()
		if len(line) > rd.width {
			line = line[:rd.width]
		}

		// Place in buffer
		for i, char := range line {
			if i >= len(buffer[y]) {
				break
			}
			buffer[y][i] = string(char)
		}

		y++
	}
}

// renderStatusIndicators renders status indicators at the bottom right
func (rd *RealtimeDisplay) renderStatusIndicators(buffer [][]string) {
	y := rd.height - 1
	x := rd.width

	for _, indicator := range rd.statusIndicators {
		statusStr := fmt.Sprintf("%s: %s", indicator.Label, indicator.Status)

		// Handle blinking
		if indicator.Blinking && time.Now().UnixNano()/1e8%2 == 0 {
			continue // Skip rendering on blink off
		}

		x -= len(statusStr) + 2
		if x < 0 {
			break
		}

		// Place in buffer
		for i, char := range statusStr {
			if x+i >= len(buffer[y]) {
				break
			}
			buffer[y][x+i] = string(char)
		}
	}
}

// outputBuffer outputs the buffer to the console
func (rd *RealtimeDisplay) outputBuffer(buffer [][]string) {
	// In a real implementation, you would optimize this to only update changed areas
	for _, row := range buffer {
		line := strings.Join(row, "")
		fmt.Println(strings.TrimRight(line, " "))
	}
}

// Helper methods

// triggerUpdate triggers a display update
func (rd *RealtimeDisplay) triggerUpdate() {
	select {
	case rd.updateChan <- true:
	default:
		// Channel is full, skip this update
	}
}

// cleanupExpiredNotifications removes expired notifications
func (rd *RealtimeDisplay) cleanupExpiredNotifications() {
	rd.mu.Lock()
	defer rd.mu.Unlock()

	now := time.Now()
	var activeNotifications []*Notification

	for _, notif := range rd.notifications {
		if notif.Duration > 0 && now.Sub(notif.Created) > notif.Duration {
			continue // Skip expired notification
		}
		activeNotifications = append(activeNotifications, notif)
	}

	rd.notifications = activeNotifications
}

// Utility methods

// SetRefreshRate sets the display refresh rate
func (rd *RealtimeDisplay) SetRefreshRate(rate time.Duration) {
	rd.refreshRate = rate
}

// GetRegion returns a region by ID
func (rd *RealtimeDisplay) GetRegion(id string) (*DisplayRegion, bool) {
	rd.mu.RLock()
	defer rd.mu.RUnlock()

	region, exists := rd.regions[id]
	return region, exists
}

// ClearNotifications clears all notifications
func (rd *RealtimeDisplay) ClearNotifications() {
	rd.mu.Lock()
	rd.notifications = rd.notifications[:0]
	rd.mu.Unlock()

	rd.triggerUpdate()
}

// RemoveRegion removes a region
func (rd *RealtimeDisplay) RemoveRegion(id string) {
	rd.mu.Lock()
	delete(rd.regions, id)
	rd.mu.Unlock()

	rd.triggerUpdate()
}

// RemoveProgressBar removes a progress bar
func (rd *RealtimeDisplay) RemoveProgressBar(id string) {
	rd.mu.Lock()
	delete(rd.progressBars, id)
	rd.mu.Unlock()

	rd.triggerUpdate()
}
