package cli

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/fatih/color"
)

// ProgressBar represents a progress indicator
type ProgressBar struct {
	title       string
	total       int
	current     int
	width       int
	startTime   time.Time
	lastUpdate  time.Time
	finished    bool
	mu          sync.Mutex
	showPercent bool
	showETA     bool
	showSpeed   bool
	template    string
	colors      *ProgressColors
}

// ProgressColors defines colors for progress bar elements
type ProgressColors struct {
	Complete   *color.Color
	Incomplete *color.Color
	Percentage *color.Color
	Title      *color.Color
	ETA        *color.Color
}

// NewProgressBar creates a new progress bar
func NewProgressBar(title string, total int) *ProgressBar {
	return &ProgressBar{
		title:       title,
		total:       total,
		width:       50,
		startTime:   time.Now(),
		lastUpdate:  time.Now(),
		showPercent: true,
		showETA:     true,
		showSpeed:   false,
		template:    "[{{.Bar}}] {{.Percent}}% {{.Title}} {{.ETA}}",
		colors: &ProgressColors{
			Complete:   color.New(color.FgGreen),
			Incomplete: color.New(color.FgWhite),
			Percentage: color.New(color.FgCyan),
			Title:      color.New(color.FgWhite),
			ETA:        color.New(color.FgYellow),
		},
	}
}

// SetWidth sets the progress bar width
func (pb *ProgressBar) SetWidth(width int) {
	pb.mu.Lock()
	defer pb.mu.Unlock()
	pb.width = width
}

// SetTemplate sets the display template
func (pb *ProgressBar) SetTemplate(template string) {
	pb.mu.Lock()
	defer pb.mu.Unlock()
	pb.template = template
}

// SetColors sets custom colors
func (pb *ProgressBar) SetColors(colors *ProgressColors) {
	pb.mu.Lock()
	defer pb.mu.Unlock()
	pb.colors = colors
}

// Update updates the progress bar
func (pb *ProgressBar) Update(current int, message ...string) {
	pb.mu.Lock()
	defer pb.mu.Unlock()

	pb.current = current
	pb.lastUpdate = time.Now()

	if len(message) > 0 {
		pb.title = message[0]
	}

	pb.render()
}

// Increment increments the progress by 1
func (pb *ProgressBar) Increment(message ...string) {
	pb.Update(pb.current+1, message...)
}

// Finish completes the progress bar
func (pb *ProgressBar) Finish(message ...string) {
	pb.mu.Lock()
	defer pb.mu.Unlock()

	pb.current = pb.total
	pb.finished = true

	if len(message) > 0 {
		pb.title = message[0]
	}

	pb.render()
	fmt.Println() // New line after completion
}

// render renders the progress bar
func (pb *ProgressBar) render() {
	if pb.total <= 0 {
		return
	}

	// Calculate progress
	percent := float64(pb.current) / float64(pb.total) * 100
	if percent > 100 {
		percent = 100
	}

	// Build progress bar
	completed := int(float64(pb.width) * (float64(pb.current) / float64(pb.total)))
	if completed > pb.width {
		completed = pb.width
	}

	bar := strings.Repeat("█", completed) + strings.Repeat("░", pb.width-completed)

	// Color the bar
	if pb.colors.Complete != nil && pb.colors.Incomplete != nil {
		coloredBar := pb.colors.Complete.Sprint(strings.Repeat("█", completed)) +
			pb.colors.Incomplete.Sprint(strings.Repeat("░", pb.width-completed))
		bar = coloredBar
	}

	// Build display string
	var parts []string

	// Add bar
	parts = append(parts, fmt.Sprintf("[%s]", bar))

	// Add percentage
	if pb.showPercent {
		percentStr := fmt.Sprintf("%.1f%%", percent)
		if pb.colors.Percentage != nil {
			percentStr = pb.colors.Percentage.Sprint(percentStr)
		}
		parts = append(parts, percentStr)
	}

	// Add title
	if pb.title != "" {
		titleStr := pb.title
		if pb.colors.Title != nil {
			titleStr = pb.colors.Title.Sprint(titleStr)
		}
		parts = append(parts, titleStr)
	}

	// Add ETA
	if pb.showETA && pb.current > 0 {
		elapsed := time.Since(pb.startTime)
		rate := float64(pb.current) / elapsed.Seconds()
		if rate > 0 {
			remaining := float64(pb.total-pb.current) / rate
			eta := time.Duration(remaining) * time.Second
			etaStr := fmt.Sprintf("ETA: %s", eta.Round(time.Second))
			if pb.colors.ETA != nil {
				etaStr = pb.colors.ETA.Sprint(etaStr)
			}
			parts = append(parts, etaStr)
		}
	}

	// Add speed
	if pb.showSpeed && pb.current > 0 {
		elapsed := time.Since(pb.startTime)
		rate := float64(pb.current) / elapsed.Seconds()
		speedStr := fmt.Sprintf("%.1f/s", rate)
		parts = append(parts, speedStr)
	}

	// Print the progress line
	line := strings.Join(parts, " ")
	fmt.Printf("\r%s", line)
}

// Spinner represents a spinning progress indicator
type Spinner struct {
	chars   []string
	current int
	title   string
	active  bool
	stopCh  chan bool
	mu      sync.Mutex
}

// NewSpinner creates a new spinner
func NewSpinner(title string) *Spinner {
	return &Spinner{
		chars:  []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"},
		title:  title,
		stopCh: make(chan bool, 1),
	}
}

// Start starts the spinner
func (s *Spinner) Start() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.active {
		return
	}

	s.active = true

	go func() {
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-s.stopCh:
				return
			case <-ticker.C:
				s.mu.Lock()
				if s.active {
					fmt.Printf("\r%s %s", s.chars[s.current], s.title)
					s.current = (s.current + 1) % len(s.chars)
				}
				s.mu.Unlock()
			}
		}
	}()
}

// UpdateTitle updates the spinner title
func (s *Spinner) UpdateTitle(title string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.title = title
}

// Stop stops the spinner
func (s *Spinner) Stop(finalMessage string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.active {
		return
	}

	s.active = false
	s.stopCh <- true

	// Clear the line and show final message
	fmt.Printf("\r%s\n", finalMessage)
}

// MultiProgressManager manages multiple progress indicators
type MultiProgressManager struct {
	bars     map[string]*ProgressBar
	spinners map[string]*Spinner
	mu       sync.RWMutex
}

// NewMultiProgressManager creates a new multi-progress manager
func NewMultiProgressManager() *MultiProgressManager {
	return &MultiProgressManager{
		bars:     make(map[string]*ProgressBar),
		spinners: make(map[string]*Spinner),
	}
}

// AddProgressBar adds a progress bar with a unique ID
func (m *MultiProgressManager) AddProgressBar(id, title string, total int) *ProgressBar {
	m.mu.Lock()
	defer m.mu.Unlock()

	pb := NewProgressBar(title, total)
	m.bars[id] = pb
	return pb
}

// AddSpinner adds a spinner with a unique ID
func (m *MultiProgressManager) AddSpinner(id, title string) *Spinner {
	m.mu.Lock()
	defer m.mu.Unlock()

	spinner := NewSpinner(title)
	m.spinners[id] = spinner
	return spinner
}

// GetProgressBar gets a progress bar by ID
func (m *MultiProgressManager) GetProgressBar(id string) *ProgressBar {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.bars[id]
}

// GetSpinner gets a spinner by ID
func (m *MultiProgressManager) GetSpinner(id string) *Spinner {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.spinners[id]
}

// RemoveProgressBar removes a progress bar
func (m *MultiProgressManager) RemoveProgressBar(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.bars, id)
}

// RemoveSpinner removes a spinner
func (m *MultiProgressManager) RemoveSpinner(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if spinner, exists := m.spinners[id]; exists {
		spinner.Stop("")
		delete(m.spinners, id)
	}
}

// StepProgress represents a multi-step progress indicator
type StepProgress struct {
	steps     []string
	current   int
	startTime time.Time
	stepTimes []time.Time
	mu        sync.Mutex
}

// NewStepProgress creates a new step progress indicator
func NewStepProgress(steps []string) *StepProgress {
	return &StepProgress{
		steps:     steps,
		startTime: time.Now(),
		stepTimes: make([]time.Time, len(steps)+1),
	}
}

// Start starts the step progress
func (sp *StepProgress) Start() {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	sp.current = 0
	sp.stepTimes[0] = time.Now()
	sp.render()
}

// NextStep advances to the next step
func (sp *StepProgress) NextStep() {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	if sp.current < len(sp.steps) {
		sp.current++
		sp.stepTimes[sp.current] = time.Now()
		sp.render()
	}
}

// Finish completes all steps
func (sp *StepProgress) Finish() {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	sp.current = len(sp.steps)
	sp.stepTimes[sp.current] = time.Now()
	sp.render()
	fmt.Println()
}

// render renders the step progress
func (sp *StepProgress) render() {
	fmt.Printf("\r")

	for i, step := range sp.steps {
		if i < sp.current {
			fmt.Printf("✅ ")
		} else if i == sp.current {
			fmt.Printf("🔄 ")
		} else {
			fmt.Printf("⏳ ")
		}

		fmt.Printf("%s", step)

		// Show duration for completed steps
		if i < sp.current && i < len(sp.stepTimes)-1 {
			duration := sp.stepTimes[i+1].Sub(sp.stepTimes[i])
			fmt.Printf(" (%.1fs)", duration.Seconds())
		}

		if i < len(sp.steps)-1 {
			fmt.Printf(" → ")
		}
	}
}

// TaskProgress represents progress for a specific task
type TaskProgress struct {
	name      string
	total     int64
	current   int64
	unit      string
	startTime time.Time
	mu        sync.Mutex
}

// NewTaskProgress creates a new task progress indicator
func NewTaskProgress(name string, total int64, unit string) *TaskProgress {
	return &TaskProgress{
		name:      name,
		total:     total,
		unit:      unit,
		startTime: time.Now(),
	}
}

// Update updates the task progress
func (tp *TaskProgress) Update(current int64) {
	tp.mu.Lock()
	defer tp.mu.Unlock()

	tp.current = current
	tp.render()
}

// Add adds to the current progress
func (tp *TaskProgress) Add(amount int64) {
	tp.Update(tp.current + amount)
}

// Finish completes the task
func (tp *TaskProgress) Finish() {
	tp.mu.Lock()
	defer tp.mu.Unlock()

	tp.current = tp.total
	tp.render()
	fmt.Println()
}

// render renders the task progress
func (tp *TaskProgress) render() {
	percent := float64(tp.current) / float64(tp.total) * 100
	if percent > 100 {
		percent = 100
	}

	elapsed := time.Since(tp.startTime)
	rate := float64(tp.current) / elapsed.Seconds()

	var eta string
	if rate > 0 && tp.current < tp.total {
		remaining := float64(tp.total-tp.current) / rate
		eta = fmt.Sprintf(" ETA: %s", time.Duration(remaining*float64(time.Second)).Round(time.Second))
	}

	fmt.Printf("\r%s: %d/%d %s (%.1f%%) %.1f %s/s%s",
		tp.name,
		tp.current,
		tp.total,
		tp.unit,
		percent,
		rate,
		tp.unit,
		eta,
	)
}
