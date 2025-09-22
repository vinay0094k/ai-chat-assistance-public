package cli

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// HistoryEntry represents a single command history entry
type HistoryEntry struct {
	Command   string    `json:"command"`
	Timestamp time.Time `json:"timestamp"`
	SessionID string    `json:"session_id"`
	Success   bool      `json:"success"`
	Duration  int64     `json:"duration_ms"`
}

// History manages command history
type History struct {
	entries  []HistoryEntry
	maxSize  int
	filePath string
	mu       sync.RWMutex
	dirty    bool
	autosave bool
}

// NewHistory creates a new command history
func NewHistory(maxSize int) *History {
	homeDir, _ := os.UserHomeDir()
	historyPath := filepath.Join(homeDir, ".ai-assistant", "history.txt")

	history := &History{
		entries:  make([]HistoryEntry, 0),
		maxSize:  maxSize,
		filePath: historyPath,
		autosave: true,
	}

	// Load existing history
	history.loadFromFile()

	return history
}

// NewHistoryWithFile creates a new command history with custom file path
func NewHistoryWithFile(maxSize int, filePath string) *History {
	history := &History{
		entries:  make([]HistoryEntry, 0),
		maxSize:  maxSize,
		filePath: filePath,
		autosave: true,
	}

	history.loadFromFile()
	return history
}

// Add adds a command to the history
func (h *History) Add(command string) {
	h.AddWithDetails(command, "", true, 0)
}

// AddWithDetails adds a command with additional details
func (h *History) AddWithDetails(command, sessionID string, success bool, durationMs int64) {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Don't add empty commands or duplicates of the last command
	if command == "" || (len(h.entries) > 0 && h.entries[len(h.entries)-1].Command == command) {
		return
	}

	entry := HistoryEntry{
		Command:   command,
		Timestamp: time.Now(),
		SessionID: sessionID,
		Success:   success,
		Duration:  durationMs,
	}

	h.entries = append(h.entries, entry)

	// Trim if over max size
	if len(h.entries) > h.maxSize {
		h.entries = h.entries[len(h.entries)-h.maxSize:]
	}

	h.dirty = true

	if h.autosave {
		go h.saveToFile()
	}
}

// GetAll returns all history entries
func (h *History) GetAll() []string {
	h.mu.RLock()
	defer h.mu.RUnlock()

	commands := make([]string, len(h.entries))
	for i, entry := range h.entries {
		commands[i] = entry.Command
	}

	return commands
}

// GetEntries returns all history entries with metadata
func (h *History) GetEntries() []HistoryEntry {
	h.mu.RLock()
	defer h.mu.RUnlock()

	entries := make([]HistoryEntry, len(h.entries))
	copy(entries, h.entries)

	return entries
}

// GetLast returns the last n commands
func (h *History) GetLast(n int) []string {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if n <= 0 || len(h.entries) == 0 {
		return nil
	}

	start := len(h.entries) - n
	if start < 0 {
		start = 0
	}

	commands := make([]string, len(h.entries)-start)
	for i, entry := range h.entries[start:] {
		commands[i] = entry.Command
	}

	return commands
}

// Search searches for commands containing the given text
func (h *History) Search(query string) []HistoryEntry {
	h.mu.RLock()
	defer h.mu.RUnlock()

	query = strings.ToLower(query)
	var results []HistoryEntry

	for _, entry := range h.entries {
		if strings.Contains(strings.ToLower(entry.Command), query) {
			results = append(results, entry)
		}
	}

	return results
}

// GetMostUsed returns the most frequently used commands
func (h *History) GetMostUsed(limit int) []CommandFrequency {
	h.mu.RLock()
	defer h.mu.RUnlock()

	frequency := make(map[string]int)

	for _, entry := range h.entries {
		// Get base command (first word)
		parts := strings.Fields(entry.Command)
		if len(parts) > 0 {
			baseCommand := parts[0]
			frequency[baseCommand]++
		}
	}

	// Convert to slice and sort
	var results []CommandFrequency
	for cmd, count := range frequency {
		results = append(results, CommandFrequency{
			Command: cmd,
			Count:   count,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Count > results[j].Count
	})

	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}

	return results
}

// CommandFrequency represents command usage frequency
type CommandFrequency struct {
	Command string `json:"command"`
	Count   int    `json:"count"`
}

// GetRecent returns recent commands within the specified duration
func (h *History) GetRecent(duration time.Duration) []HistoryEntry {
	h.mu.RLock()
	defer h.mu.RUnlock()

	cutoff := time.Now().Add(-duration)
	var results []HistoryEntry

	for _, entry := range h.entries {
		if entry.Timestamp.After(cutoff) {
			results = append(results, entry)
		}
	}

	return results
}

// Clear clears the history
func (h *History) Clear() {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.entries = h.entries[:0]
	h.dirty = true

	if h.autosave {
		go h.saveToFile()
	}
}

// Size returns the number of entries in history
func (h *History) Size() int {
	h.mu.RLock()
	defer h.mu.RUnlock()

	return len(h.entries)
}

// SetMaxSize sets the maximum history size
func (h *History) SetMaxSize(size int) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.maxSize = size

	// Trim if current size exceeds new max
	if len(h.entries) > size {
		h.entries = h.entries[len(h.entries)-size:]
		h.dirty = true
	}
}

// SetAutosave enables or disables autosave
func (h *History) SetAutosave(enabled bool) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.autosave = enabled
}

// SaveToFile saves history to file
func (h *History) SaveToFile() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	return h.saveToFile()
}

// saveToFile internal save method (must be called with lock held)
func (h *History) saveToFile() error {
	if h.filePath == "" {
		return nil
	}

	// Ensure directory exists
	dir := filepath.Dir(h.filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create history directory: %v", err)
	}

	// Write to temporary file first
	tmpFile := h.filePath + ".tmp"
	file, err := os.Create(tmpFile)
	if err != nil {
		return fmt.Errorf("failed to create history file: %v", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)

	for _, entry := range h.entries {
		line := fmt.Sprintf("%s\t%d\t%s\t%t\t%d\n",
			entry.Command,
			entry.Timestamp.Unix(),
			entry.SessionID,
			entry.Success,
			entry.Duration,
		)
		if _, err := writer.WriteString(line); err != nil {
			return fmt.Errorf("failed to write history entry: %v", err)
		}
	}

	if err := writer.Flush(); err != nil {
		return fmt.Errorf("failed to flush history file: %v", err)
	}

	// Atomic rename
	if err := os.Rename(tmpFile, h.filePath); err != nil {
		return fmt.Errorf("failed to rename history file: %v", err)
	}

	h.dirty = false
	return nil
}

// loadFromFile loads history from file
func (h *History) loadFromFile() error {
	if h.filePath == "" {
		return nil
	}

	file, err := os.Open(h.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // File doesn't exist, that's ok
		}
		return fmt.Errorf("failed to open history file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var entries []HistoryEntry

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		parts := strings.Split(line, "\t")
		if len(parts) < 2 {
			continue // Skip malformed lines
		}

		entry := HistoryEntry{
			Command: parts[0],
		}

		// Parse timestamp
		if len(parts) > 1 {
			if timestamp, err := time.Parse("1136239445", parts[1]); err == nil {
				entry.Timestamp = timestamp
			} else if timestamp, err := time.Parse(time.RFC3339, parts[1]); err == nil {
				entry.Timestamp = timestamp
			}
		}

		// Parse session ID
		if len(parts) > 2 {
			entry.SessionID = parts[2]
		}

		// Parse success
		if len(parts) > 3 {
			entry.Success = parts[3] == "true"
		}

		// Parse duration
		if len(parts) > 4 {
			if duration, err := time.ParseDuration(parts[4] + "ms"); err == nil {
				entry.Duration = duration.Nanoseconds() / 1e6
			}
		}

		entries = append(entries, entry)
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("failed to read history file: %v", err)
	}

	// Keep only the most recent entries if over max size
	if len(entries) > h.maxSize {
		entries = entries[len(entries)-h.maxSize:]
	}

	h.mu.Lock()
	h.entries = entries
	h.mu.Unlock()

	return nil
}

// GetStatistics returns history statistics
func (h *History) GetStatistics() HistoryStatistics {
	h.mu.RLock()
	defer h.mu.RUnlock()

	stats := HistoryStatistics{
		TotalCommands: len(h.entries),
		CommandCounts: make(map[string]int),
	}

	if len(h.entries) == 0 {
		return stats
	}

	var totalDuration int64
	successCount := 0

	for _, entry := range h.entries {
		// Get base command
		parts := strings.Fields(entry.Command)
		if len(parts) > 0 {
			baseCommand := parts[0]
			stats.CommandCounts[baseCommand]++
		}

		totalDuration += entry.Duration
		if entry.Success {
			successCount++
		}

		if stats.FirstCommand.IsZero() || entry.Timestamp.Before(stats.FirstCommand) {
			stats.FirstCommand = entry.Timestamp
		}

		if entry.Timestamp.After(stats.LastCommand) {
			stats.LastCommand = entry.Timestamp
		}
	}

	stats.AverageDuration = float64(totalDuration) / float64(len(h.entries))
	stats.SuccessRate = float64(successCount) / float64(len(h.entries))

	return stats
}

// HistoryStatistics represents history statistics
type HistoryStatistics struct {
	TotalCommands   int            `json:"total_commands"`
	CommandCounts   map[string]int `json:"command_counts"`
	AverageDuration float64        `json:"average_duration_ms"`
	SuccessRate     float64        `json:"success_rate"`
	FirstCommand    time.Time      `json:"first_command"`
	LastCommand     time.Time      `json:"last_command"`
}
