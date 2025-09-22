package indexer

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
)

// FileWatcher monitors file system changes in real-time
type FileWatcher struct {
	watcher       *fsnotify.Watcher
	changeQueue   chan WatchEvent
	debouncer     *Debouncer
	config        *WatcherConfig
	watchedPaths  map[string]bool
	changeHandler ChangeHandler
	mu            sync.RWMutex
	running       bool
	stopChan      chan struct{}
	wg            sync.WaitGroup
	stats         *WatcherStatistics
}

// WatcherConfig contains configuration for file watching
type WatcherConfig struct {
	DebounceDelay    time.Duration `json:"debounce_delay"`    // Delay before processing changes
	BatchSize        int           `json:"batch_size"`        // Max events in a batch
	QueueSize        int           `json:"queue_size"`        // Size of event queue
	IgnorePatterns   []string      `json:"ignore_patterns"`   // Patterns to ignore
	IgnoreExtensions []string      `json:"ignore_extensions"` // Extensions to ignore
	WatchHidden      bool          `json:"watch_hidden"`      // Watch hidden files
	Recursive        bool          `json:"recursive"`         // Watch subdirectories
	FollowSymlinks   bool          `json:"follow_symlinks"`   // Follow symbolic links
}

// WatchEvent represents a file system event
type WatchEvent struct {
	Path      string                 `json:"path"`
	Operation Operation              `json:"operation"`
	Timestamp time.Time              `json:"timestamp"`
	Size      int64                  `json:"size,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// Operation represents the type of file system operation
type Operation string

const (
	OpCreate Operation = "create"
	OpWrite  Operation = "write"
	OpRemove Operation = "remove"
	OpRename Operation = "rename"
	OpChmod  Operation = "chmod"
)

// ChangeHandler defines the interface for handling file changes
type ChangeHandler interface {
	HandleChanges(events []WatchEvent) error
}

// WatcherStatistics tracks file watcher statistics
type WatcherStatistics struct {
	EventsProcessed  int64     `json:"events_processed"`
	EventsIgnored    int64     `json:"events_ignored"`
	BatchesProcessed int64     `json:"batches_processed"`
	ErrorCount       int64     `json:"error_count"`
	StartTime        time.Time `json:"start_time"`
	LastEventTime    time.Time `json:"last_event_time"`
	AverageEventRate float64   `json:"average_event_rate"`
	PathsWatched     int       `json:"paths_watched"`
	mu               sync.RWMutex
}

// Debouncer handles debouncing of file system events
type Debouncer struct {
	delay    time.Duration
	events   map[string]*WatchEvent
	timer    *time.Timer
	callback func(map[string]*WatchEvent)
	mu       sync.Mutex
}

// NewFileWatcher creates a new file watcher
func NewFileWatcher(config *WatcherConfig, handler ChangeHandler) (*FileWatcher, error) {
	if config == nil {
		config = &WatcherConfig{
			DebounceDelay:    time.Millisecond * 500,
			BatchSize:        100,
			QueueSize:        1000,
			Recursive:        true,
			FollowSymlinks:   false,
			WatchHidden:      false,
			IgnoreExtensions: []string{".tmp", ".log", ".cache", ".swp"},
		}
	}

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, fmt.Errorf("failed to create file watcher: %v", err)
	}

	fw := &FileWatcher{
		watcher:       watcher,
		changeQueue:   make(chan WatchEvent, config.QueueSize),
		config:        config,
		changeHandler: handler,
		watchedPaths:  make(map[string]bool),
		stopChan:      make(chan struct{}),
		stats: &WatcherStatistics{
			StartTime: time.Now(),
		},
	}

	// Initialize debouncer
	fw.debouncer = &Debouncer{
		delay:  config.DebounceDelay,
		events: make(map[string]*WatchEvent),
		callback: func(events map[string]*WatchEvent) {
			fw.processDebouncedEvents(events)
		},
	}

	return fw, nil
}

// AddPath adds a path to be watched
func (fw *FileWatcher) AddPath(path string) error {
	fw.mu.Lock()
	defer fw.mu.Unlock()

	// Check if already watching
	if fw.watchedPaths[path] {
		return nil
	}

	err := fw.watcher.Add(path)
	if err != nil {
		return fmt.Errorf("failed to add path to watcher: %v", err)
	}

	fw.watchedPaths[path] = true

	// If recursive, add subdirectories
	if fw.config.Recursive {
		err = filepath.Walk(path, func(walkPath string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if info.IsDir() && walkPath != path {
				if fw.shouldWatchPath(walkPath) {
					if err := fw.watcher.Add(walkPath); err == nil {
						fw.watchedPaths[walkPath] = true
					}
				}
			}

			return nil
		})

		if err != nil {
			return fmt.Errorf("failed to add subdirectories: %v", err)
		}
	}

	fw.updatePathCount()
	return nil
}

// RemovePath removes a path from being watched
func (fw *FileWatcher) RemovePath(path string) error {
	fw.mu.Lock()
	defer fw.mu.Unlock()

	if !fw.watchedPaths[path] {
		return nil
	}

	err := fw.watcher.Remove(path)
	if err != nil {
		return fmt.Errorf("failed to remove path from watcher: %v", err)
	}

	delete(fw.watchedPaths, path)
	fw.updatePathCount()
	return nil
}

// Start starts the file watcher
func (fw *FileWatcher) Start(ctx context.Context) error {
	fw.mu.Lock()
	if fw.running {
		fw.mu.Unlock()
		return fmt.Errorf("file watcher is already running")
	}
	fw.running = true
	fw.mu.Unlock()

	// Start event processing goroutine
	fw.wg.Add(1)
	go fw.processEvents(ctx)

	// Start file system event monitoring goroutine
	fw.wg.Add(1)
	go fw.monitorFileSystem(ctx)

	log.Printf("File watcher started, watching %d paths", len(fw.watchedPaths))
	return nil
}

// Stop stops the file watcher
func (fw *FileWatcher) Stop() error {
	fw.mu.Lock()
	if !fw.running {
		fw.mu.Unlock()
		return nil
	}
	fw.running = false
	fw.mu.Unlock()

	// Signal stop
	close(fw.stopChan)

	// Wait for goroutines to finish
	fw.wg.Wait()

	// Close the watcher
	err := fw.watcher.Close()
	if err != nil {
		return fmt.Errorf("failed to close file watcher: %v", err)
	}

	log.Println("File watcher stopped")
	return nil
}

// monitorFileSystem monitors file system events
func (fw *FileWatcher) monitorFileSystem(ctx context.Context) {
	defer fw.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-fw.stopChan:
			return
		case event, ok := <-fw.watcher.Events:
			if !ok {
				return
			}
			fw.handleFileSystemEvent(event)
		case err, ok := <-fw.watcher.Errors:
			if !ok {
				return
			}
			fw.handleWatcherError(err)
		}
	}
}

// handleFileSystemEvent processes a file system event
func (fw *FileWatcher) handleFileSystemEvent(event fsnotify.Event) {
	// Check if we should ignore this event
	if fw.shouldIgnoreEvent(event) {
		fw.incrementIgnored()
		return
	}

	// Convert fsnotify event to our WatchEvent
	watchEvent := fw.convertEvent(event)

	// Add to debouncer
	fw.debouncer.AddEvent(watchEvent)

	fw.incrementProcessed()
}

// convertEvent converts fsnotify.Event to WatchEvent
func (fw *FileWatcher) convertEvent(event fsnotify.Event) WatchEvent {
	var op Operation

	switch {
	case event.Op&fsnotify.Create == fsnotify.Create:
		op = OpCreate
	case event.Op&fsnotify.Write == fsnotify.Write:
		op = OpWrite
	case event.Op&fsnotify.Remove == fsnotify.Remove:
		op = OpRemove
	case event.Op&fsnotify.Rename == fsnotify.Rename:
		op = OpRename
	case event.Op&fsnotify.Chmod == fsnotify.Chmod:
		op = OpChmod
	default:
		op = OpWrite // Default to write
	}

	watchEvent := WatchEvent{
		Path:      event.Name,
		Operation: op,
		Timestamp: time.Now(),
		Metadata:  make(map[string]interface{}),
	}

	// Add file size if file exists
	if info, err := os.Stat(event.Name); err == nil && !info.IsDir() {
		watchEvent.Size = info.Size()
		watchEvent.Metadata["is_dir"] = false
		watchEvent.Metadata["mod_time"] = info.ModTime()
		watchEvent.Metadata["permissions"] = info.Mode().String()
	} else {
		watchEvent.Metadata["is_dir"] = true
	}

	return watchEvent
}

// shouldIgnoreEvent checks if an event should be ignored
func (fw *FileWatcher) shouldIgnoreEvent(event fsnotify.Event) bool {
	path := event.Name

	// Ignore hidden files if configured
	if !fw.config.WatchHidden && strings.HasPrefix(filepath.Base(path), ".") {
		return true
	}

	// Check ignore patterns
	for _, pattern := range fw.config.IgnorePatterns {
		if strings.Contains(path, pattern) {
			return true
		}
	}

	// Check ignore extensions
	ext := strings.ToLower(filepath.Ext(path))
	for _, ignoreExt := range fw.config.IgnoreExtensions {
		if ext == ignoreExt {
			return true
		}
	}

	// Ignore common temporary files
	base := filepath.Base(path)
	if strings.HasSuffix(base, "~") || strings.HasPrefix(base, ".#") {
		return true
	}

	return false
}

// shouldWatchPath checks if a path should be watched
func (fw *FileWatcher) shouldWatchPath(path string) bool {
	// Don't watch hidden directories unless configured
	if !fw.config.WatchHidden && strings.HasPrefix(filepath.Base(path), ".") {
		return false
	}

	// Common directories to ignore
	ignoreDirs := []string{
		".git", ".svn", ".hg", "node_modules", "vendor", "__pycache__",
		".vscode", ".idea", "target", "dist", "build", ".next",
	}

	base := filepath.Base(path)
	for _, ignoreDir := range ignoreDirs {
		if base == ignoreDir {
			return false
		}
	}

	return true
}

// processEvents processes queued events
func (fw *FileWatcher) processEvents(ctx context.Context) {
	defer fw.wg.Done()

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	batch := make([]WatchEvent, 0, fw.config.BatchSize)

	for {
		select {
		case <-ctx.Done():
			return
		case <-fw.stopChan:
			return
		case event := <-fw.changeQueue:
			batch = append(batch, event)

			// Process batch when full or on ticker
			if len(batch) >= fw.config.BatchSize {
				fw.processBatch(batch)
				batch = batch[:0] // Reset slice
			}

		case <-ticker.C:
			// Process any pending events
			if len(batch) > 0 {
				fw.processBatch(batch)
				batch = batch[:0] // Reset slice
			}
		}
	}
}

// processBatch processes a batch of events
func (fw *FileWatcher) processBatch(events []WatchEvent) {
	if len(events) == 0 {
		return
	}

	// Call the change handler
	if fw.changeHandler != nil {
		if err := fw.changeHandler.HandleChanges(events); err != nil {
			fw.incrementErrors()
			log.Printf("Error handling file changes: %v", err)
		}
	}

	fw.incrementBatches()
	fw.updateLastEventTime()
}

// Debouncer methods

// AddEvent adds an event to the debouncer
func (d *Debouncer) AddEvent(event WatchEvent) {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Store the latest event for each path
	d.events[event.Path] = &event

	// Reset timer
	if d.timer != nil {
		d.timer.Stop()
	}

	d.timer = time.AfterFunc(d.delay, func() {
		d.mu.Lock()
		events := d.events
		d.events = make(map[string]*WatchEvent)
		d.mu.Unlock()

		if len(events) > 0 {
			d.callback(events)
		}
	})
}

// processDebouncedEvents processes debounced events
func (fw *FileWatcher) processDebouncedEvents(events map[string]*WatchEvent) {
	batch := make([]WatchEvent, 0, len(events))

	for _, event := range events {
		batch = append(batch, *event)
	}

	// Send to processing queue
	for _, event := range batch {
		select {
		case fw.changeQueue <- event:
			// Successfully queued
		default:
			// Queue is full, skip this event
			fw.incrementIgnored()
		}
	}
}

// handleWatcherError handles watcher errors
func (fw *FileWatcher) handleWatcherError(err error) {
	fw.incrementErrors()
	log.Printf("File watcher error: %v", err)
}

// Statistics methods

func (fw *FileWatcher) incrementProcessed() {
	fw.stats.mu.Lock()
	defer fw.stats.mu.Unlock()
	fw.stats.EventsProcessed++
	fw.updateEventRate()
}

func (fw *FileWatcher) incrementIgnored() {
	fw.stats.mu.Lock()
	defer fw.stats.mu.Unlock()
	fw.stats.EventsIgnored++
}

func (fw *FileWatcher) incrementBatches() {
	fw.stats.mu.Lock()
	defer fw.stats.mu.Unlock()
	fw.stats.BatchesProcessed++
}

func (fw *FileWatcher) incrementErrors() {
	fw.stats.mu.Lock()
	defer fw.stats.mu.Unlock()
	fw.stats.ErrorCount++
}

func (fw *FileWatcher) updateLastEventTime() {
	fw.stats.mu.Lock()
	defer fw.stats.mu.Unlock()
	fw.stats.LastEventTime = time.Now()
}

func (fw *FileWatcher) updatePathCount() {
	fw.stats.mu.Lock()
	defer fw.stats.mu.Unlock()
	fw.stats.PathsWatched = len(fw.watchedPaths)
}

func (fw *FileWatcher) updateEventRate() {
	elapsed := time.Since(fw.stats.StartTime).Seconds()
	if elapsed > 0 {
		fw.stats.AverageEventRate = float64(fw.stats.EventsProcessed) / elapsed
	}
}

// GetStatistics returns watcher statistics
func (fw *FileWatcher) GetStatistics() *WatcherStatistics {
	fw.stats.mu.RLock()
	defer fw.stats.mu.RUnlock()

	// Return a copy
	return &WatcherStatistics{
		EventsProcessed:  fw.stats.EventsProcessed,
		EventsIgnored:    fw.stats.EventsIgnored,
		BatchesProcessed: fw.stats.BatchesProcessed,
		ErrorCount:       fw.stats.ErrorCount,
		StartTime:        fw.stats.StartTime,
		LastEventTime:    fw.stats.LastEventTime,
		AverageEventRate: fw.stats.AverageEventRate,
		PathsWatched:     fw.stats.PathsWatched,
	}
}

// GetWatchedPaths returns list of watched paths
func (fw *FileWatcher) GetWatchedPaths() []string {
	fw.mu.RLock()
	defer fw.mu.RUnlock()

	paths := make([]string, 0, len(fw.watchedPaths))
	for path := range fw.watchedPaths {
		paths = append(paths, path)
	}

	return paths
}

// IsWatching checks if a path is being watched
func (fw *FileWatcher) IsWatching(path string) bool {
	fw.mu.RLock()
	defer fw.mu.RUnlock()

	return fw.watchedPaths[path]
}
