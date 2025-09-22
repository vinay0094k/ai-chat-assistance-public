package indexer

import (
	"context"
	"crypto/md5"
	"fmt"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/yourusername/ai-code-assistant/storage"
)

// BatchProcessor handles processing files in optimized batches
type BatchProcessor struct {
	codeParser     *CodeParser
	changeDetector *ChangeDetector
	db             *storage.SQLiteDB
	config         *BatchConfig
	stats          *BatchStatistics
	progressChan   chan *ProgressUpdate
	errorChan      chan error
	workQueue      chan *WorkItem
	resultQueue    chan *ProcessResult
	workers        []*BatchWorker
	wg             sync.WaitGroup
	running        int32
	stopChan       chan struct{}
}

// BatchConfig contains configuration for batch processing
type BatchConfig struct {
	BatchSize       int           `json:"batch_size"`        // Files per batch
	WorkerCount     int           `json:"worker_count"`      // Number of worker goroutines
	QueueSize       int           `json:"queue_size"`        // Size of work queue
	MemoryLimit     int64         `json:"memory_limit"`      // Memory limit in bytes
	TimeoutPerFile  time.Duration `json:"timeout_per_file"`  // Timeout for processing single file
	TimeoutPerBatch time.Duration `json:"timeout_per_batch"` // Timeout for entire batch
	RetryAttempts   int           `json:"retry_attempts"`    // Number of retry attempts
	RetryDelay      time.Duration `json:"retry_delay"`       // Delay between retries
	EnableMetrics   bool          `json:"enable_metrics"`    // Enable detailed metrics collection
}

// WorkItem represents a unit of work for batch processing
type WorkItem struct {
	ID          string                 `json:"id"`
	FilePath    string                 `json:"file_path"`
	Priority    int                    `json:"priority"` // Higher number = higher priority
	RetryCount  int                    `json:"retry_count"`
	CreatedAt   time.Time              `json:"created_at"`
	ProcessType string                 `json:"process_type"` // parse, reparse, delete
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// ProcessResult represents the result of processing a work item
type ProcessResult struct {
	WorkItem    *WorkItem     `json:"work_item"`
	ParseResult *ParseResult  `json:"parse_result,omitempty"`
	Success     bool          `json:"success"`
	Error       error         `json:"error,omitempty"`
	ProcessTime time.Duration `json:"process_time"`
	MemoryUsed  int64         `json:"memory_used"`
	ChunkCount  int           `json:"chunk_count"`
	CompletedAt time.Time     `json:"completed_at"`
}

// ProgressUpdate represents progress information
type ProgressUpdate struct {
	Total                 int           `json:"total"`
	Completed             int           `json:"completed"`
	Failed                int           `json:"failed"`
	Percentage            float64       `json:"percentage"`
	ElapsedTime           time.Duration `json:"elapsed_time"`
	ETA                   time.Duration `json:"eta"`
	CurrentFile           string        `json:"current_file"`
	ThroughputFilesPerSec float64       `json:"throughput_files_per_sec"`
	ThroughputMBPerSec    float64       `json:"throughput_mb_per_sec"`
}

// BatchStatistics tracks batch processing statistics
type BatchStatistics struct {
	TotalFiles       int64         `json:"total_files"`
	CompletedFiles   int64         `json:"completed_files"`
	FailedFiles      int64         `json:"failed_files"`
	TotalChunks      int64         `json:"total_chunks"`
	TotalBytes       int64         `json:"total_bytes"`
	ProcessedBytes   int64         `json:"processed_bytes"`
	StartTime        time.Time     `json:"start_time"`
	EndTime          time.Time     `json:"end_time"`
	Duration         time.Duration `json:"duration"`
	AverageFileTime  time.Duration `json:"average_file_time"`
	PeakMemoryUsage  int64         `json:"peak_memory_usage"`
	BatchesProcessed int64         `json:"batches_processed"`
	RetryCount       int64         `json:"retry_count"`
	mu               sync.RWMutex
}

// BatchWorker represents a worker that processes batches
type BatchWorker struct {
	ID            int
	processor     *BatchProcessor
	workQueue     <-chan *WorkItem
	resultQueue   chan<- *ProcessResult
	stopChan      <-chan struct{}
	currentMemory int64
	stats         *WorkerStats
}

// WorkerStats tracks individual worker statistics
type WorkerStats struct {
	WorkerID        int           `json:"worker_id"`
	ItemsProcessed  int64         `json:"items_processed"`
	ItemsFailed     int64         `json:"items_failed"`
	TotalTime       time.Duration `json:"total_time"`
	AverageTime     time.Duration `json:"average_time"`
	PeakMemoryUsage int64         `json:"peak_memory_usage"`
	LastActiveTime  time.Time     `json:"last_active_time"`
	mu              sync.RWMutex
}

// NewBatchProcessor creates a new batch processor
func NewBatchProcessor(codeParser *CodeParser, changeDetector *ChangeDetector, db *storage.SQLiteDB, config *BatchConfig) *BatchProcessor {
	if config == nil {
		config = &BatchConfig{
			BatchSize:       50,
			WorkerCount:     runtime.NumCPU(),
			QueueSize:       1000,
			MemoryLimit:     1024 * 1024 * 1024, // 1GB
			TimeoutPerFile:  time.Second * 30,
			TimeoutPerBatch: time.Minute * 5,
			RetryAttempts:   3,
			RetryDelay:      time.Second * 2,
			EnableMetrics:   true,
		}
	}

	// Ensure worker count doesn't exceed reasonable limits
	if config.WorkerCount > runtime.NumCPU()*2 {
		config.WorkerCount = runtime.NumCPU() * 2
	}

	bp := &BatchProcessor{
		codeParser:     codeParser,
		changeDetector: changeDetector,
		db:             db,
		config:         config,
		stats:          &BatchStatistics{},
		progressChan:   make(chan *ProgressUpdate, 100),
		errorChan:      make(chan error, 100),
		workQueue:      make(chan *WorkItem, config.QueueSize),
		resultQueue:    make(chan *ProcessResult, config.QueueSize),
		stopChan:       make(chan struct{}),
	}

	return bp
}

// Start starts the batch processor
func (bp *BatchProcessor) Start(ctx context.Context) error {
	if !atomic.CompareAndSwapInt32(&bp.running, 0, 1) {
		return fmt.Errorf("batch processor is already running")
	}

	bp.stats.StartTime = time.Now()

	// Create workers
	bp.workers = make([]*BatchWorker, bp.config.WorkerCount)
	for i := 0; i < bp.config.WorkerCount; i++ {
		worker := &BatchWorker{
			ID:          i,
			processor:   bp,
			workQueue:   bp.workQueue,
			resultQueue: bp.resultQueue,
			stopChan:    bp.stopChan,
			stats: &WorkerStats{
				WorkerID: i,
			},
		}
		bp.workers[i] = worker

		// Start worker
		bp.wg.Add(1)
		go worker.Run(ctx)
	}

	// Start result processor
	bp.wg.Add(1)
	go bp.processResults(ctx)

	// Start progress monitor
	bp.wg.Add(1)
	go bp.monitorProgress(ctx)

	return nil
}

// Stop stops the batch processor
func (bp *BatchProcessor) Stop() error {
	if !atomic.CompareAndSwapInt32(&bp.running, 1, 0) {
		return nil // Already stopped
	}

	close(bp.stopChan)
	bp.wg.Wait()

	bp.stats.EndTime = time.Now()
	bp.stats.Duration = bp.stats.EndTime.Sub(bp.stats.StartTime)

	return nil
}

// ProcessFiles processes a list of files in batches
func (bp *BatchProcessor) ProcessFiles(ctx context.Context, filePaths []string) error {
	if !bp.IsRunning() {
		return fmt.Errorf("batch processor is not running")
	}

	// Update total files count
	bp.stats.mu.Lock()
	bp.stats.TotalFiles = int64(len(filePaths))
	bp.stats.mu.Unlock()

	// Create work items
	workItems := make([]*WorkItem, len(filePaths))
	for i, filePath := range filePaths {
		workItems[i] = &WorkItem{
			ID:          bp.generateWorkItemID(filePath),
			FilePath:    filePath,
			Priority:    bp.calculatePriority(filePath),
			ProcessType: "parse",
			CreatedAt:   time.Now(),
		}
	}

	// Sort by priority (higher priority first)
	bp.sortWorkItemsByPriority(workItems)

	// Submit work items in batches
	return bp.submitWorkItems(ctx, workItems)
}

// ProcessChangedFiles processes only files that have changed
func (bp *BatchProcessor) ProcessChangedFiles(ctx context.Context, changes []*FileChange) error {
	if !bp.IsRunning() {
		return fmt.Errorf("batch processor is not running")
	}

	var workItems []*WorkItem

	for _, change := range changes {
		var processType string
		switch change.ChangeType {
		case ChangeTypeAdded, ChangeTypeModified:
			processType = "parse"
		case ChangeTypeDeleted:
			processType = "delete"
		case ChangeTypeRenamed, ChangeTypeMoved:
			processType = "reparse"
		default:
			continue
		}

		workItem := &WorkItem{
			ID:          bp.generateWorkItemID(change.FilePath),
			FilePath:    change.FilePath,
			Priority:    bp.calculatePriorityForChange(change),
			ProcessType: processType,
			CreatedAt:   time.Now(),
			Metadata: map[string]interface{}{
				"change_type": change.ChangeType,
				"file_size":   change.FileSize,
				"language":    change.Language,
			},
		}

		workItems = append(workItems, workItem)
	}

	// Update total files count
	bp.stats.mu.Lock()
	bp.stats.TotalFiles = int64(len(workItems))
	bp.stats.mu.Unlock()

	// Sort by priority
	bp.sortWorkItemsByPriority(workItems)

	return bp.submitWorkItems(ctx, workItems)
}

// submitWorkItems submits work items to the queue
func (bp *BatchProcessor) submitWorkItems(ctx context.Context, workItems []*WorkItem) error {
	for _, item := range workItems {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-bp.stopChan:
			return fmt.Errorf("batch processor stopped")
		case bp.workQueue <- item:
			// Successfully queued
		default:
			// Queue is full, wait a bit and try again
			time.Sleep(time.Millisecond * 100)
			select {
			case bp.workQueue <- item:
				// Successfully queued after wait
			default:
				return fmt.Errorf("work queue is full, cannot submit more items")
			}
		}
	}

	return nil
}

// processResults processes results from workers
func (bp *BatchProcessor) processResults(ctx context.Context) {
	defer bp.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-bp.stopChan:
			return
		case result := <-bp.resultQueue:
			bp.handleResult(result)
		}
	}
}

// handleResult handles a processing result
func (bp *BatchProcessor) handleResult(result *ProcessResult) {
	bp.stats.mu.Lock()
	defer bp.stats.mu.Unlock()

	if result.Success {
		bp.stats.CompletedFiles++
		if result.ParseResult != nil {
			bp.stats.TotalChunks += int64(len(result.ParseResult.Chunks))
			bp.stats.ProcessedBytes += result.ParseResult.Size
		}
	} else {
		bp.stats.FailedFiles++

		// Retry if needed
		if result.WorkItem.RetryCount < bp.config.RetryAttempts {
			time.AfterFunc(bp.config.RetryDelay, func() {
				result.WorkItem.RetryCount++
				bp.stats.mu.Lock()
				bp.stats.RetryCount++
				bp.stats.mu.Unlock()

				select {
				case bp.workQueue <- result.WorkItem:
					// Successfully requeued
				default:
					// Queue full, increment failed count
					bp.stats.mu.Lock()
					bp.stats.FailedFiles++
					bp.stats.mu.Unlock()
				}
			})
		}
	}

	// Update peak memory usage
	if result.MemoryUsed > bp.stats.PeakMemoryUsage {
		bp.stats.PeakMemoryUsage = result.MemoryUsed
	}

	// Calculate average file time
	totalCompleted := bp.stats.CompletedFiles + bp.stats.FailedFiles
	if totalCompleted > 0 {
		totalTime := time.Since(bp.stats.StartTime)
		bp.stats.AverageFileTime = totalTime / time.Duration(totalCompleted)
	}
}

// monitorProgress monitors and reports progress
func (bp *BatchProcessor) monitorProgress(ctx context.Context) {
	defer bp.wg.Done()

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	var lastCompleted int64
	var lastBytes int64
	lastUpdate := time.Now()

	for {
		select {
		case <-ctx.Done():
			return
		case <-bp.stopChan:
			return
		case <-ticker.C:
			bp.stats.mu.RLock()

			now := time.Now()
			timeDelta := now.Sub(lastUpdate).Seconds()

			// Calculate throughput
			completedDelta := bp.stats.CompletedFiles - lastCompleted
			bytesDelta := bp.stats.ProcessedBytes - lastBytes

			filesPerSec := float64(completedDelta) / timeDelta
			mbPerSec := float64(bytesDelta) / (1024 * 1024) / timeDelta

			// Calculate progress
			var percentage float64
			if bp.stats.TotalFiles > 0 {
				percentage = float64(bp.stats.CompletedFiles) / float64(bp.stats.TotalFiles) * 100
			}

			// Calculate ETA
			var eta time.Duration
			if filesPerSec > 0 {
				remaining := bp.stats.TotalFiles - bp.stats.CompletedFiles
				eta = time.Duration(float64(remaining)/filesPerSec) * time.Second
			}

			progress := &ProgressUpdate{
				Total:                 int(bp.stats.TotalFiles),
				Completed:             int(bp.stats.CompletedFiles),
				Failed:                int(bp.stats.FailedFiles),
				Percentage:            percentage,
				ElapsedTime:           time.Since(bp.stats.StartTime),
				ETA:                   eta,
				ThroughputFilesPerSec: filesPerSec,
				ThroughputMBPerSec:    mbPerSec,
			}

			bp.stats.mu.RUnlock()

			// Send progress update
			select {
			case bp.progressChan <- progress:
			default:
				// Channel full, skip this update
			}

			// Update counters for next iteration
			lastCompleted = bp.stats.CompletedFiles
			lastBytes = bp.stats.ProcessedBytes
			lastUpdate = now
		}
	}
}

// Worker methods

// Run runs the batch worker
func (bw *BatchWorker) Run(ctx context.Context) {
	defer bw.processor.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-bw.stopChan:
			return
		case workItem := <-bw.workQueue:
			result := bw.processWorkItem(ctx, workItem)

			select {
			case bw.resultQueue <- result:
				// Result sent successfully
			default:
				// Result queue full, log error
				bw.processor.errorChan <- fmt.Errorf("result queue full, dropping result for %s", workItem.FilePath)
			}
		}
	}
}

// processWorkItem processes a single work item
func (bw *BatchWorker) processWorkItem(ctx context.Context, item *WorkItem) *ProcessResult {
	start := time.Now()
	startMemory := bw.getCurrentMemoryUsage()

	bw.stats.mu.Lock()
	bw.stats.LastActiveTime = start
	bw.stats.mu.Unlock()

	result := &ProcessResult{
		WorkItem:    item,
		Success:     false,
		CompletedAt: time.Now(),
	}

	// Create timeout context
	timeoutCtx, cancel := context.WithTimeout(ctx, bw.processor.config.TimeoutPerFile)
	defer cancel()

	switch item.ProcessType {
	case "parse":
		result.ParseResult, result.Error = bw.parseFile(timeoutCtx, item.FilePath)
		result.Success = result.Error == nil
		if result.ParseResult != nil {
			result.ChunkCount = len(result.ParseResult.Chunks)
		}

	case "reparse":
		// Same as parse for now
		result.ParseResult, result.Error = bw.parseFile(timeoutCtx, item.FilePath)
		result.Success = result.Error == nil
		if result.ParseResult != nil {
			result.ChunkCount = len(result.ParseResult.Chunks)
		}

	case "delete":
		result.Error = bw.deleteFile(timeoutCtx, item.FilePath)
		result.Success = result.Error == nil

	default:
		result.Error = fmt.Errorf("unknown process type: %s", item.ProcessType)
	}

	// Calculate metrics
	result.ProcessTime = time.Since(start)
	result.MemoryUsed = bw.getCurrentMemoryUsage() - startMemory

	// Update worker stats
	bw.updateStats(result)

	return result
}

// parseFile parses a file using the code parser
func (bw *BatchWorker) parseFile(ctx context.Context, filePath string) (*ParseResult, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	return bw.processor.codeParser.ParseFile(filePath)
}

// deleteFile handles file deletion
func (bw *BatchWorker) deleteFile(ctx context.Context, filePath string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// Remove from change detector
	bw.processor.changeDetector.RemoveFile(filePath)

	// Remove from database (this would need to be implemented)
	// For now, just return success
	return nil
}

// updateStats updates worker statistics
func (bw *BatchWorker) updateStats(result *ProcessResult) {
	bw.stats.mu.Lock()
	defer bw.stats.mu.Unlock()

	if result.Success {
		bw.stats.ItemsProcessed++
	} else {
		bw.stats.ItemsFailed++
	}

	bw.stats.TotalTime += result.ProcessTime

	totalItems := bw.stats.ItemsProcessed + bw.stats.ItemsFailed
	if totalItems > 0 {
		bw.stats.AverageTime = bw.stats.TotalTime / time.Duration(totalItems)
	}

	if result.MemoryUsed > bw.stats.PeakMemoryUsage {
		bw.stats.PeakMemoryUsage = result.MemoryUsed
	}
}

// getCurrentMemoryUsage returns current memory usage (simplified)
func (bw *BatchWorker) getCurrentMemoryUsage() int64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.Alloc)
}

// Utility methods

func (bp *BatchProcessor) generateWorkItemID(filePath string) string {
	return fmt.Sprintf("wi_%s_%d", bp.calculateHash(filePath), time.Now().UnixNano())
}

func (bp *BatchProcessor) calculateHash(content string) string {
	return fmt.Sprintf("%x", md5.Sum([]byte(content)))
}

func (bp *BatchProcessor) calculatePriority(filePath string) int {
	// Higher priority for certain file types
	ext := strings.ToLower(filepath.Ext(filePath))

	priorityMap := map[string]int{
		".go":   100,
		".py":   90,
		".js":   80,
		".ts":   80,
		".java": 70,
		".rs":   70,
		".cpp":  60,
		".c":    60,
	}

	if priority, exists := priorityMap[ext]; exists {
		return priority
	}

	return 50 // Default priority
}

func (bp *BatchProcessor) calculatePriorityForChange(change *FileChange) int {
	basePriority := bp.calculatePriority(change.FilePath)

	// Boost priority for certain change types
	switch change.ChangeType {
	case ChangeTypeAdded:
		return basePriority + 20
	case ChangeTypeModified:
		return basePriority + 10
	case ChangeTypeDeleted:
		return basePriority + 30 // High priority to clean up quickly
	default:
		return basePriority
	}
}

func (bp *BatchProcessor) sortWorkItemsByPriority(items []*WorkItem) {
	sort.Slice(items, func(i, j int) bool {
		return items[i].Priority > items[j].Priority
	})
}

// Getters and utility methods

func (bp *BatchProcessor) IsRunning() bool {
	return atomic.LoadInt32(&bp.running) == 1
}

func (bp *BatchProcessor) GetStatistics() *BatchStatistics {
	bp.stats.mu.RLock()
	defer bp.stats.mu.RUnlock()

	// Return a copy
	stats := *bp.stats
	return &stats
}

func (bp *BatchProcessor) GetProgressChannel() <-chan *ProgressUpdate {
	return bp.progressChan
}

func (bp *BatchProcessor) GetErrorChannel() <-chan error {
	return bp.errorChan
}

func (bp *BatchProcessor) GetWorkerStats() []*WorkerStats {
	var stats []*WorkerStats

	for _, worker := range bp.workers {
		worker.stats.mu.RLock()
		workerStats := *worker.stats
		worker.stats.mu.RUnlock()
		stats = append(stats, &workerStats)
	}

	return stats
}

func (bp *BatchProcessor) GetQueueLength() int {
	return len(bp.workQueue)
}

func (bp *BatchProcessor) GetResultQueueLength() int {
	return len(bp.resultQueue)
}
