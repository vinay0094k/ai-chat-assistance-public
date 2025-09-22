package indexer

import (
	"context"
	"crypto/md5"
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ParallelProcessor handles parallel processing of multiple files simultaneously
type ParallelProcessor struct {
	batchProcessor *BatchProcessor
	config         *ParallelConfig
	stats          *ParallelStatistics
	workerPools    []*WorkerPool
	loadBalancer   *LoadBalancer
	scheduler      *TaskScheduler
	running        int32
	stopChan       chan struct{}
	wg             sync.WaitGroup
}

// ParallelConfig contains configuration for parallel processing
type ParallelConfig struct {
	MaxWorkerPools      int           `json:"max_worker_pools"`      // Maximum number of worker pools
	WorkersPerPool      int           `json:"workers_per_pool"`      // Workers per pool
	TaskQueueSize       int           `json:"task_queue_size"`       // Size of task queue per pool
	LoadBalanceStrategy string        `json:"load_balance_strategy"` // round_robin, least_loaded, priority
	SchedulingStrategy  string        `json:"scheduling_strategy"`   // fifo, priority, shortest_first
	MaxConcurrentFiles  int           `json:"max_concurrent_files"`  // Maximum files processed concurrently
	MemoryThreshold     float64       `json:"memory_threshold"`      // Memory usage threshold (0.0-1.0)
	CPUThreshold        float64       `json:"cpu_threshold"`         // CPU usage threshold (0.0-1.0)
	AdaptiveScaling     bool          `json:"adaptive_scaling"`      // Enable adaptive scaling
	ScaleCheckInterval  time.Duration `json:"scale_check_interval"`  // How often to check scaling needs
}

// WorkerPool represents a pool of workers for processing tasks
type WorkerPool struct {
	ID          int
	workers     []*ParallelWorker
	taskQueue   chan *ParallelTask
	resultQueue chan *ParallelResult
	stats       *PoolStatistics
	active      int32
	memoryUsage int64
	cpuUsage    float64
	mu          sync.RWMutex
}

// ParallelWorker represents a worker in a parallel processing pool
type ParallelWorker struct {
	ID          int
	poolID      int
	processor   *ParallelProcessor
	taskQueue   <-chan *ParallelTask
	resultQueue chan<- *ParallelResult
	stats       *WorkerStatistics
	active      int32
	stopChan    <-chan struct{}
}

// ParallelTask represents a task for parallel processing
type ParallelTask struct {
	ID            string                 `json:"id"`
	Type          TaskType               `json:"type"`
	FilePath      string                 `json:"file_path"`
	Priority      int                    `json:"priority"`
	Dependencies  []string               `json:"dependencies"`
	EstimatedTime time.Duration          `json:"estimated_time"`
	MemoryNeeded  int64                  `json:"memory_needed"`
	CreatedAt     time.Time              `json:"created_at"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	Context       context.Context        `json:"-"`
}

// TaskType represents the type of parallel task
type TaskType string

const (
	TaskTypeParse    TaskType = "parse"
	TaskTypeAnalyze  TaskType = "analyze"
	TaskTypeIndex    TaskType = "index"
	TaskTypeEmbed    TaskType = "embed"
	TaskTypeValidate TaskType = "validate"
)

// ParallelResult represents the result of parallel processing
type ParallelResult struct {
	Task        *ParallelTask `json:"task"`
	Success     bool          `json:"success"`
	Error       error         `json:"error,omitempty"`
	ParseResult *ParseResult  `json:"parse_result,omitempty"`
	ProcessTime time.Duration `json:"process_time"`
	MemoryUsed  int64         `json:"memory_used"`
	WorkerID    int           `json:"worker_id"`
	PoolID      int           `json:"pool_id"`
	CompletedAt time.Time     `json:"completed_at"`
}

// LoadBalancer distributes tasks across worker pools
type LoadBalancer struct {
	strategy string
	pools    []*WorkerPool
	current  int32 // For round-robin
	mu       sync.RWMutex
}

// TaskScheduler schedules tasks based on priorities and dependencies
type TaskScheduler struct {
	strategy     string
	pendingTasks []*ParallelTask
	readyQueue   chan *ParallelTask
	waitingTasks map[string]*ParallelTask // Tasks waiting for dependencies
	completed    map[string]bool          // Completed task IDs
	mu           sync.RWMutex
}

// Statistics structures
type ParallelStatistics struct {
	TotalTasks     int64         `json:"total_tasks"`
	CompletedTasks int64         `json:"completed_tasks"`
	FailedTasks    int64         `json:"failed_tasks"`
	ActiveTasks    int64         `json:"active_tasks"`
	TotalPools     int           `json:"total_pools"`
	ActivePools    int           `json:"active_pools"`
	TotalWorkers   int           `json:"total_workers"`
	ActiveWorkers  int           `json:"active_workers"`
	AverageTime    time.Duration `json:"average_time"`
	TotalMemory    int64         `json:"total_memory"`
	PeakMemory     int64         `json:"peak_memory"`
	ThroughputTPS  float64       `json:"throughput_tps"` // Tasks per second
	StartTime      time.Time     `json:"start_time"`
	mu             sync.RWMutex
}

type PoolStatistics struct {
	PoolID         int           `json:"pool_id"`
	ActiveWorkers  int           `json:"active_workers"`
	QueueLength    int           `json:"queue_length"`
	TasksProcessed int64         `json:"tasks_processed"`
	TasksFailed    int64         `json:"tasks_failed"`
	AverageTime    time.Duration `json:"average_time"`
	MemoryUsage    int64         `json:"memory_usage"`
	CPUUsage       float64       `json:"cpu_usage"`
	LastActiveTime time.Time     `json:"last_active_time"`
	mu             sync.RWMutex
}

type WorkerStatistics struct {
	WorkerID       int           `json:"worker_id"`
	PoolID         int           `json:"pool_id"`
	TasksProcessed int64         `json:"tasks_processed"`
	TasksFailed    int64         `json:"tasks_failed"`
	TotalTime      time.Duration `json:"total_time"`
	AverageTime    time.Duration `json:"average_time"`
	LastTaskTime   time.Time     `json:"last_task_time"`
	MemoryUsage    int64         `json:"memory_usage"`
	mu             sync.RWMutex
}

// NewParallelProcessor creates a new parallel processor
func NewParallelProcessor(batchProcessor *BatchProcessor, config *ParallelConfig) *ParallelProcessor {
	if config == nil {
		config = &ParallelConfig{
			MaxWorkerPools:      runtime.NumCPU() / 2,
			WorkersPerPool:      4,
			TaskQueueSize:       100,
			LoadBalanceStrategy: "least_loaded",
			SchedulingStrategy:  "priority",
			MaxConcurrentFiles:  runtime.NumCPU() * 4,
			MemoryThreshold:     0.8,
			CPUThreshold:        0.8,
			AdaptiveScaling:     true,
			ScaleCheckInterval:  time.Second * 10,
		}
	}

	// Ensure reasonable limits
	if config.MaxWorkerPools <= 0 {
		config.MaxWorkerPools = 1
	}
	if config.WorkersPerPool <= 0 {
		config.WorkersPerPool = 2
	}

	pp := &ParallelProcessor{
		batchProcessor: batchProcessor,
		config:         config,
		stats: &ParallelStatistics{
			StartTime: time.Now(),
		},
		stopChan: make(chan struct{}),
	}

	// Initialize load balancer
	pp.loadBalancer = &LoadBalancer{
		strategy: config.LoadBalanceStrategy,
	}

	// Initialize task scheduler
	pp.scheduler = &TaskScheduler{
		strategy:     config.SchedulingStrategy,
		readyQueue:   make(chan *ParallelTask, config.TaskQueueSize*config.MaxWorkerPools),
		waitingTasks: make(map[string]*ParallelTask),
		completed:    make(map[string]bool),
	}

	return pp
}

// Start starts the parallel processor
func (pp *ParallelProcessor) Start(ctx context.Context) error {
	if !atomic.CompareAndSwapInt32(&pp.running, 0, 1) {
		return fmt.Errorf("parallel processor is already running")
	}

	// Create initial worker pools
	initialPools := pp.config.MaxWorkerPools
	if initialPools > 4 {
		initialPools = 4 // Start with fewer pools, scale up as needed
	}

	for i := 0; i < initialPools; i++ {
		if err := pp.createWorkerPool(i); err != nil {
			return fmt.Errorf("failed to create worker pool %d: %v", i, err)
		}
	}

	// Start task scheduler
	pp.wg.Add(1)
	go pp.runTaskScheduler(ctx)

	// Start adaptive scaling if enabled
	if pp.config.AdaptiveScaling {
		pp.wg.Add(1)
		go pp.runAdaptiveScaling(ctx)
	}

	// Start statistics monitor
	pp.wg.Add(1)
	go pp.monitorStatistics(ctx)

	return nil
}

// Stop stops the parallel processor
func (pp *ParallelProcessor) Stop() error {
	if !atomic.CompareAndSwapInt32(&pp.running, 1, 0) {
		return nil
	}

	close(pp.stopChan)
	pp.wg.Wait()

	// Stop all worker pools
	for _, pool := range pp.workerPools {
		pool.Stop()
	}

	return nil
}

// ProcessTasks processes a batch of tasks in parallel
func (pp *ParallelProcessor) ProcessTasks(ctx context.Context, tasks []*ParallelTask) error {
	if !pp.IsRunning() {
		return fmt.Errorf("parallel processor is not running")
	}

	// Update total tasks count
	pp.stats.mu.Lock()
	pp.stats.TotalTasks += int64(len(tasks))
	pp.stats.mu.Unlock()

	// Submit tasks to scheduler
	for _, task := range tasks {
		task.Context = ctx
		pp.scheduler.SubmitTask(task)
	}

	return nil
}

// ProcessFilesParallel processes files in parallel
func (pp *ParallelProcessor) ProcessFilesParallel(ctx context.Context, filePaths []string) error {
	tasks := make([]*ParallelTask, len(filePaths))

	for i, filePath := range filePaths {
		tasks[i] = &ParallelTask{
			ID:            pp.generateTaskID(filePath),
			Type:          TaskTypeParse,
			FilePath:      filePath,
			Priority:      pp.calculateTaskPriority(filePath),
			EstimatedTime: pp.estimateProcessingTime(filePath),
			MemoryNeeded:  pp.estimateMemoryNeeded(filePath),
			CreatedAt:     time.Now(),
		}
	}

	return pp.ProcessTasks(ctx, tasks)
}

// createWorkerPool creates a new worker pool
func (pp *ParallelProcessor) createWorkerPool(poolID int) error {
	pool := &WorkerPool{
		ID:          poolID,
		taskQueue:   make(chan *ParallelTask, pp.config.TaskQueueSize),
		resultQueue: make(chan *ParallelResult, pp.config.TaskQueueSize),
		stats: &PoolStatistics{
			PoolID: poolID,
		},
	}

	// Create workers for this pool
	pool.workers = make([]*ParallelWorker, pp.config.WorkersPerPool)
	for i := 0; i < pp.config.WorkersPerPool; i++ {
		worker := &ParallelWorker{
			ID:          i,
			poolID:      poolID,
			processor:   pp,
			taskQueue:   pool.taskQueue,
			resultQueue: pool.resultQueue,
			stats: &WorkerStatistics{
				WorkerID: i,
				PoolID:   poolID,
			},
			stopChan: pp.stopChan,
		}

		pool.workers[i] = worker

		// Start worker
		pp.wg.Add(1)
		go worker.Run()
	}

	// Start result processor for this pool
	pp.wg.Add(1)
	go pp.processPoolResults(pool)

	// Add to pools list
	pp.workerPools = append(pp.workerPools, pool)
	pp.loadBalancer.AddPool(pool)

	// Update statistics
	pp.stats.mu.Lock()
	pp.stats.TotalPools++
	pp.stats.TotalWorkers += pp.config.WorkersPerPool
	pp.stats.mu.Unlock()

	return nil
}

// runTaskScheduler runs the task scheduler
func (pp *ParallelProcessor) runTaskScheduler(ctx context.Context) {
	defer pp.wg.Done()

	ticker := time.NewTicker(time.Millisecond * 100)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-pp.stopChan:
			return
		case <-ticker.C:
			pp.scheduler.ScheduleTasks(pp.loadBalancer)
		}
	}
}

// runAdaptiveScaling runs the adaptive scaling logic
func (pp *ParallelProcessor) runAdaptiveScaling(ctx context.Context) {
	defer pp.wg.Done()

	ticker := time.NewTicker(pp.config.ScaleCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-pp.stopChan:
			return
		case <-ticker.C:
			pp.checkAndScale()
		}
	}
}

// checkAndScale checks if scaling is needed and performs it
func (pp *ParallelProcessor) checkAndScale() {
	memUsage := pp.getCurrentMemoryUsage()
	cpuUsage := pp.getCurrentCPUUsage()

	// Scale up if needed
	if pp.shouldScaleUp(memUsage, cpuUsage) {
		pp.scaleUp()
	}

	// Scale down if needed
	if pp.shouldScaleDown(memUsage, cpuUsage) {
		pp.scaleDown()
	}
}

// shouldScaleUp determines if we should scale up
func (pp *ParallelProcessor) shouldScaleUp(memUsage, cpuUsage float64) bool {
	// Check if we have pending tasks and resources available
	pendingTasks := pp.scheduler.GetPendingTaskCount()
	if pendingTasks == 0 {
		return false
	}

	// Check resource usage
	if memUsage > pp.config.MemoryThreshold || cpuUsage > pp.config.CPUThreshold {
		return false
	}

	// Check if we can create more pools
	if len(pp.workerPools) >= pp.config.MaxWorkerPools {
		return false
	}

	// Check queue lengths
	averageQueueLength := pp.getAverageQueueLength()
	return averageQueueLength > pp.config.TaskQueueSize/2
}

// shouldScaleDown determines if we should scale down
func (pp *ParallelProcessor) shouldScaleDown(memUsage, cpuUsage float64) bool {
	// Don't scale down if we have only one pool
	if len(pp.workerPools) <= 1 {
		return false
	}

	// Check if we have low resource usage and low pending tasks
	pendingTasks := pp.scheduler.GetPendingTaskCount()
	averageQueueLength := pp.getAverageQueueLength()

	return pendingTasks < 10 &&
		averageQueueLength < pp.config.TaskQueueSize/4 &&
		memUsage < pp.config.MemoryThreshold*0.5 &&
		cpuUsage < pp.config.CPUThreshold*0.5
}

// scaleUp creates additional worker pools
func (pp *ParallelProcessor) scaleUp() {
	if len(pp.workerPools) < pp.config.MaxWorkerPools {
		poolID := len(pp.workerPools)
		if err := pp.createWorkerPool(poolID); err == nil {
			fmt.Printf("Scaled up: created worker pool %d\n", poolID)
		}
	}
}

// scaleDown removes excess worker pools
func (pp *ParallelProcessor) scaleDown() {
	if len(pp.workerPools) > 1 {
		// Remove the last pool
		lastPool := pp.workerPools[len(pp.workerPools)-1]
		lastPool.Stop()
		pp.workerPools = pp.workerPools[:len(pp.workerPools)-1]
		pp.loadBalancer.RemovePool(lastPool)

		fmt.Printf("Scaled down: removed worker pool %d\n", lastPool.ID)
	}
}

// Worker Pool methods

// Stop stops the worker pool
func (wp *WorkerPool) Stop() {
	atomic.StoreInt32(&wp.active, 0)
	close(wp.taskQueue)
}

// IsActive returns whether the pool is active
func (wp *WorkerPool) IsActive() bool {
	return atomic.LoadInt32(&wp.active) == 1
}

// GetQueueLength returns the current queue length
func (wp *WorkerPool) GetQueueLength() int {
	return len(wp.taskQueue)
}

// GetLoadScore returns a load score for this pool
func (wp *WorkerPool) GetLoadScore() float64 {
	queueLoad := float64(wp.GetQueueLength()) / float64(cap(wp.taskQueue))
	memLoad := float64(wp.memoryUsage) / (1024 * 1024 * 1024) // Normalize to GB
	cpuLoad := wp.cpuUsage

	return (queueLoad + memLoad + cpuLoad) / 3.0
}

// Parallel Worker methods

// Run runs the parallel worker
func (pw *ParallelWorker) Run() {
	defer pw.processor.wg.Done()

	atomic.StoreInt32(&pw.active, 1)
	defer atomic.StoreInt32(&pw.active, 0)

	for {
		select {
		case <-pw.stopChan:
			return
		case task, ok := <-pw.taskQueue:
			if !ok {
				return
			}

			result := pw.processTask(task)

			select {
			case pw.resultQueue <- result:
				// Result sent successfully
			default:
				// Result queue full
				fmt.Printf("Result queue full for worker %d in pool %d\n", pw.ID, pw.poolID)
			}
		}
	}
}

// processTask processes a single task
func (pw *ParallelWorker) processTask(task *ParallelTask) *ParallelResult {
	start := time.Now()
	startMemory := pw.getCurrentMemoryUsage()

	result := &ParallelResult{
		Task:        task,
		WorkerID:    pw.ID,
		PoolID:      pw.poolID,
		CompletedAt: time.Now(),
	}

	// Update active tasks count
	atomic.AddInt64(&pw.processor.stats.ActiveTasks, 1)
	defer atomic.AddInt64(&pw.processor.stats.ActiveTasks, -1)

	switch task.Type {
	case TaskTypeParse:
		parseResult, err := pw.processor.batchProcessor.codeParser.ParseFile(task.FilePath)
		result.ParseResult = parseResult
		result.Error = err
		result.Success = err == nil

	case TaskTypeAnalyze:
		// Implement analysis logic
		result.Success = true

	case TaskTypeIndex:
		// Implement indexing logic
		result.Success = true

	case TaskTypeEmbed:
		// Implement embedding logic
		result.Success = true

	case TaskTypeValidate:
		// Implement validation logic
		result.Success = true

	default:
		result.Error = fmt.Errorf("unknown task type: %s", task.Type)
		result.Success = false
	}

	result.ProcessTime = time.Since(start)
	result.MemoryUsed = pw.getCurrentMemoryUsage() - startMemory

	// Update worker statistics
	pw.updateStats(result)

	return result
}

// updateStats updates worker statistics
func (pw *ParallelWorker) updateStats(result *ParallelResult) {
	pw.stats.mu.Lock()
	defer pw.stats.mu.Unlock()

	if result.Success {
		pw.stats.TasksProcessed++
	} else {
		pw.stats.TasksFailed++
	}

	pw.stats.TotalTime += result.ProcessTime
	pw.stats.LastTaskTime = result.CompletedAt

	totalTasks := pw.stats.TasksProcessed + pw.stats.TasksFailed
	if totalTasks > 0 {
		pw.stats.AverageTime = pw.stats.TotalTime / time.Duration(totalTasks)
	}

	if result.MemoryUsed > pw.stats.MemoryUsage {
		pw.stats.MemoryUsage = result.MemoryUsed
	}
}

// getCurrentMemoryUsage returns current memory usage
func (pw *ParallelWorker) getCurrentMemoryUsage() int64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.Alloc)
}

// Load Balancer methods

// AddPool adds a pool to the load balancer
func (lb *LoadBalancer) AddPool(pool *WorkerPool) {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	lb.pools = append(lb.pools, pool)
}

// RemovePool removes a pool from the load balancer
func (lb *LoadBalancer) RemovePool(pool *WorkerPool) {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	for i, p := range lb.pools {
		if p.ID == pool.ID {
			lb.pools = append(lb.pools[:i], lb.pools[i+1:]...)
			break
		}
	}
}

// SelectPool selects the best pool for a task
func (lb *LoadBalancer) SelectPool(task *ParallelTask) *WorkerPool {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	if len(lb.pools) == 0 {
		return nil
	}

	switch lb.strategy {
	case "round_robin":
		current := atomic.AddInt32(&lb.current, 1)
		return lb.pools[int(current-1)%len(lb.pools)]

	case "least_loaded":
		var bestPool *WorkerPool
		var bestScore float64 = 999999

		for _, pool := range lb.pools {
			if pool.IsActive() {
				score := pool.GetLoadScore()
				if score < bestScore {
					bestScore = score
					bestPool = pool
				}
			}
		}

		return bestPool

	case "priority":
		// Select pool based on task priority
		for _, pool := range lb.pools {
			if pool.IsActive() && pool.GetQueueLength() < cap(pool.taskQueue) {
				return pool
			}
		}
		return nil

	default:
		return lb.pools[0]
	}
}

// Task Scheduler methods

// SubmitTask submits a task to the scheduler
func (ts *TaskScheduler) SubmitTask(task *ParallelTask) {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	// Check if task has dependencies
	if len(task.Dependencies) == 0 {
		// No dependencies, can schedule immediately
		select {
		case ts.readyQueue <- task:
			// Successfully queued
		default:
			// Queue full, add to pending
			ts.pendingTasks = append(ts.pendingTasks, task)
		}
	} else {
		// Has dependencies, add to waiting
		ts.waitingTasks[task.ID] = task
	}
}

// ScheduleTasks schedules ready tasks to worker pools
func (ts *TaskScheduler) ScheduleTasks(loadBalancer *LoadBalancer) {
	// Process ready tasks
	for {
		select {
		case task := <-ts.readyQueue:
			pool := loadBalancer.SelectPool(task)
			if pool != nil {
				select {
				case pool.taskQueue <- task:
					// Successfully scheduled
				default:
					// Pool queue full, put back in ready queue
					select {
					case ts.readyQueue <- task:
						return // Try again later
					default:
						// Ready queue full too, add to pending
						ts.mu.Lock()
						ts.pendingTasks = append(ts.pendingTasks, task)
						ts.mu.Unlock()
						return
					}
				}
			}
		default:
			// No more ready tasks
			break
		}
	}

	// Move pending tasks to ready queue if there's space
	ts.mu.Lock()
	var remainingPending []*ParallelTask

	for _, task := range ts.pendingTasks {
		select {
		case ts.readyQueue <- task:
			// Successfully queued
		default:
			// Queue still full
			remainingPending = append(remainingPending, task)
		}
	}

	ts.pendingTasks = remainingPending
	ts.mu.Unlock()

	// Check for tasks that can be unblocked
	ts.checkDependencies()
}

// checkDependencies checks if any waiting tasks can be unblocked
func (ts *TaskScheduler) checkDependencies() {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	var stillWaiting = make(map[string]*ParallelTask)

	for taskID, task := range ts.waitingTasks {
		allDepsCompleted := true

		for _, depID := range task.Dependencies {
			if !ts.completed[depID] {
				allDepsCompleted = false
				break
			}
		}

		if allDepsCompleted {
			// All dependencies completed, move to ready queue
			select {
			case ts.readyQueue <- task:
				// Successfully queued
			default:
				// Queue full, keep waiting
				stillWaiting[taskID] = task
			}
		} else {
			stillWaiting[taskID] = task
		}
	}

	ts.waitingTasks = stillWaiting
}

// MarkCompleted marks a task as completed
func (ts *TaskScheduler) MarkCompleted(taskID string) {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	ts.completed[taskID] = true
}

// GetPendingTaskCount returns the number of pending tasks
func (ts *TaskScheduler) GetPendingTaskCount() int {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	return len(ts.pendingTasks) + len(ts.waitingTasks) + len(ts.readyQueue)
}

// Utility methods for ParallelProcessor

func (pp *ParallelProcessor) processPoolResults(pool *WorkerPool) {
	defer pp.wg.Done()

	for {
		select {
		case <-pp.stopChan:
			return
		case result, ok := <-pool.resultQueue:
			if !ok {
				return
			}

			pp.handleParallelResult(result)
		}
	}
}

func (pp *ParallelProcessor) handleParallelResult(result *ParallelResult) {
	// Update statistics
	pp.stats.mu.Lock()
	if result.Success {
		pp.stats.CompletedTasks++
	} else {
		pp.stats.FailedTasks++
	}

	// Update peak memory
	if result.MemoryUsed > pp.stats.PeakMemory {
		pp.stats.PeakMemory = result.MemoryUsed
	}

	pp.stats.mu.Unlock()

	// Mark task as completed in scheduler
	pp.scheduler.MarkCompleted(result.Task.ID)
}

func (pp *ParallelProcessor) monitorStatistics(ctx context.Context) {
	defer pp.wg.Done()

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	var lastCompleted int64
	lastUpdate := time.Now()

	for {
		select {
		case <-ctx.Done():
			return
		case <-pp.stopChan:
			return
		case <-ticker.C:
			pp.stats.mu.Lock()

			now := time.Now()
			timeDelta := now.Sub(lastUpdate).Seconds()

			// Calculate throughput
			completedDelta := pp.stats.CompletedTasks - lastCompleted
			tps := float64(completedDelta) / timeDelta
			pp.stats.ThroughputTPS = tps

			// Update average time
			totalCompleted := pp.stats.CompletedTasks + pp.stats.FailedTasks
			if totalCompleted > 0 {
				totalTime := time.Since(pp.stats.StartTime)
				pp.stats.AverageTime = totalTime / time.Duration(totalCompleted)
			}

			// Count active pools and workers
			pp.stats.ActivePools = 0
			pp.stats.ActiveWorkers = 0
			for _, pool := range pp.workerPools {
				if pool.IsActive() {
					pp.stats.ActivePools++
					for _, worker := range pool.workers {
						if atomic.LoadInt32(&worker.active) == 1 {
							pp.stats.ActiveWorkers++
						}
					}
				}
			}

			pp.stats.mu.Unlock()

			lastCompleted = pp.stats.CompletedTasks
			lastUpdate = now
		}
	}
}

// Utility methods

func (pp *ParallelProcessor) IsRunning() bool {
	return atomic.LoadInt32(&pp.running) == 1
}

func (pp *ParallelProcessor) generateTaskID(filePath string) string {
	return fmt.Sprintf("task_%s_%d", pp.calculateHash(filePath), time.Now().UnixNano())
}

func (pp *ParallelProcessor) calculateHash(content string) string {
	return fmt.Sprintf("%x", md5.Sum([]byte(content)))
}

func (pp *ParallelProcessor) calculateTaskPriority(filePath string) int {
	// Same logic as batch processor
	ext := strings.ToLower(filepath.Ext(filePath))

	priorityMap := map[string]int{
		".go":   100,
		".py":   90,
		".js":   80,
		".ts":   80,
		".java": 70,
	}

	if priority, exists := priorityMap[ext]; exists {
		return priority
	}

	return 50
}

func (pp *ParallelProcessor) estimateProcessingTime(filePath string) time.Duration {
	// Simple estimation based on file extension and size
	// In practice, this would use historical data
	return time.Millisecond * 100
}

func (pp *ParallelProcessor) estimateMemoryNeeded(filePath string) int64 {
	// Simple estimation
	return 1024 * 1024 // 1MB
}

func (pp *ParallelProcessor) getCurrentMemoryUsage() float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Return as percentage of total system memory
	// This is simplified - in practice you'd use system-specific calls
	return float64(m.Alloc) / (1024 * 1024 * 1024) // Normalize to GB
}

func (pp *ParallelProcessor) getCurrentCPUUsage() float64 {
	// This would require platform-specific code to get actual CPU usage
	// For now, return a placeholder
	return 0.5
}

func (pp *ParallelProcessor) getAverageQueueLength() float64 {
	if len(pp.workerPools) == 0 {
		return 0
	}

	total := 0
	for _, pool := range pp.workerPools {
		total += pool.GetQueueLength()
	}

	return float64(total) / float64(len(pp.workerPools))
}

// Getters

func (pp *ParallelProcessor) GetStatistics() *ParallelStatistics {
	pp.stats.mu.RLock()
	defer pp.stats.mu.RUnlock()

	stats := *pp.stats
	return &stats
}

func (pp *ParallelProcessor) GetPoolStatistics() []*PoolStatistics {
	var stats []*PoolStatistics

	for _, pool := range pp.workerPools {
		pool.stats.mu.RLock()
		poolStats := *pool.stats
		pool.stats.mu.RUnlock()
		stats = append(stats, &poolStats)
	}

	return stats
}

func (pp *ParallelProcessor) GetWorkerStatistics() []*WorkerStatistics {
	var stats []*WorkerStatistics

	for _, pool := range pp.workerPools {
		for _, worker := range pool.workers {
			worker.stats.mu.RLock()
			workerStats := *worker.stats
			worker.stats.mu.RUnlock()
			stats = append(stats, &workerStats)
		}
	}

	return stats
}
