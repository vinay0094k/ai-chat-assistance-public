package indexer

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// IncrementalUpdater handles intelligent incremental updates to the index
type IncrementalUpdater struct {
	// Core components
	indexer        *UltraFastIndexer
	graphBuilder   *GraphBuilder
	diffEngine     *DiffEngine
	changeAnalyzer *ChangeAnalyzer

	// Configuration
	config *IncrementalConfig

	// State management
	updateGraph    *UpdateGraph
	pendingUpdates map[string]*IncrementalUpdate
	updateHistory  *UpdateHistory

	// Concurrency control
	running        int32
	updateLock     sync.RWMutex
	dependencyLock sync.RWMutex

	// Queues and channels
	updateQueue     chan *IncrementalUpdate
	dependencyQueue chan *DependencyUpdate
	resultChan      chan *UpdateResult

	// Background processing
	workers  []*IncrementalWorker
	wg       sync.WaitGroup
	stopChan chan struct{}

	// Statistics
	stats *IncrementalStatistics
}

// IncrementalConfig contains configuration for incremental updates
type IncrementalConfig struct {
	// Update strategy
	UpdateStrategy   string `json:"update_strategy"`   // immediate, batched, smart
	DiffGranularity  string `json:"diff_granularity"`  // line, function, class, file
	PropagationDepth int    `json:"propagation_depth"` // Maximum dependency depth

	// Performance tuning
	WorkerCount int `json:"worker_count"` // Number of update workers
	BatchSize   int `json:"batch_size"`   // Updates per batch
	QueueSize   int `json:"queue_size"`   // Size of update queue

	// Timing
	BatchInterval      time.Duration `json:"batch_interval"`      // How often to process batches
	AnalysisTimeout    time.Duration `json:"analysis_timeout"`    // Timeout for change analysis
	PropagationTimeout time.Duration `json:"propagation_timeout"` // Timeout for dependency propagation

	// Features
	EnableSmartDiff      bool `json:"enable_smart_diff"`      // Enable intelligent diffing
	EnablePropagation    bool `json:"enable_propagation"`     // Enable dependency propagation
	EnableConflictRes    bool `json:"enable_conflict_res"`    // Enable conflict resolution
	EnableImpactAnalysis bool `json:"enable_impact_analysis"` // Enable impact analysis

	// Thresholds
	MaxDependencies   int           `json:"max_dependencies"`    // Maximum dependencies to track
	MaxUpdateSize     int64         `json:"max_update_size"`     // Maximum size of single update
	MinUpdateInterval time.Duration `json:"min_update_interval"` // Minimum interval between updates
}

// IncrementalUpdate represents an incremental update operation
type IncrementalUpdate struct {
	ID            string                 `json:"id"`
	Type          UpdateType             `json:"type"`
	FilePath      string                 `json:"file_path"`
	OldContent    string                 `json:"old_content,omitempty"`
	NewContent    string                 `json:"new_content,omitempty"`
	Changes       []*ChangeEntry         `json:"changes"`
	Dependencies  []string               `json:"dependencies"`
	AffectedFiles []string               `json:"affected_files"`
	Priority      int                    `json:"priority"`
	CreatedAt     time.Time              `json:"created_at"`
	ProcessedAt   time.Time              `json:"processed_at,omitempty"`
	Status        UpdateStatus           `json:"status"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	Context       context.Context        `json:"-"`
}

// ChangeEntry represents a specific change within a file
type ChangeEntry struct {
	Type     ChangeEntryType        `json:"type"`
	Location *ChangeLocation        `json:"location"`
	OldValue string                 `json:"old_value,omitempty"`
	NewValue string                 `json:"new_value"`
	Symbol   string                 `json:"symbol,omitempty"` // Function/class/variable name
	Scope    string                 `json:"scope,omitempty"`  // Scope where change occurred
	Language string                 `json:"language"`
	Impact   ChangeImpact           `json:"impact"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// ChangeEntryType represents the type of change
type ChangeEntryType string

const (
	ChangeEntryAdd       ChangeEntryType = "add"
	ChangeEntryModify    ChangeEntryType = "modify"
	ChangeEntryDelete    ChangeEntryType = "delete"
	ChangeEntryMove      ChangeEntryType = "move"
	ChangeEntryRename    ChangeEntryType = "rename"
	ChangeEntrySignature ChangeEntryType = "signature" // Function signature change
	ChangeEntryImport    ChangeEntryType = "import"    // Import/dependency change
)

// ChangeLocation represents the location of a change
type ChangeLocation struct {
	StartLine   int `json:"start_line"`
	EndLine     int `json:"end_line"`
	StartColumn int `json:"start_column"`
	EndColumn   int `json:"end_column"`
}

// ChangeImpact represents the impact level of a change
type ChangeImpact string

const (
	ImpactLow      ChangeImpact = "low"      // Local change, no dependencies affected
	ImpactMedium   ChangeImpact = "medium"   // Some dependencies may be affected
	ImpactHigh     ChangeImpact = "high"     // Many dependencies likely affected
	ImpactCritical ChangeImpact = "critical" // Breaking change, widespread impact
)

// UpdateStatus represents the status of an update
type UpdateStatus string

const (
	StatusPending    UpdateStatus = "pending"
	StatusProcessing UpdateStatus = "processing"
	StatusCompleted  UpdateStatus = "completed"
	StatusFailed     UpdateStatus = "failed"
	StatusConflicted UpdateStatus = "conflicted"
)

// DependencyUpdate represents an update triggered by dependency changes
type DependencyUpdate struct {
	ID           string    `json:"id"`
	SourceUpdate string    `json:"source_update"` // ID of update that triggered this
	FilePath     string    `json:"file_path"`
	Reason       string    `json:"reason"` // Why this update is needed
	Priority     int       `json:"priority"`
	CreatedAt    time.Time `json:"created_at"`
}

// UpdateResult represents the result of processing an update
type UpdateResult struct {
	Update              *IncrementalUpdate `json:"update"`
	Success             bool               `json:"success"`
	Error               error              `json:"error,omitempty"`
	ProcessTime         time.Duration      `json:"process_time"`
	ChangesApplied      int                `json:"changes_applied"`
	DependenciesUpdated int                `json:"dependencies_updated"`
	NewDependencies     []string           `json:"new_dependencies,omitempty"`
	CompletedAt         time.Time          `json:"completed_at"`
}

// UpdateGraph tracks dependencies between code entities
type UpdateGraph struct {
	nodes   map[string]*GraphNode
	edges   map[string][]*GraphEdge
	symbols map[string]*SymbolNode // Symbol name -> node
	files   map[string]*FileNode   // File path -> node
	mu      sync.RWMutex
}

// GraphNode represents a node in the update graph
type GraphNode struct {
	ID           string                 `json:"id"`
	Type         NodeType               `json:"type"`
	FilePath     string                 `json:"file_path"`
	Symbol       string                 `json:"symbol,omitempty"`
	Location     *ChangeLocation        `json:"location,omitempty"`
	Dependencies []*GraphNode           `json:"dependencies,omitempty"`
	Dependents   []*GraphNode           `json:"dependents,omitempty"`
	LastUpdated  time.Time              `json:"last_updated"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// NodeType represents the type of graph node
type NodeType string

const (
	NodeTypeFile     NodeType = "file"
	NodeTypeFunction NodeType = "function"
	NodeTypeClass    NodeType = "class"
	NodeTypeVariable NodeType = "variable"
	NodeTypeImport   NodeType = "import"
	NodeTypeModule   NodeType = "module"
)

// GraphEdge represents an edge in the update graph
type GraphEdge struct {
	ID        string                 `json:"id"`
	From      *GraphNode             `json:"from"`
	To        *GraphNode             `json:"to"`
	Type      EdgeType               `json:"type"`
	Weight    float64                `json:"weight"` // Strength of dependency
	CreatedAt time.Time              `json:"created_at"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// EdgeType represents the type of dependency edge
type EdgeType string

const (
	EdgeTypeImport    EdgeType = "import"
	EdgeTypeCall      EdgeType = "call"
	EdgeTypeInherit   EdgeType = "inherit"
	EdgeTypeCompose   EdgeType = "compose"
	EdgeTypeReference EdgeType = "reference"
	EdgeTypeDefine    EdgeType = "define"
)

// IncrementalStatistics tracks incremental update statistics
type IncrementalStatistics struct {
	TotalUpdates      int64 `json:"total_updates"`
	ProcessedUpdates  int64 `json:"processed_updates"`
	FailedUpdates     int64 `json:"failed_updates"`
	ConflictedUpdates int64 `json:"conflicted_updates"`

	// Performance metrics
	AverageUpdateTime time.Duration `json:"average_update_time"`
	UpdateThroughput  float64       `json:"update_throughput"` // Updates per second

	// Change analysis
	ChangesByType   map[ChangeEntryType]int64 `json:"changes_by_type"`
	ChangesByImpact map[ChangeImpact]int64    `json:"changes_by_impact"`

	// Dependency propagation
	DependencyUpdates int64   `json:"dependency_updates"`
	PropagationDepth  int     `json:"propagation_depth"`
	AverageFanout     float64 `json:"average_fanout"`

	// Graph statistics
	GraphNodes int64 `json:"graph_nodes"`
	GraphEdges int64 `json:"graph_edges"`

	mu sync.RWMutex
}

// NewIncrementalUpdater creates a new incremental updater
func NewIncrementalUpdater(indexer *UltraFastIndexer, config *IncrementalConfig) *IncrementalUpdater {
	if config == nil {
		config = &IncrementalConfig{
			UpdateStrategy:       "smart",
			DiffGranularity:      "function",
			PropagationDepth:     3,
			WorkerCount:          4,
			BatchSize:            10,
			QueueSize:            1000,
			BatchInterval:        time.Millisecond * 100,
			AnalysisTimeout:      time.Second * 10,
			PropagationTimeout:   time.Second * 30,
			EnableSmartDiff:      true,
			EnablePropagation:    true,
			EnableConflictRes:    true,
			EnableImpactAnalysis: true,
			MaxDependencies:      1000,
			MaxUpdateSize:        10 * 1024 * 1024, // 10MB
			MinUpdateInterval:    time.Millisecond * 50,
		}
	}

	iu := &IncrementalUpdater{
		indexer:         indexer,
		config:          config,
		pendingUpdates:  make(map[string]*IncrementalUpdate),
		updateQueue:     make(chan *IncrementalUpdate, config.QueueSize),
		dependencyQueue: make(chan *DependencyUpdate, config.QueueSize),
		resultChan:      make(chan *UpdateResult, config.QueueSize),
		stopChan:        make(chan struct{}),
		stats: &IncrementalStatistics{
			ChangesByType:   make(map[ChangeEntryType]int64),
			ChangesByImpact: make(map[ChangeImpact]int64),
		},
	}

	// Initialize components
	iu.updateGraph = NewUpdateGraph()
	iu.graphBuilder = NewGraphBuilder(iu.updateGraph, config)
	iu.diffEngine = NewDiffEngine(config)
	iu.changeAnalyzer = NewChangeAnalyzer(config)
	iu.updateHistory = NewUpdateHistory()

	return iu
}

// Start starts the incremental updater
func (iu *IncrementalUpdater) Start(ctx context.Context) error {
	if !atomic.CompareAndSwapInt32(&iu.running, 0, 1) {
		return fmt.Errorf("incremental updater is already running")
	}

	// Start workers
	iu.workers = make([]*IncrementalWorker, iu.config.WorkerCount)
	for i := 0; i < iu.config.WorkerCount; i++ {
		worker := &IncrementalWorker{
			ID:       i,
			updater:  iu,
			stopChan: iu.stopChan,
		}
		iu.workers[i] = worker

		iu.wg.Add(1)
		go worker.Run(ctx)
	}

	// Start dependency processor
	iu.wg.Add(1)
	go iu.processDependencyUpdates(ctx)

	// Start result processor
	iu.wg.Add(1)
	go iu.processResults(ctx)

	// Start batch processor if using batched strategy
	if iu.config.UpdateStrategy == "batched" {
		iu.wg.Add(1)
		go iu.processBatches(ctx)
	}

	fmt.Println("Incremental updater started")
	return nil
}

// Stop stops the incremental updater
func (iu *IncrementalUpdater) Stop() error {
	if !atomic.CompareAndSwapInt32(&iu.running, 1, 0) {
		return nil
	}

	close(iu.stopChan)
	iu.wg.Wait()

	fmt.Println("Incremental updater stopped")
	return nil
}

// ProcessFileChange processes a file change incrementally
func (iu *IncrementalUpdater) ProcessFileChange(ctx context.Context, filePath, oldContent, newContent string) error {
	if !iu.IsRunning() {
		return fmt.Errorf("incremental updater is not running")
	}

	// Analyze the changes
	changes, err := iu.diffEngine.AnalyzeChanges(filePath, oldContent, newContent)
	if err != nil {
		return fmt.Errorf("failed to analyze changes: %v", err)
	}

	if len(changes) == 0 {
		return nil // No changes detected
	}

	// Create incremental update
	update := &IncrementalUpdate{
		ID:         iu.generateUpdateID(),
		Type:       UpdateTypeModify,
		FilePath:   filePath,
		OldContent: oldContent,
		NewContent: newContent,
		Changes:    changes,
		Priority:   iu.calculateUpdatePriority(changes),
		CreatedAt:  time.Now(),
		Status:     StatusPending,
		Context:    ctx,
	}

	// Analyze impact if enabled
	if iu.config.EnableImpactAnalysis {
		impact, err := iu.changeAnalyzer.AnalyzeImpact(update)
		if err == nil {
			update.AffectedFiles = impact.AffectedFiles
			update.Dependencies = impact.Dependencies
		}
	}

	return iu.submitUpdate(update)
}

// ProcessBulkChanges processes multiple file changes in a coordinated way
func (iu *IncrementalUpdater) ProcessBulkChanges(ctx context.Context, changes []*FileChange) error {
	if !iu.IsRunning() {
		return fmt.Errorf("incremental updater is not running")
	}

	// Group changes by file
	fileChanges := make(map[string][]*FileChange)
	for _, change := range changes {
		fileChanges[change.FilePath] = append(fileChanges[change.FilePath], change)
	}

	// Process each file's changes
	for filePath, fileChangeList := range fileChanges {
		// For now, treat as a single update per file
		// In a real implementation, we'd merge changes intelligently

		var changeEntries []*ChangeEntry
		for _, fc := range fileChangeList {
			entry := &ChangeEntry{
				Type:     iu.mapChangeTypeToEntryType(fc.ChangeType),
				Language: fc.Language,
				Impact:   iu.assessChangeImpact(fc),
			}
			changeEntries = append(changeEntries, entry)
		}

		update := &IncrementalUpdate{
			ID:        iu.generateUpdateID(),
			Type:      UpdateTypeBulk,
			FilePath:  filePath,
			Changes:   changeEntries,
			Priority:  50, // Medium priority for bulk updates
			CreatedAt: time.Now(),
			Status:    StatusPending,
			Context:   ctx,
		}

		if err := iu.submitUpdate(update); err != nil {
			return err
		}
	}

	return nil
}

// submitUpdate submits an update for processing
func (iu *IncrementalUpdater) submitUpdate(update *IncrementalUpdate) error {
	// Check for conflicts with pending updates
	if iu.config.EnableConflictRes {
		if conflict := iu.checkForConflicts(update); conflict != nil {
			return iu.resolveConflict(update, conflict)
		}
	}

	// Add to pending updates
	iu.updateLock.Lock()
	iu.pendingUpdates[update.ID] = update
	iu.updateLock.Unlock()

	// Submit to appropriate queue based on strategy
	switch iu.config.UpdateStrategy {
	case "immediate":
		return iu.submitImmediateUpdate(update)
	case "batched":
		return iu.submitBatchedUpdate(update)
	case "smart":
		return iu.submitSmartUpdate(update)
	default:
		return fmt.Errorf("unknown update strategy: %s", iu.config.UpdateStrategy)
	}
}

// submitImmediateUpdate submits an update for immediate processing
func (iu *IncrementalUpdater) submitImmediateUpdate(update *IncrementalUpdate) error {
	select {
	case iu.updateQueue <- update:
		return nil
	default:
		return fmt.Errorf("update queue is full")
	}
}

// submitBatchedUpdate submits an update for batched processing
func (iu *IncrementalUpdater) submitBatchedUpdate(update *IncrementalUpdate) error {
	// For batched updates, we just add to pending and let the batch processor handle it
	return nil
}

// submitSmartUpdate submits an update using smart routing
func (iu *IncrementalUpdater) submitSmartUpdate(update *IncrementalUpdate) error {
	// Smart strategy: immediate for high-impact changes, batched for low-impact
	highImpact := false
	for _, change := range update.Changes {
		if change.Impact == ImpactHigh || change.Impact == ImpactCritical {
			highImpact = true
			break
		}
	}

	if highImpact {
		return iu.submitImmediateUpdate(update)
	} else {
		return iu.submitBatchedUpdate(update)
	}
}

// Worker implementation

// IncrementalWorker processes incremental updates
type IncrementalWorker struct {
	ID       int
	updater  *IncrementalUpdater
	stopChan <-chan struct{}
	stats    *WorkerStatistics
}

// Run runs the incremental worker
func (iw *IncrementalWorker) Run(ctx context.Context) {
	defer iw.updater.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-iw.stopChan:
			return
		case update := <-iw.updater.updateQueue:
			result := iw.processUpdate(update)

			select {
			case iw.updater.resultChan <- result:
				// Result submitted successfully
			default:
				// Result channel full
				fmt.Printf("Worker %d: result channel full\n", iw.ID)
			}
		}
	}
}

// processUpdate processes a single incremental update
func (iw *IncrementalWorker) processUpdate(update *IncrementalUpdate) *UpdateResult {
	start := time.Now()

	result := &UpdateResult{
		Update:      update,
		CompletedAt: time.Now(),
	}

	// Update status
	update.Status = StatusProcessing
	update.ProcessedAt = time.Now()

	// Apply changes based on type
	switch update.Type {
	case UpdateTypeModify:
		err := iw.processModifyUpdate(update)
		result.Success = err == nil
		result.Error = err

	case UpdateTypeDelete:
		err := iw.processDeleteUpdate(update)
		result.Success = err == nil
		result.Error = err

	case UpdateTypeBulk:
		err := iw.processBulkUpdate(update)
		result.Success = err == nil
		result.Error = err

	default:
		result.Success = false
		result.Error = fmt.Errorf("unknown update type: %s", update.Type)
	}

	// Calculate metrics
	result.ProcessTime = time.Since(start)
	result.ChangesApplied = len(update.Changes)

	// Process dependencies if enabled and update was successful
	if result.Success && iw.updater.config.EnablePropagation {
		dependencies := iw.processDependencies(update)
		result.DependenciesUpdated = len(dependencies)
		result.NewDependencies = dependencies
	}

	// Update final status
	if result.Success {
		update.Status = StatusCompleted
	} else {
		update.Status = StatusFailed
	}

	return result
}

// processModifyUpdate processes a file modification update
func (iw *IncrementalWorker) processModifyUpdate(update *IncrementalUpdate) error {
	// Parse the new content
	parseResult, err := iw.updater.indexer.codeParser.ParseFile(update.FilePath)
	if err != nil {
		return fmt.Errorf("failed to parse updated file: %v", err)
	}

	// Update the index with new chunks
	if err := iw.updateIndex(update.FilePath, parseResult); err != nil {
		return fmt.Errorf("failed to update index: %v", err)
	}

	// Update the dependency graph
	if err := iw.updater.graphBuilder.UpdateFileInGraph(update.FilePath, parseResult); err != nil {
		return fmt.Errorf("failed to update graph: %v", err)
	}

	return nil
}

// processDeleteUpdate processes a file deletion update
func (iw *IncrementalWorker) processDeleteUpdate(update *IncrementalUpdate) error {
	// Remove from index
	if err := iw.removeFromIndex(update.FilePath); err != nil {
		return fmt.Errorf("failed to remove from index: %v", err)
	}

	// Remove from dependency graph
	if err := iw.updater.graphBuilder.RemoveFileFromGraph(update.FilePath); err != nil {
		return fmt.Errorf("failed to remove from graph: %v", err)
	}

	return nil
}

// processBulkUpdate processes a bulk update
func (iw *IncrementalWorker) processBulkUpdate(update *IncrementalUpdate) error {
	// Process each change in the bulk update
	for _, change := range update.Changes {
		if err := iw.processChangeEntry(update.FilePath, change); err != nil {
			return fmt.Errorf("failed to process change %+v: %v", change, err)
		}
	}

	return nil
}

// processChangeEntry processes a single change entry
func (iw *IncrementalWorker) processChangeEntry(filePath string, change *ChangeEntry) error {
	switch change.Type {
	case ChangeEntryAdd:
		return iw.processAddChange(filePath, change)
	case ChangeEntryModify:
		return iw.processModifyChange(filePath, change)
	case ChangeEntryDelete:
		return iw.processDeleteChange(filePath, change)
	case ChangeEntrySignature:
		return iw.processSignatureChange(filePath, change)
	case ChangeEntryImport:
		return iw.processImportChange(filePath, change)
	default:
		return fmt.Errorf("unknown change entry type: %s", change.Type)
	}
}

// processDependencies processes dependency updates
func (iw *IncrementalWorker) processDependencies(update *IncrementalUpdate) []string {
	var newDependencies []string

	// Find all nodes that depend on this file
	dependents := iw.updater.updateGraph.GetDependents(update.FilePath)

	for _, dependent := range dependents {
		// Create dependency update
		depUpdate := &DependencyUpdate{
			ID:           iw.updater.generateDependencyUpdateID(),
			SourceUpdate: update.ID,
			FilePath:     dependent.FilePath,
			Reason:       fmt.Sprintf("depends on %s", update.FilePath),
			Priority:     update.Priority - 10, // Lower priority than source
			CreatedAt:    time.Now(),
		}

		// Submit dependency update
		select {
		case iw.updater.dependencyQueue <- depUpdate:
			newDependencies = append(newDependencies, dependent.FilePath)
		default:
			// Queue full, skip this dependency
		}
	}

	return newDependencies
}

// Batch processing

// processBatches processes updates in batches
func (iu *IncrementalUpdater) processBatches(ctx context.Context) {
	defer iu.wg.Done()

	ticker := time.NewTicker(iu.config.BatchInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-iu.stopChan:
			return
		case <-ticker.C:
			iu.processPendingBatch()
		}
	}
}

// processPendingBatch processes a batch of pending updates
func (iu *IncrementalUpdater) processPendingBatch() {
	iu.updateLock.Lock()

	// Collect pending updates
	var batch []*IncrementalUpdate
	count := 0
	for _, update := range iu.pendingUpdates {
		if update.Status == StatusPending && count < iu.config.BatchSize {
			batch = append(batch, update)
			count++
		}
	}

	iu.updateLock.Unlock()

	if len(batch) == 0 {
		return
	}

	// Sort batch by priority
	sort.Slice(batch, func(i, j int) bool {
		return batch[i].Priority > batch[j].Priority
	})

	// Process batch
	fmt.Printf("Processing batch of %d updates\n", len(batch))

	for _, update := range batch {
		select {
		case iu.updateQueue <- update:
			// Successfully queued
		default:
			// Queue full, will retry in next batch
			fmt.Printf("Queue full, skipping update %s\n", update.ID)
		}
	}
}

// Dependency processing

// processDependencyUpdates processes dependency updates
func (iu *IncrementalUpdater) processDependencyUpdates(ctx context.Context) {
	defer iu.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-iu.stopChan:
			return
		case depUpdate := <-iu.dependencyQueue:
			iu.processDependencyUpdate(depUpdate)
		}
	}
}

// processDependencyUpdate processes a single dependency update
func (iu *IncrementalUpdater) processDependencyUpdate(depUpdate *DependencyUpdate) {
	// Create a new incremental update for the dependent file
	update := &IncrementalUpdate{
		ID:        iu.generateUpdateID(),
		Type:      UpdateTypeModify,
		FilePath:  depUpdate.FilePath,
		Priority:  depUpdate.Priority,
		CreatedAt: time.Now(),
		Status:    StatusPending,
		Metadata: map[string]interface{}{
			"source_update":     depUpdate.SourceUpdate,
			"reason":            depUpdate.Reason,
			"dependency_update": true,
		},
	}

	// Submit for processing
	select {
	case iu.updateQueue <- update:
		iu.stats.mu.Lock()
		iu.stats.DependencyUpdates++
		iu.stats.mu.Unlock()
	default:
		fmt.Printf("Failed to queue dependency update for %s\n", depUpdate.FilePath)
	}
}

// Result processing

// processResults processes update results
func (iu *IncrementalUpdater) processResults(ctx context.Context) {
	defer iu.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-iu.stopChan:
			return
		case result := <-iu.resultChan:
			iu.handleUpdateResult(result)
		}
	}
}

// handleUpdateResult handles an update result
func (iu *IncrementalUpdater) handleUpdateResult(result *UpdateResult) {
	// Remove from pending updates
	iu.updateLock.Lock()
	delete(iu.pendingUpdates, result.Update.ID)
	iu.updateLock.Unlock()

	// Update statistics
	iu.updateStatistics(result)

	// Add to history
	iu.updateHistory.AddResult(result)

	// Log result
	if result.Success {
		fmt.Printf("Update %s completed successfully in %v\n",
			result.Update.ID, result.ProcessTime)
	} else {
		fmt.Printf("Update %s failed: %v\n",
			result.Update.ID, result.Error)
	}
}

// updateStatistics updates incremental statistics
func (iu *IncrementalUpdater) updateStatistics(result *UpdateResult) {
	iu.stats.mu.Lock()
	defer iu.stats.mu.Unlock()

	iu.stats.TotalUpdates++

	if result.Success {
		iu.stats.ProcessedUpdates++
	} else {
		iu.stats.FailedUpdates++
		if result.Update.Status == StatusConflicted {
			iu.stats.ConflictedUpdates++
		}
	}

	// Update timing statistics
	if iu.stats.AverageUpdateTime == 0 {
		iu.stats.AverageUpdateTime = result.ProcessTime
	} else {
		iu.stats.AverageUpdateTime = (iu.stats.AverageUpdateTime + result.ProcessTime) / 2
	}

	// Update change statistics
	for _, change := range result.Update.Changes {
		iu.stats.ChangesByType[change.Type]++
		iu.stats.ChangesByImpact[change.Impact]++
	}

	// Calculate throughput
	elapsed := time.Since(iu.stats.startTime)
	if elapsed.Seconds() > 0 {
		iu.stats.UpdateThroughput = float64(iu.stats.ProcessedUpdates) / elapsed.Seconds()
	}
}

// Utility methods

func (iu *IncrementalUpdater) IsRunning() bool {
	return atomic.LoadInt32(&iu.running) == 1
}

func (iu *IncrementalUpdater) generateUpdateID() string {
	return fmt.Sprintf("inc_%d", time.Now().UnixNano())
}

func (iu *IncrementalUpdater) generateDependencyUpdateID() string {
	return fmt.Sprintf("dep_%d", time.Now().UnixNano())
}

func (iu *IncrementalUpdater) calculateUpdatePriority(changes []*ChangeEntry) int {
	maxPriority := 0

	for _, change := range changes {
		priority := 50 // Base priority

		// Adjust based on impact
		switch change.Impact {
		case ImpactCritical:
			priority += 50
		case ImpactHigh:
			priority += 30
		case ImpactMedium:
			priority += 10
		}

		// Adjust based on change type
		switch change.Type {
		case ChangeEntrySignature:
			priority += 20
		case ChangeEntryImport:
			priority += 15
		case ChangeEntryAdd:
			priority += 10
		}

		if priority > maxPriority {
			maxPriority = priority
		}
	}

	return maxPriority
}

func (iu *IncrementalUpdater) mapChangeTypeToEntryType(changeType ChangeType) ChangeEntryType {
	switch changeType {
	case ChangeTypeAdded:
		return ChangeEntryAdd
	case ChangeTypeModified:
		return ChangeEntryModify
	case ChangeTypeDeleted:
		return ChangeEntryDelete
	case ChangeTypeRenamed:
		return ChangeEntryRename
	case ChangeTypeMoved:
		return ChangeEntryMove
	default:
		return ChangeEntryModify
	}
}

func (iu *IncrementalUpdater) assessChangeImpact(change *FileChange) ChangeImpact {
	// Simple heuristic based on file type and size
	if change.FileSize > 100000 { // Large files likely have high impact
		return ImpactHigh
	}

	// Check file extension for critical files
	if strings.HasSuffix(change.FilePath, ".go") ||
		strings.HasSuffix(change.FilePath, ".py") ||
		strings.HasSuffix(change.FilePath, ".js") {
		return ImpactMedium
	}

	return ImpactLow
}

// checkForConflicts checks for conflicts with pending updates
func (iu *IncrementalUpdater) checkForConflicts(update *IncrementalUpdate) *IncrementalUpdate {
	iu.updateLock.RLock()
	defer iu.updateLock.RUnlock()

	for _, pending := range iu.pendingUpdates {
		if pending.FilePath == update.FilePath && pending.Status == StatusPending {
			return pending
		}
	}

	return nil
}

// resolveConflict resolves conflicts between updates
func (iu *IncrementalUpdater) resolveConflict(newUpdate, conflictUpdate *IncrementalUpdate) error {
	// Simple conflict resolution: merge updates
	fmt.Printf("Resolving conflict between updates %s and %s\n",
		newUpdate.ID, conflictUpdate.ID)

	// Merge changes
	mergedChanges := append(conflictUpdate.Changes, newUpdate.Changes...)

	// Update the existing update
	conflictUpdate.Changes = mergedChanges
	conflictUpdate.Priority = max(conflictUpdate.Priority, newUpdate.Priority)

	return nil
}

// Helper methods for workers

func (iw *IncrementalWorker) updateIndex(filePath string, parseResult *ParseResult) error {
	// This would update the index with new parse results
	// For now, just update the cache
	iw.updater.indexer.hotDataCache.mu.Lock()
	defer iw.updater.indexer.hotDataCache.mu.Unlock()

	cachedData := &CachedFileData{
		FilePath:    filePath,
		ParseResult: parseResult,
		Chunks:      parseResult.Chunks,
		Hash:        parseResult.FileHash,
		CachedAt:    time.Now(),
		LastAccess:  time.Now(),
	}

	iw.updater.indexer.hotDataCache.hotFiles[filePath] = cachedData
	return nil
}

func (iw *IncrementalWorker) removeFromIndex(filePath string) error {
	// Remove from cache
	iw.updater.indexer.hotDataCache.mu.Lock()
	defer iw.updater.indexer.hotDataCache.mu.Unlock()

	delete(iw.updater.indexer.hotDataCache.hotFiles, filePath)
	delete(iw.updater.indexer.hotDataCache.coldFiles, filePath)

	return nil
}

func (iw *IncrementalWorker) processAddChange(filePath string, change *ChangeEntry) error {
	fmt.Printf("Processing add change in %s: %s\n", filePath, change.Symbol)
	return nil
}

func (iw *IncrementalWorker) processModifyChange(filePath string, change *ChangeEntry) error {
	fmt.Printf("Processing modify change in %s: %s\n", filePath, change.Symbol)
	return nil
}

func (iw *IncrementalWorker) processDeleteChange(filePath string, change *ChangeEntry) error {
	fmt.Printf("Processing delete change in %s: %s\n", filePath, change.Symbol)
	return nil
}

func (iw *IncrementalWorker) processSignatureChange(filePath string, change *ChangeEntry) error {
	fmt.Printf("Processing signature change in %s: %s\n", filePath, change.Symbol)
	return nil
}

func (iw *IncrementalWorker) processImportChange(filePath string, change *ChangeEntry) error {
	fmt.Printf("Processing import change in %s: %s\n", filePath, change.Symbol)
	return nil
}

// Getters

func (iu *IncrementalUpdater) GetStatistics() *IncrementalStatistics {
	iu.stats.mu.RLock()
	defer iu.stats.mu.RUnlock()

	stats := *iu.stats
	return &stats
}

func (iu *IncrementalUpdater) GetPendingUpdates() []*IncrementalUpdate {
	iu.updateLock.RLock()
	defer iu.updateLock.RUnlock()

	var updates []*IncrementalUpdate
	for _, update := range iu.pendingUpdates {
		updates = append(updates, update)
	}

	return updates
}

// Helper function
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
