package indexer

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/cache"
	"github.com/yourusername/ai-code-assistant/storage"
)

// UltraFastIndexer is the high-performance indexing engine
type UltraFastIndexer struct {
	// Core components
	codeParser         *CodeParser
	changeDetector     *ChangeDetector
	batchProcessor     *BatchProcessor
	parallelProcessor  *ParallelProcessor
	incrementalUpdater *IncrementalUpdater

	// Storage and caching
	db                *storage.SQLiteDB
	memoryCache       *cache.LRUCache
	bloomFilter       *BloomFilter
	hotDataCache      *HotDataCache
	compressionEngine *CompressionEngine

	// Configuration
	config *UltraFastConfig

	// State management
	stats           *UltraFastStatistics
	indexState      *IndexState
	snapshotManager *SnapshotManager

	// Concurrency control
	running         int32
	indexLock       sync.RWMutex
	updateQueue     chan *UpdateRequest
	compactionQueue chan *CompactionTask

	// Background workers
	backgroundWorkers sync.WaitGroup
	stopChan          chan struct{}

	// Performance optimization
	memoryPool     *MemoryPool
	stringInterner *StringInterner
	fastHasher     *FastHasher
}

// UltraFastConfig contains configuration for ultra-fast indexing
type UltraFastConfig struct {
	// Performance settings
	MaxMemoryUsage   int64 `json:"max_memory_usage"`  // Maximum memory usage in bytes
	CacheSize        int   `json:"cache_size"`        // LRU cache size
	BloomFilterSize  int   `json:"bloom_filter_size"` // Bloom filter size
	CompressionLevel int   `json:"compression_level"` // Compression level (1-9)

	// Indexing strategy
	IndexingStrategy string `json:"indexing_strategy"` // aggressive, balanced, conservative
	UpdateStrategy   string `json:"update_strategy"`   // immediate, batched, background
	CachingStrategy  string `json:"caching_strategy"`  // aggressive, normal, minimal

	// Background processing
	CompactionInterval time.Duration `json:"compaction_interval"` // How often to compact data
	SnapshotInterval   time.Duration `json:"snapshot_interval"`   // How often to create snapshots
	BackgroundWorkers  int           `json:"background_workers"`  // Number of background workers

	// Optimization flags
	EnableCompression  bool `json:"enable_compression"`   // Enable data compression
	EnableMMap         bool `json:"enable_mmap"`          // Enable memory-mapped files
	EnablePrefetch     bool `json:"enable_prefetch"`      // Enable data prefetching
	EnableStringIntern bool `json:"enable_string_intern"` // Enable string interning

	// Queue sizes
	UpdateQueueSize     int `json:"update_queue_size"`     // Update queue size
	CompactionQueueSize int `json:"compaction_queue_size"` // Compaction queue size

	// Timeouts
	IndexTimeout  time.Duration `json:"index_timeout"`  // Timeout for indexing operations
	UpdateTimeout time.Duration `json:"update_timeout"` // Timeout for update operations
}

// UpdateRequest represents a request to update the index
type UpdateRequest struct {
	ID          string                 `json:"id"`
	Type        UpdateType             `json:"type"`
	FilePath    string                 `json:"file_path"`
	Changes     []*FileChange          `json:"changes"`
	Priority    int                    `json:"priority"`
	Metadata    map[string]interface{} `json:"metadata"`
	RequestedAt time.Time              `json:"requested_at"`
	Context     context.Context        `json:"-"`
}

// UpdateType represents the type of update
type UpdateType string

const (
	UpdateTypeAdd      UpdateType = "add"
	UpdateTypeModify   UpdateType = "modify"
	UpdateTypeDelete   UpdateType = "delete"
	UpdateTypeBulk     UpdateType = "bulk"
	UpdateTypeSnapshot UpdateType = "snapshot"
)

// IndexState represents the current state of the index
type IndexState struct {
	Version          int64                 `json:"version"`
	FileCount        int64                 `json:"file_count"`
	ChunkCount       int64                 `json:"chunk_count"`
	TotalSize        int64                 `json:"total_size"`
	LastUpdated      time.Time             `json:"last_updated"`
	LastSnapshot     time.Time             `json:"last_snapshot"`
	ActiveFiles      map[string]*FileEntry `json:"active_files"`
	IndexedLanguages map[string]int64      `json:"indexed_languages"`
	mu               sync.RWMutex
}

// FileEntry represents an entry in the index
type FileEntry struct {
	FilePath       string                 `json:"file_path"`
	Hash           string                 `json:"hash"`
	Language       string                 `json:"language"`
	ChunkCount     int                    `json:"chunk_count"`
	Size           int64                  `json:"size"`
	IndexedAt      time.Time              `json:"indexed_at"`
	UpdatedAt      time.Time              `json:"updated_at"`
	Dependencies   []string               `json:"dependencies"`
	Metadata       map[string]interface{} `json:"metadata"`
	CacheKey       string                 `json:"cache_key"`
	CompressedSize int64                  `json:"compressed_size"`
}

// UltraFastStatistics tracks ultra-fast indexer statistics
type UltraFastStatistics struct {
	// Indexing performance
	IndexingSpeed  float64       `json:"indexing_speed"`  // Files per second
	ThroughputMBPS float64       `json:"throughput_mbps"` // Megabytes per second
	AverageLatency time.Duration `json:"average_latency"` // Average operation latency

	// Cache performance
	CacheHitRate      float64 `json:"cache_hit_rate"`      // Cache hit rate (0.0-1.0)
	CacheMissRate     float64 `json:"cache_miss_rate"`     // Cache miss rate (0.0-1.0)
	BloomFilterHits   int64   `json:"bloom_filter_hits"`   // Bloom filter true positives
	BloomFilterMisses int64   `json:"bloom_filter_misses"` // Bloom filter true negatives

	// Memory usage
	MemoryUsage      int64 `json:"memory_usage"`       // Current memory usage
	PeakMemoryUsage  int64 `json:"peak_memory_usage"`  // Peak memory usage
	CacheMemoryUsage int64 `json:"cache_memory_usage"` // Memory used by caches

	// Operations
	IndexOperations  int64 `json:"index_operations"`  // Total index operations
	UpdateOperations int64 `json:"update_operations"` // Total update operations
	CompactionOps    int64 `json:"compaction_ops"`    // Total compaction operations
	SnapshotOps      int64 `json:"snapshot_ops"`      // Total snapshot operations

	// Background processing
	QueueLength     int   `json:"queue_length"`     // Current update queue length
	CompactionQueue int   `json:"compaction_queue"` // Current compaction queue length
	BackgroundTasks int64 `json:"background_tasks"` // Active background tasks

	// Compression
	CompressionRatio  float64 `json:"compression_ratio"`  // Average compression ratio
	CompressedBytes   int64   `json:"compressed_bytes"`   // Total compressed bytes
	UncompressedBytes int64   `json:"uncompressed_bytes"` // Total uncompressed bytes

	// Timing
	StartTime      time.Time `json:"start_time"`
	LastUpdateTime time.Time `json:"last_update_time"`
	UptimeSeconds  float64   `json:"uptime_seconds"`

	mu sync.RWMutex
}

// BloomFilter provides fast membership testing
type BloomFilter struct {
	bits      []uint64
	size      uint64
	hashFuncs int
	count     uint64
	mu        sync.RWMutex
}

// HotDataCache caches frequently accessed data in memory
type HotDataCache struct {
	hotFiles    map[string]*CachedFileData
	coldFiles   map[string]*CachedFileData
	accessCount map[string]int64
	maxHotFiles int
	mu          sync.RWMutex
}

// CachedFileData represents cached file data
type CachedFileData struct {
	FilePath       string       `json:"file_path"`
	ParseResult    *ParseResult `json:"parse_result"`
	Chunks         []*CodeChunk `json:"chunks"`
	Hash           string       `json:"hash"`
	AccessCount    int64        `json:"access_count"`
	CachedAt       time.Time    `json:"cached_at"`
	LastAccess     time.Time    `json:"last_access"`
	Compressed     bool         `json:"compressed"`
	CompressedData []byte       `json:"compressed_data,omitempty"`
}

// MemoryPool manages reusable memory allocations
type MemoryPool struct {
	pools map[int]*sync.Pool
	mu    sync.RWMutex
}

// StringInterner interns strings to reduce memory usage
type StringInterner struct {
	strings map[string]string
	mu      sync.RWMutex
}

// FastHasher provides fast hashing for hot paths
type FastHasher struct {
	hashPool sync.Pool
}

// CompactionTask represents a background compaction task
type CompactionTask struct {
	ID        string          `json:"id"`
	Type      string          `json:"type"` // incremental, full, hot_data
	Priority  int             `json:"priority"`
	FilePaths []string        `json:"file_paths"`
	CreatedAt time.Time       `json:"created_at"`
	Context   context.Context `json:"-"`
}

// NewUltraFastIndexer creates a new ultra-fast indexer
func NewUltraFastIndexer(
	codeParser *CodeParser,
	changeDetector *ChangeDetector,
	batchProcessor *BatchProcessor,
	parallelProcessor *ParallelProcessor,
	db *storage.SQLiteDB,
	config *UltraFastConfig,
) *UltraFastIndexer {
	if config == nil {
		config = &UltraFastConfig{
			MaxMemoryUsage:      2 * 1024 * 1024 * 1024, // 2GB
			CacheSize:           10000,
			BloomFilterSize:     1000000,
			CompressionLevel:    6,
			IndexingStrategy:    "aggressive",
			UpdateStrategy:      "immediate",
			CachingStrategy:     "aggressive",
			CompactionInterval:  time.Minute * 15,
			SnapshotInterval:    time.Hour,
			BackgroundWorkers:   runtime.NumCPU(),
			EnableCompression:   true,
			EnableMMap:          true,
			EnablePrefetch:      true,
			EnableStringIntern:  true,
			UpdateQueueSize:     10000,
			CompactionQueueSize: 1000,
			IndexTimeout:        time.Minute * 5,
			UpdateTimeout:       time.Second * 30,
		}
	}

	ufi := &UltraFastIndexer{
		codeParser:        codeParser,
		changeDetector:    changeDetector,
		batchProcessor:    batchProcessor,
		parallelProcessor: parallelProcessor,
		db:                db,
		config:            config,
		stats: &UltraFastStatistics{
			StartTime: time.Now(),
		},
		indexState: &IndexState{
			ActiveFiles:      make(map[string]*FileEntry),
			IndexedLanguages: make(map[string]int64),
		},
		updateQueue:     make(chan *UpdateRequest, config.UpdateQueueSize),
		compactionQueue: make(chan *CompactionTask, config.CompactionQueueSize),
		stopChan:        make(chan struct{}),
	}

	// Initialize components
	ufi.initializeComponents()

	return ufi
}

// initializeComponents initializes all sub-components
func (ufi *UltraFastIndexer) initializeComponents() {
	// Initialize memory cache
	ufi.memoryCache = cache.NewLRUCache(ufi.config.CacheSize)

	// Initialize bloom filter
	ufi.bloomFilter = NewBloomFilter(ufi.config.BloomFilterSize, 3)

	// Initialize hot data cache
	ufi.hotDataCache = &HotDataCache{
		hotFiles:    make(map[string]*CachedFileData),
		coldFiles:   make(map[string]*CachedFileData),
		accessCount: make(map[string]int64),
		maxHotFiles: ufi.config.CacheSize / 10, // 10% for hot data
	}

	// Initialize compression engine
	if ufi.config.EnableCompression {
		ufi.compressionEngine = NewCompressionEngine(ufi.config.CompressionLevel)
	}

	// Initialize memory pool
	ufi.memoryPool = NewMemoryPool()

	// Initialize string interner
	if ufi.config.EnableStringIntern {
		ufi.stringInterner = &StringInterner{
			strings: make(map[string]string),
		}
	}

	// Initialize fast hasher
	ufi.fastHasher = &FastHasher{
		hashPool: sync.Pool{
			New: func() interface{} {
				return &FastHash{}
			},
		},
	}

	// Initialize incremental updater
	ufi.incrementalUpdater = NewIncrementalUpdater(ufi, ufi.config)

	// Initialize snapshot manager
	ufi.snapshotManager = NewSnapshotManager(ufi, ufi.config)
}

// Start starts the ultra-fast indexer
func (ufi *UltraFastIndexer) Start(ctx context.Context) error {
	if !atomic.CompareAndSwapInt32(&ufi.running, 0, 1) {
		return fmt.Errorf("ultra-fast indexer is already running")
	}

	// Load existing index state
	if err := ufi.loadIndexState(); err != nil {
		return fmt.Errorf("failed to load index state: %v", err)
	}

	// Start background workers
	for i := 0; i < ufi.config.BackgroundWorkers; i++ {
		ufi.backgroundWorkers.Add(1)
		go ufi.runUpdateWorker(ctx, i)
	}

	// Start compaction worker
	ufi.backgroundWorkers.Add(1)
	go ufi.runCompactionWorker(ctx)

	// Start snapshot worker
	ufi.backgroundWorkers.Add(1)
	go ufi.runSnapshotWorker(ctx)

	// Start statistics monitor
	ufi.backgroundWorkers.Add(1)
	go ufi.monitorStatistics(ctx)

	// Start incremental updater
	if err := ufi.incrementalUpdater.Start(ctx); err != nil {
		return fmt.Errorf("failed to start incremental updater: %v", err)
	}

	fmt.Println("Ultra-fast indexer started successfully")
	return nil
}

// Stop stops the ultra-fast indexer
func (ufi *UltraFastIndexer) Stop() error {
	if !atomic.CompareAndSwapInt32(&ufi.running, 1, 0) {
		return nil
	}

	// Signal stop
	close(ufi.stopChan)

	// Stop incremental updater
	ufi.incrementalUpdater.Stop()

	// Wait for workers to finish
	ufi.backgroundWorkers.Wait()

	// Save final index state
	if err := ufi.saveIndexState(); err != nil {
		fmt.Printf("Warning: failed to save index state: %v\n", err)
	}

	fmt.Println("Ultra-fast indexer stopped")
	return nil
}

// IndexFiles indexes a list of files with ultra-fast performance
func (ufi *UltraFastIndexer) IndexFiles(ctx context.Context, filePaths []string) error {
	if !ufi.IsRunning() {
		return fmt.Errorf("ultra-fast indexer is not running")
	}

	start := time.Now()

	// Pre-filter files using bloom filter
	filteredPaths := ufi.preFilterFiles(filePaths)

	// Check cache for already processed files
	newFiles, cachedFiles := ufi.partitionFilesFromCache(filteredPaths)

	fmt.Printf("Indexing %d files (%d new, %d from cache)\n",
		len(filePaths), len(newFiles), len(cachedFiles))

	// Process cached files immediately
	ufi.processCachedFiles(cachedFiles)

	// Process new files with parallel processing
	if len(newFiles) > 0 {
		switch ufi.config.IndexingStrategy {
		case "aggressive":
			err := ufi.indexFilesAggressive(ctx, newFiles)
			if err != nil {
				return err
			}
		case "balanced":
			err := ufi.indexFilesBalanced(ctx, newFiles)
			if err != nil {
				return err
			}
		case "conservative":
			err := ufi.indexFilesConservative(ctx, newFiles)
			if err != nil {
				return err
			}
		default:
			return fmt.Errorf("unknown indexing strategy: %s", ufi.config.IndexingStrategy)
		}
	}

	// Update statistics
	ufi.updateIndexingStats(len(filePaths), time.Since(start))

	return nil
}

// indexFilesAggressive uses maximum resources for fastest indexing
func (ufi *UltraFastIndexer) indexFilesAggressive(ctx context.Context, filePaths []string) error {
	// Use all available CPU cores and memory
	return ufi.parallelProcessor.ProcessFilesParallel(ctx, filePaths)
}

// indexFilesBalanced balances speed with resource usage
func (ufi *UltraFastIndexer) indexFilesBalanced(ctx context.Context, filePaths []string) error {
	// Use batch processing with parallel workers
	return ufi.batchProcessor.ProcessFiles(ctx, filePaths)
}

// indexFilesConservative uses minimal resources for stable performance
func (ufi *UltraFastIndexer) indexFilesConservative(ctx context.Context, filePaths []string) error {
	// Process files sequentially in small batches
	batchSize := 10
	for i := 0; i < len(filePaths); i += batchSize {
		end := i + batchSize
		if end > len(filePaths) {
			end = len(filePaths)
		}

		batch := filePaths[i:end]
		if err := ufi.batchProcessor.ProcessFiles(ctx, batch); err != nil {
			return err
		}
	}

	return nil
}

// UpdateFile updates a single file in the index
func (ufi *UltraFastIndexer) UpdateFile(ctx context.Context, filePath string) error {
	if !ufi.IsRunning() {
		return fmt.Errorf("ultra-fast indexer is not running")
	}

	request := &UpdateRequest{
		ID:          ufi.generateRequestID(),
		Type:        UpdateTypeModify,
		FilePath:    filePath,
		Priority:    100,
		RequestedAt: time.Now(),
		Context:     ctx,
	}

	return ufi.submitUpdateRequest(request)
}

// DeleteFile removes a file from the index
func (ufi *UltraFastIndexer) DeleteFile(ctx context.Context, filePath string) error {
	if !ufi.IsRunning() {
		return fmt.Errorf("ultra-fast indexer is not running")
	}

	request := &UpdateRequest{
		ID:          ufi.generateRequestID(),
		Type:        UpdateTypeDelete,
		FilePath:    filePath,
		Priority:    80,
		RequestedAt: time.Now(),
		Context:     ctx,
	}

	return ufi.submitUpdateRequest(request)
}

// Background workers

// runUpdateWorker runs a background update worker
func (ufi *UltraFastIndexer) runUpdateWorker(ctx context.Context, workerID int) {
	defer ufi.backgroundWorkers.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ufi.stopChan:
			return
		case request := <-ufi.updateQueue:
			ufi.processUpdateRequest(request, workerID)
		}
	}
}

// processUpdateRequest processes an update request
func (ufi *UltraFastIndexer) processUpdateRequest(request *UpdateRequest, workerID int) {
	start := time.Now()

	switch request.Type {
	case UpdateTypeAdd, UpdateTypeModify:
		err := ufi.processFileUpdate(request)
		if err != nil {
			fmt.Printf("Worker %d: failed to update file %s: %v\n", workerID, request.FilePath, err)
		}

	case UpdateTypeDelete:
		err := ufi.processFileDelete(request)
		if err != nil {
			fmt.Printf("Worker %d: failed to delete file %s: %v\n", workerID, request.FilePath, err)
		}

	case UpdateTypeBulk:
		err := ufi.processBulkUpdate(request)
		if err != nil {
			fmt.Printf("Worker %d: failed to process bulk update: %v\n", workerID, err)
		}
	}

	// Update statistics
	ufi.stats.mu.Lock()
	ufi.stats.UpdateOperations++
	ufi.stats.AverageLatency = (ufi.stats.AverageLatency + time.Since(start)) / 2
	ufi.stats.mu.Unlock()
}

// processFileUpdate processes a file update
func (ufi *UltraFastIndexer) processFileUpdate(request *UpdateRequest) error {
	// Parse the file
	parseResult, err := ufi.codeParser.ParseFile(request.FilePath)
	if err != nil {
		return fmt.Errorf("failed to parse file: %v", err)
	}

	// Update bloom filter
	ufi.bloomFilter.Add(request.FilePath)

	// Cache the results
	ufi.cacheFileData(request.FilePath, parseResult)

	// Update index state
	ufi.updateIndexState(request.FilePath, parseResult)

	return nil
}

// processFileDelete processes a file deletion
func (ufi *UltraFastIndexer) processFileDelete(request *UpdateRequest) error {
	// Remove from cache
	ufi.removeCachedFile(request.FilePath)

	// Update index state
	ufi.removeFromIndexState(request.FilePath)

	return nil
}

// processBulkUpdate processes a bulk update
func (ufi *UltraFastIndexer) processBulkUpdate(request *UpdateRequest) error {
	// Process each change in the bulk update
	for _, change := range request.Changes {
		subRequest := &UpdateRequest{
			ID:       ufi.generateRequestID(),
			FilePath: change.FilePath,
			Context:  request.Context,
		}

		switch change.ChangeType {
		case ChangeTypeAdded, ChangeTypeModified:
			subRequest.Type = UpdateTypeModify
		case ChangeTypeDeleted:
			subRequest.Type = UpdateTypeDelete
		default:
			continue
		}

		if err := ufi.processUpdateRequest(subRequest, -1); err != nil {
			return err
		}
	}

	return nil
}

// runCompactionWorker runs the background compaction worker
func (ufi *UltraFastIndexer) runCompactionWorker(ctx context.Context) {
	defer ufi.backgroundWorkers.Done()

	ticker := time.NewTicker(ufi.config.CompactionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ufi.stopChan:
			return
		case <-ticker.C:
			ufi.performScheduledCompaction()
		case task := <-ufi.compactionQueue:
			ufi.performCompaction(task)
		}
	}
}

// performScheduledCompaction performs scheduled compaction
func (ufi *UltraFastIndexer) performScheduledCompaction() {
	task := &CompactionTask{
		ID:        ufi.generateTaskID(),
		Type:      "incremental",
		Priority:  50,
		CreatedAt: time.Now(),
	}

	select {
	case ufi.compactionQueue <- task:
		// Successfully queued
	default:
		// Queue full, skip this compaction
	}
}

// performCompaction performs data compaction
func (ufi *UltraFastIndexer) performCompaction(task *CompactionTask) {
	start := time.Now()

	switch task.Type {
	case "incremental":
		ufi.performIncrementalCompaction()
	case "full":
		ufi.performFullCompaction()
	case "hot_data":
		ufi.performHotDataCompaction()
	}

	// Update statistics
	ufi.stats.mu.Lock()
	ufi.stats.CompactionOps++
	ufi.stats.mu.Unlock()

	fmt.Printf("Compaction %s completed in %v\n", task.Type, time.Since(start))
}

// Cache management methods

// preFilterFiles pre-filters files using bloom filter
func (ufi *UltraFastIndexer) preFilterFiles(filePaths []string) []string {
	var filtered []string

	for _, filePath := range filePaths {
		// Always include files that might be new
		if !ufi.bloomFilter.Contains(filePath) {
			filtered = append(filtered, filePath)
		} else {
			// Check if file actually exists in index
			if !ufi.isFileInIndex(filePath) {
				filtered = append(filtered, filePath)
			}
		}
	}

	return filtered
}

// partitionFilesFromCache partitions files into new and cached
func (ufi *UltraFastIndexer) partitionFilesFromCache(filePaths []string) ([]string, []string) {
	var newFiles, cachedFiles []string

	for _, filePath := range filePaths {
		if ufi.isFileCached(filePath) {
			cachedFiles = append(cachedFiles, filePath)
		} else {
			newFiles = append(newFiles, filePath)
		}
	}

	return newFiles, cachedFiles
}

// processCachedFiles processes files that are already cached
func (ufi *UltraFastIndexer) processCachedFiles(filePaths []string) {
	for _, filePath := range filePaths {
		// Update access count and move to hot cache if needed
		ufi.hotDataCache.mu.Lock()
		ufi.hotDataCache.accessCount[filePath]++
		ufi.hotDataCache.mu.Unlock()

		// Update cache hit statistics
		ufi.stats.mu.Lock()
		ufi.stats.CacheHitRate = (ufi.stats.CacheHitRate + 1.0) / 2.0
		ufi.stats.mu.Unlock()
	}
}

// cacheFileData caches file data
func (ufi *UltraFastIndexer) cacheFileData(filePath string, parseResult *ParseResult) {
	cachedData := &CachedFileData{
		FilePath:    filePath,
		ParseResult: parseResult,
		Chunks:      parseResult.Chunks,
		Hash:        parseResult.FileHash,
		AccessCount: 1,
		CachedAt:    time.Now(),
		LastAccess:  time.Now(),
	}

	// Compress data if enabled
	if ufi.config.EnableCompression {
		compressed, err := ufi.compressionEngine.Compress(parseResult)
		if err == nil {
			cachedData.Compressed = true
			cachedData.CompressedData = compressed
		}
	}

	// Add to cache
	ufi.memoryCache.Set(filePath, cachedData)

	// Add to hot data cache
	ufi.hotDataCache.mu.Lock()
	ufi.hotDataCache.hotFiles[filePath] = cachedData
	ufi.hotDataCache.mu.Unlock()
}

// Utility methods

func (ufi *UltraFastIndexer) IsRunning() bool {
	return atomic.LoadInt32(&ufi.running) == 1
}

func (ufi *UltraFastIndexer) generateRequestID() string {
	return fmt.Sprintf("req_%d", time.Now().UnixNano())
}

func (ufi *UltraFastIndexer) generateTaskID() string {
	return fmt.Sprintf("task_%d", time.Now().UnixNano())
}

func (ufi *UltraFastIndexer) submitUpdateRequest(request *UpdateRequest) error {
	select {
	case ufi.updateQueue <- request:
		return nil
	default:
		return fmt.Errorf("update queue is full")
	}
}

func (ufi *UltraFastIndexer) isFileInIndex(filePath string) bool {
	ufi.indexState.mu.RLock()
	defer ufi.indexState.mu.RUnlock()

	_, exists := ufi.indexState.ActiveFiles[filePath]
	return exists
}

func (ufi *UltraFastIndexer) isFileCached(filePath string) bool {
	_, exists := ufi.memoryCache.Get(filePath)
	return exists
}

// Bloom Filter implementation

// NewBloomFilter creates a new bloom filter
func NewBloomFilter(size int, hashFuncs int) *BloomFilter {
	return &BloomFilter{
		bits:      make([]uint64, (size+63)/64),
		size:      uint64(size),
		hashFuncs: hashFuncs,
	}
}

// Add adds an item to the bloom filter
func (bf *BloomFilter) Add(item string) {
	bf.mu.Lock()
	defer bf.mu.Unlock()

	hash := bf.hash(item)
	for i := 0; i < bf.hashFuncs; i++ {
		bit := (hash + uint64(i)*hash) % bf.size
		bf.bits[bit/64] |= 1 << (bit % 64)
	}
	bf.count++
}

// Contains checks if an item might be in the bloom filter
func (bf *BloomFilter) Contains(item string) bool {
	bf.mu.RLock()
	defer bf.mu.RUnlock()

	hash := bf.hash(item)
	for i := 0; i < bf.hashFuncs; i++ {
		bit := (hash + uint64(i)*hash) % bf.size
		if bf.bits[bit/64]&(1<<(bit%64)) == 0 {
			return false
		}
	}
	return true
}

// hash calculates hash for bloom filter
func (bf *BloomFilter) hash(item string) uint64 {
	// Simple FNV-1a hash
	hash := uint64(14695981039346656037)
	for _, b := range []byte(item) {
		hash ^= uint64(b)
		hash *= 1099511628211
	}
	return hash
}

// Statistics and monitoring

// updateIndexingStats updates indexing statistics
func (ufi *UltraFastIndexer) updateIndexingStats(fileCount int, duration time.Duration) {
	ufi.stats.mu.Lock()
	defer ufi.stats.mu.Unlock()

	ufi.stats.IndexOperations++
	seconds := duration.Seconds()
	if seconds > 0 {
		ufi.stats.IndexingSpeed = float64(fileCount) / seconds
	}
	ufi.stats.LastUpdateTime = time.Now()
}

// monitorStatistics monitors and updates statistics
func (ufi *UltraFastIndexer) monitorStatistics(ctx context.Context) {
	defer ufi.backgroundWorkers.Done()

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ufi.stopChan:
			return
		case <-ticker.C:
			ufi.updateRuntimeStatistics()
		}
	}
}

// updateRuntimeStatistics updates runtime statistics
func (ufi *UltraFastIndexer) updateRuntimeStatistics() {
	ufi.stats.mu.Lock()
	defer ufi.stats.mu.Unlock()

	// Update uptime
	ufi.stats.UptimeSeconds = time.Since(ufi.stats.StartTime).Seconds()

	// Update queue lengths
	ufi.stats.QueueLength = len(ufi.updateQueue)
	ufi.stats.CompactionQueue = len(ufi.compactionQueue)

	// Update memory usage
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	ufi.stats.MemoryUsage = int64(m.Alloc)
	if ufi.stats.MemoryUsage > ufi.stats.PeakMemoryUsage {
		ufi.stats.PeakMemoryUsage = ufi.stats.MemoryUsage
	}

	// Update cache memory usage
	ufi.stats.CacheMemoryUsage = ufi.memoryCache.Size()
}

// loadIndexState loads the index state from storage
func (ufi *UltraFastIndexer) loadIndexState() error {
	// This would load from database or file
	// For now, just initialize
	ufi.indexState.Version = 1
	ufi.indexState.LastUpdated = time.Now()
	return nil
}

// saveIndexState saves the index state to storage
func (ufi *UltraFastIndexer) saveIndexState() error {
	// This would save to database or file
	// For now, just log
	fmt.Println("Index state saved")
	return nil
}

// updateIndexState updates the index state
func (ufi *UltraFastIndexer) updateIndexState(filePath string, parseResult *ParseResult) {
	ufi.indexState.mu.Lock()
	defer ufi.indexState.mu.Unlock()

	entry := &FileEntry{
		FilePath:   filePath,
		Hash:       parseResult.FileHash,
		Language:   parseResult.Language,
		ChunkCount: len(parseResult.Chunks),
		Size:       parseResult.Size,
		IndexedAt:  parseResult.ParsedAt,
		UpdatedAt:  time.Now(),
		CacheKey:   ufi.generateCacheKey(filePath),
	}

	// Update or add file entry
	oldEntry := ufi.indexState.ActiveFiles[filePath]
	ufi.indexState.ActiveFiles[filePath] = entry

	// Update counters
	if oldEntry == nil {
		ufi.indexState.FileCount++
		ufi.indexState.ChunkCount += int64(entry.ChunkCount)
		ufi.indexState.TotalSize += entry.Size
	} else {
		ufi.indexState.ChunkCount += int64(entry.ChunkCount) - int64(oldEntry.ChunkCount)
		ufi.indexState.TotalSize += entry.Size - oldEntry.Size
	}

	// Update language count
	ufi.indexState.IndexedLanguages[parseResult.Language]++

	// Update version and timestamp
	ufi.indexState.Version++
	ufi.indexState.LastUpdated = time.Now()
}

// removeFromIndexState removes a file from index state
func (ufi *UltraFastIndexer) removeFromIndexState(filePath string) {
	ufi.indexState.mu.Lock()
	defer ufi.indexState.mu.Unlock()

	if entry, exists := ufi.indexState.ActiveFiles[filePath]; exists {
		delete(ufi.indexState.ActiveFiles, filePath)

		ufi.indexState.FileCount--
		ufi.indexState.ChunkCount -= int64(entry.ChunkCount)
		ufi.indexState.TotalSize -= entry.Size
		ufi.indexState.IndexedLanguages[entry.Language]--

		ufi.indexState.Version++
		ufi.indexState.LastUpdated = time.Now()
	}
}

// removeCachedFile removes a file from cache
func (ufi *UltraFastIndexer) removeCachedFile(filePath string) {
	ufi.memoryCache.Delete(filePath)

	ufi.hotDataCache.mu.Lock()
	delete(ufi.hotDataCache.hotFiles, filePath)
	delete(ufi.hotDataCache.coldFiles, filePath)
	delete(ufi.hotDataCache.accessCount, filePath)
	ufi.hotDataCache.mu.Unlock()
}

// generateCacheKey generates a cache key for a file
func (ufi *UltraFastIndexer) generateCacheKey(filePath string) string {
	return fmt.Sprintf("file_%s", ufi.fastHasher.Hash(filePath))
}

// Getters

func (ufi *UltraFastIndexer) GetStatistics() *UltraFastStatistics {
	ufi.stats.mu.RLock()
	defer ufi.stats.mu.RUnlock()

	stats := *ufi.stats
	return &stats
}

func (ufi *UltraFastIndexer) GetIndexState() *IndexState {
	ufi.indexState.mu.RLock()
	defer ufi.indexState.mu.RUnlock()

	// Return a copy
	state := &IndexState{
		Version:          ufi.indexState.Version,
		FileCount:        ufi.indexState.FileCount,
		ChunkCount:       ufi.indexState.ChunkCount,
		TotalSize:        ufi.indexState.TotalSize,
		LastUpdated:      ufi.indexState.LastUpdated,
		LastSnapshot:     ufi.indexState.LastSnapshot,
		ActiveFiles:      make(map[string]*FileEntry),
		IndexedLanguages: make(map[string]int64),
	}

	for k, v := range ufi.indexState.ActiveFiles {
		entry := *v
		state.ActiveFiles[k] = &entry
	}

	for k, v := range ufi.indexState.IndexedLanguages {
		state.IndexedLanguages[k] = v
	}

	return state
}
