package vectordb

import (
	"context"
	"crypto/md5"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/indexer"
	ai "github.com/yourusername/ai-code-assistant/internal/llm"
)

// EmbeddingsManager handles embedding generation and management
type EmbeddingsManager struct {
	// Core components
	qdrantClient *QdrantClient
	aiProvider   ai.Provider

	// Configuration
	config *EmbeddingsConfig

	// Caching and optimization
	embeddingCache  *EmbeddingCache
	batchProcessor  *EmbeddingBatchProcessor
	vectorOptimizer *VectorOptimizer

	// State management
	collections map[string]*EmbeddingCollection
	stats       *EmbeddingStatistics
	mu          sync.RWMutex

	// Background processing
	processQueue chan *EmbeddingTask
	resultQueue  chan *EmbeddingResult
	workers      []*EmbeddingWorker
	wg           sync.WaitGroup
	stopChan     chan struct{}
	running      bool
}

// EmbeddingsConfig contains embedding configuration
type EmbeddingsConfig struct {
	// Model settings
	ModelName  string `json:"model_name"`  // e.g., "all-MiniLM-L6-v2"
	VectorSize int    `json:"vector_size"` // Embedding dimension
	MaxTokens  int    `json:"max_tokens"`  // Maximum tokens per chunk

	// Processing settings
	BatchSize   int           `json:"batch_size"`   // Embeddings per batch
	WorkerCount int           `json:"worker_count"` // Number of workers
	QueueSize   int           `json:"queue_size"`   // Task queue size
	MaxRetries  int           `json:"max_retries"`  // Retry attempts
	RetryDelay  time.Duration `json:"retry_delay"`  // Delay between retries

	// Caching settings
	EnableCaching bool          `json:"enable_caching"` // Enable embedding cache
	CacheSize     int           `json:"cache_size"`     // Cache size
	CacheTTL      time.Duration `json:"cache_ttl"`      // Cache TTL

	// Optimization settings
	EnableNormalization bool `json:"enable_normalization"` // Normalize vectors
	EnableQuantization  bool `json:"enable_quantization"`  // Quantize vectors
	QuantizationBits    int  `json:"quantization_bits"`    // Bits for quantization

	// Collection settings
	Collections map[string]*CollectionConfig `json:"collections"`

	// Performance tuning
	Timeout       time.Duration `json:"timeout"`        // Request timeout
	RateLimit     int           `json:"rate_limit"`     // Requests per second
	EnableMetrics bool          `json:"enable_metrics"` // Enable detailed metrics
}

// CollectionConfig contains collection-specific configuration
type CollectionConfig struct {
	Name           string            `json:"name"`
	VectorSize     int               `json:"vector_size"`
	MetricType     string            `json:"metric_type"`
	ChunkStrategy  string            `json:"chunk_strategy"`  // function, class, file, smart
	PreprocessFunc string            `json:"preprocess_func"` // Function to preprocess text
	FilterRules    []string          `json:"filter_rules"`    // Rules to filter content
	Metadata       map[string]string `json:"metadata"`        // Additional metadata
}

// EmbeddingCollection represents a collection of embeddings
type EmbeddingCollection struct {
	Name        string                     `json:"name"`
	Config      *CollectionConfig          `json:"config"`
	VectorCount int64                      `json:"vector_count"`
	LastUpdated time.Time                  `json:"last_updated"`
	Statistics  *CollectionStatistics      `json:"statistics"`
	Index       map[string]*VectorMetadata `json:"index"` // Fast lookup by content hash
	mu          sync.RWMutex
}

// EmbeddingTask represents a task to generate embeddings
type EmbeddingTask struct {
	ID             string                 `json:"id"`
	Type           TaskType               `json:"type"`
	CollectionName string                 `json:"collection_name"`
	Content        string                 `json:"content"`
	Metadata       map[string]interface{} `json:"metadata"`
	ChunkInfo      *indexer.CodeChunk     `json:"chunk_info,omitempty"`
	Priority       int                    `json:"priority"`
	CreatedAt      time.Time              `json:"created_at"`
	Context        context.Context        `json:"-"`
}

type TaskType string

const (
	TaskTypeGenerate TaskType = "generate"
	TaskTypeUpdate   TaskType = "update"
	TaskTypeDelete   TaskType = "delete"
	TaskTypeBulk     TaskType = "bulk"
)

// EmbeddingResult represents the result of embedding generation
type EmbeddingResult struct {
	Task        *EmbeddingTask `json:"task"`
	Vector      []float32      `json:"vector,omitempty"`
	Success     bool           `json:"success"`
	Error       error          `json:"error,omitempty"`
	ProcessTime time.Duration  `json:"process_time"`
	TokenCount  int            `json:"token_count"`
	CacheHit    bool           `json:"cache_hit"`
	CompletedAt time.Time      `json:"completed_at"`
}

// VectorMetadata contains metadata about stored vectors
type VectorMetadata struct {
	ID            string                 `json:"id"`
	ContentHash   string                 `json:"content_hash"`
	OriginalText  string                 `json:"original_text"`
	ProcessedText string                 `json:"processed_text"`
	TokenCount    int                    `json:"token_count"`
	VectorNorm    float32                `json:"vector_norm"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// EmbeddingStatistics tracks embedding generation statistics
type EmbeddingStatistics struct {
	// Generation stats
	TotalGenerated int64 `json:"total_generated"`
	TotalFailed    int64 `json:"total_failed"`
	TotalTokens    int64 `json:"total_tokens"`

	// Performance metrics
	AvgGenerationTime   time.Duration `json:"avg_generation_time"`
	TotalGenerationTime time.Duration `json:"total_generation_time"`
	TokensPerSecond     float64       `json:"tokens_per_second"`
	VectorsPerSecond    float64       `json:"vectors_per_second"`

	// Cache performance
	CacheHits    int64   `json:"cache_hits"`
	CacheMisses  int64   `json:"cache_misses"`
	CacheHitRate float64 `json:"cache_hit_rate"`

	// Collection statistics
	CollectionCounts map[string]int64 `json:"collection_counts"`

	// Resource usage
	MemoryUsage  int64 `json:"memory_usage"`
	NetworkUsage int64 `json:"network_usage"`

	// Error tracking
	ErrorCounts   map[string]int64 `json:"error_counts"`
	LastError     string           `json:"last_error,omitempty"`
	LastErrorTime time.Time        `json:"last_error_time,omitempty"`

	mu sync.RWMutex
}

// NewEmbeddingsManager creates a new embeddings manager
func NewEmbeddingsManager(qdrantClient *QdrantClient, aiProvider ai.Provider, config *EmbeddingsConfig) (*EmbeddingsManager, error) {
	if config == nil {
		config = &EmbeddingsConfig{
			ModelName:           "all-MiniLM-L6-v2",
			VectorSize:          384,
			MaxTokens:           512,
			BatchSize:           32,
			WorkerCount:         4,
			QueueSize:           1000,
			MaxRetries:          3,
			RetryDelay:          time.Second * 2,
			EnableCaching:       true,
			CacheSize:           10000,
			CacheTTL:            time.Hour * 24,
			EnableNormalization: true,
			EnableQuantization:  false,
			QuantizationBits:    8,
			Timeout:             time.Second * 30,
			RateLimit:           100,
			EnableMetrics:       true,
			Collections:         make(map[string]*CollectionConfig),
		}
	}

	em := &EmbeddingsManager{
		qdrantClient: qdrantClient,
		aiProvider:   aiProvider,
		config:       config,
		collections:  make(map[string]*EmbeddingCollection),
		processQueue: make(chan *EmbeddingTask, config.QueueSize),
		resultQueue:  make(chan *EmbeddingResult, config.QueueSize),
		stopChan:     make(chan struct{}),
		stats: &EmbeddingStatistics{
			CollectionCounts: make(map[string]int64),
			ErrorCounts:      make(map[string]int64),
		},
	}

	// Initialize components
	em.initializeComponents()

	return em, nil
}

// Start starts the embeddings manager
func (em *EmbeddingsManager) Start(ctx context.Context) error {
	em.mu.Lock()
	defer em.mu.Unlock()

	if em.running {
		return fmt.Errorf("embeddings manager is already running")
	}

	// Start workers
	em.workers = make([]*EmbeddingWorker, em.config.WorkerCount)
	for i := 0; i < em.config.WorkerCount; i++ {
		worker := &EmbeddingWorker{
			ID:       i,
			manager:  em,
			stopChan: em.stopChan,
		}
		em.workers[i] = worker

		em.wg.Add(1)
		go worker.Run(ctx)
	}

	// Start result processor
	em.wg.Add(1)
	go em.processResults(ctx)

	// Start batch processor
	em.wg.Add(1)
	go em.batchProcessor.Start(ctx)

	em.running = true
	fmt.Println("Embeddings manager started")
	return nil
}

// Stop stops the embeddings manager
func (em *EmbeddingsManager) Stop() error {
	em.mu.Lock()
	defer em.mu.Unlock()

	if !em.running {
		return nil
	}

	close(em.stopChan)
	em.wg.Wait()

	em.running = false
	fmt.Println("Embeddings manager stopped")
	return nil
}

// GenerateEmbedding generates an embedding for a single text
func (em *EmbeddingsManager) GenerateEmbedding(ctx context.Context, collectionName, text string, metadata map[string]interface{}) ([]float32, error) {
	if !em.running {
		return nil, fmt.Errorf("embeddings manager is not running")
	}

	// Check cache first
	if em.config.EnableCaching {
		if cached := em.embeddingCache.Get(text); cached != nil {
			em.updateCacheStats(true)
			return cached, nil
		}
	}

	// Create embedding task
	task := &EmbeddingTask{
		ID:             em.generateTaskID(),
		Type:           TaskTypeGenerate,
		CollectionName: collectionName,
		Content:        text,
		Metadata:       metadata,
		Priority:       50,
		CreatedAt:      time.Now(),
		Context:        ctx,
	}

	return em.processEmbeddingTask(task)
}

// GenerateEmbeddingsForChunks generates embeddings for code chunks
func (em *EmbeddingsManager) GenerateEmbeddingsForChunks(ctx context.Context, collectionName string, chunks []*indexer.CodeChunk) error {
	if !em.running {
		return fmt.Errorf("embeddings manager is not running")
	}

	// Create tasks for each chunk
	for _, chunk := range chunks {
		// Preprocess chunk content
		content := em.preprocessChunkContent(chunk)

		// Create metadata
		metadata := map[string]interface{}{
			"file_path":  chunk.FilePath,
			"chunk_type": chunk.ChunkType,
			"language":   chunk.Language,
			"start_line": chunk.StartLine,
			"end_line":   chunk.EndLine,
			"name":       chunk.Name,
			"signature":  chunk.Signature,
			"complexity": chunk.Complexity,
			"hash":       chunk.Hash,
		}

		task := &EmbeddingTask{
			ID:             em.generateTaskID(),
			Type:           TaskTypeGenerate,
			CollectionName: collectionName,
			Content:        content,
			Metadata:       metadata,
			ChunkInfo:      chunk,
			Priority:       em.calculatePriority(chunk),
			CreatedAt:      time.Now(),
			Context:        ctx,
		}

		// Submit task
		select {
		case em.processQueue <- task:
			// Successfully queued
		default:
			return fmt.Errorf("embedding queue is full")
		}
	}

	return nil
}

// UpdateEmbedding updates an existing embedding
func (em *EmbeddingsManager) UpdateEmbedding(ctx context.Context, collectionName, id, newText string, metadata map[string]interface{}) error {
	if !em.running {
		return fmt.Errorf("embeddings manager is not running")
	}

	task := &EmbeddingTask{
		ID:             id,
		Type:           TaskTypeUpdate,
		CollectionName: collectionName,
		Content:        newText,
		Metadata:       metadata,
		Priority:       70,
		CreatedAt:      time.Now(),
		Context:        ctx,
	}

	select {
	case em.processQueue <- task:
		return nil
	default:
		return fmt.Errorf("embedding queue is full")
	}
}

// DeleteEmbedding deletes an embedding
func (em *EmbeddingsManager) DeleteEmbedding(ctx context.Context, collectionName, id string) error {
	if !em.running {
		return fmt.Errorf("embeddings manager is not running")
	}

	task := &EmbeddingTask{
		ID:             id,
		Type:           TaskTypeDelete,
		CollectionName: collectionName,
		Priority:       80,
		CreatedAt:      time.Now(),
		Context:        ctx,
	}

	select {
	case em.processQueue <- task:
		return nil
	default:
		return fmt.Errorf("embedding queue is full")
	}
}

// processEmbeddingTask processes a single embedding task synchronously
func (em *EmbeddingsManager) processEmbeddingTask(task *EmbeddingTask) ([]float32, error) {
	start := time.Now()

	// Preprocess text
	processedText := em.preprocessText(task.Content)

	// Generate embedding using AI provider
	response, err := em.aiProvider.GenerateEmbedding(task.Context, &ai.EmbeddingRequest{
		Text:  processedText,
		Model: em.config.ModelName,
	})

	if err != nil {
		em.updateErrorStats(err)
		return nil, fmt.Errorf("failed to generate embedding: %v", err)
	}

	vector := response.Embedding

	// Apply optimizations
	if em.config.EnableNormalization {
		vector = em.vectorOptimizer.Normalize(vector)
	}

	if em.config.EnableQuantization {
		vector = em.vectorOptimizer.Quantize(vector, em.config.QuantizationBits)
	}

	// Cache the result
	if em.config.EnableCaching {
		em.embeddingCache.Set(task.Content, vector)
		em.updateCacheStats(false)
	}

	// Update statistics
	em.updateGenerationStats(1, len(strings.Fields(processedText)), time.Since(start))

	return vector, nil
}

// Worker implementation

// EmbeddingWorker processes embedding tasks
type EmbeddingWorker struct {
	ID       int
	manager  *EmbeddingsManager
	stopChan <-chan struct{}
	stats    *WorkerStats
}

type WorkerStats struct {
	TasksProcessed int64         `json:"tasks_processed"`
	TasksFailed    int64         `json:"tasks_failed"`
	TotalTime      time.Duration `json:"total_time"`
	AverageTime    time.Duration `json:"average_time"`
	LastActiveTime time.Time     `json:"last_active_time"`
	mu             sync.RWMutex
}

// Run runs the embedding worker
func (ew *EmbeddingWorker) Run(ctx context.Context) {
	defer ew.manager.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ew.stopChan:
			return
		case task := <-ew.manager.processQueue:
			result := ew.processTask(task)

			select {
			case ew.manager.resultQueue <- result:
				// Result sent successfully
			default:
				// Result queue full
				fmt.Printf("Worker %d: result queue full\n", ew.ID)
			}
		}
	}
}

// processTask processes a single embedding task
func (ew *EmbeddingWorker) processTask(task *EmbeddingTask) *EmbeddingResult {
	start := time.Now()

	result := &EmbeddingResult{
		Task:        task,
		CompletedAt: time.Now(),
	}

	switch task.Type {
	case TaskTypeGenerate:
		vector, err := ew.manager.processEmbeddingTask(task)
		result.Vector = vector
		result.Success = err == nil
		result.Error = err

		if err == nil {
			// Store in Qdrant
			point := &VectorPoint{
				ID:      task.ID,
				Vector:  vector,
				Payload: task.Metadata,
			}

			err = ew.manager.qdrantClient.UpsertPoints(task.Context, task.CollectionName, []*VectorPoint{point})
			if err != nil {
				result.Success = false
				result.Error = fmt.Errorf("failed to store vector: %v", err)
			}
		}

	case TaskTypeUpdate:
		// Similar to generate but also update existing vector
		vector, err := ew.manager.processEmbeddingTask(task)
		result.Vector = vector
		result.Success = err == nil
		result.Error = err

		if err == nil {
			point := &VectorPoint{
				ID:      task.ID,
				Vector:  vector,
				Payload: task.Metadata,
			}

			err = ew.manager.qdrantClient.UpsertPoints(task.Context, task.CollectionName, []*VectorPoint{point})
			if err != nil {
				result.Success = false
				result.Error = fmt.Errorf("failed to update vector: %v", err)
			}
		}

	case TaskTypeDelete:
		err := ew.manager.qdrantClient.DeletePoints(task.Context, task.CollectionName, []string{task.ID})
		result.Success = err == nil
		result.Error = err

	default:
		result.Success = false
		result.Error = fmt.Errorf("unknown task type: %s", task.Type)
	}

	result.ProcessTime = time.Since(start)
	result.TokenCount = len(strings.Fields(task.Content))

	// Update worker statistics
	ew.updateStats(result)

	return result
}

// updateStats updates worker statistics
func (ew *EmbeddingWorker) updateStats(result *EmbeddingResult) {
	ew.stats.mu.Lock()
	defer ew.stats.mu.Unlock()

	if result.Success {
		ew.stats.TasksProcessed++
	} else {
		ew.stats.TasksFailed++
	}

	ew.stats.TotalTime += result.ProcessTime
	ew.stats.LastActiveTime = result.CompletedAt

	totalTasks := ew.stats.TasksProcessed + ew.stats.TasksFailed
	if totalTasks > 0 {
		ew.stats.AverageTime = ew.stats.TotalTime / time.Duration(totalTasks)
	}
}

// Result processing

// processResults processes embedding results
func (em *EmbeddingsManager) processResults(ctx context.Context) {
	defer em.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-em.stopChan:
			return
		case result := <-em.resultQueue:
			em.handleResult(result)
		}
	}
}

// handleResult handles an embedding result
func (em *EmbeddingsManager) handleResult(result *EmbeddingResult) {
	// Update collection statistics
	if result.Success {
		em.updateCollectionStats(result.Task.CollectionName, result)
	}

	// Update global statistics
	em.updateGlobalStats(result)

	// Log result
	if result.Success {
		fmt.Printf("Embedding task %s completed successfully in %v\n",
			result.Task.ID, result.ProcessTime)
	} else {
		fmt.Printf("Embedding task %s failed: %v\n",
			result.Task.ID, result.Error)
	}
}

// Helper methods

func (em *EmbeddingsManager) initializeComponents() {
	// Initialize embedding cache
	if em.config.EnableCaching {
		em.embeddingCache = NewEmbeddingCache(em.config.CacheSize, em.config.CacheTTL)
	}

	// Initialize batch processor
	em.batchProcessor = NewEmbeddingBatchProcessor(em.config.BatchSize, em.config.Timeout)

	// Initialize vector optimizer
	em.vectorOptimizer = NewVectorOptimizer(em.config)

	// Initialize collections from config
	for name, config := range em.config.Collections {
		collection := &EmbeddingCollection{
			Name:        name,
			Config:      config,
			Statistics:  &CollectionStatistics{},
			Index:       make(map[string]*VectorMetadata),
			LastUpdated: time.Now(),
		}
		em.collections[name] = collection
	}
}

func (em *EmbeddingsManager) preprocessChunkContent(chunk *indexer.CodeChunk) string {
	var content strings.Builder

	// Include chunk name if available
	if chunk.Name != "" {
		content.WriteString(fmt.Sprintf("Function: %s\n", chunk.Name))
	}

	// Include signature if available
	if chunk.Signature != "" {
		content.WriteString(fmt.Sprintf("Signature: %s\n", chunk.Signature))
	}

	// Include docstring if available
	if chunk.DocString != "" {
		content.WriteString(fmt.Sprintf("Documentation: %s\n", chunk.DocString))
	}

	// Include the main content
	content.WriteString(chunk.Content)

	return content.String()
}

func (em *EmbeddingsManager) preprocessText(text string) string {
	// Remove excessive whitespace
	text = strings.TrimSpace(text)

	// Normalize line endings
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")

	// Remove excessive newlines
	for strings.Contains(text, "\n\n\n") {
		text = strings.ReplaceAll(text, "\n\n\n", "\n\n")
	}

	// Truncate if too long
	words := strings.Fields(text)
	if len(words) > em.config.MaxTokens {
		words = words[:em.config.MaxTokens]
		text = strings.Join(words, " ")
	}

	return text
}

func (em *EmbeddingsManager) calculatePriority(chunk *indexer.CodeChunk) int {
	priority := 50 // Base priority

	// Higher priority for functions and classes
	switch chunk.ChunkType {
	case "function":
		priority += 20
	case "class":
		priority += 15
	case "type":
		priority += 10
	}

	// Higher priority for exported symbols
	if len(chunk.Name) > 0 && chunk.Name[0] >= 'A' && chunk.Name[0] <= 'Z' {
		priority += 10
	}

	// Higher priority for complex code
	if chunk.Complexity > 5 {
		priority += 5
	}

	return priority
}

func (em *EmbeddingsManager) generateTaskID() string {
	return fmt.Sprintf("emb_%d", time.Now().UnixNano())
}

func (em *EmbeddingsManager) calculateContentHash(content string) string {
	hash := md5.Sum([]byte(content))
	return fmt.Sprintf("%x", hash)
}

// Statistics methods

func (em *EmbeddingsManager) updateGenerationStats(count int, tokens int, duration time.Duration) {
	em.stats.mu.Lock()
	defer em.stats.mu.Unlock()

	em.stats.TotalGenerated += int64(count)
	em.stats.TotalTokens += int64(tokens)
	em.stats.TotalGenerationTime += duration

	if em.stats.TotalGenerated > 0 {
		em.stats.AvgGenerationTime = em.stats.TotalGenerationTime / time.Duration(em.stats.TotalGenerated)
	}

	// Calculate throughput
	totalSeconds := em.stats.TotalGenerationTime.Seconds()
	if totalSeconds > 0 {
		em.stats.TokensPerSecond = float64(em.stats.TotalTokens) / totalSeconds
		em.stats.VectorsPerSecond = float64(em.stats.TotalGenerated) / totalSeconds
	}
}

func (em *EmbeddingsManager) updateCacheStats(hit bool) {
	em.stats.mu.Lock()
	defer em.stats.mu.Unlock()

	if hit {
		em.stats.CacheHits++
	} else {
		em.stats.CacheMisses++
	}

	total := em.stats.CacheHits + em.stats.CacheMisses
	if total > 0 {
		em.stats.CacheHitRate = float64(em.stats.CacheHits) / float64(total)
	}
}

func (em *EmbeddingsManager) updateErrorStats(err error) {
	em.stats.mu.Lock()
	defer em.stats.mu.Unlock()

	em.stats.TotalFailed++
	errorType := "unknown"
	if err != nil {
		errorType = fmt.Sprintf("%T", err)
		em.stats.LastError = err.Error()
		em.stats.LastErrorTime = time.Now()
	}
	em.stats.ErrorCounts[errorType]++
}

func (em *EmbeddingsManager) updateCollectionStats(collectionName string, result *EmbeddingResult) {
	em.stats.mu.Lock()
	defer em.stats.mu.Unlock()

	em.stats.CollectionCounts[collectionName]++

	// Update collection-specific stats
	if collection, exists := em.collections[collectionName]; exists {
		collection.mu.Lock()
		collection.VectorCount++
		collection.LastUpdated = time.Now()
		collection.mu.Unlock()
	}
}

func (em *EmbeddingsManager) updateGlobalStats(result *EmbeddingResult) {
	// This would update global statistics
	// Implementation depends on specific metrics needed
}

// Public API

func (em *EmbeddingsManager) IsRunning() bool {
	em.mu.RLock()
	defer em.mu.RUnlock()
	return em.running
}

func (em *EmbeddingsManager) GetStatistics() *EmbeddingStatistics {
	em.stats.mu.RLock()
	defer em.stats.mu.RUnlock()

	stats := *em.stats
	return &stats
}

func (em *EmbeddingsManager) GetCollections() map[string]*EmbeddingCollection {
	em.mu.RLock()
	defer em.mu.RUnlock()

	collections := make(map[string]*EmbeddingCollection)
	for k, v := range em.collections {
		collection := *v
		collections[k] = &collection
	}

	return collections
}

func (em *EmbeddingsManager) GetQueueLength() int {
	return len(em.processQueue)
}

func (em *EmbeddingsManager) GetWorkerStats() []*WorkerStats {
	var stats []*WorkerStats

	for _, worker := range em.workers {
		if worker.stats != nil {
			worker.stats.mu.RLock()
			workerStats := *worker.stats
			worker.stats.mu.RUnlock()
			stats = append(stats, &workerStats)
		}
	}

	return stats
}
