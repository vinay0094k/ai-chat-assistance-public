package vectordb
package vectordb

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/indexer"
	"github.com/yourusername/ai-code-assistant/storage"
)

// SemanticIndex manages the semantic index of code, mapping code chunks to vectors
type SemanticIndex struct {
	// Core components
	qdrantClient      *QdrantClient
	similarityEngine  *SimilarityEngine
	db                *storage.SQLiteDB
	
	// Configuration
	config            *SemanticIndexConfig
	
	// Index management
	collections       map[string]*IndexedCollection
	vectorMappings    map[string]*VectorMapping
	metadataIndex     map[string]*MetadataEntry
	
	// Caching and optimization
	indexCache        *IndexCache
	queryCache        *QueryCache
	bloomFilter       *IndexBloomFilter
	
	// Background processing
	indexer           *BackgroundIndexer
	compactor         *IndexCompactor
	optimizer         *IndexOptimizer
	
	// Analytics and monitoring
	stats             *SemanticIndexStatistics
	healthMonitor     *IndexHealthMonitor
	
	// State management
	mu                sync.RWMutex
	indexLock         sync.RWMutex
	updateLock        sync.Mutex
	version           int64
	lastCompaction    time.Time
}

// SemanticIndexConfig contains semantic index configuration
type SemanticIndexConfig struct {
	// Index structure
	DefaultCollection    string            `json:"default_collection"`
	MaxCollections      int               `json:"max_collections"`
	VectorDimensions    int               `json:"vector_dimensions"`
	DefaultMetric       string            `json:"default_metric"`
	
	// Performance settings
	IndexingStrategy    IndexingStrategy  `json:"indexing_strategy"`
	BatchSize          int               `json:"batch_size"`
	FlushInterval      time.Duration     `json:"flush_interval"`
	CompactionInterval time.Duration     `json:"compaction_interval"`
	OptimizationInterval time.Duration   `json:"optimization_interval"`
	
	// Caching
	EnableCaching      bool              `json:"enable_caching"`
	CacheSize          int               `json:"cache_size"`
	CacheTTL           time.Duration     `json:"cache_ttl"`
	EnableBloomFilter  bool              `json:"enable_bloom_filter"`
	BloomFilterSize    int               `json:"bloom_filter_size"`
	
	// Quality and consistency
	EnableVersioning   bool              `json:"enable_versioning"`
	EnableConsistencyCheck bool          `json:"enable_consistency_check"`
	ConsistencyThreshold float32         `json:"consistency_threshold"`
	
	// Background processing
	EnableBackgroundIndexing bool        `json:"enable_background_indexing"`
	BackgroundWorkers  int               `json:"background_workers"`
	EnableAutoCompaction bool            `json:"enable_auto_compaction"`
	EnableAutoOptimization bool          `json:"enable_auto_optimization"`
	
	// Storage optimization
	EnableCompression  bool              `json:"enable_compression"`
	CompressionLevel   int               `json:"compression_level"`
	EnableQuantization bool              `json:"enable_quantization"`
	QuantizationBits   int               `json:"quantization_bits"`
	
	// Search optimization
	DefaultSearchLimit int               `json:"default_search_limit"`
	MaxSearchLimit     int               `json:"max_search_limit"`
	SearchTimeout      time.Duration     `json:"search_timeout"`
	EnableApproximateSearch bool         `json:"enable_approximate_search"`
}

// IndexingStrategy represents different indexing approaches
type IndexingStrategy string

const (
	StrategyImmediate    IndexingStrategy = "immediate"    // Index immediately
	StrategyBatched      IndexingStrategy = "batched"      // Batch and index
	StrategyBackground   IndexingStrategy = "background"   // Background indexing
	StrategyAdaptive     IndexingStrategy = "adaptive"     // Adaptive based on load
)

// IndexedCollection represents a collection in the semantic index
type IndexedCollection struct {
	Name              string                    `json:"name"`
	VectorCount       int64                     `json:"vector_count"`
	Dimensions        int                       `json:"dimensions"`
	Metric            string                    `json:"metric"`
	CreatedAt         time.Time                 `json:"created_at"`
	UpdatedAt         time.Time                 `json:"updated_at"`
	LastOptimized     time.Time                 `json:"last_optimized"`
	Config            *CollectionIndexConfig    `json:"config"`
	Statistics        *CollectionIndexStats     `json:"statistics"`
	Health            *CollectionHealth         `json:"health"`
	mu                sync.RWMutex
}

// CollectionIndexConfig contains collection-specific configuration
type CollectionIndexConfig struct {
	IndexType         string            `json:"index_type"`         // hnsw, ivf, flat
	IndexParameters   map[string]interface{} `json:"index_parameters"`
	ReplicationFactor int               `json:"replication_factor"`
	ShardCount        int               `json:"shard_count"`
	OptimizeFor       string            `json:"optimize_for"`       // speed, memory, accuracy
}

// CollectionIndexStats tracks collection statistics
type CollectionIndexStats struct {
	TotalInserts      int64         `json:"total_inserts"`
	TotalUpdates      int64         `json:"total_updates"`
	TotalDeletes      int64         `json:"total_deletes"`
	TotalQueries      int64         `json:"total_queries"`
	AverageQueryTime  time.Duration `json:"average_query_time"`
	IndexSizeBytes    int64         `json:"index_size_bytes"`
	VectorSizeBytes   int64         `json:"vector_size_bytes"`
	CompressionRatio  float64       `json:"compression_ratio"`
	FragmentationRatio float64      `json:"fragmentation_ratio"`
	mu                sync.RWMutex
}

// CollectionHealth represents the health status of a collection
type CollectionHealth struct {
	Status            HealthStatus      `json:"status"`
	LastHealthCheck   time.Time         `json:"last_health_check"`
	Issues            []HealthIssue     `json:"issues"`
	PerformanceScore  float64           `json:"performance_score"`
	ConsistencyScore  float64           `json:"consistency_score"`
	QualityScore      float64           `json:"quality_score"`
}

type HealthStatus string

const (
	HealthStatusHealthy   HealthStatus = "healthy"
	HealthStatusWarning   HealthStatus = "warning"
	HealthStatusCritical  HealthStatus = "critical"
	HealthStatusUnknown   HealthStatus = "unknown"
)

type HealthIssue struct {
	Type        string    `json:"type"`
	Severity    string    `json:"severity"`
	Description string    `json:"description"`
	DetectedAt  time.Time `json:"detected_at"`
	Suggestion  string    `json:"suggestion"`
}

// VectorMapping maps code chunks to their vector representations
type VectorMapping struct {
	ID                string                 `json:"id"`
	ChunkID           string                 `json:"chunk_id"`
	VectorID          string                 `json:"vector_id"`
	CollectionName    string                 `json:"collection_name"`
	Vector            []float32              `json:"vector,omitempty"`
	Metadata          *VectorMetadata        `json:"metadata"`
	Quality           *VectorQuality         `json:"quality"`
	CreatedAt         time.Time              `json:"created_at"`
	UpdatedAt         time.Time              `json:"updated_at"`
	Version           int64                  `json:"version"`
	Status            MappingStatus          `json:"status"`
}

type MappingStatus string

const (
	StatusActive      MappingStatus = "active"
	StatusPending     MappingStatus = "pending"
	StatusUpdating    MappingStatus = "updating"
	StatusDeleted     MappingStatus = "deleted"
	StatusCorrupted   MappingStatus = "corrupted"
)

// VectorQuality represents quality metrics for a vector
type VectorQuality struct {
	EmbeddingQuality  float32   `json:"embedding_quality"`   // Quality of the embedding
	SemanticClarity   float32   `json:"semantic_clarity"`    // How clear the semantic meaning is
	Uniqueness        float32   `json:"uniqueness"`          // How unique the vector is
	Consistency       float32   `json:"consistency"`         // Consistency with similar code
	Completeness      float32   `json:"completeness"`        // Completeness of the representation
	OverallScore      float32   `json:"overall_score"`       // Overall quality score
	LastEvaluated     time.Time `json:"last_evaluated"`
}

// MetadataEntry represents an entry in the metadata index
type MetadataEntry struct {
	Key               string                 `json:"key"`
	Value             string                 `json:"value"`
	VectorIDs         []string               `json:"vector_ids"`
	Count             int64                  `json:"count"`
	LastUpdated       time.Time              `json:"last_updated"`
}

// SemanticSearchRequest represents a semantic search request
type SemanticSearchRequest struct {
	Query             string                 `json:"query"`
	QueryVector       []float32              `json:"query_vector,omitempty"`
	Collections       []string               `json:"collections,omitempty"`
	Filters           *SemanticFilters       `json:"filters,omitempty"`
	Limit             int                    `json:"limit"`
	Threshold         float32                `json:"threshold"`
	IncludeMetadata   bool                   `json:"include_metadata"`
	IncludeVectors    bool                   `json:"include_vectors"`
	SearchMode        SemanticSearchMode     `json:"search_mode"`
	RankingMode       string                 `json:"ranking_mode"`
	Context           context.Context        `json:"-"`
}

type SemanticSearchMode string

const (
	SearchModeExact        SemanticSearchMode = "exact"
	SearchModeSimilar      SemanticSearchMode = "similar"
	SearchModeExpanded     SemanticSearchMode = "expanded"
	SearchModeHybridSem    SemanticSearchMode = "hybrid"
)

// SemanticFilters represents filters for semantic search
type SemanticFilters struct {
	Languages         []string               `json:"languages,omitempty"`
	FileTypes         []string               `json:"file_types,omitempty"`
	ChunkTypes        []string               `json:"chunk_types,omitempty"`
	Complexity        *RangeFilter           `json:"complexity,omitempty"`
	Size              *RangeFilter           `json:"size,omitempty"`
	Recency           *TimeRangeFilter       `json:"recency,omitempty"`
	Quality           *RangeFilter           `json:"quality,omitempty"`
	CustomFilters     map[string]interface{} `json:"custom_filters,omitempty"`
}

type RangeFilter struct {
	Min               *float64               `json:"min,omitempty"`
	Max               *float64               `json:"max,omitempty"`
}

type TimeRangeFilter struct {
	After             *time.Time             `json:"after,omitempty"`
	Before            *time.Time             `json:"before,omitempty"`
}

// SemanticSearchResponse represents semantic search results
type SemanticSearchResponse struct {
	Results           []*SemanticSearchResult `json:"results"`
	TotalFound        int                     `json:"total_found"`
	QueryTime         time.Duration           `json:"query_time"`
	CacheHit          bool                    `json:"cache_hit"`
	Collections       []string                `json:"collections"`
	QualityScore      float64                 `json:"quality_score"`
	Suggestions       []string                `json:"suggestions,omitempty"`
}

// SemanticSearchResult represents a single semantic search result
type SemanticSearchResult struct {
	ID                string                 `json:"id"`
	ChunkID           string                 `json:"chunk_id"`
	Score             float32                `json:"score"`
	Vector            []float32              `json:"vector,omitempty"`
	Metadata          *VectorMetadata        `json:"metadata"`
	Quality           *VectorQuality         `json:"quality,omitempty"`
	Explanation       *SearchExplanation     `json:"explanation,omitempty"`
	CollectionName    string                 `json:"collection_name"`
}

// SearchExplanation explains why a result was returned
type SearchExplanation struct {
	MatchType         string                 `json:"match_type"`
	MatchFactors      []string               `json:"match_factors"`
	SimilarityScore   float32                `json:"similarity_score"`
	QualityBonus      float32                `json:"quality_bonus"`
	RelevanceFactors  map[string]float32     `json:"relevance_factors"`
}

// SemanticIndexStatistics tracks overall semantic index statistics
type SemanticIndexStatistics struct {
	TotalCollections    int64                    `json:"total_collections"`
	TotalVectors        int64                    `json:"total_vectors"`
	TotalSize           int64                    `json:"total_size"`
	IndexSize           int64                    `json:"index_size"`
	CompressionRatio    float64                  `json:"compression_ratio"`
	
	// Operation statistics
	IndexOperations     int64                    `json:"index_operations"`
	SearchOperations    int64                    `json:"search_operations"`
	UpdateOperations    int64                    `json:"update_operations"`
	DeleteOperations    int64                    `json:"delete_operations"`
	
	// Performance metrics
	AverageIndexTime    time.Duration            `json:"average_index_time"`
	AverageSearchTime   time.Duration            `json:"average_search_time"`
	ThroughputOpsPerSec float64                  `json:"throughput_ops_per_sec"`
	
	// Quality metrics
	AverageQualityScore float64                  `json:"average_quality_score"`
	ConsistencyScore    float64                  `json:"consistency_score"`
	
	// Cache metrics
	CacheHitRate        float64                  `json:"cache_hit_rate"`
	CacheMissRate       float64                  `json:"cache_miss_rate"`
	
	// Health metrics
	HealthyCollections  int64                    `json:"healthy_collections"`
	WarningCollections  int64                    `json:"warning_collections"`
	CriticalCollections int64                    `json:"critical_collections"`
	
	// Background processing
	BackgroundTasks     int64                    `json:"background_tasks"`
	CompactionCount     int64                    `json:"compaction_count"`
	OptimizationCount   int64                    `json:"optimization_count"`
	
	LastUpdated         time.Time                `json:"last_updated"`
	mu                  sync.RWMutex
}

// NewSemanticIndex creates a new semantic index
func NewSemanticIndex(qdrantClient *QdrantClient, similarityEngine *SimilarityEngine, db *storage.SQLiteDB, config *SemanticIndexConfig) (*SemanticIndex, error) {
	if config == nil {
		config = &SemanticIndexConfig{
			DefaultCollection:        "code",
			MaxCollections:          10,
			VectorDimensions:        384,
			DefaultMetric:           "cosine",
			IndexingStrategy:        StrategyBatched,
			BatchSize:              100,
			FlushInterval:          time.Minute * 5,
			CompactionInterval:     time.Hour * 6,
			OptimizationInterval:   time.Hour * 24,
			EnableCaching:          true,
			CacheSize:              10000,
			CacheTTL:               time.Hour,
			EnableBloomFilter:      true,
			BloomFilterSize:        100000,
			EnableVersioning:       true,
			EnableConsistencyCheck: true,
			ConsistencyThreshold:   0.8,
			EnableBackgroundIndexing: true,
			BackgroundWorkers:      4,
			EnableAutoCompaction:   true,
			EnableAutoOptimization: true,
			EnableCompression:      true,
			CompressionLevel:       6,
			EnableQuantization:     false,
			QuantizationBits:       8,
			DefaultSearchLimit:     50,
			MaxSearchLimit:         1000,
			SearchTimeout:          time.Second * 30,
			EnableApproximateSearch: true,
		}
	}

	si := &SemanticIndex{
		qdrantClient:     qdrantClient,
		similarityEngine: similarityEngine,
		db:              db,
		config:          config,
		collections:     make(map[string]*IndexedCollection),
		vectorMappings:  make(map[string]*VectorMapping),
		metadataIndex:   make(map[string]*MetadataEntry),
		version:         1,
		stats: &SemanticIndexStatistics{
			LastUpdated: time.Now(),
		},
	}

	// Initialize components
	if err := si.initializeComponents(); err != nil {
		return nil, fmt.Errorf("failed to initialize semantic index: %v", err)
	}

	// Load existing index
	if err := si.loadIndex(); err != nil {
		return nil, fmt.Errorf("failed to load existing index: %v", err)
	}

	// Start background services
	si.startBackgroundServices()

	return si, nil
}

// IndexCodeChunk indexes a code chunk with its vector representation
func (si *SemanticIndex) IndexCodeChunk(ctx context.Context, chunk *indexer.CodeChunk, vector []float32, collectionName string) error {
	if collectionName == "" {
		collectionName = si.config.DefaultCollection
	}

	start := time.Now()

	// Validate inputs
	if len(vector) != si.config.VectorDimensions {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", si.config.VectorDimensions, len(vector))
	}

	// Generate unique IDs
	mappingID := si.generateMappingID(chunk)
	vectorID := si.generateVectorID(chunk, vector)

	// Create vector mapping
	mapping := &VectorMapping{
		ID:             mappingID,
		ChunkID:        chunk.ID,
		VectorID:       vectorID,
		CollectionName: collectionName,
		Vector:         vector,
		Metadata: &VectorMetadata{
			ID:           vectorID,
			ContentHash:  chunk.Hash,
			OriginalText: chunk.Content,
			TokenCount:   len(strings.Fields(chunk.Content)),
			CreatedAt:    time.Now(),
			UpdatedAt:    time.Now(),
			Metadata: map[string]interface{}{
				"file_path":   chunk.FilePath,
				"language":    chunk.Language,
				"chunk_type":  chunk.ChunkType,
				"start_line":  chunk.StartLine,
				"end_line":    chunk.EndLine,
				"name":        chunk.Name,
				"signature":   chunk.Signature,
				"complexity":  chunk.Complexity,
			},
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Version:   si.version,
		Status:    StatusPending,
	}

	// Calculate vector quality
	mapping.Quality = si.calculateVectorQuality(chunk, vector)

	// Check if collection exists
	collection, err := si.ensureCollection(collectionName)
	if err != nil {
		return fmt.Errorf("failed to ensure collection: %v", err)
	}

	// Index based on strategy
	switch si.config.IndexingStrategy {
	case StrategyImmediate:
		err = si.indexImmediately(ctx, mapping, collection)
	case StrategyBatched:
		err = si.addToBatch(mapping, collection)
	case StrategyBackground:
		err = si.addToBackgroundQueue(mapping, collection)
	case StrategyAdaptive:
		err = si.indexAdaptively(ctx, mapping, collection)
	default:
		err = si.indexImmediately(ctx, mapping, collection)
	}

	if err != nil {
		return fmt.Errorf("indexing failed: %v", err)
	}

	// Update local mappings
	si.updateLock.Lock()
	si.vectorMappings[mappingID] = mapping
	si.updateMetadataIndex(mapping)
	si.updateLock.Unlock()

	// Update bloom filter
	if si.config.EnableBloomFilter {
		si.bloomFilter.Add(chunk.ID)
		si.bloomFilter.Add(vectorID)
	}

	// Update statistics
	si.updateIndexingStats(time.Since(start))

	return nil
}

// SearchSemantic performs semantic search across the index
func (si *SemanticIndex) SearchSemantic(ctx context.Context, request *SemanticSearchRequest) (*SemanticSearchResponse, error) {
	start := time.Now()

	// Validate request
	if err := si.validateSearchRequest(request); err != nil {
		return nil, fmt.Errorf("invalid search request: %v", err)
	}

	// Check cache
	cacheKey := si.generateSearchCacheKey(request)
	if si.config.EnableCaching {
		if cached := si.queryCache.Get(cacheKey); cached != nil {
			cached.CacheHit = true
			cached.QueryTime = time.Since(start)
			si.updateSearchStats(time.Since(start), true)
			return cached, nil
		}
	}

	// Determine collections to search
	collections := request.Collections
	if len(collections) == 0 {
		collections = []string{si.config.DefaultCollection}
	}

	// Perform search based on mode
	var results []*SemanticSearchResult
	var err error

	switch request.SearchMode {
	case SearchModeExact:
		results, err = si.performExactSearch(ctx, request, collections)
	case SearchModeSimilar:
		results, err = si.performSimilaritySearch(ctx, request, collections)
	case SearchModeExpanded:
		results, err = si.performExpandedSearch(ctx, request, collections)
	case SearchModeHybridSem:
		results, err = si.performHybridSemanticSearch(ctx, request, collections)
	default:
		results, err = si.performSimilaritySearch(ctx, request, collections)
	}

	if err != nil {
		return nil, fmt.Errorf("search failed: %v", err)
	}

	// Apply post-processing
	results = si.postProcessResults(results, request)

	// Create response
	response := &SemanticSearchResponse{
		Results:      results,
		TotalFound:   len(results),
		QueryTime:    time.Since(start),
		CacheHit:     false,
		Collections:  collections,
		QualityScore: si.calculateResultsQuality(results),
	}

	// Generate suggestions if results are limited
	if len(results) < request.Limit/2 {
		response.Suggestions = si.generateSearchSuggestions(request, results)
	}

	// Cache results
	if si.config.EnableCaching {
		si.queryCache.Set(cacheKey, response)
	}

	// Update statistics
	si.updateSearchStats(time.Since(start), false)

	return response, nil
}

// UpdateVector updates an existing vector in the index
func (si *SemanticIndex) UpdateVector(ctx context.Context, chunkID string, newVector []float32) error {
	start := time.Now()

	si.updateLock.Lock()
	defer si.updateLock.Unlock()

	// Find existing mapping
	var mapping *VectorMapping
	for _, m := range si.vectorMappings {
		if m.ChunkID == chunkID {
			mapping = m
			break
		}
	}

	if mapping == nil {
		return fmt.Errorf("mapping not found for chunk ID: %s", chunkID)
	}

	// Update vector
	oldVector := mapping.Vector
	mapping.Vector = newVector
	mapping.UpdatedAt = time.Now()
	mapping.Version = si.version
	mapping.Status = StatusUpdating

	// Update in Qdrant
	point := &VectorPoint{
		ID:      mapping.VectorID,
		Vector:  newVector,
		Payload: si.createPayload(mapping.Metadata),
	}

	err := si.qdrantClient.UpsertPoints(ctx, mapping.CollectionName, []*VectorPoint{point})
	if err != nil {
		// Rollback
		mapping.Vector = oldVector
		mapping.Status = StatusActive
		return fmt.Errorf("failed to update vector in Qdrant: %v", err)
	}

	mapping.Status = StatusActive

	// Update statistics
	si.updateUpdateStats(time.Since(start))

	return nil
}

// DeleteVector removes a vector from the index
func (si *SemanticIndex) DeleteVector(ctx context.Context, chunkID string) error {
	start := time.Now()

	si.updateLock.Lock()
	defer si.updateLock.Unlock()

	// Find mapping
	var mappingID string
	var mapping *VectorMapping
	for id, m := range si.vectorMappings {
		if m.ChunkID == chunkID {
			mappingID = id
			mapping = m
			break
		}
	}

	if mapping == nil {
		return fmt.Errorf("mapping not found for chunk ID: %s", chunkID)
	}

	// Delete from Qdrant
	err := si.qdrantClient.DeletePoints(ctx, mapping.CollectionName, []string{mapping.VectorID})
	if err != nil {
		return fmt.Errorf("failed to delete from Qdrant: %v", err)
	}

	// Update mapping status
	mapping.Status = StatusDeleted
	mapping.UpdatedAt = time.Now()

	// Remove from local mappings after a delay (for potential rollback)
	time.AfterFunc(time.Hour, func() {
		si.updateLock.Lock()
		defer si.updateLock.Unlock()
		delete(si.vectorMappings, mappingID)
	})

	// Update statistics
	si.updateDeleteStats(time.Since(start))

	return nil
}

// GetVectorByChunkID retrieves a vector by chunk ID
func (si *SemanticIndex) GetVectorByChunkID(chunkID string) (*VectorMapping, error) {
	si.mu.RLock()
	defer si.mu.RUnlock()

	for _, mapping := range si.vectorMappings {
		if mapping.ChunkID == chunkID && mapping.Status == StatusActive {
			return mapping, nil
		}
	}

	return nil, fmt.Errorf("vector not found for chunk ID: %s", chunkID)
}

// EnhanceResults enhances search results with semantic understanding
func (si *SemanticIndex) EnhanceResults(results []*SearchResult, request *SearchRequest) []*SearchResult {
	enhanced := make([]*SearchResult, len(results))
	copy(enhanced, results)

	for _, result := range enhanced {
		// Add semantic explanation
		if mapping, err := si.GetVectorByChunkID(result.ID); err == nil {
			result.Metadata["semantic_quality"] = mapping.Quality.OverallScore
			result.Metadata["embedding_quality"] = mapping.Quality.EmbeddingQuality
			result.Metadata["uniqueness"] = mapping.Quality.Uniqueness
		}

		// Enhance with similar code
		similar := si.findSimilarCode(result, 3)
		if len(similar) > 0 {
			result.Metadata["similar_code"] = similar
		}
	}

	return enhanced
}

// Private methods for indexing strategies

func (si *SemanticIndex) indexImmediately(ctx context.Context, mapping *VectorMapping, collection *IndexedCollection) error {
	// Create Qdrant point
	point := &VectorPoint{
		ID:      mapping.VectorID,
		Vector:  mapping.Vector,
		Payload: si.createPayload(mapping.Metadata),
		Version: mapping.Version,
	}

	// Insert into Qdrant
	err := si.qdrantClient.UpsertPoints(ctx, collection.Name, []*VectorPoint{point})
	if err != nil {
		return err
	}

	mapping.Status = StatusActive
	collection.VectorCount++
	collection.UpdatedAt = time.Now()

	return nil
}

func (si *SemanticIndex) addToBatch(mapping *VectorMapping, collection *IndexedCollection) error {
	// Add to background indexer batch
	return si.indexer.AddToBatch(mapping, collection)
}

func (si *SemanticIndex) addToBackgroundQueue(mapping *VectorMapping, collection *IndexedCollection) error {
	// Add to background processing queue
	return si.indexer.AddToQueue(mapping, collection)
}

func (si *SemanticIndex) indexAdaptively(ctx context.Context, mapping *VectorMapping, collection *IndexedCollection) error {
	// Choose strategy based on current load
	currentLoad := si.getCurrentLoad()
	
	if currentLoad < 0.5 {
		return si.indexImmediately(ctx, mapping, collection)
	} else if currentLoad < 0.8 {
		return si.addToBatch(mapping, collection)
	} else {
		return si.addToBackgroundQueue(mapping, collection)
	}
}

// Private methods for search

func (si *SemanticIndex) performSimilaritySearch(ctx context.Context, request *SemanticSearchRequest, collections []string) ([]*SemanticSearchResult, error) {
	var allResults []*SemanticSearchResult

	for _, collectionName := range collections {
		// Create Qdrant search request
		searchReq := &SearchRequest{
			CollectionName: collectionName,
			Vector:         request.QueryVector,
			Limit:          request.Limit,
			ScoreThreshold: request.Threshold,
			WithPayload:    request.IncludeMetadata,
			WithVector:     request.IncludeVectors,
			Filter:         si.convertFilters(request.Filters),
		}

		// Execute search
		results, err := si.qdrantClient.SearchPoints(ctx, searchReq)
		if err != nil {
			continue // Skip failed collections
		}

		// Convert results
		for _, result := range results {
			semanticResult := si.convertToSemanticResult(result, collectionName)
			allResults = append(allResults, semanticResult)
		}
	}

	return allResults, nil
}

func (si *SemanticIndex) performExactSearch(ctx context.Context, request *SemanticSearchRequest, collections []string) ([]*SemanticSearchResult, error) {
	// For exact search, we look for vectors with very high similarity (>0.95)
	request.Threshold = 0.95
	return si.performSimilaritySearch(ctx, request, collections)
}

func (si *SemanticIndex) performExpandedSearch(ctx context.Context, request *SemanticSearchRequest, collections []string) ([]*SemanticSearchResult, error) {
	// First perform regular search
	results, err := si.performSimilaritySearch(ctx, request, collections)
	if err != nil {
		return nil, err
	}

	// Then expand with related vectors
	expanded := make([]*SemanticSearchResult, 0, len(results)*2)
	expanded = append(expanded, results...)

	for _, result := range results {
		related := si.findRelatedVectors(result, 2) // Find 2 related vectors per result
		expanded = append(expanded, related...)
	}

	// Remove duplicates and re-sort
	expanded = si.deduplicateResults(expanded)
	si.sortResultsByScore(expanded)

	// Limit results
	if len(expanded) > request.Limit {
		expanded = expanded[:request.Limit]
	}

	return expanded, nil
}

func (si *SemanticIndex) performHybridSemanticSearch(ctx context.Context, request *SemanticSearchRequest, collections []string) ([]*SemanticSearchResult, error) {
	// Combine similarity search with metadata-based search
	similarityResults, err := si.performSimilaritySearch(ctx, request, collections)
	if err != nil {
		return nil, err
	}

	// Perform metadata-based search
	metadataResults := si.searchByMetadata(request, collections)

	// Combine and re-rank
	combined := si.combineAndRerankResults(similarityResults, metadataResults, 0.7, 0.3)

	return combined, nil
}

// Utility methods

func (si *SemanticIndex) ensureCollection(name string) (*IndexedCollection, error) {
	si.indexLock.Lock()
	defer si.indexLock.Unlock()

	if collection, exists := si.collections[name]; exists {
		return collection, nil
	}

	// Create new collection
	err := si.qdrantClient.CreateCollection(context.Background(), name, si.config.VectorDimensions, si.config.DefaultMetric)
	if err != nil {
		return nil, err
	}

	collection := &IndexedCollection{
		Name:        name,
		Dimensions:  si.config.VectorDimensions,
		Metric:      si.config.DefaultMetric,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Config: &CollectionIndexConfig{
			IndexType:         "hnsw",
			IndexParameters:   make(map[string]interface{}),
			ReplicationFactor: 1,
			ShardCount:        1,
			OptimizeFor:       "balanced",
		},
		Statistics: &CollectionIndexStats{},
		Health: &CollectionHealth{
			Status:           HealthStatusHealthy,
			LastHealthCheck:  time.Now(),
			Issues:           []HealthIssue{},
			PerformanceScore: 1.0,
			ConsistencyScore: 1.0,
			QualityScore:     1.0,
		},
	}

	si.collections[name] = collection
	return collection, nil
}

func (si *SemanticIndex) calculateVectorQuality(chunk *indexer.CodeChunk, vector []float32) *VectorQuality {
	quality := &VectorQuality{
		LastEvaluated: time.Now(),
	}

	// Calculate embedding quality based on vector properties
	quality.EmbeddingQuality = si.calculateEmbeddingQuality(vector)
	
	// Calculate semantic clarity based on code properties
	quality.SemanticClarity = si.calculateSemanticClarity(chunk)
	
	// Calculate uniqueness by comparing with existing vectors
	quality.Uniqueness = si.calculateUniqueness(vector)
	
	// Calculate consistency with similar code
	quality.Consistency = si.calculateConsistency(chunk, vector)
	
	// Calculate completeness
	quality.Completeness = si.calculateCompleteness(chunk)
	
	// Calculate overall score
	quality.OverallScore = (quality.EmbeddingQuality*0.25 + 
						   quality.SemanticClarity*0.25 + 
						   quality.Uniqueness*0.2 + 
						   quality.Consistency*0.2 + 
						   quality.Completeness*0.1)

	return quality
}

func (si *SemanticIndex) calculateEmbeddingQuality(vector []float32) float32 {
	if len(vector) == 0 {
		return 0.0
	}

	// Calculate vector norm
	var norm float64
	for _, v := range vector {
		norm += float64(v * v)
	}
	norm = math.Sqrt(norm)

	// Quality based on norm (should be close to 1 for normalized vectors)
	normQuality := 1.0 - math.Abs(norm-1.0)
	if normQuality < 0 {
		normQuality = 0
	}

	// Calculate entropy (diversity of values)
	entropy := si.calculateVectorEntropy(vector)
	entropyQuality := math.Min(entropy/5.0, 1.0) // Normalize to 0-1

	// Combine metrics
	quality := (normQuality + entropyQuality) / 2.0
	return float32(quality)
}

func (si *SemanticIndex) calculateVectorEntropy(vector []float32) float64 {
	// Simple entropy calculation based on value distribution
	if len(vector) == 0 {
		return 0
	}

	// Quantize values into buckets
	buckets := make(map[int]int)
	for _, v := range vector {
		bucket := int(v * 10) // Simple quantization
		buckets[bucket]++
	}

	// Calculate entropy
	var entropy float64
	total := float64(len(vector))
	for _, count := range buckets {
		if count > 0 {
			p := float64(count) / total
			entropy -= p * math.Log2(p)
		}
	}

	return entropy
}

func (si *SemanticIndex) calculateSemanticClarity(chunk *indexer.CodeChunk) float32 {
	clarity := float32(0.5) // Base clarity

	// Bonus for having a name
	if chunk.Name != "" {
		clarity += 0.2
	}

	// Bonus for having documentation
	if chunk.DocString != "" {
		clarity += 0.2
	}

	// Bonus for having a signature
	if chunk.Signature != "" {
		clarity += 0.1
	}

	// Adjust for complexity (moderate complexity is clearer)
	if chunk.Complexity >= 3 && chunk.Complexity <= 10 {
		clarity += 0.1
	} else if chunk.Complexity > 15 {
		clarity -= 0.2
	}

	// Ensure valid range
	if clarity > 1.0 {
		clarity = 1.0
	}
	if clarity < 0.0 {
		clarity = 0.0
	}

	return clarity
}

func (si *SemanticIndex) calculateUniqueness(vector []float32) float32 {
	// Compare with a sample of existing vectors to calculate uniqueness
	// This is expensive, so we use a simplified approach
	
	// For now, return a default value
	// In practice, this would compare with existing vectors
	return 0.7
}

func (si *SemanticIndex) calculateConsistency(chunk *indexer.CodeChunk, vector []float32) float32 {
	// Calculate consistency with similar code chunks
	// This is also expensive, so simplified for now
	return 0.8
}

func (si *SemanticIndex) calculateCompleteness(chunk *indexer.CodeChunk) float32 {
	completeness := float32(0.5) // Base completeness

	// Check for various completeness indicators
	if chunk.Content != "" {
		completeness += 0.3
	}

	if len(chunk.Content) > 50 { // Reasonable size
		completeness += 0.1
	}

	if chunk.ChunkType != "" {
		completeness += 0.1
	}

	if completeness > 1.0 {
		completeness = 1.0
	}

	return completeness
}

func (si *SemanticIndex) createPayload(metadata *VectorMetadata) map[string]interface{} {
	payload := make(map[string]interface{})
	
	payload["content_hash"] = metadata.ContentHash
	payload["token_count"] = metadata.TokenCount
	payload["created_at"] = metadata.CreatedAt.Unix()
	payload["updated_at"] = metadata.UpdatedAt.Unix()
	
	// Add all metadata
	for key, value := range metadata.Metadata {
		payload[key] = value
	}
	
	return payload
}

func (si *SemanticIndex) convertFilters(filters *SemanticFilters) *FilterCondition {
	if filters == nil {
		return nil
	}

	filter := &FilterCondition{
		Must: make([]*FieldCondition, 0),
	}

	// Language filter
	if len(filters.Languages) > 0 {
		condition := &FieldCondition{
			Key:   "language",
			Match: filters.Languages[0], // Simplified - would need OR logic
		}
		filter.Must = append(filter.Must, condition)
	}

	// Complexity filter
	if filters.Complexity != nil {
		condition := &FieldCondition{
			Key: "complexity",
			Range: &RangeCondition{
				GTE: filters.Complexity.Min,
				LTE: filters.Complexity.Max,
			},
		}
		filter.Must = append(filter.Must, condition)
	}

	return filter
}

func (si *SemanticIndex) convertToSemanticResult(result *SearchResult, collectionName string) *SemanticSearchResult {
	semanticResult := &SemanticSearchResult{
		ID:             result.ID,
		ChunkID:        result.ID, // Assuming they're the same for now
		Score:          result.Score,
		Vector:         result.Vector,
		CollectionName: collectionName,
	}

	// Find mapping for additional metadata
	if mapping, exists := si.vectorMappings[result.ID]; exists {
		semanticResult.Metadata = mapping.Metadata
		semanticResult.Quality = mapping.Quality
	}

	// Generate explanation
	semanticResult.Explanation = &SearchExplanation{
		MatchType:       "semantic_similarity",
		MatchFactors:    []string{"vector_similarity"},
		SimilarityScore: result.Score,
		QualityBonus:    0.0,
		RelevanceFactors: map[string]float32{
			"similarity": result.Score,
		},
	}

	return semanticResult
}

// Component initialization and management

func (si *SemanticIndex) initializeComponents() error {
	// Initialize caches
	if si.config.EnableCaching {
		si.indexCache = NewIndexCache(si.config.CacheSize, si.config.CacheTTL)
		si.queryCache = NewQueryCache(si.config.CacheSize, si.config.CacheTTL)
	}

	// Initialize bloom filter
	if si.config.EnableBloomFilter {
		si.bloomFilter = NewIndexBloomFilter(si.config.BloomFilterSize, 3)
	}

	// Initialize background components
	si.indexer = NewBackgroundIndexer(si.config.BackgroundWorkers, si.config.BatchSize)
	si.compactor = NewIndexCompactor(si.config.CompactionInterval)
	si.optimizer = NewIndexOptimizer(si.config.OptimizationInterval)

	// Initialize health monitor
	si.healthMonitor = NewIndexHealthMonitor(si.config)

	return nil
}

func (si *SemanticIndex) loadIndex() error {
	// Load existing collections, mappings, and metadata from database
	// This is a simplified implementation
	
	// In practice, this would load from persistent storage
	return nil
}

func (si *SemanticIndex) startBackgroundServices() {
	// Start background indexer
	if si.config.EnableBackgroundIndexing {
		go si.indexer.Start()
	}

	// Start auto-compaction
	if si.config.EnableAutoCompaction {
		go si.runPeriodicCompaction()
	}

	// Start auto-optimization
	if si.config.EnableAutoOptimization {
		go si.runPeriodicOptimization()
	}

	// Start health monitoring
	go si.healthMonitor.Start()
}

func (si *SemanticIndex) runPeriodicCompaction() {
	ticker := time.NewTicker(si.config.CompactionInterval)
	defer ticker.Stop()

	for range ticker.C {
		if time.Since(si.lastCompaction) >= si.config.CompactionInterval {
			si.performCompaction()
		}
	}
}

func (si *SemanticIndex) runPeriodicOptimization() {
	ticker := time.NewTicker(si.config.OptimizationInterval)
	defer ticker.Stop()

	for range ticker.C {
		si.performOptimization()
	}
}

func (si *SemanticIndex) performCompaction() {
	// Implement index compaction
	si.lastCompaction = time.Now()
	si.stats.mu.Lock()
	si.stats.CompactionCount++
	si.stats.mu.Unlock()
}

func (si *SemanticIndex) performOptimization() {
	// Implement index optimization
	si.stats.mu.Lock()
	si.stats.OptimizationCount++
	si.stats.mu.Unlock()
}

// Utility and helper methods

func (si *SemanticIndex) generateMappingID(chunk *indexer.CodeChunk) string {
	return fmt.Sprintf("mapping_%s_%d", chunk.Hash, time.Now().UnixNano())
}

func (si *SemanticIndex) generateVectorID(chunk *indexer.CodeChunk, vector []float32) string {
	// Generate a deterministic ID based on chunk and vector properties
	vectorHash := si.hashVector(vector)
	return fmt.Sprintf("vec_%s_%s", chunk.Hash[:8], vectorHash[:8])
}

func (si *SemanticIndex) hashVector(vector []float32) string {
	// Simple hash of vector for ID generation
	var hash uint64
	for i, v := range vector {
		if i%4 == 0 { // Sample every 4th element
			hash = hash*31 + uint64(math.Float32bits(v))
		}
	}
	return fmt.Sprintf("%016x", hash)
}

func (si *SemanticIndex) validateSearchRequest(request *SemanticSearchRequest) error {
	if request.Query == "" && len(request.QueryVector) == 0 {
		return fmt.Errorf("query or query vector must be provided")
	}

	if request.Limit <= 0 {
		request.Limit = si.config.DefaultSearchLimit
	}

	if request.Limit > si.config.MaxSearchLimit {
		request.Limit = si.config.MaxSearchLimit
	}

	return nil
}

func (si *SemanticIndex) generateSearchCacheKey(request *SemanticSearchRequest) string {
	// Generate a cache key for the search request
	return fmt.Sprintf("search_%s_%d_%f", 
		si.hashString(request.Query), request.Limit, request.Threshold)
}

func (si *SemanticIndex) hashString(s string) string {
	var hash uint64
	for _, c := range s {
		hash = hash*31 + uint64(c)
	}
	return fmt.Sprintf("%016x", hash)
}

func (si *SemanticIndex) getCurrentLoad() float64 {
	// Calculate current system load
	// This is simplified - in practice would consider CPU, memory, queue sizes
	return 0.5
}

func (si *SemanticIndex) updateMetadataIndex(mapping *VectorMapping) {
	// Update metadata index for faster searching
	for key, value := range mapping.Metadata.Metadata {
		if valueStr, ok := value.(string); ok {
			entryKey := fmt.Sprintf("%s:%s", key, valueStr)
			if entry, exists := si.metadataIndex[entryKey]; exists {
				entry.VectorIDs = append(entry.VectorIDs, mapping.VectorID)
				entry.Count++
				entry.LastUpdated = time.Now()
			} else {
				si.metadataIndex[entryKey] = &MetadataEntry{
					Key:         key,
					Value:       valueStr,
					VectorIDs:   []string{mapping.VectorID},
					Count:       1,
					LastUpdated: time.Now(),
				}
			}
		}
	}
}

// Additional helper methods for search operations

func (si *SemanticIndex) postProcessResults(results []*SemanticSearchResult, request *SemanticSearchRequest) []*SemanticSearchResult {
	// Apply additional filtering and ranking
	filtered := make([]*SemanticSearchResult, 0, len(results))
	
	for _, result := range results {
		if result.Score >= request.Threshold {
			filtered = append(filtered, result)
		}
	}
	
	// Sort by score
	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].Score > filtered[j].Score
	})
	
	return filtered
}

func (si *SemanticIndex) calculateResultsQuality(results []*SemanticSearchResult) float64 {
	if len(results) == 0 {
		return 0.0
	}
	
	var totalQuality float64
	for _, result := range results {
		if result.Quality != nil {
			totalQuality += float64(result.Quality.OverallScore)
		} else {
			totalQuality += 0.5 // Default quality
		}
	}
	
	return totalQuality / float64(len(results))
}

func (si *SemanticIndex) generateSearchSuggestions(request *SemanticSearchRequest, results []*SemanticSearchResult) []string {
	// Generate search suggestions based on query and results
	var suggestions []string
	
	// This is simplified - in practice would use more sophisticated suggestion logic
	if len(results) == 0 {
		suggestions = append(suggestions, "Try using different keywords")
		suggestions = append(suggestions, "Check spelling and try again")
	} else if len(results) < request.Limit/4 {
		suggestions = append(suggestions, "Try broader search terms")
		suggestions = append(suggestions, "Remove filters to see more results")
	}
	
	return suggestions
}

// Placeholder methods for complex operations

func (si *SemanticIndex) findSimilarCode(result *SearchResult, limit int) []string {
	// Find similar code chunks
	return []string{}
}

func (si *SemanticIndex) findRelatedVectors(result *SemanticSearchResult, limit int) []*SemanticSearchResult {
	// Find related vectors
	return []*SemanticSearchResult{}
}

func (si *SemanticIndex) deduplicateResults(results []*SemanticSearchResult) []*SemanticSearchResult {
	seen := make(map[string]bool)
	var unique []*SemanticSearchResult
	
	for _, result := range results {
		if !seen[result.ID] {
			seen[result.ID] = true
			unique = append(unique, result)
		}
	}
	
	return unique
}

func (si *SemanticIndex) sortResultsByScore(results []*SemanticSearchResult) {
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
}

func (si *SemanticIndex) searchByMetadata(request *SemanticSearchRequest, collections []string) []*SemanticSearchResult {
	// Search based on metadata
	return []*SemanticSearchResult{}
}

func (si *SemanticIndex) combineAndRerankResults(similarityResults, metadataResults []*SemanticSearchResult, simWeight, metaWeight float64) []*SemanticSearchResult {
	// Combine and re-rank results
	resultMap := make(map[string]*SemanticSearchResult)
	
	// Add similarity results
	for _, result := range similarityResults {
		result.Score = result.Score * float32(simWeight)
		resultMap[result.ID] = result
	}
	
	// Add/merge metadata results
	for _, result := range metadataResults {
		if existing, exists := resultMap[result.ID]; exists {
			existing.Score += result.Score * float32(metaWeight)
		} else {
			result.Score = result.Score * float32(metaWeight)
			resultMap[result.ID] = result
		}
	}
	
	// Convert back to slice and sort
	var combined []*SemanticSearchResult
	for _, result := range resultMap {
		combined = append(combined, result)
	}
	
	si.sortResultsByScore(combined)
	return combined
}

// Statistics methods

func (si *SemanticIndex) updateIndexingStats(duration time.Duration) {
	si.stats.mu.Lock()
	defer si.stats.mu.Unlock()
	
	si.stats.IndexOperations++
	
	if si.stats.AverageIndexTime == 0 {
		si.stats.AverageIndexTime = duration
	} else {
		si.stats.AverageIndexTime = (si.stats.AverageIndexTime + duration) / 2
	}
}

func (si *SemanticIndex) updateSearchStats(duration time.Duration, cacheHit bool) {
	si.stats.mu.Lock()
	defer si.stats.mu.Unlock()
	
	si.stats.SearchOperations++
	
	if si.stats.AverageSearchTime == 0 {
		si.stats.AverageSearchTime = duration
	} else {
		si.stats.AverageSearchTime = (si.stats.AverageSearchTime + duration) / 2
	}
	
	// Update cache stats
	totalOps := si.stats.SearchOperations
	if cacheHit {
		si.stats.CacheHitRate = (si.stats.CacheHitRate*float64(totalOps-1) + 1.0) / float64(totalOps)
	} else {
		si.stats.CacheMissRate = (si.stats.CacheMissRate*float64(totalOps-1) + 1.0) / float64(totalOps)
	}
}

func (si *SemanticIndex) updateUpdateStats(duration time.Duration) {
	si.stats.mu.Lock()
	defer si.stats.mu.Unlock()
	
	si.stats.UpdateOperations++
}

func (si *SemanticIndex) updateDeleteStats(duration time.Duration) {
	si.stats.mu.Lock()
	defer si.stats.mu.Unlock()
	
	si.stats.DeleteOperations++
}

// Public API

func (si *SemanticIndex) GetStatistics() *SemanticIndexStatistics {
	si.stats.mu.RLock()
	defer si.stats.mu.RUnlock()
	
	stats := *si.stats
	stats.LastUpdated = time.Now()
	return &stats
}

func (si *SemanticIndex) GetCollections() map[string]*IndexedCollection {
	si.mu.RLock()
	defer si.mu.RUnlock()
	
	collections := make(map[string]*IndexedCollection)
	for k, v := range si.collections {
		collection := *v
		collections[k] = &collection
	}
	
	return collections
}

func (si *SemanticIndex) GetHealth() map[string]interface{} {
	si.mu.RLock()
	defer si.mu.RUnlock()
	
	health := map[string]interface{}{
		"total_collections": len(si.collections),
		"total_vectors":    len(si.vectorMappings),
		"version":          si.version,
		"last_compaction":  si.lastCompaction,
	}
	
	// Add collection health
	healthyCollections := 0
	for _, collection := range si.collections {
		if collection.Health.Status == HealthStatusHealthy {
			healthyCollections++
		}
	}
	
	health["healthy_collections"] = healthyCollections
	health["health_percentage"] = float64(healthyCollections) / float64(len(si.collections)) * 100
	
	return health
}

func (si *SemanticIndex) GetVersion() int64 {
	si.mu.RLock()
	defer si.mu.RUnlock()
	
	return si.version
}

func (si *SemanticIndex) IncrementVersion() {
	si.mu.Lock()
	defer si.mu.Unlock()
	
	si.version++
}