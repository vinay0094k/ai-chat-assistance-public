package vectordb

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// QdrantClient provides high-performance interface to Qdrant vector database
type QdrantClient struct {
	// Core client
	client qdrant.QdrantClient
	conn   *grpc.ClientConn
	config *QdrantConfig

	// Connection management
	pool          *ConnectionPool
	healthChecker *HealthChecker

	// Performance optimization
	batchManager *BatchManager
	queryCache   *QueryCache
	compression  *CompressionEngine

	// Monitoring
	stats   *QdrantStatistics
	metrics *PerformanceMetrics

	// State management
	collections  map[string]*CollectionInfo
	mu           sync.RWMutex
	connected    bool
	reconnecting bool
}

// QdrantConfig contains Qdrant client configuration
type QdrantConfig struct {
	// Connection settings
	Host     string `json:"host"`
	Port     int    `json:"port"`
	UseHTTPS bool   `json:"use_https"`
	APIKey   string `json:"api_key,omitempty"`

	// Connection pooling
	MaxConnections     int           `json:"max_connections"`
	MaxIdleConnections int           `json:"max_idle_connections"`
	ConnectionTimeout  time.Duration `json:"connection_timeout"`
	KeepAlive          time.Duration `json:"keep_alive"`

	// Performance tuning
	BatchSize         int           `json:"batch_size"`
	MaxBatchWait      time.Duration `json:"max_batch_wait"`
	EnableCompression bool          `json:"enable_compression"`
	EnableCaching     bool          `json:"enable_caching"`
	CacheSize         int           `json:"cache_size"`
	CacheTTL          time.Duration `json:"cache_ttl"`

	// Retry and reliability
	MaxRetries          int           `json:"max_retries"`
	RetryDelay          time.Duration `json:"retry_delay"`
	HealthCheckInterval time.Duration `json:"health_check_interval"`
	EnableAutoReconnect bool          `json:"enable_auto_reconnect"`

	// Vector settings
	DefaultVectorSize  int    `json:"default_vector_size"`
	DefaultMetricType  string `json:"default_metric_type"` // cosine, euclidean, dot
	EnableQuantization bool   `json:"enable_quantization"`

	// Indexing settings
	IndexType  string      `json:"index_type"` // hnsw, ivf
	HNSWConfig *HNSWConfig `json:"hnsw_config,omitempty"`
	IVFConfig  *IVFConfig  `json:"ivf_config,omitempty"`
}

// Collection-specific configurations
type CollectionInfo struct {
	Name           string                 `json:"name"`
	VectorSize     int                    `json:"vector_size"`
	MetricType     string                 `json:"metric_type"`
	IndexType      string                 `json:"index_type"`
	Status         string                 `json:"status"`
	PointCount     int64                  `json:"point_count"`
	IndexedVectors int64                  `json:"indexed_vectors"`
	CreatedAt      time.Time              `json:"created_at"`
	UpdatedAt      time.Time              `json:"updated_at"`
	Config         map[string]interface{} `json:"config"`
}

// Index configurations
type HNSWConfig struct {
	M           int     `json:"m"`            // Number of bidirectional links for each node
	EfConstruct int     `json:"ef_construct"` // Size of the dynamic candidate list
	MaxM        int     `json:"max_m"`        // Maximum number of bidirectional links
	MaxM0       int     `json:"max_m0"`       // Maximum number of bidirectional links for layer 0
	MlCoeff     float64 `json:"ml_coeff"`     // Level generation factor
	OnDisk      bool    `json:"on_disk"`      // Store vectors on disk
}

type IVFConfig struct {
	NClusters    int    `json:"n_clusters"`   // Number of clusters
	NProbe       int    `json:"n_probe"`      // Number of clusters to search
	Quantization string `json:"quantization"` // none, scalar, product
	OnDisk       bool   `json:"on_disk"`      // Store vectors on disk
}

// Vector operations
type VectorPoint struct {
	ID      string                 `json:"id"`
	Vector  []float32              `json:"vector"`
	Payload map[string]interface{} `json:"payload"`
	Version int64                  `json:"version,omitempty"`
}

type SearchRequest struct {
	CollectionName string                 `json:"collection_name"`
	Vector         []float32              `json:"vector"`
	Limit          int                    `json:"limit"`
	Offset         int                    `json:"offset"`
	Filter         *FilterCondition       `json:"filter,omitempty"`
	WithPayload    bool                   `json:"with_payload"`
	WithVector     bool                   `json:"with_vector"`
	ScoreThreshold float32                `json:"score_threshold,omitempty"`
	SearchParams   map[string]interface{} `json:"search_params,omitempty"`
}

type SearchResult struct {
	ID      string                 `json:"id"`
	Score   float32                `json:"score"`
	Vector  []float32              `json:"vector,omitempty"`
	Payload map[string]interface{} `json:"payload,omitempty"`
	Version int64                  `json:"version,omitempty"`
}

type FilterCondition struct {
	Must    []*FieldCondition `json:"must,omitempty"`
	MustNot []*FieldCondition `json:"must_not,omitempty"`
	Should  []*FieldCondition `json:"should,omitempty"`
}

type FieldCondition struct {
	Key       string              `json:"key"`
	Match     interface{}         `json:"match,omitempty"`
	Range     *RangeCondition     `json:"range,omitempty"`
	GeoRadius *GeoRadiusCondition `json:"geo_radius,omitempty"`
}

type RangeCondition struct {
	LT  *float64 `json:"lt,omitempty"`
	GT  *float64 `json:"gt,omitempty"`
	LTE *float64 `json:"lte,omitempty"`
	GTE *float64 `json:"gte,omitempty"`
}

type GeoRadiusCondition struct {
	Center GeoPoint `json:"center"`
	Radius float64  `json:"radius"`
}

type GeoPoint struct {
	Lat float64 `json:"lat"`
	Lon float64 `json:"lon"`
}

// Statistics and monitoring
type QdrantStatistics struct {
	// Connection stats
	TotalConnections  int64 `json:"total_connections"`
	ActiveConnections int64 `json:"active_connections"`
	FailedConnections int64 `json:"failed_connections"`
	Reconnections     int64 `json:"reconnections"`

	// Operation stats
	SearchOperations int64 `json:"search_operations"`
	UpsertOperations int64 `json:"upsert_operations"`
	DeleteOperations int64 `json:"delete_operations"`
	BatchOperations  int64 `json:"batch_operations"`

	// Performance metrics
	AvgSearchLatency time.Duration `json:"avg_search_latency"`
	AvgUpsertLatency time.Duration `json:"avg_upsert_latency"`
	TotalSearchTime  time.Duration `json:"total_search_time"`
	TotalUpsertTime  time.Duration `json:"total_upsert_time"`

	// Cache statistics
	CacheHits    int64   `json:"cache_hits"`
	CacheMisses  int64   `json:"cache_misses"`
	CacheHitRate float64 `json:"cache_hit_rate"`

	// Error tracking
	Errors        int64     `json:"errors"`
	Timeouts      int64     `json:"timeouts"`
	LastError     string    `json:"last_error,omitempty"`
	LastErrorTime time.Time `json:"last_error_time,omitempty"`

	// Resource usage
	MemoryUsage     int64 `json:"memory_usage"`
	DiskUsage       int64 `json:"disk_usage"`
	NetworkBytesIn  int64 `json:"network_bytes_in"`
	NetworkBytesOut int64 `json:"network_bytes_out"`

	mu sync.RWMutex
}

// NewQdrantClient creates a new Qdrant client
func NewQdrantClient(config *QdrantConfig) (*QdrantClient, error) {
	if config == nil {
		config = &QdrantConfig{
			Host:                "localhost",
			Port:                6334,
			MaxConnections:      10,
			MaxIdleConnections:  5,
			ConnectionTimeout:   time.Second * 30,
			KeepAlive:           time.Minute * 5,
			BatchSize:           100,
			MaxBatchWait:        time.Millisecond * 100,
			EnableCompression:   true,
			EnableCaching:       true,
			CacheSize:           1000,
			CacheTTL:            time.Minute * 15,
			MaxRetries:          3,
			RetryDelay:          time.Second * 2,
			HealthCheckInterval: time.Minute,
			EnableAutoReconnect: true,
			DefaultVectorSize:   384,
			DefaultMetricType:   "cosine",
			IndexType:           "hnsw",
		}
	}

	qc := &QdrantClient{
		config:      config,
		collections: make(map[string]*CollectionInfo),
		stats:       &QdrantStatistics{},
	}

	// Initialize components
	qc.initializeComponents()

	// Establish connection
	if err := qc.Connect(); err != nil {
		return nil, fmt.Errorf("failed to connect to Qdrant: %v", err)
	}

	// Start background services
	qc.startBackgroundServices()

	return qc, nil
}

// Connect establishes connection to Qdrant
func (qc *QdrantClient) Connect() error {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	if qc.connected {
		return nil
	}

	// Create connection
	address := fmt.Sprintf("%s:%d", qc.config.Host, qc.config.Port)

	var opts []grpc.DialOption
	if !qc.config.UseHTTPS {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	ctx, cancel := context.WithTimeout(context.Background(), qc.config.ConnectionTimeout)
	defer cancel()

	conn, err := grpc.DialContext(ctx, address, opts...)
	if err != nil {
		qc.stats.mu.Lock()
		qc.stats.FailedConnections++
		qc.stats.mu.Unlock()
		return fmt.Errorf("failed to dial Qdrant: %v", err)
	}

	qc.conn = conn
	qc.client = qdrant.NewQdrantClient(conn)
	qc.connected = true

	// Update statistics
	qc.stats.mu.Lock()
	qc.stats.TotalConnections++
	qc.stats.ActiveConnections++
	qc.stats.mu.Unlock()

	// Load existing collections
	if err := qc.loadCollections(); err != nil {
		return fmt.Errorf("failed to load collections: %v", err)
	}

	fmt.Printf("Connected to Qdrant at %s\n", address)
	return nil
}

// Disconnect closes connection to Qdrant
func (qc *QdrantClient) Disconnect() error {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	if !qc.connected {
		return nil
	}

	if qc.conn != nil {
		if err := qc.conn.Close(); err != nil {
			return fmt.Errorf("failed to close connection: %v", err)
		}
	}

	qc.connected = false
	qc.client = nil
	qc.conn = nil

	// Update statistics
	qc.stats.mu.Lock()
	qc.stats.ActiveConnections--
	qc.stats.mu.Unlock()

	fmt.Println("Disconnected from Qdrant")
	return nil
}

// Collection Management

// CreateCollection creates a new collection
func (qc *QdrantClient) CreateCollection(ctx context.Context, name string, vectorSize int, metricType string) error {
	if !qc.connected {
		return fmt.Errorf("not connected to Qdrant")
	}

	// Convert metric type
	var distance qdrant.Distance
	switch metricType {
	case "cosine":
		distance = qdrant.Distance_Cosine
	case "euclidean":
		distance = qdrant.Distance_Euclid
	case "dot":
		distance = qdrant.Distance_Dot
	default:
		return fmt.Errorf("unsupported metric type: %s", metricType)
	}

	// Create vector params
	vectorParams := &qdrant.VectorParams{
		Size:     uint64(vectorSize),
		Distance: distance,
	}

	// Create HNSW config if specified
	var hnswConfig *qdrant.HnswConfigDiff
	if qc.config.HNSWConfig != nil {
		hnswConfig = &qdrant.HnswConfigDiff{
			M:           uint64(qc.config.HNSWConfig.M),
			EfConstruct: uint64(qc.config.HNSWConfig.EfConstruct),
			MaxM:        uint64(qc.config.HNSWConfig.MaxM),
			MaxM0:       uint64(qc.config.HNSWConfig.MaxM0),
			OnDisk:      &qc.config.HNSWConfig.OnDisk,
		}
	}

	// Create collection request
	req := &qdrant.CreateCollection{
		CollectionName: name,
		VectorsConfig: &qdrant.VectorsConfig{
			Config: &qdrant.VectorsConfig_Params{
				Params: vectorParams,
			},
		},
		HnswConfig: hnswConfig,
	}

	// Execute creation
	_, err := qc.client.CreateCollection(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to create collection: %v", err)
	}

	// Add to local collection info
	qc.mu.Lock()
	qc.collections[name] = &CollectionInfo{
		Name:       name,
		VectorSize: vectorSize,
		MetricType: metricType,
		IndexType:  qc.config.IndexType,
		Status:     "created",
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}
	qc.mu.Unlock()

	fmt.Printf("Created collection: %s\n", name)
	return nil
}

// DeleteCollection deletes a collection
func (qc *QdrantClient) DeleteCollection(ctx context.Context, name string) error {
	if !qc.connected {
		return fmt.Errorf("not connected to Qdrant")
	}

	req := &qdrant.DeleteCollection{
		CollectionName: name,
	}

	_, err := qc.client.DeleteCollection(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to delete collection: %v", err)
	}

	// Remove from local collection info
	qc.mu.Lock()
	delete(qc.collections, name)
	qc.mu.Unlock()

	fmt.Printf("Deleted collection: %s\n", name)
	return nil
}

// GetCollectionInfo gets information about a collection
func (qc *QdrantClient) GetCollectionInfo(ctx context.Context, name string) (*CollectionInfo, error) {
	if !qc.connected {
		return nil, fmt.Errorf("not connected to Qdrant")
	}

	req := &qdrant.GetCollectionInfoRequest{
		CollectionName: name,
	}

	resp, err := qc.client.GetCollectionInfo(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to get collection info: %v", err)
	}

	// Convert response to CollectionInfo
	info := &CollectionInfo{
		Name:           name,
		Status:         resp.Result.Status.String(),
		PointCount:     int64(resp.Result.PointsCount),
		IndexedVectors: int64(resp.Result.IndexedVectorsCount),
		UpdatedAt:      time.Now(),
	}

	// Update local collection info
	qc.mu.Lock()
	if existing, exists := qc.collections[name]; exists {
		info.CreatedAt = existing.CreatedAt
		info.VectorSize = existing.VectorSize
		info.MetricType = existing.MetricType
		info.IndexType = existing.IndexType
	}
	qc.collections[name] = info
	qc.mu.Unlock()

	return info, nil
}

// Vector Operations

// UpsertPoints upserts vector points
func (qc *QdrantClient) UpsertPoints(ctx context.Context, collectionName string, points []*VectorPoint) error {
	if !qc.connected {
		return fmt.Errorf("not connected to Qdrant")
	}

	start := time.Now()

	// Convert points to Qdrant format
	qdrantPoints := make([]*qdrant.PointStruct, len(points))
	for i, point := range points {
		// Convert payload
		payloadData, err := json.Marshal(point.Payload)
		if err != nil {
			return fmt.Errorf("failed to marshal payload: %v", err)
		}

		var payload map[string]*qdrant.Value
		if err := json.Unmarshal(payloadData, &payload); err != nil {
			return fmt.Errorf("failed to convert payload: %v", err)
		}

		qdrantPoints[i] = &qdrant.PointStruct{
			Id: &qdrant.PointId{
				PointIdOptions: &qdrant.PointId_Uuid{
					Uuid: point.ID,
				},
			},
			Vectors: &qdrant.Vectors{
				VectorsOptions: &qdrant.Vectors_Vector{
					Vector: &qdrant.Vector{
						Data: point.Vector,
					},
				},
			},
			Payload: payload,
		}
	}

	// Create upsert request
	req := &qdrant.UpsertPoints{
		CollectionName: collectionName,
		Points:         qdrantPoints,
		Wait:           true,
	}

	// Execute upsert
	_, err := qc.client.Upsert(ctx, req)
	if err != nil {
		qc.updateErrorStats(err)
		return fmt.Errorf("failed to upsert points: %v", err)
	}

	// Update statistics
	qc.stats.mu.Lock()
	qc.stats.UpsertOperations++
	duration := time.Since(start)
	qc.stats.TotalUpsertTime += duration
	qc.stats.AvgUpsertLatency = qc.stats.TotalUpsertTime / time.Duration(qc.stats.UpsertOperations)
	qc.stats.mu.Unlock()

	return nil
}

// SearchPoints performs vector similarity search
func (qc *QdrantClient) SearchPoints(ctx context.Context, req *SearchRequest) ([]*SearchResult, error) {
	if !qc.connected {
		return nil, fmt.Errorf("not connected to Qdrant")
	}

	start := time.Now()

	// Check cache first
	if qc.config.EnableCaching {
		if cached := qc.queryCache.Get(req); cached != nil {
			qc.stats.mu.Lock()
			qc.stats.CacheHits++
			qc.stats.mu.Unlock()
			return cached, nil
		}
	}

	// Convert filter if present
	var filter *qdrant.Filter
	if req.Filter != nil {
		filter = qc.convertFilter(req.Filter)
	}

	// Create search request
	searchReq := &qdrant.SearchPoints{
		CollectionName: req.CollectionName,
		Vector:         req.Vector,
		Limit:          uint64(req.Limit),
		Offset:         uint64(req.Offset),
		WithPayload:    &qdrant.WithPayloadSelector{SelectorOptions: &qdrant.WithPayloadSelector_Enable{Enable: req.WithPayload}},
		WithVectors:    &qdrant.WithVectorsSelector{SelectorOptions: &qdrant.WithVectorsSelector_Enable{Enable: req.WithVector}},
		Filter:         filter,
	}

	if req.ScoreThreshold > 0 {
		searchReq.ScoreThreshold = &req.ScoreThreshold
	}

	// Execute search
	resp, err := qc.client.Search(ctx, searchReq)
	if err != nil {
		qc.updateErrorStats(err)
		return nil, fmt.Errorf("failed to search points: %v", err)
	}

	// Convert results
	results := make([]*SearchResult, len(resp.Result))
	for i, point := range resp.Result {
		result := &SearchResult{
			ID:    point.Id.GetUuid(),
			Score: point.Score,
		}

		if req.WithVector && point.Vectors != nil {
			if vectorData := point.Vectors.GetVector(); vectorData != nil {
				result.Vector = vectorData.Data
			}
		}

		if req.WithPayload && point.Payload != nil {
			payloadMap := make(map[string]interface{})
			for key, value := range point.Payload {
				payloadMap[key] = qc.convertValue(value)
			}
			result.Payload = payloadMap
		}

		results[i] = result
	}

	// Cache results
	if qc.config.EnableCaching {
		qc.queryCache.Set(req, results)
		qc.stats.mu.Lock()
		qc.stats.CacheMisses++
		qc.stats.mu.Unlock()
	}

	// Update statistics
	qc.stats.mu.Lock()
	qc.stats.SearchOperations++
	duration := time.Since(start)
	qc.stats.TotalSearchTime += duration
	qc.stats.AvgSearchLatency = qc.stats.TotalSearchTime / time.Duration(qc.stats.SearchOperations)

	// Update cache hit rate
	totalCacheOps := qc.stats.CacheHits + qc.stats.CacheMisses
	if totalCacheOps > 0 {
		qc.stats.CacheHitRate = float64(qc.stats.CacheHits) / float64(totalCacheOps)
	}
	qc.stats.mu.Unlock()

	return results, nil
}

// DeletePoints deletes points by IDs
func (qc *QdrantClient) DeletePoints(ctx context.Context, collectionName string, ids []string) error {
	if !qc.connected {
		return fmt.Errorf("not connected to Qdrant")
	}

	// Convert IDs to Qdrant format
	pointIds := make([]*qdrant.PointId, len(ids))
	for i, id := range ids {
		pointIds[i] = &qdrant.PointId{
			PointIdOptions: &qdrant.PointId_Uuid{
				Uuid: id,
			},
		}
	}

	// Create delete request
	req := &qdrant.DeletePoints{
		CollectionName: collectionName,
		Points: &qdrant.PointsSelector{
			PointsSelectorOneOf: &qdrant.PointsSelector_Points{
				Points: &qdrant.PointsIdsList{
					Ids: pointIds,
				},
			},
		},
		Wait: true,
	}

	// Execute delete
	_, err := qc.client.Delete(ctx, req)
	if err != nil {
		qc.updateErrorStats(err)
		return fmt.Errorf("failed to delete points: %v", err)
	}

	// Update statistics
	qc.stats.mu.Lock()
	qc.stats.DeleteOperations++
	qc.stats.mu.Unlock()

	return nil
}

// Helper methods

func (qc *QdrantClient) initializeComponents() {
	// Initialize connection pool
	qc.pool = NewConnectionPool(qc.config)

	// Initialize health checker
	qc.healthChecker = NewHealthChecker(qc, qc.config.HealthCheckInterval)

	// Initialize batch manager
	qc.batchManager = NewBatchManager(qc.config.BatchSize, qc.config.MaxBatchWait)

	// Initialize query cache
	if qc.config.EnableCaching {
		qc.queryCache = NewQueryCache(qc.config.CacheSize, qc.config.CacheTTL)
	}

	// Initialize compression engine
	if qc.config.EnableCompression {
		qc.compression = NewCompressionEngine(6) // Default compression level
	}

	// Initialize performance metrics
	qc.metrics = NewPerformanceMetrics()
}

func (qc *QdrantClient) startBackgroundServices() {
	// Start health checker
	if qc.config.EnableAutoReconnect {
		go qc.healthChecker.Start()
	}

	// Start metrics collection
	go qc.metrics.Start()
}

func (qc *QdrantClient) loadCollections() error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()

	req := &qdrant.ListCollectionsRequest{}
	resp, err := qc.client.ListCollections(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to list collections: %v", err)
	}

	for _, collection := range resp.Collections {
		info := &CollectionInfo{
			Name:      collection.Name,
			Status:    "loaded",
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		qc.collections[collection.Name] = info
	}

	fmt.Printf("Loaded %d collections\n", len(resp.Collections))
	return nil
}

func (qc *QdrantClient) convertFilter(filter *FilterCondition) *qdrant.Filter {
	// This would convert our filter format to Qdrant's filter format
	// Implementation depends on specific filter requirements
	return nil
}

func (qc *QdrantClient) convertValue(value *qdrant.Value) interface{} {
	switch v := value.Kind.(type) {
	case *qdrant.Value_StringValue:
		return v.StringValue
	case *qdrant.Value_IntegerValue:
		return v.IntegerValue
	case *qdrant.Value_DoubleValue:
		return v.DoubleValue
	case *qdrant.Value_BoolValue:
		return v.BoolValue
	default:
		return nil
	}
}

func (qc *QdrantClient) updateErrorStats(err error) {
	qc.stats.mu.Lock()
	defer qc.stats.mu.Unlock()

	qc.stats.Errors++
	qc.stats.LastError = err.Error()
	qc.stats.LastErrorTime = time.Now()
}

// Public API

func (qc *QdrantClient) IsConnected() bool {
	qc.mu.RLock()
	defer qc.mu.RUnlock()
	return qc.connected
}

func (qc *QdrantClient) GetStatistics() *QdrantStatistics {
	qc.stats.mu.RLock()
	defer qc.stats.mu.RUnlock()

	stats := *qc.stats
	return &stats
}

func (qc *QdrantClient) GetCollections() map[string]*CollectionInfo {
	qc.mu.RLock()
	defer qc.mu.RUnlock()

	collections := make(map[string]*CollectionInfo)
	for k, v := range qc.collections {
		info := *v
		collections[k] = &info
	}

	return collections
}

func (qc *QdrantClient) GetHealth() map[string]interface{} {
	return map[string]interface{}{
		"connected":    qc.IsConnected(),
		"reconnecting": qc.reconnecting,
		"collections":  len(qc.collections),
		"last_error":   qc.stats.LastError,
		"error_count":  qc.stats.Errors,
	}
}
