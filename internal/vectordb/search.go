package vectordb

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	ai "github.com/yourusername/ai-code-assistant/internal/llm"
)

// VectorSearchEngine provides intelligent semantic search capabilities
type VectorSearchEngine struct {
	// Core components
	qdrantClient      *QdrantClient
	embeddingsManager *EmbeddingsManager
	contextRetriever  *ContextRetriever
	rankingAlgorithm  *RankingAlgorithm
	semanticIndex     *SemanticIndex

	// AI providers
	aiProvider ai.Provider

	// Configuration
	config *SearchConfig

	// Caching and optimization
	searchCache      *SearchCache
	queryOptimizer   *QueryOptimizer
	resultAggregator *ResultAggregator

	// Analytics and learning
	searchAnalytics *SearchAnalytics
	learningEngine  *SearchLearningEngine

	// State management
	collections map[string]*SearchCollection
	stats       *SearchStatistics
	mu          sync.RWMutex
}

// SearchConfig contains search engine configuration
type SearchConfig struct {
	// Search behavior
	DefaultCollections []string      `json:"default_collections"`
	MaxResults         int           `json:"max_results"`
	DefaultThreshold   float32       `json:"default_threshold"`
	TimeoutDuration    time.Duration `json:"timeout_duration"`

	// Search modes
	EnableHybridSearch     bool `json:"enable_hybrid_search"`     // Combine vector + keyword
	EnableContextExpansion bool `json:"enable_context_expansion"` // Expand results with context
	EnableRanking          bool `json:"enable_ranking"`           // Use advanced ranking
	EnableCaching          bool `json:"enable_caching"`           // Cache search results
	EnableAnalytics        bool `json:"enable_analytics"`         // Track search analytics

	// Performance tuning
	VectorSearchWeight  float32 `json:"vector_search_weight"`  // Weight for vector similarity
	KeywordSearchWeight float32 `json:"keyword_search_weight"` // Weight for keyword matching
	ContextWindowSize   int     `json:"context_window_size"`   // Lines of context to include
	ParallelQueries     int     `json:"parallel_queries"`      // Number of parallel searches

	// Ranking configuration
	RankingFactors    []RankingFactor `json:"ranking_factors"`
	LearningRate      float64         `json:"learning_rate"`
	AdaptationEnabled bool            `json:"adaptation_enabled"`

	// Cache settings
	CacheSize int           `json:"cache_size"`
	CacheTTL  time.Duration `json:"cache_ttl"`

	// Filter settings
	DefaultFilters map[string]interface{} `json:"default_filters"`
	FilterWeight   float32                `json:"filter_weight"`
}

// SearchRequest represents a search query
type SearchRequest struct {
	// Query information
	Query       string    `json:"query"`
	QueryType   QueryType `json:"query_type"`
	QueryVector []float32 `json:"query_vector,omitempty"`

	// Search parameters
	Collections    []string `json:"collections,omitempty"`
	MaxResults     int      `json:"max_results,omitempty"`
	ScoreThreshold float32  `json:"score_threshold,omitempty"`
	IncludeContext bool     `json:"include_context"`
	ContextSize    int      `json:"context_size,omitempty"`

	// Filtering
	Filters         *SearchFilters `json:"filters,omitempty"`
	LanguageFilters []string       `json:"language_filters,omitempty"`
	FileTypeFilters []string       `json:"file_type_filters,omitempty"`
	TimeRange       *TimeRange     `json:"time_range,omitempty"`

	// Search modes
	SearchMode    SearchMode     `json:"search_mode"`
	RankingMode   RankingMode    `json:"ranking_mode"`
	HybridWeights *HybridWeights `json:"hybrid_weights,omitempty"`

	// User context
	UserID          string   `json:"user_id,omitempty"`
	SessionID       string   `json:"session_id,omitempty"`
	PreviousQueries []string `json:"previous_queries,omitempty"`

	// Metadata
	RequestID string          `json:"request_id"`
	Timestamp time.Time       `json:"timestamp"`
	Context   context.Context `json:"-"`
}

// QueryType represents the type of search query
type QueryType string

const (
	QueryTypeNatural     QueryType = "natural"     // Natural language query
	QueryTypeCode        QueryType = "code"        // Code snippet query
	QueryTypeFunction    QueryType = "function"    // Function signature query
	QueryTypeError       QueryType = "error"       // Error message query
	QueryTypeDescription QueryType = "description" // Functionality description
	QueryTypeHybrid      QueryType = "hybrid"      // Mixed query type
)

// SearchMode represents the search strategy
type SearchMode string

const (
	SearchModeVector   SearchMode = "vector"   // Pure vector similarity search
	SearchModeKeyword  SearchMode = "keyword"  // Traditional keyword search
	SearchModeHybrid   SearchMode = "hybrid"   // Combined vector + keyword
	SearchModeSemantic SearchMode = "semantic" // Enhanced semantic search
	SearchModeExact    SearchMode = "exact"    // Exact matching
)

// RankingMode represents the ranking strategy
type RankingMode string

const (
	RankingModeSimple   RankingMode = "simple"   // Simple similarity-based ranking
	RankingModeAdvanced RankingMode = "advanced" // Multi-factor ranking
	RankingModeLearning RankingMode = "learning" // Machine learning-based ranking
	RankingModePersonal RankingMode = "personal" // Personalized ranking
)

// SearchResponse represents search results
type SearchResponse struct {
	// Results
	Results         []*SearchResult `json:"results"`
	TotalResults    int             `json:"total_results"`
	ResultsReturned int             `json:"results_returned"`
	HasMore         bool            `json:"has_more"`

	// Search metadata
	QueryInfo     *QueryInfo     `json:"query_info"`
	SearchMetrics *SearchMetrics `json:"search_metrics"`
	RankingInfo   *RankingInfo   `json:"ranking_info,omitempty"`

	// Performance metrics
	SearchTime     time.Duration `json:"search_time"`
	ProcessingTime time.Duration `json:"processing_time"`
	CacheHit       bool          `json:"cache_hit"`

	// Suggestions
	Suggestions    []string `json:"suggestions,omitempty"`
	RelatedQueries []string `json:"related_queries,omitempty"`
	DidYouMean     string   `json:"did_you_mean,omitempty"`

	// Metadata
	RequestID string    `json:"request_id"`
	Timestamp time.Time `json:"timestamp"`
}

// SearchResult represents a single search result
type SearchResult struct {
	// Core information
	ID         string  `json:"id"`
	Score      float32 `json:"score"`
	Relevance  float32 `json:"relevance"`
	Confidence float32 `json:"confidence"`

	// Content
	Content string `json:"content"`
	Title   string `json:"title,omitempty"`
	Summary string `json:"summary,omitempty"`
	Snippet string `json:"snippet"`

	// Context
	Context         *ResultContext   `json:"context,omitempty"`
	ExpandedContext *ExpandedContext `json:"expanded_context,omitempty"`

	// Metadata
	Metadata map[string]interface{} `json:"metadata"`
	Source   *SourceInfo            `json:"source"`

	// Ranking information
	RankingFactors  map[string]float32 `json:"ranking_factors,omitempty"`
	MatchType       MatchType          `json:"match_type"`
	MatchHighlights []Highlight        `json:"match_highlights,omitempty"`

	// Vector information
	Vector            []float32 `json:"vector,omitempty"`
	VectorSimilarity  float32   `json:"vector_similarity"`
	KeywordSimilarity float32   `json:"keyword_similarity,omitempty"`

	// Timestamps
	IndexedAt    time.Time `json:"indexed_at"`
	LastModified time.Time `json:"last_modified,omitempty"`
}

// Supporting structures
type SearchFilters struct {
	Languages   []string               `json:"languages,omitempty"`
	FileTypes   []string               `json:"file_types,omitempty"`
	Directories []string               `json:"directories,omitempty"`
	Authors     []string               `json:"authors,omitempty"`
	MinScore    float32                `json:"min_score,omitempty"`
	MaxAge      time.Duration          `json:"max_age,omitempty"`
	Custom      map[string]interface{} `json:"custom,omitempty"`
}

type TimeRange struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

type HybridWeights struct {
	VectorWeight   float32 `json:"vector_weight"`
	KeywordWeight  float32 `json:"keyword_weight"`
	MetadataWeight float32 `json:"metadata_weight"`
	RecencyWeight  float32 `json:"recency_weight"`
}

type QueryInfo struct {
	OriginalQuery    string    `json:"original_query"`
	ProcessedQuery   string    `json:"processed_query"`
	QueryType        QueryType `json:"query_type"`
	DetectedLanguage string    `json:"detected_language,omitempty"`
	QueryComplexity  float32   `json:"query_complexity"`
	KeywordCount     int       `json:"keyword_count"`
	VectorGenerated  bool      `json:"vector_generated"`
}

type SearchMetrics struct {
	VectorSearchTime   time.Duration `json:"vector_search_time"`
	KeywordSearchTime  time.Duration `json:"keyword_search_time,omitempty"`
	RankingTime        time.Duration `json:"ranking_time"`
	ContextTime        time.Duration `json:"context_time,omitempty"`
	FilterTime         time.Duration `json:"filter_time,omitempty"`
	CandidatesFound    int           `json:"candidates_found"`
	CandidatesFiltered int           `json:"candidates_filtered"`
	CandidatesRanked   int           `json:"candidates_ranked"`
}

type SourceInfo struct {
	FilePath     string `json:"file_path"`
	Language     string `json:"language"`
	ChunkType    string `json:"chunk_type"`
	FunctionName string `json:"function_name,omitempty"`
	ClassName    string `json:"class_name,omitempty"`
	StartLine    int    `json:"start_line"`
	EndLine      int    `json:"end_line"`
	Repository   string `json:"repository,omitempty"`
	Branch       string `json:"branch,omitempty"`
}

type MatchType string

const (
	MatchTypeExact    MatchType = "exact"
	MatchTypeSemantic MatchType = "semantic"
	MatchTypePartial  MatchType = "partial"
	MatchTypeFuzzy    MatchType = "fuzzy"
	MatchTypeContext  MatchType = "context"
)

type Highlight struct {
	Start int    `json:"start"`
	End   int    `json:"end"`
	Text  string `json:"text"`
	Type  string `json:"type"`
}

// NewVectorSearchEngine creates a new vector search engine
func NewVectorSearchEngine(
	qdrantClient *QdrantClient,
	embeddingsManager *EmbeddingsManager,
	aiProvider ai.Provider,
	config *SearchConfig,
) (*VectorSearchEngine, error) {
	if config == nil {
		config = &SearchConfig{
			MaxResults:             50,
			DefaultThreshold:       0.7,
			TimeoutDuration:        time.Second * 30,
			EnableHybridSearch:     true,
			EnableContextExpansion: true,
			EnableRanking:          true,
			EnableCaching:          true,
			EnableAnalytics:        true,
			VectorSearchWeight:     0.7,
			KeywordSearchWeight:    0.3,
			ContextWindowSize:      10,
			ParallelQueries:        3,
			CacheSize:              1000,
			CacheTTL:               time.Minute * 15,
			LearningRate:           0.1,
			AdaptationEnabled:      true,
		}
	}

	vse := &VectorSearchEngine{
		qdrantClient:      qdrantClient,
		embeddingsManager: embeddingsManager,
		aiProvider:        aiProvider,
		config:            config,
		collections:       make(map[string]*SearchCollection),
		stats: &SearchStatistics{
			QueryCounts:   make(map[QueryType]int64),
			SearchModes:   make(map[SearchMode]int64),
			ResponseTimes: make(map[string]time.Duration),
		},
	}

	// Initialize components
	vse.initializeComponents()

	return vse, nil
}

// Search performs intelligent semantic search
func (vse *VectorSearchEngine) Search(ctx context.Context, request *SearchRequest) (*SearchResponse, error) {
	start := time.Now()

	// Validate and enrich request
	if err := vse.validateSearchRequest(request); err != nil {
		return nil, fmt.Errorf("invalid search request: %v", err)
	}

	vse.enrichSearchRequest(request)

	// Check cache first
	if vse.config.EnableCaching {
		if cached := vse.searchCache.Get(request); cached != nil {
			cached.CacheHit = true
			vse.updateCacheStats(true)
			return cached, nil
		}
	}

	// Analyze query
	queryInfo, err := vse.analyzeQuery(request)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze query: %v", err)
	}

	// Generate query vector if needed
	var queryVector []float32
	if request.QueryVector != nil {
		queryVector = request.QueryVector
	} else {
		queryVector, err = vse.generateQueryVector(ctx, request.Query)
		if err != nil {
			return nil, fmt.Errorf("failed to generate query vector: %v", err)
		}
	}

	// Perform search based on mode
	var results []*SearchResult
	var searchMetrics *SearchMetrics

	switch request.SearchMode {
	case SearchModeVector:
		results, searchMetrics, err = vse.performVectorSearch(ctx, request, queryVector)
	case SearchModeKeyword:
		results, searchMetrics, err = vse.performKeywordSearch(ctx, request)
	case SearchModeHybrid:
		results, searchMetrics, err = vse.performHybridSearch(ctx, request, queryVector)
	case SearchModeSemantic:
		results, searchMetrics, err = vse.performSemanticSearch(ctx, request, queryVector)
	default:
		results, searchMetrics, err = vse.performHybridSearch(ctx, request, queryVector)
	}

	if err != nil {
		return nil, fmt.Errorf("search failed: %v", err)
	}

	// Apply filters
	if request.Filters != nil {
		results = vse.applyFilters(results, request.Filters)
	}

	// Rank results
	if vse.config.EnableRanking {
		results = vse.rankingAlgorithm.RankResults(results, request, queryInfo)
	}

	// Expand context if requested
	if request.IncludeContext {
		results = vse.expandResultsContext(ctx, results, request)
	}

	// Limit results
	if len(results) > request.MaxResults {
		results = results[:request.MaxResults]
	}

	// Create response
	response := &SearchResponse{
		Results:         results,
		TotalResults:    len(results),
		ResultsReturned: len(results),
		HasMore:         false, // TODO: Implement pagination
		QueryInfo:       queryInfo,
		SearchMetrics:   searchMetrics,
		SearchTime:      time.Since(start),
		RequestID:       request.RequestID,
		Timestamp:       time.Now(),
	}

	// Cache results
	if vse.config.EnableCaching {
		vse.searchCache.Set(request, response)
		vse.updateCacheStats(false)
	}

	// Update analytics
	if vse.config.EnableAnalytics {
		vse.searchAnalytics.RecordSearch(request, response)
	}

	// Update statistics
	vse.updateSearchStats(request, response)

	return response, nil
}

// performVectorSearch performs pure vector similarity search
func (vse *VectorSearchEngine) performVectorSearch(ctx context.Context, request *SearchRequest, queryVector []float32) ([]*SearchResult, *SearchMetrics, error) {
	start := time.Now()

	var allResults []*SearchResult
	metrics := &SearchMetrics{}

	// Search in specified collections
	collections := request.Collections
	if len(collections) == 0 {
		collections = vse.config.DefaultCollections
	}

	for _, collectionName := range collections {
		// Create Qdrant search request
		searchReq := &SearchRequest{
			CollectionName: collectionName,
			Vector:         queryVector,
			Limit:          request.MaxResults * 2, // Get more candidates for ranking
			ScoreThreshold: request.ScoreThreshold,
			WithPayload:    true,
			WithVector:     false, // Don't return vectors to save bandwidth
		}

		// Convert filters if present
		if request.Filters != nil {
			searchReq.Filter = vse.convertToQdrantFilter(request.Filters)
		}

		// Execute search
		qdrantResults, err := vse.qdrantClient.SearchPoints(ctx, searchReq)
		if err != nil {
			return nil, nil, fmt.Errorf("vector search failed: %v", err)
		}

		// Convert results
		for _, result := range qdrantResults {
			searchResult := vse.convertQdrantResult(result, collectionName)
			searchResult.VectorSimilarity = result.Score
			searchResult.MatchType = MatchTypeSemantic
			allResults = append(allResults, searchResult)
		}

		metrics.CandidatesFound += len(qdrantResults)
	}

	metrics.VectorSearchTime = time.Since(start)
	return allResults, metrics, nil
}

// performKeywordSearch performs traditional keyword search
func (vse *VectorSearchEngine) performKeywordSearch(ctx context.Context, request *SearchRequest) ([]*SearchResult, *SearchMetrics, error) {
	start := time.Now()

	// This would integrate with a text search engine like Elasticsearch
	// For now, we'll simulate keyword search using metadata filtering

	var results []*SearchResult
	metrics := &SearchMetrics{
		KeywordSearchTime: time.Since(start),
	}

	// TODO: Implement actual keyword search
	// This could use:
	// - Elasticsearch/OpenSearch
	// - Lucene-based search
	// - SQLite FTS
	// - Custom inverted index

	return results, metrics, nil
}

// performHybridSearch combines vector and keyword search
func (vse *VectorSearchEngine) performHybridSearch(ctx context.Context, request *SearchRequest, queryVector []float32) ([]*SearchResult, *SearchMetrics, error) {
	// Get vector results
	vectorResults, vectorMetrics, err := vse.performVectorSearch(ctx, request, queryVector)
	if err != nil {
		return nil, nil, fmt.Errorf("vector search failed: %v", err)
	}

	// Get keyword results
	keywordResults, keywordMetrics, err := vse.performKeywordSearch(ctx, request)
	if err != nil {
		return nil, nil, fmt.Errorf("keyword search failed: %v", err)
	}

	// Merge and re-rank results
	mergedResults := vse.mergeSearchResults(vectorResults, keywordResults, request.HybridWeights)

	// Combine metrics
	combinedMetrics := &SearchMetrics{
		VectorSearchTime:  vectorMetrics.VectorSearchTime,
		KeywordSearchTime: keywordMetrics.KeywordSearchTime,
		CandidatesFound:   vectorMetrics.CandidatesFound + keywordMetrics.CandidatesFound,
	}

	return mergedResults, combinedMetrics, nil
}

// performSemanticSearch performs enhanced semantic search
func (vse *VectorSearchEngine) performSemanticSearch(ctx context.Context, request *SearchRequest, queryVector []float32) ([]*SearchResult, *SearchMetrics, error) {
	// Start with vector search
	results, metrics, err := vse.performVectorSearch(ctx, request, queryVector)
	if err != nil {
		return nil, nil, err
	}

	// Enhance with semantic understanding
	enhancedResults := vse.semanticIndex.EnhanceResults(results, request)

	return enhancedResults, metrics, nil
}

// Helper methods

func (vse *VectorSearchEngine) initializeComponents() {
	// Initialize context retriever
	vse.contextRetriever = NewContextRetriever(vse.config)

	// Initialize ranking algorithm
	vse.rankingAlgorithm = NewRankingAlgorithm(vse.config)

	// Initialize semantic index
	vse.semanticIndex = NewSemanticIndex(vse.config)

	// Initialize search cache
	if vse.config.EnableCaching {
		vse.searchCache = NewSearchCache(vse.config.CacheSize, vse.config.CacheTTL)
	}

	// Initialize query optimizer
	vse.queryOptimizer = NewQueryOptimizer(vse.config)

	// Initialize result aggregator
	vse.resultAggregator = NewResultAggregator(vse.config)

	// Initialize analytics
	if vse.config.EnableAnalytics {
		vse.searchAnalytics = NewSearchAnalytics(vse.config)
	}

	// Initialize learning engine
	if vse.config.AdaptationEnabled {
		vse.learningEngine = NewSearchLearningEngine(vse.config)
	}
}

func (vse *VectorSearchEngine) validateSearchRequest(request *SearchRequest) error {
	if request.Query == "" && request.QueryVector == nil {
		return fmt.Errorf("query or query vector must be provided")
	}

	if request.MaxResults <= 0 {
		request.MaxResults = vse.config.MaxResults
	}

	if request.ScoreThreshold <= 0 {
		request.ScoreThreshold = vse.config.DefaultThreshold
	}

	if request.RequestID == "" {
		request.RequestID = vse.generateRequestID()
	}

	if request.Timestamp.IsZero() {
		request.Timestamp = time.Now()
	}

	return nil
}

func (vse *VectorSearchEngine) enrichSearchRequest(request *SearchRequest) {
	// Set defaults
	if request.SearchMode == "" {
		if vse.config.EnableHybridSearch {
			request.SearchMode = SearchModeHybrid
		} else {
			request.SearchMode = SearchModeVector
		}
	}

	if request.RankingMode == "" {
		if vse.config.EnableRanking {
			request.RankingMode = RankingModeAdvanced
		} else {
			request.RankingMode = RankingModeSimple
		}
	}

	if request.HybridWeights == nil && request.SearchMode == SearchModeHybrid {
		request.HybridWeights = &HybridWeights{
			VectorWeight:   vse.config.VectorSearchWeight,
			KeywordWeight:  vse.config.KeywordSearchWeight,
			MetadataWeight: 0.1,
			RecencyWeight:  0.1,
		}
	}

	if request.ContextSize == 0 {
		request.ContextSize = vse.config.ContextWindowSize
	}
}

func (vse *VectorSearchEngine) analyzeQuery(request *SearchRequest) (*QueryInfo, error) {
	info := &QueryInfo{
		OriginalQuery:   request.Query,
		ProcessedQuery:  strings.TrimSpace(request.Query),
		QueryType:       request.QueryType,
		KeywordCount:    len(strings.Fields(request.Query)),
		VectorGenerated: request.QueryVector != nil,
	}

	// Detect query type if not specified
	if info.QueryType == "" {
		info.QueryType = vse.detectQueryType(request.Query)
	}

	// Calculate query complexity
	info.QueryComplexity = vse.calculateQueryComplexity(request.Query)

	// Detect programming language if applicable
	if strings.Contains(request.Query, "(") && strings.Contains(request.Query, ")") {
		info.DetectedLanguage = vse.detectProgrammingLanguage(request.Query)
	}

	return info, nil
}

func (vse *VectorSearchEngine) generateQueryVector(ctx context.Context, query string) ([]float32, error) {
	// Use embeddings manager to generate vector
	vector, err := vse.embeddingsManager.GenerateEmbedding(ctx, "query", query, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query vector: %v", err)
	}

	return vector, nil
}

func (vse *VectorSearchEngine) convertQdrantResult(result *SearchResult, collectionName string) *SearchResult {
	searchResult := &SearchResult{
		ID:               result.ID,
		Score:            result.Score,
		Content:          extractContent(result.Payload),
		Metadata:         result.Payload,
		VectorSimilarity: result.Score,
		MatchType:        MatchTypeSemantic,
	}

	// Extract source information from metadata
	if source := extractSourceInfo(result.Payload); source != nil {
		searchResult.Source = source
	}

	// Generate snippet
	searchResult.Snippet = vse.generateSnippet(searchResult.Content, 200)

	return searchResult
}

func (vse *VectorSearchEngine) applyFilters(results []*SearchResult, filters *SearchFilters) []*SearchResult {
	var filtered []*SearchResult

	for _, result := range results {
		if vse.passesFilters(result, filters) {
			filtered = append(filtered, result)
		}
	}

	return filtered
}

func (vse *VectorSearchEngine) passesFilters(result *SearchResult, filters *SearchFilters) bool {
	// Language filter
	if len(filters.Languages) > 0 {
		language := result.Source.Language
		found := false
		for _, lang := range filters.Languages {
			if strings.EqualFold(language, lang) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// File type filter
	if len(filters.FileTypes) > 0 {
		fileType := extractFileType(result.Source.FilePath)
		found := false
		for _, ft := range filters.FileTypes {
			if strings.EqualFold(fileType, ft) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Score threshold
	if filters.MinScore > 0 && result.Score < filters.MinScore {
		return false
	}

	// Age filter
	if filters.MaxAge > 0 {
		age := time.Since(result.LastModified)
		if age > filters.MaxAge {
			return false
		}
	}

	return true
}

func (vse *VectorSearchEngine) expandResultsContext(ctx context.Context, results []*SearchResult, request *SearchRequest) []*SearchResult {
	if !vse.config.EnableContextExpansion {
		return results
	}

	for _, result := range results {
		if request.IncludeContext {
			expandedContext := vse.contextRetriever.GetExpandedContext(result, request.ContextSize)
			result.ExpandedContext = expandedContext
		}
	}

	return results
}

func (vse *VectorSearchEngine) mergeSearchResults(vectorResults, keywordResults []*SearchResult, weights *HybridWeights) []*SearchResult {
	// Create a map to combine results by ID
	resultMap := make(map[string]*SearchResult)

	// Add vector results
	for _, result := range vectorResults {
		result.Score = result.VectorSimilarity * weights.VectorWeight
		resultMap[result.ID] = result
	}

	// Add or merge keyword results
	for _, result := range keywordResults {
		if existing, exists := resultMap[result.ID]; exists {
			// Merge scores
			existing.Score += result.KeywordSimilarity * weights.KeywordWeight
			existing.KeywordSimilarity = result.KeywordSimilarity
		} else {
			result.Score = result.KeywordSimilarity * weights.KeywordWeight
			resultMap[result.ID] = result
		}
	}

	// Convert back to slice and sort by score
	var merged []*SearchResult
	for _, result := range resultMap {
		merged = append(merged, result)
	}

	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Score > merged[j].Score
	})

	return merged
}

// Utility methods

func (vse *VectorSearchEngine) detectQueryType(query string) QueryType {
	query = strings.ToLower(query)

	// Check for function signatures
	if strings.Contains(query, "(") && strings.Contains(query, ")") {
		return QueryTypeFunction
	}

	// Check for error messages
	if strings.Contains(query, "error") || strings.Contains(query, "exception") {
		return QueryTypeError
	}

	// Check for code patterns
	if strings.Contains(query, "func") || strings.Contains(query, "def") ||
		strings.Contains(query, "class") || strings.Contains(query, "import") {
		return QueryTypeCode
	}

	// Check for natural language patterns
	if strings.Contains(query, "how to") || strings.Contains(query, "what is") ||
		strings.Contains(query, "how do") || strings.Contains(query, "can you") {
		return QueryTypeNatural
	}

	return QueryTypeDescription
}

func (vse *VectorSearchEngine) calculateQueryComplexity(query string) float32 {
	// Simple complexity calculation based on various factors
	complexity := float32(0.0)

	// Length factor
	wordCount := len(strings.Fields(query))
	complexity += float32(wordCount) * 0.1

	// Special characters
	if strings.Contains(query, "(") || strings.Contains(query, "{") {
		complexity += 0.2
	}

	// Programming keywords
	progKeywords := []string{"function", "class", "method", "variable", "loop", "condition"}
	for _, keyword := range progKeywords {
		if strings.Contains(strings.ToLower(query), keyword) {
			complexity += 0.1
		}
	}

	// Cap at 1.0
	if complexity > 1.0 {
		complexity = 1.0
	}

	return complexity
}

func (vse *VectorSearchEngine) detectProgrammingLanguage(query string) string {
	query = strings.ToLower(query)

	// Language-specific patterns
	patterns := map[string][]string{
		"go":         {"func", "package", "import", "type", "struct", "interface"},
		"python":     {"def", "class", "import", "from", "lambda", "self"},
		"javascript": {"function", "const", "let", "var", "=>", "async"},
		"java":       {"public", "private", "class", "interface", "extends", "implements"},
		"rust":       {"fn", "struct", "impl", "trait", "use", "mod"},
		"cpp":        {"#include", "class", "namespace", "template", "std::"},
	}

	maxMatches := 0
	detectedLang := ""

	for lang, keywords := range patterns {
		matches := 0
		for _, keyword := range keywords {
			if strings.Contains(query, keyword) {
				matches++
			}
		}
		if matches > maxMatches {
			maxMatches = matches
			detectedLang = lang
		}
	}

	return detectedLang
}

func (vse *VectorSearchEngine) generateSnippet(content string, maxLength int) string {
	if len(content) <= maxLength {
		return content
	}

	// Try to break at word boundaries
	words := strings.Fields(content)
	snippet := ""

	for _, word := range words {
		if len(snippet)+len(word)+1 > maxLength {
			break
		}
		if snippet != "" {
			snippet += " "
		}
		snippet += word
	}

	if len(snippet) < len(content) {
		snippet += "..."
	}

	return snippet
}

func (vse *VectorSearchEngine) generateRequestID() string {
	return fmt.Sprintf("req_%d", time.Now().UnixNano())
}

func (vse *VectorSearchEngine) convertToQdrantFilter(filters *SearchFilters) *FilterCondition {
	// Convert our filters to Qdrant filter format
	// This is a simplified implementation
	filter := &FilterCondition{}

	if len(filters.Languages) > 0 {
		condition := &FieldCondition{
			Key:   "language",
			Match: filters.Languages[0], // Simplified - would need OR logic for multiple
		}
		filter.Must = append(filter.Must, condition)
	}

	return filter
}

// Statistics methods

func (vse *VectorSearchEngine) updateSearchStats(request *SearchRequest, response *SearchResponse) {
	vse.stats.mu.Lock()
	defer vse.stats.mu.Unlock()

	vse.stats.TotalSearches++
	vse.stats.QueryCounts[request.QueryType]++
	vse.stats.SearchModes[request.SearchMode]++

	// Update average response time
	if vse.stats.AverageResponseTime == 0 {
		vse.stats.AverageResponseTime = response.SearchTime
	} else {
		vse.stats.AverageResponseTime = (vse.stats.AverageResponseTime + response.SearchTime) / 2
	}

	vse.stats.ResultsReturned += int64(response.ResultsReturned)
	vse.stats.LastSearchTime = time.Now()
}

func (vse *VectorSearchEngine) updateCacheStats(hit bool) {
	vse.stats.mu.Lock()
	defer vse.stats.mu.Unlock()

	if hit {
		vse.stats.CacheHits++
	} else {
		vse.stats.CacheMisses++
	}

	total := vse.stats.CacheHits + vse.stats.CacheMisses
	if total > 0 {
		vse.stats.CacheHitRate = float64(vse.stats.CacheHits) / float64(total)
	}
}

// Helper functions

func extractContent(payload map[string]interface{}) string {
	if content, exists := payload["content"]; exists {
		if contentStr, ok := content.(string); ok {
			return contentStr
		}
	}
	return ""
}

func extractSourceInfo(payload map[string]interface{}) *SourceInfo {
	source := &SourceInfo{}

	if filePath, exists := payload["file_path"]; exists {
		if fp, ok := filePath.(string); ok {
			source.FilePath = fp
		}
	}

	if language, exists := payload["language"]; exists {
		if lang, ok := language.(string); ok {
			source.Language = lang
		}
	}

	if chunkType, exists := payload["chunk_type"]; exists {
		if ct, ok := chunkType.(string); ok {
			source.ChunkType = ct
		}
	}

	if startLine, exists := payload["start_line"]; exists {
		if sl, ok := startLine.(float64); ok {
			source.StartLine = int(sl)
		}
	}

	if endLine, exists := payload["end_line"]; exists {
		if el, ok := endLine.(float64); ok {
			source.EndLine = int(el)
		}
	}

	return source
}

func extractFileType(filePath string) string {
	parts := strings.Split(filePath, ".")
	if len(parts) > 1 {
		return parts[len(parts)-1]
	}
	return ""
}

// SearchStatistics tracks search engine performance
type SearchStatistics struct {
	TotalSearches       int64                    `json:"total_searches"`
	QueryCounts         map[QueryType]int64      `json:"query_counts"`
	SearchModes         map[SearchMode]int64     `json:"search_modes"`
	AverageResponseTime time.Duration            `json:"average_response_time"`
	ResponseTimes       map[string]time.Duration `json:"response_times"`
	CacheHits           int64                    `json:"cache_hits"`
	CacheMisses         int64                    `json:"cache_misses"`
	CacheHitRate        float64                  `json:"cache_hit_rate"`
	ResultsReturned     int64                    `json:"results_returned"`
	LastSearchTime      time.Time                `json:"last_search_time"`
	mu                  sync.RWMutex
}

// Public API

func (vse *VectorSearchEngine) GetStatistics() *SearchStatistics {
	vse.stats.mu.RLock()
	defer vse.stats.mu.RUnlock()

	stats := *vse.stats
	return &stats
}

func (vse *VectorSearchEngine) GetCollections() map[string]*SearchCollection {
	vse.mu.RLock()
	defer vse.mu.RUnlock()

	collections := make(map[string]*SearchCollection)
	for k, v := range vse.collections {
		collection := *v
		collections[k] = &collection
	}

	return collections
}
