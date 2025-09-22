package agents

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/app"
	"github.com/yourusername/ai-code-assistant/internal/indexer"
	"github.com/yourusername/ai-code-assistant/internal/logger"
	"github.com/yourusername/ai-code-assistant/internal/vectordb"
)

// Component interfaces for dependency injection
type TextSearchEngine interface {
	Search(query string, options *SearchOptions) ([]*SearchResult, error)
}

type QueryAnalyzer interface {
	AnalyzeQuery(query string) (*QueryAnalysis, error)
}

type SearchResultRanker interface {
	RankResults(results []*SearchResult, query string) []*SearchResult
}

type SearchCache interface {
	Get(key string) ([]*SearchResult, bool)
	Set(key string, results []*SearchResult)
	Clear()
}

// SearchAgent performs intelligent searches within the codebase
type SearchAgent struct {
	// Core components
	indexer        *indexer.UltraFastIndexer
	vectorDB       *vectordb.VectorDB
	contextManager *app.ContextManager

	//Injected dependencies
	textSearchEngine TextSearchEngine
	queryAnalyzer    QueryAnalyzer
	resultRanker     SearchResultRanker
	searchCache      SearchCache

	// Agent configuration
	config *SearchAgentConfig
	logger logger.Logger

	// Capabilities
	capabilities *AgentCapabilities

	// Search engines
	textSearchEngine     *TextSearchEngine
	semanticSearchEngine *SemanticSearchEngine
	codeSearchEngine     *CodeSearchEngine
	symbolSearchEngine   *SymbolSearchEngine

	// Query processing
	queryAnalyzer *QueryAnalyzer
	queryExpander *QueryExpander
	resultRanker  *SearchResultRanker

	// Performance optimization
	searchCache *SearchCache
	indexCache  *IndexCache

	// Statistics and monitoring
	metrics *SearchAgentMetrics

	// State management
	mu     sync.RWMutex
	status AgentStatus
}

// SearchAgentConfig contains search agent configuration
type SearchAgentConfig struct {
	// Search behavior
	EnableTextSearch     bool `json:"enable_text_search"`
	EnableSemanticSearch bool `json:"enable_semantic_search"`
	EnableCodeSearch     bool `json:"enable_code_search"`
	EnableSymbolSearch   bool `json:"enable_symbol_search"`
	EnableFuzzySearch    bool `json:"enable_fuzzy_search"`

	// Search limits and performance
	MaxResults           int           `json:"max_results"`
	DefaultResultLimit   int           `json:"default_result_limit"`
	SearchTimeout        time.Duration `json:"search_timeout"`
	EnableParallelSearch bool          `json:"enable_parallel_search"`

	// Query processing
	EnableQueryExpansion   bool `json:"enable_query_expansion"`
	EnableSpellCheck       bool `json:"enable_spell_check"`
	EnableSynonymExpansion bool `json:"enable_synonym_expansion"`
	EnableAutoComplete     bool `json:"enable_auto_complete"`

	// Result ranking
	EnableRanking         bool            `json:"enable_ranking"`
	RankingWeights        *RankingWeights `json:"ranking_weights"`
	EnablePersonalization bool            `json:"enable_personalization"`

	// Caching
	EnableCaching bool          `json:"enable_caching"`
	CacheSize     int           `json:"cache_size"`
	CacheTTL      time.Duration `json:"cache_ttl"`

	// File filtering
	IncludedFileTypes   []string `json:"included_file_types"`
	ExcludedFileTypes   []string `json:"excluded_file_types"`
	IncludedDirectories []string `json:"included_directories"`
	ExcludedDirectories []string `json:"excluded_directories"`

	// Advanced search
	EnableRegexSearch   bool `json:"enable_regex_search"`
	EnableCaseSensitive bool `json:"enable_case_sensitive"`
	EnableWholeWord     bool `json:"enable_whole_word"`
}

// SearchRequest represents a search request
type SearchRequest struct {
	Query      string         `json:"query"`
	SearchType SearchType     `json:"search_type"`
	Filters    *SearchFilters `json:"filters,omitempty"`
	Options    *SearchOptions `json:"options,omitempty"`
	Context    *SearchContext `json:"context,omitempty"`
	Limit      int            `json:"limit"`
	Offset     int            `json:"offset"`
}

// SearchResponse represents search results
type SearchResponse struct {
	Results        []*SearchResult        `json:"results"`
	TotalFound     int                    `json:"total_found"`
	QueryTime      time.Duration          `json:"query_time"`
	SearchStrategy string                 `json:"search_strategy"`
	Suggestions    []string               `json:"suggestions,omitempty"`
	RelatedQueries []string               `json:"related_queries,omitempty"`
	Facets         map[string][]FacetItem `json:"facets,omitempty"`
}

type SearchType string

const (
	SearchTypeText     SearchType = "text"
	SearchTypeSemantic SearchType = "semantic"
	SearchTypeCode     SearchType = "code"
	SearchTypeSymbol   SearchType = "symbol"
	SearchTypeRegex    SearchType = "regex"
	SearchTypeAuto     SearchType = "auto" // Automatically determine best search type
)

type SearchFilters struct {
	FileTypes   []string     `json:"file_types,omitempty"`
	Languages   []string     `json:"languages,omitempty"`
	Directories []string     `json:"directories,omitempty"`
	Authors     []string     `json:"authors,omitempty"`
	DateRange   *DateRange   `json:"date_range,omitempty"`
	SizeRange   *SizeRange   `json:"size_range,omitempty"`
	Complexity  *RangeFilter `json:"complexity,omitempty"`
	ChunkTypes  []string     `json:"chunk_types,omitempty"`
}

type SearchOptions struct {
	CaseSensitive  bool         `json:"case_sensitive"`
	WholeWord      bool         `json:"whole_word"`
	Fuzzy          bool         `json:"fuzzy"`
	FuzzyThreshold float32      `json:"fuzzy_threshold"`
	IncludeContext bool         `json:"include_context"`
	ContextLines   int          `json:"context_lines"`
	Highlight      bool         `json:"highlight"`
	SortBy         SortCriteria `json:"sort_by"`
	SortOrder      SortOrder    `json:"sort_order"`
}

type SearchContext struct {
	WorkingDirectory string                 `json:"working_directory,omitempty"`
	CurrentFile      string                 `json:"current_file,omitempty"`
	RecentFiles      []string               `json:"recent_files,omitempty"`
	UserPreferences  map[string]interface{} `json:"user_preferences,omitempty"`
}

type SearchResult struct {
	ID           string                 `json:"id"`
	FilePath     string                 `json:"file_path"`
	LineNumber   int                    `json:"line_number"`
	ColumnNumber int                    `json:"column_number"`
	Content      string                 `json:"content"`
	MatchedText  string                 `json:"matched_text"`
	Context      *ResultContext         `json:"context,omitempty"`
	Score        float32                `json:"score"`
	Relevance    float32                `json:"relevance"`
	Highlights   []Highlight            `json:"highlights,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	Type         ResultType             `json:"type"`
}

type ResultContext struct {
	BeforeLines []string `json:"before_lines,omitempty"`
	AfterLines  []string `json:"after_lines,omitempty"`
	Function    string   `json:"function,omitempty"`
	Class       string   `json:"class,omitempty"`
	Scope       string   `json:"scope,omitempty"`
}

type Highlight struct {
	Start int    `json:"start"`
	End   int    `json:"end"`
	Text  string `json:"text"`
	Type  string `json:"type"`
}

type ResultType string

const (
	ResultTypeCode     ResultType = "code"
	ResultTypeFunction ResultType = "function"
	ResultTypeClass    ResultType = "class"
	ResultTypeVariable ResultType = "variable"
	ResultTypeComment  ResultType = "comment"
	ResultTypeString   ResultType = "string"
	ResultTypeImport   ResultType = "import"
)

type DateRange struct {
	Start *time.Time `json:"start,omitempty"`
	End   *time.Time `json:"end,omitempty"`
}

type SizeRange struct {
	Min *int64 `json:"min,omitempty"`
	Max *int64 `json:"max,omitempty"`
}

type SortCriteria string

const (
	SortByRelevance SortCriteria = "relevance"
	SortByDate      SortCriteria = "date"
	SortBySize      SortCriteria = "size"
	SortByName      SortCriteria = "name"
	SortByPath      SortCriteria = "path"
	SortByScore     SortCriteria = "score"
)

type SortOrder string

const (
	SortOrderAsc  SortOrder = "asc"
	SortOrderDesc SortOrder = "desc"
)

type FacetItem struct {
	Value string `json:"value"`
	Count int    `json:"count"`
}

// SearchAgentMetrics tracks search performance
type SearchAgentMetrics struct {
	TotalSearches      int64                `json:"total_searches"`
	SearchesByType     map[SearchType]int64 `json:"searches_by_type"`
	AverageSearchTime  time.Duration        `json:"average_search_time"`
	CacheHitRate       float64              `json:"cache_hit_rate"`
	AverageResultCount float64              `json:"average_result_count"`
	PopularQueries     map[string]int64     `json:"popular_queries"`
	SuccessRate        float64              `json:"success_rate"`
	LastSearch         time.Time            `json:"last_search"`
	mu                 sync.RWMutex
}

// NewSearchAgent creates a new search agent
func NewSearchAgent(
	indexer *indexer.UltraFastIndexer,
	vectorDB *vectordb.VectorDB,
	config *SearchAgentConfig,
	logger logger.Logger,
	textSearchEngine TextSearchEngine,
	queryAnalyzer QueryAnalyzer,
	resultRanker SearchResultRanker,
	searchCache SearchCache,
) *SearchAgent {

	if config == nil {
		config = &SearchAgentConfig{
			EnableTextSearch:       true,
			EnableSemanticSearch:   true,
			EnableCodeSearch:       true,
			EnableSymbolSearch:     true,
			EnableFuzzySearch:      true,
			MaxResults:             1000,
			DefaultResultLimit:     50,
			SearchTimeout:          time.Second * 10,
			EnableParallelSearch:   true,
			EnableQueryExpansion:   true,
			EnableSpellCheck:       true,
			EnableSynonymExpansion: true,
			EnableAutoComplete:     true,
			EnableRanking:          true,
			EnablePersonalization:  false,
			EnableCaching:          true,
			CacheSize:              1000,
			CacheTTL:               time.Minute * 15,
			EnableRegexSearch:      true,
			EnableCaseSensitive:    false,
			EnableWholeWord:        false,
			RankingWeights: &RankingWeights{
				Relevance:  0.4,
				Freshness:  0.2,
				Popularity: 0.2,
				Context:    0.2,
			},
		}
	}

	agent := &SearchAgent{
		indexer:          indexer,
		vectorDB:         vectorDB,
		config:           config,
		logger:           logger,
		textSearchEngine: textSearchEngine,
		queryAnalyzer:    queryAnalyzer,
		resultRanker:     resultRanker,
		searchCache:      searchCache,
		status:           StatusIdle,
		metrics: &SearchAgentMetrics{
			SearchesByType: make(map[SearchType]int64),
			PopularQueries: make(map[string]int64),
		},
	}

	// Initialize capabilities
	agent.initializeCapabilities()

	// Initialize components
	agent.initializeComponents()

	return agent
}

// ProcessRequest processes a search request
func (sa *SearchAgent) ProcessRequest(ctx context.Context, request *AgentRequest) (*AgentResponse, error) {
	start := time.Now()
	sa.status = StatusBusy
	defer func() { sa.status = StatusIdle }()

	// Parse search request
	searchRequest, err := sa.parseSearchRequest(request)
	if err != nil {
		return nil, fmt.Errorf("failed to parse search request: %v", err)
	}

	// Apply timeout
	searchCtx := ctx
	if sa.config.SearchTimeout > 0 {
		var cancel context.CancelFunc
		searchCtx, cancel = context.WithTimeout(ctx, sa.config.SearchTimeout)
		defer cancel()
	}

	// Perform search
	searchResponse, err := sa.performSearch(searchCtx, searchRequest)
	if err != nil {
		sa.updateMetrics(searchRequest.SearchType, false, time.Since(start), 0)
		return nil, fmt.Errorf("search failed: %v", err)
	}

	// Create agent response
	response := &AgentResponse{
		RequestID:      request.ID,
		AgentType:      sa.GetType(),
		AgentVersion:   sa.GetVersion(),
		Result:         searchResponse,
		Confidence:     sa.calculateConfidence(searchRequest, searchResponse),
		ProcessingTime: time.Since(start),
		CreatedAt:      time.Now(),
	}

	// Update metrics
	sa.updateMetrics(searchRequest.SearchType, true, time.Since(start), len(searchResponse.Results))

	return response, nil
}

// performSearch executes the search based on request type
func (sa *SearchAgent) performSearch(ctx context.Context, request *SearchRequest) (*SearchResponse, error) {
	// Check cache first
	if sa.config.EnableCaching {
		if cached := sa.getFromCache(request); cached != nil {
			return cached, nil
		}
	}

	// Determine search strategy
	searchType := request.SearchType
	if searchType == SearchTypeAuto {
		searchType = sa.determineSearchType(request.Query)
	}

	// Preprocess query
	processedQuery, err := sa.preprocessQuery(request.Query)
	if err != nil {
		return nil, fmt.Errorf("query preprocessing failed: %v", err)
	}

	// Perform search based on type
	var results []*SearchResult
	var searchStrategy string

	switch searchType {
	case SearchTypeText:
		results, err = sa.performTextSearch(ctx, processedQuery, request)
		searchStrategy = "text_search"
	case SearchTypeSemantic:
		results, err = sa.performSemanticSearch(ctx, processedQuery, request)
		searchStrategy = "semantic_search"
	case SearchTypeCode:
		results, err = sa.performCodeSearch(ctx, processedQuery, request)
		searchStrategy = "code_search"
	case SearchTypeSymbol:
		results, err = sa.performSymbolSearch(ctx, processedQuery, request)
		searchStrategy = "symbol_search"
	case SearchTypeRegex:
		results, err = sa.performRegexSearch(ctx, processedQuery, request)
		searchStrategy = "regex_search"
	default:
		return nil, fmt.Errorf("unsupported search type: %s", searchType)
	}

	if err != nil {
		return nil, fmt.Errorf("search execution failed: %v", err)
	}

	// Apply filters
	if request.Filters != nil {
		results = sa.applyFilters(results, request.Filters)
	}

	// Rank results
	if sa.config.EnableRanking {
		results = sa.rankResults(results, request)
	}

	// Apply sorting
	if request.Options != nil && request.Options.SortBy != "" {
		sa.sortResults(results, request.Options.SortBy, request.Options.SortOrder)
	}

	// Apply pagination
	totalFound := len(results)
	if request.Limit > 0 {
		end := request.Offset + request.Limit
		if end > len(results) {
			end = len(results)
		}
		if request.Offset < len(results) {
			results = results[request.Offset:end]
		} else {
			results = []*SearchResult{}
		}
	}

	// Generate response
	response := &SearchResponse{
		Results:        results,
		TotalFound:     totalFound,
		QueryTime:      time.Since(time.Now().Add(-time.Millisecond)), // Approximate
		SearchStrategy: searchStrategy,
	}

	// Add suggestions and related queries
	if sa.config.EnableQueryExpansion {
		response.Suggestions = sa.generateSuggestions(request.Query, results)
		response.RelatedQueries = sa.generateRelatedQueries(request.Query)
	}

	// Add facets
	response.Facets = sa.generateFacets(results)

	// Cache result
	if sa.config.EnableCaching {
		sa.cacheResult(request, response)
	}

	return response, nil
}

// Search implementation methods

func (sa *SearchAgent) performTextSearch(ctx context.Context, query string, request *SearchRequest) ([]*SearchResult, error) {
	if !sa.config.EnableTextSearch {
		return nil, fmt.Errorf("text search is disabled")
	}

	return sa.textSearchEngine.Search(ctx, query, request)
}

func (sa *SearchAgent) performSemanticSearch(ctx context.Context, query string, request *SearchRequest) ([]*SearchResult, error) {
	if !sa.config.EnableSemanticSearch {
		return nil, fmt.Errorf("semantic search is disabled")
	}

	if sa.vectorDB == nil {
		return nil, fmt.Errorf("vector database not available for semantic search")
	}

	return sa.semanticSearchEngine.Search(ctx, query, request)
}

func (sa *SearchAgent) performCodeSearch(ctx context.Context, query string, request *SearchRequest) ([]*SearchResult, error) {
	if !sa.config.EnableCodeSearch {
		return nil, fmt.Errorf("code search is disabled")
	}

	return sa.codeSearchEngine.Search(ctx, query, request)
}

func (sa *SearchAgent) performSymbolSearch(ctx context.Context, query string, request *SearchRequest) ([]*SearchResult, error) {
	if !sa.config.EnableSymbolSearch {
		return nil, fmt.Errorf("symbol search is disabled")
	}

	return sa.symbolSearchEngine.Search(ctx, query, request)
}

func (sa *SearchAgent) performRegexSearch(ctx context.Context, query string, request *SearchRequest) ([]*SearchResult, error) {
	if !sa.config.EnableRegexSearch {
		return nil, fmt.Errorf("regex search is disabled")
	}

	// Validate regex
	_, err := regexp.Compile(query)
	if err != nil {
		return nil, fmt.Errorf("invalid regex pattern: %v", err)
	}

	return sa.textSearchEngine.SearchRegex(ctx, query, request)
}

// Query processing methods

func (sa *SearchAgent) determineSearchType(query string) SearchType {
	// Simple heuristics to determine search type
	query = strings.TrimSpace(strings.ToLower(query))

	// Check for regex patterns
	if sa.isRegexPattern(query) {
		return SearchTypeRegex
	}

	// Check for symbol search patterns
	if sa.isSymbolPattern(query) {
		return SearchTypeSymbol
	}

	// Check for code search patterns
	if sa.isCodePattern(query) {
		return SearchTypeCode
	}

	// Default to semantic search for natural language queries
	if sa.isNaturalLanguageQuery(query) {
		return SearchTypeSemantic
	}

	// Default to text search
	return SearchTypeText
}

func (sa *SearchAgent) preprocessQuery(query string) (string, error) {
	processedQuery := strings.TrimSpace(query)

	// Apply spell check if enabled
	if sa.config.EnableSpellCheck {
		corrected, err := sa.queryAnalyzer.CheckSpelling(processedQuery)
		if err == nil {
			processedQuery = corrected
		}
	}

	// Apply query expansion if enabled
	if sa.config.EnableQueryExpansion {
		expanded, err := sa.queryExpander.ExpandQuery(processedQuery)
		if err == nil {
			processedQuery = expanded
		}
	}

	return processedQuery, nil
}

// Filtering and ranking methods

func (sa *SearchAgent) applyFilters(results []*SearchResult, filters *SearchFilters) []*SearchResult {
	var filtered []*SearchResult

	for _, result := range results {
		if sa.matchesFilters(result, filters) {
			filtered = append(filtered, result)
		}
	}

	return filtered
}

func (sa *SearchAgent) matchesFilters(result *SearchResult, filters *SearchFilters) bool {
	// File type filter
	if len(filters.FileTypes) > 0 {
		fileExt := sa.getFileExtension(result.FilePath)
		if !sa.contains(filters.FileTypes, fileExt) {
			return false
		}
	}

	// Language filter
	if len(filters.Languages) > 0 {
		language := sa.detectLanguage(result.FilePath)
		if !sa.contains(filters.Languages, language) {
			return false
		}
	}

	// Directory filter
	if len(filters.Directories) > 0 {
		dir := sa.getDirectory(result.FilePath)
		if !sa.containsPath(filters.Directories, dir) {
			return false
		}
	}

	// Date range filter
	if filters.DateRange != nil {
		if modTime, ok := result.Metadata["modified_time"].(time.Time); ok {
			if filters.DateRange.Start != nil && modTime.Before(*filters.DateRange.Start) {
				return false
			}
			if filters.DateRange.End != nil && modTime.After(*filters.DateRange.End) {
				return false
			}
		}
	}

	// Size range filter
	if filters.SizeRange != nil {
		if size, ok := result.Metadata["size"].(int64); ok {
			if filters.SizeRange.Min != nil && size < *filters.SizeRange.Min {
				return false
			}
			if filters.SizeRange.Max != nil && size > *filters.SizeRange.Max {
				return false
			}
		}
	}

	return true
}

func (sa *SearchAgent) rankResults(results []*SearchResult, request *SearchRequest) []*SearchResult {
	if sa.resultRanker == nil {
		return results
	}

	return sa.resultRanker.RankResults(results, request, sa.config.RankingWeights)
}

func (sa *SearchAgent) sortResults(results []*SearchResult, sortBy SortCriteria, sortOrder SortOrder) {
	sort.Slice(results, func(i, j int) bool {
		var less bool

		switch sortBy {
		case SortByRelevance:
			less = results[i].Relevance > results[j].Relevance // Higher relevance first
		case SortByScore:
			less = results[i].Score > results[j].Score // Higher score first
		case SortByName:
			less = results[i].FilePath < results[j].FilePath
		case SortByPath:
			less = results[i].FilePath < results[j].FilePath
		case SortByDate:
			if modTime1, ok := results[i].Metadata["modified_time"].(time.Time); ok {
				if modTime2, ok := results[j].Metadata["modified_time"].(time.Time); ok {
					less = modTime1.After(modTime2) // Newer first
				}
			}
		case SortBySize:
			if size1, ok := results[i].Metadata["size"].(int64); ok {
				if size2, ok := results[j].Metadata["size"].(int64); ok {
					less = size1 < size2
				}
			}
		default:
			less = results[i].Score > results[j].Score
		}

		if sortOrder == SortOrderAsc {
			return less
		}
		return !less
	})
}

// Utility methods

func (sa *SearchAgent) parseSearchRequest(request *AgentRequest) (*SearchRequest, error) {
	// Try to parse from parameters first
	if params, ok := request.Parameters["search_request"].(*SearchRequest); ok {
		return params, nil
	}

	// Parse from query and context
	searchRequest := &SearchRequest{
		Query:      request.Query,
		SearchType: SearchTypeAuto,
		Limit:      sa.config.DefaultResultLimit,
		Options: &SearchOptions{
			IncludeContext: true,
			ContextLines:   3,
			Highlight:      true,
		},
	}

	// Infer search type from intent
	if request.Intent != nil {
		switch request.Intent.Type {
		case IntentCodeSearch:
			searchRequest.SearchType = SearchTypeCode
		default:
			searchRequest.SearchType = SearchTypeAuto
		}
	}

	// Add context from request context
	if request.Context != nil {
		searchRequest.Context = &SearchContext{
			WorkingDirectory: request.Context.WorkingDirectory,
			CurrentFile:      request.Context.CurrentFile,
			RecentFiles:      request.Context.OpenFiles,
		}

		// Add file type filter based on current context
		if request.Context.CurrentFile != "" {
			fileExt := sa.getFileExtension(request.Context.CurrentFile)
			if fileExt != "" {
				searchRequest.Filters = &SearchFilters{
					FileTypes: []string{fileExt},
				}
			}
		}
	}

	return searchRequest, nil
}

func (sa *SearchAgent) calculateConfidence(request *SearchRequest, response *SearchResponse) float64 {
	confidence := 0.7 // Base confidence

	// Adjust based on result count
	if response.TotalFound == 0 {
		confidence = 0.1 // Low confidence for no results
	} else if response.TotalFound == 1 {
		confidence = 0.9 // High confidence for single exact result
	} else if response.TotalFound <= 10 {
		confidence = 0.8 // Good confidence for few results
	}

	// Adjust based on search type
	switch request.SearchType {
	case SearchTypeText:
		confidence *= 0.8 // Lower confidence for text search
	case SearchTypeSemantic:
		confidence *= 0.9 // Higher confidence for semantic search
	case SearchTypeCode:
		confidence *= 0.85 // Good confidence for code search
	case SearchTypeSymbol:
		confidence *= 0.95 // Very high confidence for symbol search
	}

	// Cap at 1.0
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

// Pattern recognition methods

func (sa *SearchAgent) isRegexPattern(query string) bool {
	// Simple heuristics for regex patterns
	regexIndicators := []string{"^", "$", "\\", "[", "]", "{", "}", "+", "*", "?", "|"}
	for _, indicator := range regexIndicators {
		if strings.Contains(query, indicator) {
			return true
		}
	}
	return false
}

func (sa *SearchAgent) isSymbolPattern(query string) bool {
	// Check if query looks like a symbol (function, class, variable name)
	symbolPattern := regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_]*$`)
	return symbolPattern.MatchString(query)
}

func (sa *SearchAgent) isCodePattern(query string) bool {
	// Check for code-like patterns
	codeKeywords := []string{"function", "class", "def", "func", "var", "let", "const", "import", "export"}
	queryLower := strings.ToLower(query)

	for _, keyword := range codeKeywords {
		if strings.Contains(queryLower, keyword) {
			return true
		}
	}

	// Check for common code punctuation
	codePunctuation := []string{"{", "}", "(", ")", ";", ":", "=>", "->"}
	for _, punct := range codePunctuation {
		if strings.Contains(query, punct) {
			return true
		}
	}

	return false
}

func (sa *SearchAgent) isNaturalLanguageQuery(query string) bool {
	// Simple heuristic: if query contains multiple words and common English words
	words := strings.Fields(query)
	if len(words) < 2 {
		return false
	}

	commonWords := []string{"how", "what", "where", "when", "why", "find", "search", "get", "show", "list"}
	queryLower := strings.ToLower(query)

	for _, word := range commonWords {
		if strings.Contains(queryLower, word) {
			return true
		}
	}

	return false
}

// Helper methods

func (sa *SearchAgent) getFileExtension(filepath string) string {
	parts := strings.Split(filepath, ".")
	if len(parts) > 1 {
		return parts[len(parts)-1]
	}
	return ""
}

func (sa *SearchAgent) detectLanguage(filepath string) string {
	ext := sa.getFileExtension(filepath)
	langMap := map[string]string{
		"go":   "go",
		"py":   "python",
		"js":   "javascript",
		"ts":   "typescript",
		"java": "java",
		"cpp":  "cpp",
		"c":    "c",
		"cs":   "csharp",
		"rb":   "ruby",
		"php":  "php",
		"rs":   "rust",
	}

	if lang, exists := langMap[ext]; exists {
		return lang
	}
	return "unknown"
}

func (sa *SearchAgent) getDirectory(filepath string) string {
	parts := strings.Split(filepath, "/")
	if len(parts) > 1 {
		return strings.Join(parts[:len(parts)-1], "/")
	}
	return ""
}

func (sa *SearchAgent) contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func (sa *SearchAgent) containsPath(paths []string, path string) bool {
	for _, p := range paths {
		if strings.HasPrefix(path, p) {
			return true
		}
	}
	return false
}

// Suggestion and facet generation

func (sa *SearchAgent) generateSuggestions(query string, results []*SearchResult) []string {
	var suggestions []string

	// If no results, suggest alternatives
	if len(results) == 0 {
		suggestions = append(suggestions, "Try a different search term")
		suggestions = append(suggestions, "Check spelling")
		suggestions = append(suggestions, "Use broader search terms")

		if sa.config.EnableSpellCheck {
			if corrected, err := sa.queryAnalyzer.CheckSpelling(query); err == nil && corrected != query {
				suggestions = append(suggestions, fmt.Sprintf("Did you mean: %s", corrected))
			}
		}
	} else if len(results) > 100 {
		suggestions = append(suggestions, "Refine your search with filters")
		suggestions = append(suggestions, "Try more specific terms")
	}

	return suggestions
}

func (sa *SearchAgent) generateRelatedQueries(query string) []string {
	var related []string

	// Simple related query generation
	words := strings.Fields(strings.ToLower(query))
	if len(words) > 1 {
		// Try individual words
		for _, word := range words {
			if len(word) > 3 {
				related = append(related, word)
			}
		}

		// Try combinations
		if len(words) >= 2 {
			related = append(related, words[0]+" "+words[1])
		}
	}

	// Remove duplicates and limit
	seen := make(map[string]bool)
	var unique []string
	for _, r := range related {
		if !seen[r] && r != strings.ToLower(query) {
			seen[r] = true
			unique = append(unique, r)
		}
	}

	if len(unique) > 5 {
		unique = unique[:5]
	}

	return unique
}

func (sa *SearchAgent) generateFacets(results []*SearchResult) map[string][]FacetItem {
	facets := make(map[string][]FacetItem)

	// File type facet
	fileTypes := make(map[string]int)
	languages := make(map[string]int)
	directories := make(map[string]int)

	for _, result := range results {
		// File type
		ext := sa.getFileExtension(result.FilePath)
		if ext != "" {
			fileTypes[ext]++
		}

		// Language
		lang := sa.detectLanguage(result.FilePath)
		languages[lang]++

		// Directory
		dir := sa.getDirectory(result.FilePath)
		if dir != "" {
			directories[dir]++
		}
	}

	// Convert to facet items
	facets["file_types"] = sa.mapToFacetItems(fileTypes)
	facets["languages"] = sa.mapToFacetItems(languages)
	facets["directories"] = sa.mapToFacetItems(directories)

	return facets
}

func (sa *SearchAgent) mapToFacetItems(countMap map[string]int) []FacetItem {
	var items []FacetItem
	for key, count := range countMap {
		items = append(items, FacetItem{
			Value: key,
			Count: count,
		})
	}

	// Sort by count descending
	sort.Slice(items, func(i, j int) bool {
		return items[i].Count > items[j].Count
	})

	return items
}

// Caching methods

func (sa *SearchAgent) getFromCache(request *SearchRequest) *SearchResponse {
	if sa.searchCache == nil {
		return nil
	}

	key := sa.generateCacheKey(request)
	return sa.searchCache.Get(key)
}

func (sa *SearchAgent) cacheResult(request *SearchRequest, response *SearchResponse) {
	if sa.searchCache == nil {
		return
	}

	key := sa.generateCacheKey(request)
	sa.searchCache.Set(key, response)
}

func (sa *SearchAgent) generateCacheKey(request *SearchRequest) string {
	// Generate a cache key based on request parameters
	key := fmt.Sprintf("%s_%s_%d_%d", request.Query, request.SearchType, request.Limit, request.Offset)

	if request.Filters != nil {
		key += fmt.Sprintf("_ft:%v_lg:%v", request.Filters.FileTypes, request.Filters.Languages)
	}

	if request.Options != nil {
		key += fmt.Sprintf("_cs:%t_ww:%t", request.Options.CaseSensitive, request.Options.WholeWord)
	}

	return key
}

// Metrics and monitoring

func (sa *SearchAgent) updateMetrics(searchType SearchType, success bool, duration time.Duration, resultCount int) {
	sa.metrics.mu.Lock()
	defer sa.metrics.mu.Unlock()

	sa.metrics.TotalSearches++
	sa.metrics.SearchesByType[searchType]++

	// Update success rate
	if sa.metrics.TotalSearches == 1 {
		if success {
			sa.metrics.SuccessRate = 1.0
		} else {
			sa.metrics.SuccessRate = 0.0
		}
	} else {
		oldSuccessCount := int64(sa.metrics.SuccessRate * float64(sa.metrics.TotalSearches-1))
		if success {
			oldSuccessCount++
		}
		sa.metrics.SuccessRate = float64(oldSuccessCount) / float64(sa.metrics.TotalSearches)
	}

	// Update average search time
	if sa.metrics.AverageSearchTime == 0 {
		sa.metrics.AverageSearchTime = duration
	} else {
		sa.metrics.AverageSearchTime = (sa.metrics.AverageSearchTime + duration) / 2
	}

	// Update average result count
	if sa.metrics.AverageResultCount == 0 {
		sa.metrics.AverageResultCount = float64(resultCount)
	} else {
		sa.metrics.AverageResultCount = (sa.metrics.AverageResultCount + float64(resultCount)) / 2.0
	}

	sa.metrics.LastSearch = time.Now()
}

// Component initialization

func (sa *SearchAgent) initializeCapabilities() {
	sa.capabilities = &AgentCapabilities{
		AgentType: AgentTypeSearch,
		SupportedIntents: []IntentType{
			IntentCodeSearch,
		},
		SupportedLanguages: []string{
			"go", "python", "javascript", "typescript", "java",
			"cpp", "c", "csharp", "ruby", "php", "rust",
		},
		SupportedFileTypes: []string{
			"go", "py", "js", "ts", "java", "cpp", "c", "cs", "rb", "php", "rs",
			"txt", "md", "json", "yaml", "yml", "xml", "html", "css",
		},
		MaxContextSize:    4096,
		SupportsStreaming: false,
		SupportsAsync:     true,
		RequiresContext:   false,
		Capabilities: map[string]interface{}{
			"text_search":     sa.config.EnableTextSearch,
			"semantic_search": sa.config.EnableSemanticSearch,
			"code_search":     sa.config.EnableCodeSearch,
			"symbol_search":   sa.config.EnableSymbolSearch,
			"regex_search":    sa.config.EnableRegexSearch,
			"fuzzy_search":    sa.config.EnableFuzzySearch,
		},
	}
}

func (sa *SearchAgent) initializeComponents() {
	// Initialize search engines
	sa.textSearchEngine = NewTextSearchEngine(sa.indexer)
	sa.codeSearchEngine = NewCodeSearchEngine(sa.indexer)
	sa.symbolSearchEngine = NewSymbolSearchEngine(sa.indexer)

	if sa.vectorDB != nil {
		sa.semanticSearchEngine = NewSemanticSearchEngine(sa.vectorDB)
	}

	// Initialize query processing components
	sa.queryAnalyzer = NewQueryAnalyzer()
	sa.queryExpander = NewQueryExpander()

	// Initialize result ranker
	if sa.config.EnableRanking {
		sa.resultRanker = NewSearchResultRanker()
	}

	// Initialize caching
	if sa.config.EnableCaching {
		sa.searchCache = NewSearchCache(sa.config.CacheSize, sa.config.CacheTTL)
	}
}

// Required Agent interface methods

func (sa *SearchAgent) GetCapabilities() *AgentCapabilities {
	return sa.capabilities
}

func (sa *SearchAgent) GetType() AgentType {
	return AgentTypeSearch
}

func (sa *SearchAgent) GetVersion() string {
	return "1.0.0"
}

func (sa *SearchAgent) GetStatus() AgentStatus {
	sa.mu.RLock()
	defer sa.mu.RUnlock()
	return sa.status
}

func (sa *SearchAgent) Initialize(config interface{}) error {
	if cfg, ok := config.(*SearchAgentConfig); ok {
		sa.config = cfg
		sa.initializeComponents()
		return nil
	}
	return fmt.Errorf("invalid config type")
}

func (sa *SearchAgent) Start() error {
	sa.mu.Lock()
	defer sa.mu.Unlock()

	sa.status = StatusIdle
	sa.logger.Info("Search agent started")
	return nil
}

func (sa *SearchAgent) Stop() error {
	sa.mu.Lock()
	defer sa.mu.Unlock()

	sa.status = StatusStopped
	sa.logger.Info("Search agent stopped")
	return nil
}

func (sa *SearchAgent) HealthCheck() error {
	if sa.indexer == nil {
		return fmt.Errorf("indexer not configured")
	}

	// Test basic search functionality
	_, err := sa.textSearchEngine.Search(context.Background(), "test", &SearchRequest{Limit: 1})
	return err
}

func (sa *SearchAgent) GetMetrics() *AgentMetrics {
	sa.metrics.mu.RLock()
	defer sa.metrics.mu.RUnlock()

	return &AgentMetrics{
		RequestsProcessed:   sa.metrics.TotalSearches,
		AverageResponseTime: sa.metrics.AverageSearchTime,
		SuccessRate:         sa.metrics.SuccessRate,
		LastRequestAt:       sa.metrics.LastSearch,
	}
}

func (sa *SearchAgent) ResetMetrics() {
	sa.metrics.mu.Lock()
	defer sa.metrics.mu.Unlock()

	sa.metrics = &SearchAgentMetrics{
		SearchesByType: make(map[SearchType]int64),
		PopularQueries: make(map[string]int64),
	}
}

// Supporting types and placeholder implementations

type RankingWeights struct {
	Relevance  float32 `json:"relevance"`
	Freshness  float32 `json:"freshness"`
	Popularity float32 `json:"popularity"`
	Context    float32 `json:"context"`
}

// Placeholder implementations for search engines and components

type TextSearchEngine struct {
	indexer *indexer.UltraFastIndexer
}

func NewTextSearchEngine(indexer *indexer.UltraFastIndexer) *TextSearchEngine {
	return &TextSearchEngine{indexer: indexer}
}

func (tse *TextSearchEngine) Search(ctx context.Context, query string, request *SearchRequest) ([]*SearchResult, error) {
	// Placeholder implementation - would interface with indexer
	return []*SearchResult{}, nil
}

func (tse *TextSearchEngine) SearchRegex(ctx context.Context, pattern string, request *SearchRequest) ([]*SearchResult, error) {
	// Placeholder implementation for regex search
	return []*SearchResult{}, nil
}

type SemanticSearchEngine struct {
	vectorDB *vectordb.VectorDB
}

func NewSemanticSearchEngine(vectorDB *vectordb.VectorDB) *SemanticSearchEngine {
	return &SemanticSearchEngine{vectorDB: vectorDB}
}

func (sse *SemanticSearchEngine) Search(ctx context.Context, query string, request *SearchRequest) ([]*SearchResult, error) {
	// Placeholder implementation - would interface with vector DB
	return []*SearchResult{}, nil
}

type CodeSearchEngine struct {
	indexer *indexer.UltraFastIndexer
}

func NewCodeSearchEngine(indexer *indexer.UltraFastIndexer) *CodeSearchEngine {
	return &CodeSearchEngine{indexer: indexer}
}

func (cse *CodeSearchEngine) Search(ctx context.Context, query string, request *SearchRequest) ([]*SearchResult, error) {
	// Placeholder implementation - would search for code patterns
	return []*SearchResult{}, nil
}

type SymbolSearchEngine struct {
	indexer *indexer.UltraFastIndexer
}

func NewSymbolSearchEngine(indexer *indexer.UltraFastIndexer) *SymbolSearchEngine {
	return &SymbolSearchEngine{indexer: indexer}
}

func (sse *SymbolSearchEngine) Search(ctx context.Context, query string, request *SearchRequest) ([]*SearchResult, error) {
	// Placeholder implementation - would search for symbols
	return []*SearchResult{}, nil
}

type QueryAnalyzer struct{}

func NewQueryAnalyzer() *QueryAnalyzer {
	return &QueryAnalyzer{}
}

func (qa *QueryAnalyzer) CheckSpelling(query string) (string, error) {
	// Placeholder implementation
	return query, nil
}

type QueryExpander struct{}

func NewQueryExpander() *QueryExpander {
	return &QueryExpander{}
}

func (qe *QueryExpander) ExpandQuery(query string) (string, error) {
	// Placeholder implementation - would expand with synonyms
	return query, nil
}

type SearchResultRanker struct{}

func NewSearchResultRanker() *SearchResultRanker {
	return &SearchResultRanker{}
}

func (srr *SearchResultRanker) RankResults(results []*SearchResult, request *SearchRequest, weights *RankingWeights) []*SearchResult {
	// Placeholder implementation - would apply sophisticated ranking
	return results
}

type SearchCache struct {
	cache map[string]*SearchResponse
	ttl   time.Duration
	mu    sync.RWMutex
}

func NewSearchCache(size int, ttl time.Duration) *SearchCache {
	return &SearchCache{
		cache: make(map[string]*SearchResponse),
		ttl:   ttl,
	}
}

func (sc *SearchCache) Get(key string) *SearchResponse {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	// Placeholder implementation
	return sc.cache[key]
}

func (sc *SearchCache) Set(key string, response *SearchResponse) {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	sc.cache[key] = response
}

type IndexCache struct{}
