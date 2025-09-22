package vectordb

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/indexer"
	"github.com/yourusername/ai-code-assistant/storage"
)

// ContextRetriever provides intelligent context expansion for search results
type ContextRetriever struct {
	// Core components
	indexer      *indexer.UltraFastIndexer
	graphBuilder *indexer.GraphBuilder
	db           *storage.SQLiteDB

	// Configuration
	config *ContextConfig

	// Caching
	contextCache    *ContextCache
	dependencyCache *DependencyCache

	// Analytics
	stats *ContextStatistics

	// State management
	mu sync.RWMutex
}

// ContextConfig contains context retrieval configuration
type ContextConfig struct {
	// Context window settings
	DefaultWindowSize int  `json:"default_window_size"` // Lines before/after
	MaxWindowSize     int  `json:"max_window_size"`     // Maximum context size
	MinWindowSize     int  `json:"min_window_size"`     // Minimum context size
	AdaptiveWindow    bool `json:"adaptive_window"`     // Adjust window based on content

	// Context types to include
	IncludeFunctions    bool `json:"include_functions"`    // Include related functions
	IncludeClasses      bool `json:"include_classes"`      // Include parent/child classes
	IncludeImports      bool `json:"include_imports"`      // Include import statements
	IncludeComments     bool `json:"include_comments"`     // Include documentation
	IncludeDependencies bool `json:"include_dependencies"` // Include dependency context
	IncludeUsages       bool `json:"include_usages"`       // Include usage examples

	// Smart context features
	EnableSmartBoundaries bool `json:"enable_smart_boundaries"` // Respect function/class boundaries
	EnableCrossFile       bool `json:"enable_cross_file"`       // Include context from other files
	EnableSemanticContext bool `json:"enable_semantic_context"` // Include semantically related code
	EnableHierarchical    bool `json:"enable_hierarchical"`     // Include class hierarchies

	// Performance settings
	MaxCrossFileRefs int           `json:"max_cross_file_refs"` // Max cross-file references
	ContextTimeout   time.Duration `json:"context_timeout"`     // Timeout for context retrieval
	EnableCaching    bool          `json:"enable_caching"`      // Enable context caching
	CacheSize        int           `json:"cache_size"`          // Context cache size
	CacheTTL         time.Duration `json:"cache_ttl"`           // Cache time-to-live

	// Ranking for context relevance
	RelevanceThreshold float32                `json:"relevance_threshold"` // Minimum relevance score
	MaxContextItems    int                    `json:"max_context_items"`   // Maximum context items to return
	RankingWeights     *ContextRankingWeights `json:"ranking_weights"`
}

// ContextRankingWeights defines weights for ranking context relevance
type ContextRankingWeights struct {
	Proximity  float32 `json:"proximity"`  // Physical proximity in code
	Semantic   float32 `json:"semantic"`   // Semantic similarity
	Dependency float32 `json:"dependency"` // Dependency relationship
	Usage      float32 `json:"usage"`      // Usage frequency
	Recency    float32 `json:"recency"`    // Recent modifications
	Popularity float32 `json:"popularity"` // Code popularity/importance
}

// ResultContext represents the basic context around a search result
type ResultContext struct {
	// Basic context
	BeforeLines  []string `json:"before_lines"`
	AfterLines   []string `json:"after_lines"`
	FullFunction string   `json:"full_function,omitempty"`
	FullClass    string   `json:"full_class,omitempty"`

	// Metadata
	StartLineNumber int         `json:"start_line_number"`
	EndLineNumber   int         `json:"end_line_number"`
	TotalLines      int         `json:"total_lines"`
	ContextType     ContextType `json:"context_type"`
}

// ExpandedContext represents rich context with semantic relationships
type ExpandedContext struct {
	// Basic context
	BasicContext *ResultContext `json:"basic_context"`

	// Related code elements
	RelatedFunctions []*ContextItem `json:"related_functions,omitempty"`
	RelatedClasses   []*ContextItem `json:"related_classes,omitempty"`
	Dependencies     []*ContextItem `json:"dependencies,omitempty"`
	Usages           []*ContextItem `json:"usages,omitempty"`
	Imports          []*ContextItem `json:"imports,omitempty"`

	// Cross-file context
	CrossFileRefs []*CrossFileContext `json:"cross_file_refs,omitempty"`

	// Hierarchical context
	ClassHierarchy *ClassHierarchy `json:"class_hierarchy,omitempty"`
	CallHierarchy  *CallHierarchy  `json:"call_hierarchy,omitempty"`

	// Semantic context
	SemanticSimilar []*ContextItem `json:"semantic_similar,omitempty"`

	// Metadata
	RetrievalTime  time.Duration `json:"retrieval_time"`
	TotalItems     int           `json:"total_items"`
	CrossFileCount int           `json:"cross_file_count"`
	ContextScore   float32       `json:"context_score"`
}

// ContextItem represents a single piece of context
type ContextItem struct {
	// Identification
	ID   string   `json:"id"`
	Type ItemType `json:"type"`
	Name string   `json:"name"`

	// Content
	Content       string `json:"content"`
	Snippet       string `json:"snippet"`
	Signature     string `json:"signature,omitempty"`
	Documentation string `json:"documentation,omitempty"`

	// Location
	FilePath  string `json:"file_path"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`

	// Relevance
	RelevanceScore   float32          `json:"relevance_score"`
	RelationshipType RelationshipType `json:"relationship_type"`
	Distance         int              `json:"distance"` // Lines or call distance

	// Metadata
	Language     string    `json:"language"`
	LastModified time.Time `json:"last_modified,omitempty"`
	AccessCount  int64     `json:"access_count,omitempty"`
}

// CrossFileContext represents context from other files
type CrossFileContext struct {
	FilePath         string           `json:"file_path"`
	RelationshipType RelationshipType `json:"relationship_type"`
	Items            []*ContextItem   `json:"items"`
	RelevanceScore   float32          `json:"relevance_score"`
}

// ClassHierarchy represents class inheritance hierarchy
type ClassHierarchy struct {
	CurrentClass  *ContextItem   `json:"current_class"`
	ParentClasses []*ContextItem `json:"parent_classes,omitempty"`
	ChildClasses  []*ContextItem `json:"child_classes,omitempty"`
	Interfaces    []*ContextItem `json:"interfaces,omitempty"`
	Siblings      []*ContextItem `json:"siblings,omitempty"`
}

// CallHierarchy represents function call hierarchy
type CallHierarchy struct {
	CurrentFunction *ContextItem   `json:"current_function"`
	Callers         []*ContextItem `json:"callers,omitempty"`
	Callees         []*ContextItem `json:"callees,omitempty"`
	CallChain       []*ContextItem `json:"call_chain,omitempty"`
}

// Enums
type ContextType string

const (
	ContextTypeBasic    ContextType = "basic"
	ContextTypeFunction ContextType = "function"
	ContextTypeClass    ContextType = "class"
	ContextTypeFile     ContextType = "file"
	ContextTypeExpanded ContextType = "expanded"
)

type ItemType string

const (
	ItemTypeFunction  ItemType = "function"
	ItemTypeClass     ItemType = "class"
	ItemTypeVariable  ItemType = "variable"
	ItemTypeInterface ItemType = "interface"
	ItemTypeImport    ItemType = "import"
	ItemTypeComment   ItemType = "comment"
	ItemTypeUsage     ItemType = "usage"
)

type RelationshipType string

const (
	RelationshipCallFrom   RelationshipType = "calls_from"
	RelationshipCallTo     RelationshipType = "calls_to"
	RelationshipInherits   RelationshipType = "inherits"
	RelationshipImplements RelationshipType = "implements"
	RelationshipUses       RelationshipType = "uses"
	RelationshipImports    RelationshipType = "imports"
	RelationshipSimilar    RelationshipType = "similar"
	RelationshipProximity  RelationshipType = "proximity"
)

// ContextStatistics tracks context retrieval performance
type ContextStatistics struct {
	TotalRetrievals      int64         `json:"total_retrievals"`
	CacheHits            int64         `json:"cache_hits"`
	CacheMisses          int64         `json:"cache_misses"`
	AverageRetrievalTime time.Duration `json:"average_retrieval_time"`
	CrossFileRetrievals  int64         `json:"cross_file_retrievals"`
	SemanticRetrievals   int64         `json:"semantic_retrievals"`
	ErrorCount           int64         `json:"error_count"`
	LastError            string        `json:"last_error,omitempty"`
	mu                   sync.RWMutex
}

// NewContextRetriever creates a new context retriever
func NewContextRetriever(config *ContextConfig) *ContextRetriever {
	if config == nil {
		config = &ContextConfig{
			DefaultWindowSize:     10,
			MaxWindowSize:         50,
			MinWindowSize:         3,
			AdaptiveWindow:        true,
			IncludeFunctions:      true,
			IncludeClasses:        true,
			IncludeImports:        true,
			IncludeComments:       true,
			IncludeDependencies:   true,
			IncludeUsages:         false, // Expensive operation
			EnableSmartBoundaries: true,
			EnableCrossFile:       true,
			EnableSemanticContext: true,
			EnableHierarchical:    true,
			MaxCrossFileRefs:      5,
			ContextTimeout:        time.Second * 10,
			EnableCaching:         true,
			CacheSize:             1000,
			CacheTTL:              time.Minute * 15,
			RelevanceThreshold:    0.3,
			MaxContextItems:       20,
			RankingWeights: &ContextRankingWeights{
				Proximity:  0.3,
				Semantic:   0.2,
				Dependency: 0.2,
				Usage:      0.1,
				Recency:    0.1,
				Popularity: 0.1,
			},
		}
	}

	cr := &ContextRetriever{
		config: config,
		stats:  &ContextStatistics{},
	}

	// Initialize caches
	if config.EnableCaching {
		cr.contextCache = NewContextCache(config.CacheSize, config.CacheTTL)
		cr.dependencyCache = NewDependencyCache(config.CacheSize, config.CacheTTL)
	}

	return cr
}

// GetExpandedContext retrieves comprehensive context for a search result
func (cr *ContextRetriever) GetExpandedContext(result *SearchResult, windowSize int) *ExpandedContext {
	start := time.Now()

	// Generate cache key
	cacheKey := cr.generateContextCacheKey(result, windowSize)

	// Check cache
	if cr.config.EnableCaching {
		if cached := cr.contextCache.Get(cacheKey); cached != nil {
			cr.updateCacheStats(true)
			return cached
		}
	}

	// Get basic context first
	basicContext := cr.getBasicContext(result, windowSize)

	expandedContext := &ExpandedContext{
		BasicContext:  basicContext,
		RetrievalTime: time.Since(start),
	}

	// Add related functions
	if cr.config.IncludeFunctions {
		expandedContext.RelatedFunctions = cr.getRelatedFunctions(result)
	}

	// Add related classes
	if cr.config.IncludeClasses {
		expandedContext.RelatedClasses = cr.getRelatedClasses(result)
	}

	// Add dependencies
	if cr.config.IncludeDependencies {
		expandedContext.Dependencies = cr.getDependencies(result)
	}

	// Add usages (expensive operation)
	if cr.config.IncludeUsages {
		expandedContext.Usages = cr.getUsages(result)
	}

	// Add imports
	if cr.config.IncludeImports {
		expandedContext.Imports = cr.getImports(result)
	}

	// Add cross-file references
	if cr.config.EnableCrossFile {
		expandedContext.CrossFileRefs = cr.getCrossFileReferences(result)
	}

	// Add hierarchical context
	if cr.config.EnableHierarchical {
		if result.Source.ChunkType == "class" {
			expandedContext.ClassHierarchy = cr.getClassHierarchy(result)
		}
		if result.Source.ChunkType == "function" {
			expandedContext.CallHierarchy = cr.getCallHierarchy(result)
		}
	}

	// Add semantic context
	if cr.config.EnableSemanticContext {
		expandedContext.SemanticSimilar = cr.getSemanticallySimilar(result)
	}

	// Calculate context metrics
	expandedContext.TotalItems = cr.countContextItems(expandedContext)
	expandedContext.CrossFileCount = len(expandedContext.CrossFileRefs)
	expandedContext.ContextScore = cr.calculateContextScore(expandedContext)
	expandedContext.RetrievalTime = time.Since(start)

	// Cache the result
	if cr.config.EnableCaching {
		cr.contextCache.Set(cacheKey, expandedContext)
		cr.updateCacheStats(false)
	}

	// Update statistics
	cr.updateRetrievalStats(expandedContext)

	return expandedContext
}

// getBasicContext retrieves basic surrounding context
func (cr *ContextRetriever) getBasicContext(result *SearchResult, windowSize int) *ResultContext {
	// Adjust window size if adaptive
	if cr.config.AdaptiveWindow {
		windowSize = cr.calculateAdaptiveWindowSize(result, windowSize)
	}

	// Ensure window size is within bounds
	if windowSize < cr.config.MinWindowSize {
		windowSize = cr.config.MinWindowSize
	}
	if windowSize > cr.config.MaxWindowSize {
		windowSize = cr.config.MaxWindowSize
	}

	// Read file content
	fileContent, err := cr.readFileContent(result.Source.FilePath)
	if err != nil {
		return &ResultContext{
			ContextType: ContextTypeBasic,
		}
	}

	lines := strings.Split(fileContent, "\n")
	startLine := result.Source.StartLine
	endLine := result.Source.EndLine

	// Calculate context boundaries
	contextStart := startLine - windowSize - 1
	contextEnd := endLine + windowSize - 1

	// Respect smart boundaries if enabled
	if cr.config.EnableSmartBoundaries {
		contextStart, contextEnd = cr.adjustForSmartBoundaries(lines, contextStart, contextEnd, startLine, endLine)
	}

	// Ensure boundaries are valid
	if contextStart < 0 {
		contextStart = 0
	}
	if contextEnd >= len(lines) {
		contextEnd = len(lines) - 1
	}

	// Extract context lines
	var beforeLines, afterLines []string

	if contextStart < startLine-1 {
		beforeLines = lines[contextStart : startLine-1]
	}

	if contextEnd > endLine-1 {
		afterLines = lines[endLine : contextEnd+1]
	}

	context := &ResultContext{
		BeforeLines:     beforeLines,
		AfterLines:      afterLines,
		StartLineNumber: contextStart + 1,
		EndLineNumber:   contextEnd + 1,
		TotalLines:      contextEnd - contextStart + 1,
		ContextType:     ContextTypeBasic,
	}

	// Try to get full function or class if applicable
	if result.Source.ChunkType == "function" {
		context.FullFunction = cr.extractFullFunction(lines, result.Source)
		context.ContextType = ContextTypeFunction
	} else if result.Source.ChunkType == "class" {
		context.FullClass = cr.extractFullClass(lines, result.Source)
		context.ContextType = ContextTypeClass
	}

	return context
}

// getRelatedFunctions finds functions related to the search result
func (cr *ContextRetriever) getRelatedFunctions(result *SearchResult) []*ContextItem {
	var relatedFunctions []*ContextItem

	// This would query the dependency graph to find related functions
	// For now, we'll simulate this

	// Get functions in the same file
	fileFunctions := cr.getFunctionsInFile(result.Source.FilePath)

	// Score and filter functions
	for _, function := range fileFunctions {
		if function.Name == result.Source.FunctionName {
			continue // Skip self
		}

		score := cr.calculateFunctionRelevance(result, function)
		if score >= cr.config.RelevanceThreshold {
			function.RelevanceScore = score
			relatedFunctions = append(relatedFunctions, function)
		}
	}

	// Sort by relevance
	sort.Slice(relatedFunctions, func(i, j int) bool {
		return relatedFunctions[i].RelevanceScore > relatedFunctions[j].RelevanceScore
	})

	// Limit results
	if len(relatedFunctions) > cr.config.MaxContextItems {
		relatedFunctions = relatedFunctions[:cr.config.MaxContextItems]
	}

	return relatedFunctions
}

// getRelatedClasses finds classes related to the search result
func (cr *ContextRetriever) getRelatedClasses(result *SearchResult) []*ContextItem {
	var relatedClasses []*ContextItem

	// Get classes in the same file and imported classes
	fileClasses := cr.getClassesInFile(result.Source.FilePath)

	for _, class := range fileClasses {
		if class.Name == result.Source.ClassName {
			continue // Skip self
		}

		score := cr.calculateClassRelevance(result, class)
		if score >= cr.config.RelevanceThreshold {
			class.RelevanceScore = score
			relatedClasses = append(relatedClasses, class)
		}
	}

	// Sort and limit
	sort.Slice(relatedClasses, func(i, j int) bool {
		return relatedClasses[i].RelevanceScore > relatedClasses[j].RelevanceScore
	})

	if len(relatedClasses) > cr.config.MaxContextItems {
		relatedClasses = relatedClasses[:cr.config.MaxContextItems]
	}

	return relatedClasses
}

// getDependencies finds code dependencies
func (cr *ContextRetriever) getDependencies(result *SearchResult) []*ContextItem {
	// Check cache first
	cacheKey := fmt.Sprintf("deps_%s_%d", result.Source.FilePath, result.Source.StartLine)
	if cr.config.EnableCaching {
		if cached := cr.dependencyCache.Get(cacheKey); cached != nil {
			return cached
		}
	}

	var dependencies []*ContextItem

	// This would use the dependency graph to find actual dependencies
	// For now, we'll extract from imports and function calls

	dependencies = cr.extractDependenciesFromCode(result)

	// Cache the result
	if cr.config.EnableCaching {
		cr.dependencyCache.Set(cacheKey, dependencies)
	}

	return dependencies
}

// getUsages finds usage examples (expensive operation)
func (cr *ContextRetriever) getUsages(result *SearchResult) []*ContextItem {
	// This would search the entire codebase for usages
	// Very expensive operation, should be cached aggressively

	var usages []*ContextItem

	// For functions, find call sites
	if result.Source.ChunkType == "function" {
		usages = cr.findFunctionUsages(result.Source.FunctionName)
	}

	// For classes, find instantiations
	if result.Source.ChunkType == "class" {
		usages = cr.findClassUsages(result.Source.ClassName)
	}

	// Limit to prevent overwhelming results
	if len(usages) > 10 {
		usages = usages[:10]
	}

	return usages
}

// getImports gets import statements and related imports
func (cr *ContextRetriever) getImports(result *SearchResult) []*ContextItem {
	var imports []*ContextItem

	// Get imports from the file
	fileImports := cr.getFileImports(result.Source.FilePath)

	// Filter relevant imports
	for _, imp := range fileImports {
		if cr.isRelevantImport(result, imp) {
			imports = append(imports, imp)
		}
	}

	return imports
}

// getCrossFileReferences gets relevant references from other files
func (cr *ContextRetriever) getCrossFileReferences(result *SearchResult) []*CrossFileContext {
	var crossFileRefs []*CrossFileContext

	// This would use the dependency graph to find cross-file relationships
	// For now, we'll look for imports and exports

	relatedFiles := cr.findRelatedFiles(result)

	for _, filePath := range relatedFiles {
		refs := cr.getReferencesInFile(filePath, result)
		if len(refs) > 0 {
			crossFileRef := &CrossFileContext{
				FilePath:         filePath,
				RelationshipType: RelationshipImports, // Simplified
				Items:            refs,
				RelevanceScore:   cr.calculateFileRelevance(result, filePath),
			}
			crossFileRefs = append(crossFileRefs, crossFileRef)
		}

		// Limit cross-file references
		if len(crossFileRefs) >= cr.config.MaxCrossFileRefs {
			break
		}
	}

	return crossFileRefs
}

// getClassHierarchy builds class inheritance hierarchy
func (cr *ContextRetriever) getClassHierarchy(result *SearchResult) *ClassHierarchy {
	hierarchy := &ClassHierarchy{}

	// This would use the dependency graph to build the hierarchy
	// For now, we'll provide a simplified implementation

	if result.Source.ChunkType == "class" {
		hierarchy.CurrentClass = &ContextItem{
			Name:     result.Source.ClassName,
			Type:     ItemTypeClass,
			FilePath: result.Source.FilePath,
			Content:  result.Content,
		}

		// Find parent classes
		hierarchy.ParentClasses = cr.findParentClasses(result.Source.ClassName)

		// Find child classes
		hierarchy.ChildClasses = cr.findChildClasses(result.Source.ClassName)

		// Find interfaces
		hierarchy.Interfaces = cr.findImplementedInterfaces(result.Source.ClassName)
	}

	return hierarchy
}

// getCallHierarchy builds function call hierarchy
func (cr *ContextRetriever) getCallHierarchy(result *SearchResult) *CallHierarchy {
	hierarchy := &CallHierarchy{}

	if result.Source.ChunkType == "function" {
		hierarchy.CurrentFunction = &ContextItem{
			Name:     result.Source.FunctionName,
			Type:     ItemTypeFunction,
			FilePath: result.Source.FilePath,
			Content:  result.Content,
		}

		// Find callers
		hierarchy.Callers = cr.findFunctionCallers(result.Source.FunctionName)

		// Find callees
		hierarchy.Callees = cr.findFunctionCallees(result.Source.FunctionName)

		// Build call chain
		hierarchy.CallChain = cr.buildCallChain(result.Source.FunctionName)
	}

	return hierarchy
}

// getSemanticallySimilar finds semantically similar code
func (cr *ContextRetriever) getSemanticallySimilar(result *SearchResult) []*ContextItem {
	// This would use vector similarity to find semantically similar code
	// For now, we'll provide a placeholder

	var similar []*ContextItem

	// This would:
	// 1. Generate embedding for the result content
	// 2. Search for similar embeddings
	// 3. Filter and rank results

	return similar
}

// Helper methods

func (cr *ContextRetriever) calculateAdaptiveWindowSize(result *SearchResult, defaultSize int) int {
	// Adjust window size based on content type and complexity
	windowSize := defaultSize

	// Larger window for complex functions
	if result.Source.ChunkType == "function" {
		if complexity, ok := result.Metadata["complexity"].(int); ok {
			if complexity > 10 {
				windowSize = int(float64(windowSize) * 1.5)
			}
		}
	}

	// Larger window for classes
	if result.Source.ChunkType == "class" {
		windowSize = int(float64(windowSize) * 1.3)
	}

	// Smaller window for comments
	if result.Source.ChunkType == "comment" {
		windowSize = int(float64(windowSize) * 0.7)
	}

	return windowSize
}

func (cr *ContextRetriever) adjustForSmartBoundaries(lines []string, contextStart, contextEnd, resultStart, resultEnd int) (int, int) {
	// Try to respect function and class boundaries

	// Look backwards for function/class start
	for i := contextStart; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if cr.isFunctionOrClassStart(line) {
			contextStart = i
			break
		}
	}

	// Look forwards for function/class end
	for i := contextEnd; i < len(lines); i++ {
		line := strings.TrimSpace(lines[i])
		if cr.isFunctionOrClassEnd(line) {
			contextEnd = i
			break
		}
	}

	return contextStart, contextEnd
}

func (cr *ContextRetriever) isFunctionOrClassStart(line string) bool {
	// Simple heuristic for detecting function/class starts
	return strings.HasPrefix(line, "func ") ||
		strings.HasPrefix(line, "class ") ||
		strings.HasPrefix(line, "type ") ||
		strings.HasPrefix(line, "def ") ||
		strings.HasPrefix(line, "function ")
}

func (cr *ContextRetriever) isFunctionOrClassEnd(line string) bool {
	// Simple heuristic for detecting function/class ends
	return line == "}" || line == "end" || (len(line) > 0 && line[0] != ' ' && line[0] != '\t')
}

func (cr *ContextRetriever) extractFullFunction(lines []string, source *SourceInfo) string {
	if source.StartLine <= 0 || source.EndLine <= 0 || source.EndLine >= len(lines) {
		return ""
	}

	functionLines := lines[source.StartLine-1 : source.EndLine]
	return strings.Join(functionLines, "\n")
}

func (cr *ContextRetriever) extractFullClass(lines []string, source *SourceInfo) string {
	if source.StartLine <= 0 || source.EndLine <= 0 || source.EndLine >= len(lines) {
		return ""
	}

	classLines := lines[source.StartLine-1 : source.EndLine]
	return strings.Join(classLines, "\n")
}

func (cr *ContextRetriever) calculateFunctionRelevance(result *SearchResult, function *ContextItem) float32 {
	score := float32(0.0)

	// Proximity score
	distance := math.Abs(float64(function.StartLine - result.Source.StartLine))
	proximityScore := float32(1.0 / (1.0 + distance/100.0)) // Normalize by 100 lines
	score += proximityScore * cr.config.RankingWeights.Proximity

	// Name similarity score
	nameSimilarity := cr.calculateNameSimilarity(result.Source.FunctionName, function.Name)
	score += nameSimilarity * cr.config.RankingWeights.Semantic

	// Same file bonus
	if function.FilePath == result.Source.FilePath {
		score += 0.2
	}

	return score
}

func (cr *ContextRetriever) calculateClassRelevance(result *SearchResult, class *ContextItem) float32 {
	score := float32(0.0)

	// Similar to function relevance but with class-specific logic
	distance := math.Abs(float64(class.StartLine - result.Source.StartLine))
	proximityScore := float32(1.0 / (1.0 + distance/100.0))
	score += proximityScore * cr.config.RankingWeights.Proximity

	// Same file bonus
	if class.FilePath == result.Source.FilePath {
		score += 0.3
	}

	return score
}

func (cr *ContextRetriever) calculateNameSimilarity(name1, name2 string) float32 {
	// Simple string similarity calculation
	if name1 == name2 {
		return 1.0
	}

	// Convert to lowercase for comparison
	name1 = strings.ToLower(name1)
	name2 = strings.ToLower(name2)

	// Check for common prefixes/suffixes
	if strings.HasPrefix(name1, name2) || strings.HasPrefix(name2, name1) {
		return 0.8
	}

	if strings.Contains(name1, name2) || strings.Contains(name2, name1) {
		return 0.6
	}

	// Calculate edit distance (simplified)
	return cr.calculateEditDistanceSimilarity(name1, name2)
}

func (cr *ContextRetriever) calculateEditDistanceSimilarity(s1, s2 string) float32 {
	// Simplified Levenshtein distance
	if len(s1) == 0 {
		return float32(len(s2))
	}
	if len(s2) == 0 {
		return float32(len(s1))
	}

	// For simplicity, return a basic similarity
	maxLen := math.Max(float64(len(s1)), float64(len(s2)))
	return float32(1.0 - math.Abs(float64(len(s1)-len(s2)))/maxLen)
}

func (cr *ContextRetriever) calculateContextScore(context *ExpandedContext) float32 {
	score := float32(0.0)

	// Base score from number of items
	totalItems := float32(context.TotalItems)
	score += totalItems * 0.1

	// Bonus for cross-file references
	if context.CrossFileCount > 0 {
		score += float32(context.CrossFileCount) * 0.2
	}

	// Bonus for hierarchical context
	if context.ClassHierarchy != nil || context.CallHierarchy != nil {
		score += 0.3
	}

	// Bonus for semantic similarity
	if len(context.SemanticSimilar) > 0 {
		score += float32(len(context.SemanticSimilar)) * 0.15
	}

	// Cap at 1.0
	if score > 1.0 {
		score = 1.0
	}

	return score
}

func (cr *ContextRetriever) countContextItems(context *ExpandedContext) int {
	count := 0
	count += len(context.RelatedFunctions)
	count += len(context.RelatedClasses)
	count += len(context.Dependencies)
	count += len(context.Usages)
	count += len(context.Imports)
	count += len(context.SemanticSimilar)

	for _, crossFile := range context.CrossFileRefs {
		count += len(crossFile.Items)
	}

	return count
}

// Placeholder implementations for complex operations
// These would be implemented with actual dependency graph queries

func (cr *ContextRetriever) readFileContent(filePath string) (string, error) {
	// This would read from the file system or cache
	return "", fmt.Errorf("file reading not implemented")
}

func (cr *ContextRetriever) getFunctionsInFile(filePath string) []*ContextItem {
	// This would query the index for functions in the file
	return []*ContextItem{}
}

func (cr *ContextRetriever) getClassesInFile(filePath string) []*ContextItem {
	// This would query the index for classes in the file
	return []*ContextItem{}
}

func (cr *ContextRetriever) extractDependenciesFromCode(result *SearchResult) []*ContextItem {
	// This would analyze the code for dependencies
	return []*ContextItem{}
}

func (cr *ContextRetriever) findFunctionUsages(functionName string) []*ContextItem {
	// This would search for function call sites
	return []*ContextItem{}
}

func (cr *ContextRetriever) findClassUsages(className string) []*ContextItem {
	// This would search for class instantiations
	return []*ContextItem{}
}

func (cr *ContextRetriever) getFileImports(filePath string) []*ContextItem {
	// This would extract import statements from the file
	return []*ContextItem{}
}

func (cr *ContextRetriever) isRelevantImport(result *SearchResult, imp *ContextItem) bool {
	// This would determine if an import is relevant to the result
	return true
}

func (cr *ContextRetriever) findRelatedFiles(result *SearchResult) []string {
	// This would find files related through imports, dependencies, etc.
	return []string{}
}

func (cr *ContextRetriever) getReferencesInFile(filePath string, result *SearchResult) []*ContextItem {
	// This would find references to the result in the specified file
	return []*ContextItem{}
}

func (cr *ContextRetriever) calculateFileRelevance(result *SearchResult, filePath string) float32 {
	// This would calculate how relevant a file is to the result
	return 0.5
}

func (cr *ContextRetriever) findParentClasses(className string) []*ContextItem {
	// This would find parent classes using the dependency graph
	return []*ContextItem{}
}

func (cr *ContextRetriever) findChildClasses(className string) []*ContextItem {
	// This would find child classes using the dependency graph
	return []*ContextItem{}
}

func (cr *ContextRetriever) findImplementedInterfaces(className string) []*ContextItem {
	// This would find interfaces implemented by the class
	return []*ContextItem{}
}

func (cr *ContextRetriever) findFunctionCallers(functionName string) []*ContextItem {
	// This would find functions that call the specified function
	return []*ContextItem{}
}

func (cr *ContextRetriever) findFunctionCallees(functionName string) []*ContextItem {
	// This would find functions called by the specified function
	return []*ContextItem{}
}

func (cr *ContextRetriever) buildCallChain(functionName string) []*ContextItem {
	// This would build a call chain for the function
	return []*ContextItem{}
}

// Cache and statistics methods

func (cr *ContextRetriever) generateContextCacheKey(result *SearchResult, windowSize int) string {
	return fmt.Sprintf("ctx_%s_%d_%d_%d",
		result.ID, result.Source.StartLine, result.Source.EndLine, windowSize)
}

func (cr *ContextRetriever) updateCacheStats(hit bool) {
	cr.stats.mu.Lock()
	defer cr.stats.mu.Unlock()

	if hit {
		cr.stats.CacheHits++
	} else {
		cr.stats.CacheMisses++
	}
}

func (cr *ContextRetriever) updateRetrievalStats(context *ExpandedContext) {
	cr.stats.mu.Lock()
	defer cr.stats.mu.Unlock()

	cr.stats.TotalRetrievals++

	if context.CrossFileCount > 0 {
		cr.stats.CrossFileRetrievals++
	}

	if len(context.SemanticSimilar) > 0 {
		cr.stats.SemanticRetrievals++
	}

	// Update average retrieval time
	if cr.stats.AverageRetrievalTime == 0 {
		cr.stats.AverageRetrievalTime = context.RetrievalTime
	} else {
		cr.stats.AverageRetrievalTime = (cr.stats.AverageRetrievalTime + context.RetrievalTime) / 2
	}
}

// Public API

func (cr *ContextRetriever) GetStatistics() *ContextStatistics {
	cr.stats.mu.RLock()
	defer cr.stats.mu.RUnlock()

	stats := *cr.stats
	return &stats
}
