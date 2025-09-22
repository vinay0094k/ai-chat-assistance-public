package vectordb

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/indexer"
)

// SimilarityEngine provides advanced similarity calculations for code and embeddings
type SimilarityEngine struct {
	// Configuration
	config *SimilarityConfig

	// Caching
	similarityCache *SimilarityCache
	resultCache     *ResultCache

	// Optimization
	vectorOptimizer *VectorOptimizer
	batchProcessor  *SimilarityBatchProcessor

	// Analytics
	stats *SimilarityStatistics

	// State management
	mu sync.RWMutex
}

// SimilarityConfig contains similarity calculation configuration
type SimilarityConfig struct {
	// Algorithm preferences
	DefaultMetric        SimilarityMetric `json:"default_metric"`
	EnableApproximate    bool             `json:"enable_approximate"`
	ApproximateThreshold float32          `json:"approximate_threshold"`

	// Performance settings
	BatchSize       int           `json:"batch_size"`
	ParallelWorkers int           `json:"parallel_workers"`
	EnableCaching   bool          `json:"enable_caching"`
	CacheSize       int           `json:"cache_size"`
	CacheTTL        time.Duration `json:"cache_ttl"`

	// Code-specific settings
	EnableCodeSimilarity bool    `json:"enable_code_similarity"`
	WeightSyntax         float32 `json:"weight_syntax"`
	WeightSemantic       float32 `json:"weight_semantic"`
	WeightStructural     float32 `json:"weight_structural"`

	// Optimization settings
	EnableQuantization  bool `json:"enable_quantization"`
	QuantizationBits    int  `json:"quantization_bits"`
	EnableNormalization bool `json:"enable_normalization"`

	// Quality settings
	MinSimilarityThreshold float32 `json:"min_similarity_threshold"`
	MaxResults             int     `json:"max_results"`
	EnableFiltering        bool    `json:"enable_filtering"`
}

// SimilarityMetric represents different similarity calculation methods
type SimilarityMetric string

const (
	MetricCosine         SimilarityMetric = "cosine"
	MetricEuclidean      SimilarityMetric = "euclidean"
	MetricDotProduct     SimilarityMetric = "dot_product"
	MetricManhattan      SimilarityMetric = "manhattan"
	MetricJaccard        SimilarityMetric = "jaccard"
	MetricHamming        SimilarityMetric = "hamming"
	MetricKLDivergence   SimilarityMetric = "kl_divergence"
	MetricJSDistance     SimilarityMetric = "js_distance"
	MetricCodeSimilarity SimilarityMetric = "code_similarity"
)

// SimilarityResult represents the result of a similarity calculation
type SimilarityResult struct {
	ID1             string                 `json:"id1"`
	ID2             string                 `json:"id2"`
	Similarity      float32                `json:"similarity"`
	Distance        float32                `json:"distance"`
	Metric          SimilarityMetric       `json:"metric"`
	Confidence      float32                `json:"confidence"`
	Components      *SimilarityComponents  `json:"components,omitempty"`
	CalculationTime time.Duration          `json:"calculation_time"`
	CacheHit        bool                   `json:"cache_hit"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

// SimilarityComponents breaks down similarity into different aspects
type SimilarityComponents struct {
	Syntactic  float32 `json:"syntactic"`  // Syntax-level similarity
	Semantic   float32 `json:"semantic"`   // Semantic similarity
	Structural float32 `json:"structural"` // Code structure similarity
	Lexical    float32 `json:"lexical"`    // Variable/function name similarity
	Behavioral float32 `json:"behavioral"` // Execution behavior similarity
	Contextual float32 `json:"contextual"` // Context similarity
}

// BatchSimilarityRequest represents a batch similarity calculation request
type BatchSimilarityRequest struct {
	QueryVector    []float32        `json:"query_vector"`
	TargetVectors  []VectorWithID   `json:"target_vectors"`
	Metric         SimilarityMetric `json:"metric"`
	TopK           int              `json:"top_k"`
	Threshold      float32          `json:"threshold"`
	EnableParallel bool             `json:"enable_parallel"`
}

// VectorWithID represents a vector with its identifier
type VectorWithID struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// BatchSimilarityResponse represents batch similarity results
type BatchSimilarityResponse struct {
	Results        []*SimilarityResult `json:"results"`
	TotalProcessed int                 `json:"total_processed"`
	ProcessingTime time.Duration       `json:"processing_time"`
	CacheHits      int                 `json:"cache_hits"`
	CacheMisses    int                 `json:"cache_misses"`
}

// CodeSimilarityRequest represents a code-specific similarity request
type CodeSimilarityRequest struct {
	Code1         *indexer.CodeChunk `json:"code1"`
	Code2         *indexer.CodeChunk `json:"code2"`
	Vector1       []float32          `json:"vector1,omitempty"`
	Vector2       []float32          `json:"vector2,omitempty"`
	AnalysisDepth AnalysisDepth      `json:"analysis_depth"`
	WeightProfile *SimilarityWeights `json:"weight_profile,omitempty"`
}

// AnalysisDepth represents the depth of code similarity analysis
type AnalysisDepth string

const (
	DepthSurface    AnalysisDepth = "surface"    // Basic text comparison
	DepthSyntactic  AnalysisDepth = "syntactic"  // AST-based comparison
	DepthSemantic   AnalysisDepth = "semantic"   // Semantic understanding
	DepthBehavioral AnalysisDepth = "behavioral" // Execution behavior analysis
	DepthComplete   AnalysisDepth = "complete"   // Full analysis
)

// SimilarityWeights defines weights for different similarity aspects
type SimilarityWeights struct {
	Syntactic  float32 `json:"syntactic"`
	Semantic   float32 `json:"semantic"`
	Structural float32 `json:"structural"`
	Lexical    float32 `json:"lexical"`
	Behavioral float32 `json:"behavioral"`
	Contextual float32 `json:"contextual"`
}

// SimilarityStatistics tracks similarity calculation performance
type SimilarityStatistics struct {
	TotalCalculations       int64                      `json:"total_calculations"`
	CalculationsByMetric    map[SimilarityMetric]int64 `json:"calculations_by_metric"`
	AverageCalculationTime  time.Duration              `json:"average_calculation_time"`
	CacheHitRate            float64                    `json:"cache_hit_rate"`
	BatchCalculations       int64                      `json:"batch_calculations"`
	ParallelCalculations    int64                      `json:"parallel_calculations"`
	ApproximateCalculations int64                      `json:"approximate_calculations"`
	ErrorCount              int64                      `json:"error_count"`
	LastError               string                     `json:"last_error,omitempty"`
	mu                      sync.RWMutex
}

// NewSimilarityEngine creates a new similarity engine
func NewSimilarityEngine(config *SimilarityConfig) *SimilarityEngine {
	if config == nil {
		config = &SimilarityConfig{
			DefaultMetric:          MetricCosine,
			EnableApproximate:      true,
			ApproximateThreshold:   0.9,
			BatchSize:              100,
			ParallelWorkers:        4,
			EnableCaching:          true,
			CacheSize:              10000,
			CacheTTL:               time.Hour,
			EnableCodeSimilarity:   true,
			WeightSyntax:           0.3,
			WeightSemantic:         0.4,
			WeightStructural:       0.3,
			EnableQuantization:     false,
			QuantizationBits:       8,
			EnableNormalization:    true,
			MinSimilarityThreshold: 0.1,
			MaxResults:             100,
			EnableFiltering:        true,
		}
	}

	se := &SimilarityEngine{
		config: config,
		stats: &SimilarityStatistics{
			CalculationsByMetric: make(map[SimilarityMetric]int64),
		},
	}

	// Initialize components
	se.initializeComponents()

	return se
}

// CalculateSimilarity calculates similarity between two vectors
func (se *SimilarityEngine) CalculateSimilarity(ctx context.Context, vector1, vector2 []float32, metric SimilarityMetric) (*SimilarityResult, error) {
	start := time.Now()

	// Validate inputs
	if len(vector1) != len(vector2) {
		return nil, fmt.Errorf("vectors must have same length: %d != %d", len(vector1), len(vector2))
	}

	if len(vector1) == 0 {
		return nil, fmt.Errorf("vectors cannot be empty")
	}

	// Generate cache key
	cacheKey := ""
	if se.config.EnableCaching {
		cacheKey = se.generateCacheKey(vector1, vector2, metric)
		if cached := se.similarityCache.Get(cacheKey); cached != nil {
			cached.CacheHit = true
			se.updateCacheStats(true)
			return cached, nil
		}
	}

	// Use default metric if not specified
	if metric == "" {
		metric = se.config.DefaultMetric
	}

	// Calculate similarity based on metric
	var similarity float32
	var err error

	switch metric {
	case MetricCosine:
		similarity = se.calculateCosineSimilarity(vector1, vector2)
	case MetricEuclidean:
		similarity = se.calculateEuclideanSimilarity(vector1, vector2)
	case MetricDotProduct:
		similarity = se.calculateDotProductSimilarity(vector1, vector2)
	case MetricManhattan:
		similarity = se.calculateManhattanSimilarity(vector1, vector2)
	case MetricJaccard:
		similarity = se.calculateJaccardSimilarity(vector1, vector2)
	case MetricHamming:
		similarity = se.calculateHammingSimilarity(vector1, vector2)
	case MetricKLDivergence:
		similarity = se.calculateKLDivergence(vector1, vector2)
	case MetricJSDistance:
		similarity = se.calculateJSDistance(vector1, vector2)
	default:
		return nil, fmt.Errorf("unsupported similarity metric: %s", metric)
	}

	if err != nil {
		se.updateErrorStats(err)
		return nil, fmt.Errorf("similarity calculation failed: %v", err)
	}

	// Calculate distance (1 - similarity for most metrics)
	distance := 1.0 - similarity
	if metric == MetricEuclidean || metric == MetricManhattan {
		distance = similarity                 // These are already distance metrics
		similarity = 1.0 / (1.0 + similarity) // Convert to similarity
	}

	// Calculate confidence
	confidence := se.calculateConfidence(similarity, metric, len(vector1))

	result := &SimilarityResult{
		ID1:             se.generateVectorID(vector1),
		ID2:             se.generateVectorID(vector2),
		Similarity:      similarity,
		Distance:        distance,
		Metric:          metric,
		Confidence:      confidence,
		CalculationTime: time.Since(start),
		CacheHit:        false,
	}

	// Cache the result
	if se.config.EnableCaching && cacheKey != "" {
		se.similarityCache.Set(cacheKey, result)
		se.updateCacheStats(false)
	}

	// Update statistics
	se.updateCalculationStats(metric, time.Since(start))

	return result, nil
}

// BatchCalculateSimilarity calculates similarity for multiple vector pairs
func (se *SimilarityEngine) BatchCalculateSimilarity(ctx context.Context, request *BatchSimilarityRequest) (*BatchSimilarityResponse, error) {
	start := time.Now()

	if len(request.TargetVectors) == 0 {
		return &BatchSimilarityResponse{
			Results:        []*SimilarityResult{},
			TotalProcessed: 0,
			ProcessingTime: time.Since(start),
		}, nil
	}

	var results []*SimilarityResult
	var cacheHits, cacheMisses int

	if request.EnableParallel && len(request.TargetVectors) > se.config.BatchSize {
		results, cacheHits, cacheMisses = se.parallelBatchCalculation(ctx, request)
	} else {
		results, cacheHits, cacheMisses = se.sequentialBatchCalculation(ctx, request)
	}

	// Sort results by similarity if TopK is specified
	if request.TopK > 0 && len(results) > request.TopK {
		sort.Slice(results, func(i, j int) bool {
			return results[i].Similarity > results[j].Similarity
		})
		results = results[:request.TopK]
	}

	// Filter by threshold
	if request.Threshold > 0 {
		filtered := make([]*SimilarityResult, 0)
		for _, result := range results {
			if result.Similarity >= request.Threshold {
				filtered = append(filtered, result)
			}
		}
		results = filtered
	}

	response := &BatchSimilarityResponse{
		Results:        results,
		TotalProcessed: len(request.TargetVectors),
		ProcessingTime: time.Since(start),
		CacheHits:      cacheHits,
		CacheMisses:    cacheMisses,
	}

	// Update statistics
	se.updateBatchStats(len(request.TargetVectors), time.Since(start))

	return response, nil
}

// CalculateCodeSimilarity calculates similarity between code chunks
func (se *SimilarityEngine) CalculateCodeSimilarity(ctx context.Context, request *CodeSimilarityRequest) (*SimilarityResult, error) {
	if !se.config.EnableCodeSimilarity {
		return nil, fmt.Errorf("code similarity calculation is disabled")
	}

	start := time.Now()

	// Use default weights if not provided
	weights := request.WeightProfile
	if weights == nil {
		weights = &SimilarityWeights{
			Syntactic:  se.config.WeightSyntax,
			Semantic:   se.config.WeightSemantic,
			Structural: se.config.WeightStructural,
			Lexical:    0.1,
			Behavioral: 0.1,
			Contextual: 0.1,
		}
	}

	components := &SimilarityComponents{}

	// Calculate different similarity aspects based on analysis depth
	switch request.AnalysisDepth {
	case DepthSurface:
		components.Lexical = se.calculateLexicalSimilarity(request.Code1, request.Code2)
	case DepthSyntactic:
		components.Syntactic = se.calculateSyntacticSimilarity(request.Code1, request.Code2)
		components.Lexical = se.calculateLexicalSimilarity(request.Code1, request.Code2)
	case DepthSemantic:
		if len(request.Vector1) > 0 && len(request.Vector2) > 0 {
			semResult, _ := se.CalculateSimilarity(ctx, request.Vector1, request.Vector2, MetricCosine)
			if semResult != nil {
				components.Semantic = semResult.Similarity
			}
		}
		components.Syntactic = se.calculateSyntacticSimilarity(request.Code1, request.Code2)
		components.Structural = se.calculateStructuralSimilarity(request.Code1, request.Code2)
	case DepthBehavioral:
		// Full analysis including behavioral similarity (expensive)
		components.Behavioral = se.calculateBehavioralSimilarity(request.Code1, request.Code2)
		fallthrough
	case DepthComplete:
		// Calculate all aspects
		if len(request.Vector1) > 0 && len(request.Vector2) > 0 {
			semResult, _ := se.CalculateSimilarity(ctx, request.Vector1, request.Vector2, MetricCosine)
			if semResult != nil {
				components.Semantic = semResult.Similarity
			}
		}
		components.Syntactic = se.calculateSyntacticSimilarity(request.Code1, request.Code2)
		components.Structural = se.calculateStructuralSimilarity(request.Code1, request.Code2)
		components.Lexical = se.calculateLexicalSimilarity(request.Code1, request.Code2)
		components.Contextual = se.calculateContextualSimilarity(request.Code1, request.Code2)
		if request.AnalysisDepth == DepthComplete {
			components.Behavioral = se.calculateBehavioralSimilarity(request.Code1, request.Code2)
		}
	}

	// Calculate weighted overall similarity
	overallSimilarity :=
		components.Syntactic*weights.Syntactic +
			components.Semantic*weights.Semantic +
			components.Structural*weights.Structural +
			components.Lexical*weights.Lexical +
			components.Behavioral*weights.Behavioral +
			components.Contextual*weights.Contextual

	result := &SimilarityResult{
		ID1:             request.Code1.ID,
		ID2:             request.Code2.ID,
		Similarity:      overallSimilarity,
		Distance:        1.0 - overallSimilarity,
		Metric:          MetricCodeSimilarity,
		Confidence:      se.calculateCodeConfidence(components, request.AnalysisDepth),
		Components:      components,
		CalculationTime: time.Since(start),
		Metadata: map[string]interface{}{
			"analysis_depth": request.AnalysisDepth,
			"language1":      request.Code1.Language,
			"language2":      request.Code2.Language,
			"chunk_type1":    request.Code1.ChunkType,
			"chunk_type2":    request.Code2.ChunkType,
		},
	}

	return result, nil
}

// Individual similarity calculation methods

func (se *SimilarityEngine) calculateCosineSimilarity(v1, v2 []float32) float32 {
	if len(v1) != len(v2) {
		return 0
	}

	var dotProduct, normA, normB float64

	for i := 0; i < len(v1); i++ {
		dotProduct += float64(v1[i] * v2[i])
		normA += float64(v1[i] * v1[i])
		normB += float64(v2[i] * v2[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return float32(dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)))
}

func (se *SimilarityEngine) calculateEuclideanSimilarity(v1, v2 []float32) float32 {
	var sum float64
	for i := 0; i < len(v1); i++ {
		diff := float64(v1[i] - v2[i])
		sum += diff * diff
	}
	return float32(math.Sqrt(sum))
}

func (se *SimilarityEngine) calculateDotProductSimilarity(v1, v2 []float32) float32 {
	var dotProduct float32
	for i := 0; i < len(v1); i++ {
		dotProduct += v1[i] * v2[i]
	}
	return dotProduct
}

func (se *SimilarityEngine) calculateManhattanSimilarity(v1, v2 []float32) float32 {
	var sum float32
	for i := 0; i < len(v1); i++ {
		sum += float32(math.Abs(float64(v1[i] - v2[i])))
	}
	return sum
}

func (se *SimilarityEngine) calculateJaccardSimilarity(v1, v2 []float32) float32 {
	var intersection, union float32

	for i := 0; i < len(v1); i++ {
		minVal := float32(math.Min(float64(v1[i]), float64(v2[i])))
		maxVal := float32(math.Max(float64(v1[i]), float64(v2[i])))
		intersection += minVal
		union += maxVal
	}

	if union == 0 {
		return 0
	}

	return intersection / union
}

func (se *SimilarityEngine) calculateHammingSimilarity(v1, v2 []float32) float32 {
	different := 0
	threshold := float32(0.5) // Consider values > 0.5 as 1, others as 0

	for i := 0; i < len(v1); i++ {
		bit1 := v1[i] > threshold
		bit2 := v2[i] > threshold
		if bit1 != bit2 {
			different++
		}
	}

	return 1.0 - float32(different)/float32(len(v1))
}

func (se *SimilarityEngine) calculateKLDivergence(v1, v2 []float32) float32 {
	// Normalize vectors to probability distributions
	p := se.normalizeToDistribution(v1)
	q := se.normalizeToDistribution(v2)

	var kl float64
	for i := 0; i < len(p); i++ {
		if p[i] > 0 && q[i] > 0 {
			kl += float64(p[i] * float32(math.Log(float64(p[i]/q[i]))))
		}
	}

	// Convert to similarity (lower KL = higher similarity)
	return float32(1.0 / (1.0 + kl))
}

func (se *SimilarityEngine) calculateJSDistance(v1, v2 []float32) float32 {
	// Jensen-Shannon distance
	p := se.normalizeToDistribution(v1)
	q := se.normalizeToDistribution(v2)

	// Calculate M = (P + Q) / 2
	m := make([]float32, len(p))
	for i := 0; i < len(p); i++ {
		m[i] = (p[i] + q[i]) / 2
	}

	// Calculate KL divergences
	klPM := se.calculateKLDiv(p, m)
	klQM := se.calculateKLDiv(q, m)

	// JS divergence = (KL(P||M) + KL(Q||M)) / 2
	jsDiv := (klPM + klQM) / 2

	// Convert to similarity
	return float32(1.0 / (1.0 + jsDiv))
}

// Code-specific similarity methods

func (se *SimilarityEngine) calculateSyntacticSimilarity(code1, code2 *indexer.CodeChunk) float32 {
	// Compare AST structures (simplified implementation)
	if code1.Language != code2.Language {
		return 0.0 // Different languages
	}

	// Simple token-based comparison for now
	tokens1 := se.tokenizeCode(code1.Content)
	tokens2 := se.tokenizeCode(code2.Content)

	return se.calculateTokenSimilarity(tokens1, tokens2)
}

func (se *SimilarityEngine) calculateStructuralSimilarity(code1, code2 *indexer.CodeChunk) float32 {
	// Compare code structure (control flow, nesting, etc.)
	struct1 := se.extractStructuralFeatures(code1)
	struct2 := se.extractStructuralFeatures(code2)

	return se.compareStructuralFeatures(struct1, struct2)
}

func (se *SimilarityEngine) calculateLexicalSimilarity(code1, code2 *indexer.CodeChunk) float32 {
	// Compare variable names, function names, etc.
	names1 := se.extractIdentifiers(code1.Content)
	names2 := se.extractIdentifiers(code2.Content)

	return se.calculateNameSimilarity(names1, names2)
}

func (se *SimilarityEngine) calculateBehavioralSimilarity(code1, code2 *indexer.CodeChunk) float32 {
	// Compare expected behavior (very simplified)
	// In practice, this would involve more sophisticated analysis

	// For now, compare function signatures and return patterns
	sig1 := code1.Signature
	sig2 := code2.Signature

	if sig1 == "" || sig2 == "" {
		return 0.5 // Neutral similarity if no signatures
	}

	return se.compareSignatures(sig1, sig2)
}

func (se *SimilarityEngine) calculateContextualSimilarity(code1, code2 *indexer.CodeChunk) float32 {
	// Compare context (file paths, surrounding code, etc.)
	contextScore := float32(0.0)

	// Same file bonus
	if code1.FilePath == code2.FilePath {
		contextScore += 0.3
	}

	// Same chunk type bonus
	if code1.ChunkType == code2.ChunkType {
		contextScore += 0.2
	}

	// Proximity bonus (if same file)
	if code1.FilePath == code2.FilePath {
		lineDiff := math.Abs(float64(code1.StartLine - code2.StartLine))
		proximityScore := 1.0 / (1.0 + lineDiff/100.0) // Normalize by 100 lines
		contextScore += float32(proximityScore * 0.3)
	}

	// Similar complexity bonus
	complexityDiff := math.Abs(float64(code1.Complexity - code2.Complexity))
	complexityScore := 1.0 / (1.0 + complexityDiff/5.0) // Normalize by 5 complexity units
	contextScore += float32(complexityScore * 0.2)

	if contextScore > 1.0 {
		contextScore = 1.0
	}

	return contextScore
}

// Helper methods for parallel processing

func (se *SimilarityEngine) parallelBatchCalculation(ctx context.Context, request *BatchSimilarityRequest) ([]*SimilarityResult, int, int) {
	numWorkers := se.config.ParallelWorkers
	chunkSize := (len(request.TargetVectors) + numWorkers - 1) / numWorkers

	resultChan := make(chan []*SimilarityResult, numWorkers)
	statsChan := make(chan [2]int, numWorkers) // [cacheHits, cacheMisses]

	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()

			end := start + chunkSize
			if end > len(request.TargetVectors) {
				end = len(request.TargetVectors)
			}

			chunkRequest := &BatchSimilarityRequest{
				QueryVector:    request.QueryVector,
				TargetVectors:  request.TargetVectors[start:end],
				Metric:         request.Metric,
				Threshold:      request.Threshold,
				EnableParallel: false, // Prevent nested parallelization
			}

			chunkResults, hits, misses := se.sequentialBatchCalculation(ctx, chunkRequest)
			resultChan <- chunkResults
			statsChan <- [2]int{hits, misses}
		}(i * chunkSize)
	}

	wg.Wait()
	close(resultChan)
	close(statsChan)

	// Collect results
	var allResults []*SimilarityResult
	var totalHits, totalMisses int

	for results := range resultChan {
		allResults = append(allResults, results...)
	}

	for stats := range statsChan {
		totalHits += stats[0]
		totalMisses += stats[1]
	}

	se.updateParallelStats()

	return allResults, totalHits, totalMisses
}

func (se *SimilarityEngine) sequentialBatchCalculation(ctx context.Context, request *BatchSimilarityRequest) ([]*SimilarityResult, int, int) {
	var results []*SimilarityResult
	var cacheHits, cacheMisses int

	for _, target := range request.TargetVectors {
		result, err := se.CalculateSimilarity(ctx, request.QueryVector, target.Vector, request.Metric)
		if err != nil {
			continue
		}

		result.ID2 = target.ID
		if target.Metadata != nil {
			result.Metadata = target.Metadata
		}

		if result.CacheHit {
			cacheHits++
		} else {
			cacheMisses++
		}

		results = append(results, result)
	}

	return results, cacheHits, cacheMisses
}

// Utility methods

func (se *SimilarityEngine) normalizeToDistribution(v []float32) []float32 {
	var sum float32
	for _, val := range v {
		if val > 0 {
			sum += val
		}
	}

	if sum == 0 {
		// Uniform distribution
		uniform := 1.0 / float32(len(v))
		result := make([]float32, len(v))
		for i := range result {
			result[i] = uniform
		}
		return result
	}

	result := make([]float32, len(v))
	for i, val := range v {
		if val > 0 {
			result[i] = val / sum
		} else {
			result[i] = 1e-10 // Small epsilon to avoid log(0)
		}
	}

	return result
}

func (se *SimilarityEngine) calculateKLDiv(p, q []float32) float64 {
	var kl float64
	for i := 0; i < len(p); i++ {
		if p[i] > 0 && q[i] > 0 {
			kl += float64(p[i] * float32(math.Log(float64(p[i]/q[i]))))
		}
	}
	return kl
}

func (se *SimilarityEngine) tokenizeCode(code string) []string {
	// Simple tokenization (in practice, would use proper lexer)
	// Remove comments and strings for better comparison
	cleaned := se.cleanCode(code)

	// Split by common delimiters
	tokens := strings.FieldsFunc(cleaned, func(r rune) bool {
		return r == ' ' || r == '\t' || r == '\n' || r == '(' || r == ')' ||
			r == '{' || r == '}' || r == '[' || r == ']' || r == ';' ||
			r == ',' || r == '.' || r == '=' || r == '+' || r == '-'
	})

	// Filter out empty tokens
	var result []string
	for _, token := range tokens {
		if token != "" {
			result = append(result, strings.ToLower(token))
		}
	}

	return result
}

func (se *SimilarityEngine) cleanCode(code string) string {
	// Remove comments and strings (very simplified)
	lines := strings.Split(code, "\n")
	var cleaned []string

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "//") && !strings.HasPrefix(line, "#") {
			// Remove string literals (simplified)
			line = se.removeStringLiterals(line)
			cleaned = append(cleaned, line)
		}
	}

	return strings.Join(cleaned, "\n")
}

func (se *SimilarityEngine) removeStringLiterals(line string) string {
	// Very simplified string literal removal
	result := ""
	inString := false
	quote := byte(0)

	for i := 0; i < len(line); i++ {
		char := line[i]

		if !inString {
			if char == '"' || char == '\'' {
				inString = true
				quote = char
				result += " " // Replace string with space
			} else {
				result += string(char)
			}
		} else {
			if char == quote && (i == 0 || line[i-1] != '\\') {
				inString = false
			}
		}
	}

	return result
}

func (se *SimilarityEngine) calculateTokenSimilarity(tokens1, tokens2 []string) float32 {
	if len(tokens1) == 0 && len(tokens2) == 0 {
		return 1.0
	}

	if len(tokens1) == 0 || len(tokens2) == 0 {
		return 0.0
	}

	// Create frequency maps
	freq1 := make(map[string]int)
	freq2 := make(map[string]int)

	for _, token := range tokens1 {
		freq1[token]++
	}

	for _, token := range tokens2 {
		freq2[token]++
	}

	// Calculate Jaccard similarity
	intersection := 0
	union := 0

	allTokens := make(map[string]bool)
	for token := range freq1 {
		allTokens[token] = true
	}
	for token := range freq2 {
		allTokens[token] = true
	}

	for token := range allTokens {
		f1, exists1 := freq1[token]
		f2, exists2 := freq2[token]

		if exists1 && exists2 {
			intersection += int(math.Min(float64(f1), float64(f2)))
		}

		if exists1 {
			union += f1
		}
		if exists2 {
			union += f2
		}
	}

	union -= intersection // Remove double counting

	if union == 0 {
		return 1.0
	}

	return float32(intersection) / float32(union)
}

func (se *SimilarityEngine) extractStructuralFeatures(code *indexer.CodeChunk) map[string]int {
	features := make(map[string]int)
	content := code.Content

	// Count structural elements
	features["lines"] = strings.Count(content, "\n") + 1
	features["functions"] = strings.Count(strings.ToLower(content), "func")
	features["classes"] = strings.Count(strings.ToLower(content), "class")
	features["if_statements"] = strings.Count(strings.ToLower(content), "if")
	features["for_loops"] = strings.Count(strings.ToLower(content), "for")
	features["while_loops"] = strings.Count(strings.ToLower(content), "while")
	features["braces"] = strings.Count(content, "{")
	features["parentheses"] = strings.Count(content, "(")
	features["complexity"] = code.Complexity

	return features
}

func (se *SimilarityEngine) compareStructuralFeatures(features1, features2 map[string]int) float32 {
	if len(features1) == 0 && len(features2) == 0 {
		return 1.0
	}

	var totalDiff, maxPossible float64

	allFeatures := make(map[string]bool)
	for feature := range features1 {
		allFeatures[feature] = true
	}
	for feature := range features2 {
		allFeatures[feature] = true
	}

	for feature := range allFeatures {
		val1 := features1[feature]
		val2 := features2[feature]

		diff := math.Abs(float64(val1 - val2))
		maxVal := math.Max(float64(val1), float64(val2))

		totalDiff += diff
		maxPossible += maxVal
	}

	if maxPossible == 0 {
		return 1.0
	}

	return float32(1.0 - (totalDiff / maxPossible))
}

func (se *SimilarityEngine) extractIdentifiers(code string) []string {
	// Extract identifiers (variable names, function names, etc.)
	// This is a simplified implementation

	var identifiers []string

	// Simple pattern matching for identifiers
	words := strings.FieldsFunc(code, func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= 'A' && r <= 'Z') &&
			!(r >= '0' && r <= '9') && r != '_'
	})

	for _, word := range words {
		if len(word) > 1 && ((word[0] >= 'a' && word[0] <= 'z') || (word[0] >= 'A' && word[0] <= 'Z')) {
			identifiers = append(identifiers, strings.ToLower(word))
		}
	}

	return identifiers
}

func (se *SimilarityEngine) calculateNameSimilarity(names1, names2 []string) float32 {
	if len(names1) == 0 && len(names2) == 0 {
		return 1.0
	}

	// Create sets for Jaccard similarity
	set1 := make(map[string]bool)
	set2 := make(map[string]bool)

	for _, name := range names1 {
		set1[name] = true
	}

	for _, name := range names2 {
		set2[name] = true
	}

	// Calculate Jaccard similarity
	intersection := 0
	union := 0

	for name := range set1 {
		if set2[name] {
			intersection++
		}
		union++
	}

	for name := range set2 {
		if !set1[name] {
			union++
		}
	}

	if union == 0 {
		return 1.0
	}

	return float32(intersection) / float32(union)
}

func (se *SimilarityEngine) compareSignatures(sig1, sig2 string) float32 {
	// Simple signature comparison
	if sig1 == sig2 {
		return 1.0
	}

	// Compare normalized signatures
	norm1 := se.normalizeSignature(sig1)
	norm2 := se.normalizeSignature(sig2)

	if norm1 == norm2 {
		return 0.8
	}

	// Calculate edit distance similarity
	return se.calculateEditDistanceSimilarity(norm1, norm2)
}

func (se *SimilarityEngine) normalizeSignature(sig string) string {
	// Remove parameter names, keep only types
	// This is a simplified implementation
	return strings.ToLower(strings.TrimSpace(sig))
}

func (se *SimilarityEngine) calculateEditDistanceSimilarity(s1, s2 string) float32 {
	if len(s1) == 0 && len(s2) == 0 {
		return 1.0
	}

	maxLen := math.Max(float64(len(s1)), float64(len(s2)))
	if maxLen == 0 {
		return 1.0
	}

	// Simplified edit distance
	distance := math.Abs(float64(len(s1) - len(s2)))

	// Add character-by-character comparison
	minLen := int(math.Min(float64(len(s1)), float64(len(s2))))
	for i := 0; i < minLen; i++ {
		if s1[i] != s2[i] {
			distance += 1
		}
	}

	similarity := 1.0 - (distance / maxLen)
	if similarity < 0 {
		similarity = 0
	}

	return float32(similarity)
}

// Component initialization and management

func (se *SimilarityEngine) initializeComponents() {
	// Initialize caches
	if se.config.EnableCaching {
		se.similarityCache = NewSimilarityCache(se.config.CacheSize, se.config.CacheTTL)
		se.resultCache = NewResultCache(se.config.CacheSize, se.config.CacheTTL)
	}

	// Initialize batch processor
	se.batchProcessor = NewSimilarityBatchProcessor(se.config.BatchSize, se.config.ParallelWorkers)
}

func (se *SimilarityEngine) calculateConfidence(similarity float32, metric SimilarityMetric, vectorSize int) float32 {
	confidence := similarity

	// Adjust confidence based on metric reliability
	switch metric {
	case MetricCosine:
		confidence *= 0.95 // Very reliable
	case MetricEuclidean:
		confidence *= 0.85 // Less reliable for high-dimensional spaces
	case MetricDotProduct:
		confidence *= 0.80 // Depends on vector normalization
	case MetricJaccard:
		confidence *= 0.90 // Good for sparse vectors
	default:
		confidence *= 0.75 // Lower confidence for other metrics
	}

	// Adjust for vector size (higher dimensions generally more reliable)
	if vectorSize < 50 {
		confidence *= 0.8
	} else if vectorSize > 500 {
		confidence *= 1.1
		if confidence > 1.0 {
			confidence = 1.0
		}
	}

	return confidence
}

func (se *SimilarityEngine) calculateCodeConfidence(components *SimilarityComponents, depth AnalysisDepth) float32 {
	confidence := float32(0.5) // Base confidence

	// Higher confidence with more analysis
	switch depth {
	case DepthSurface:
		confidence = 0.4
	case DepthSyntactic:
		confidence = 0.6
	case DepthSemantic:
		confidence = 0.7
	case DepthBehavioral:
		confidence = 0.8
	case DepthComplete:
		confidence = 0.9
	}

	// Adjust based on component values
	avgComponent := (components.Syntactic + components.Semantic + components.Structural +
		components.Lexical + components.Behavioral + components.Contextual) / 6.0

	if avgComponent > 0.8 {
		confidence *= 1.1
	} else if avgComponent < 0.3 {
		confidence *= 0.8
	}

	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

// Cache and ID generation

func (se *SimilarityEngine) generateCacheKey(v1, v2 []float32, metric SimilarityMetric) string {
	// Generate a deterministic cache key
	id1 := se.generateVectorID(v1)
	id2 := se.generateVectorID(v2)

	// Ensure consistent ordering for cache efficiency
	if id1 > id2 {
		id1, id2 = id2, id1
	}

	return fmt.Sprintf("%s_%s_%s", id1, id2, metric)
}

func (se *SimilarityEngine) generateVectorID(vector []float32) string {
	// Generate a hash-based ID for the vector
	if len(vector) == 0 {
		return "empty"
	}

	// Use a simple hash of the first few and last few elements
	hashElements := make([]float32, 0, 8)

	// Add first 4 elements
	for i := 0; i < len(vector) && i < 4; i++ {
		hashElements = append(hashElements, vector[i])
	}

	// Add last 4 elements
	start := len(vector) - 4
	if start < 4 {
		start = 4
	}
	for i := start; i < len(vector); i++ {
		hashElements = append(hashElements, vector[i])
	}

	// Generate hash
	var hash uint64
	for _, val := range hashElements {
		hash = hash*31 + uint64(math.Float32bits(val))
	}

	return fmt.Sprintf("v_%016x", hash)
}

// Statistics methods

func (se *SimilarityEngine) updateCalculationStats(metric SimilarityMetric, duration time.Duration) {
	se.stats.mu.Lock()
	defer se.stats.mu.Unlock()

	se.stats.TotalCalculations++
	se.stats.CalculationsByMetric[metric]++

	// Update average calculation time
	if se.stats.AverageCalculationTime == 0 {
		se.stats.AverageCalculationTime = duration
	} else {
		se.stats.AverageCalculationTime = (se.stats.AverageCalculationTime + duration) / 2
	}
}

func (se *SimilarityEngine) updateBatchStats(batchSize int, duration time.Duration) {
	se.stats.mu.Lock()
	defer se.stats.mu.Unlock()

	se.stats.BatchCalculations++
}

func (se *SimilarityEngine) updateParallelStats() {
	se.stats.mu.Lock()
	defer se.stats.mu.Unlock()

	se.stats.ParallelCalculations++
}

func (se *SimilarityEngine) updateCacheStats(hit bool) {
	se.stats.mu.Lock()
	defer se.stats.mu.Unlock()

	if hit {
		// Update cache hit rate
		totalOps := se.stats.TotalCalculations
		if totalOps > 0 {
			se.stats.CacheHitRate = (se.stats.CacheHitRate*float64(totalOps-1) + 1.0) / float64(totalOps)
		}
	}
}

func (se *SimilarityEngine) updateErrorStats(err error) {
	se.stats.mu.Lock()
	defer se.stats.mu.Unlock()

	se.stats.ErrorCount++
	se.stats.LastError = err.Error()
}

// Public API

func (se *SimilarityEngine) GetStatistics() *SimilarityStatistics {
	se.stats.mu.RLock()
	defer se.stats.mu.RUnlock()

	stats := *se.stats
	return &stats
}

func (se *SimilarityEngine) GetConfig() *SimilarityConfig {
	se.mu.RLock()
	defer se.mu.RUnlock()

	return se.config
}

func (se *SimilarityEngine) UpdateConfig(config *SimilarityConfig) {
	se.mu.Lock()
	defer se.mu.Unlock()

	se.config = config
}
