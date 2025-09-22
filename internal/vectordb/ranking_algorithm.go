package vectordb

import (
	"context"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

// RankingAlgorithm provides advanced multi-factor ranking for search results
type RankingAlgorithm struct {
	// Configuration
	config *RankingConfig

	// Learning components
	learningEngine    *LearningEngine
	personalizer      *PersonalizationEngine
	feedbackCollector *FeedbackCollector

	// Ranking models
	baseRanker     *BaseRanker
	mlRanker       *MLRanker
	personalRanker *PersonalRanker

	// Analytics
	stats *RankingStatistics

	// State management
	mu sync.RWMutex
}

// RankingConfig contains ranking algorithm configuration
type RankingConfig struct {
	// Ranking factors and weights
	RankingFactors []RankingFactor `json:"ranking_factors"`
	BaseWeights    *RankingWeights `json:"base_weights"`

	// Learning settings
	EnableLearning  bool    `json:"enable_learning"`
	LearningRate    float64 `json:"learning_rate"`
	AdaptationSpeed float64 `json:"adaptation_speed"`
	MinSamples      int     `json:"min_samples"`

	// Personalization
	EnablePersonalization bool    `json:"enable_personalization"`
	UserHistoryWeight     float32 `json:"user_history_weight"`
	TeamPreferenceWeight  float32 `json:"team_preference_weight"`
	GlobalTrendWeight     float32 `json:"global_trend_weight"`

	// Performance tuning
	RankingMode    string        `json:"ranking_mode"`     // simple, advanced, ml, hybrid
	MaxRankingTime time.Duration `json:"max_ranking_time"` // Timeout for ranking
	CacheRankings  bool          `json:"cache_rankings"`   // Cache ranking results
	BatchSize      int           `json:"batch_size"`       // Results to rank in parallel

	// Quality metrics
	EnableQualityMetrics bool `json:"enable_quality_metrics"`
	TrackClickthrough    bool `json:"track_clickthrough"`
	TrackDwellTime       bool `json:"track_dwell_time"`
	TrackRelevance       bool `json:"track_relevance"`
}

// RankingFactor represents a factor used in ranking
type RankingFactor string

const (
	FactorSimilarity   RankingFactor = "similarity"    // Vector similarity score
	FactorKeywordMatch RankingFactor = "keyword_match" // Keyword matching score
	FactorRecency      RankingFactor = "recency"       // How recent the code is
	FactorPopularity   RankingFactor = "popularity"    // Usage/reference frequency
	FactorComplexity   RankingFactor = "complexity"    // Code complexity
	FactorQuality      RankingFactor = "quality"       // Code quality metrics
	FactorAuthority    RankingFactor = "authority"     // Author reputation
	FactorContext      RankingFactor = "context"       // Contextual relevance
	FactorPersonal     RankingFactor = "personal"      // Personal preferences
	FactorTeam         RankingFactor = "team"          // Team preferences
	FactorProject      RankingFactor = "project"       // Project relevance
	FactorLanguage     RankingFactor = "language"      // Language preference
	FactorFileType     RankingFactor = "file_type"     // File type preference
	FactorLocation     RankingFactor = "location"      // File location relevance
)

// RankingWeights defines weights for different ranking factors
type RankingWeights struct {
	Similarity   float32 `json:"similarity"`
	KeywordMatch float32 `json:"keyword_match"`
	Recency      float32 `json:"recency"`
	Popularity   float32 `json:"popularity"`
	Complexity   float32 `json:"complexity"`
	Quality      float32 `json:"quality"`
	Authority    float32 `json:"authority"`
	Context      float32 `json:"context"`
	Personal     float32 `json:"personal"`
	Team         float32 `json:"team"`
	Project      float32 `json:"project"`
	Language     float32 `json:"language"`
	FileType     float32 `json:"file_type"`
	Location     float32 `json:"location"`
}

// RankingInfo provides information about how results were ranked
type RankingInfo struct {
	RankingMode            string          `json:"ranking_mode"`
	FactorsUsed            []RankingFactor `json:"factors_used"`
	WeightsApplied         *RankingWeights `json:"weights_applied"`
	PersonalizationApplied bool            `json:"personalization_applied"`
	LearningApplied        bool            `json:"learning_applied"`
	RankingTime            time.Duration   `json:"ranking_time"`
	TotalResults           int             `json:"total_results"`
	RerankedResults        int             `json:"reranked_results"`
}

// RankingStatistics tracks ranking performance and quality
type RankingStatistics struct {
	TotalRankings      int64            `json:"total_rankings"`
	AverageRankingTime time.Duration    `json:"average_ranking_time"`
	RankingsByMode     map[string]int64 `json:"rankings_by_mode"`

	// Quality metrics
	ClickthroughRate float64       `json:"clickthrough_rate"`
	AverageDwellTime time.Duration `json:"average_dwell_time"`
	RelevanceScore   float64       `json:"relevance_score"`
	UserSatisfaction float64       `json:"user_satisfaction"`

	// Learning metrics
	ModelAccuracy       float64 `json:"model_accuracy"`
	AdaptationRate      float64 `json:"adaptation_rate"`
	PersonalizationGain float64 `json:"personalization_gain"`

	// Performance metrics
	CacheHitRate float64 `json:"cache_hit_rate"`
	TimeoutRate  float64 `json:"timeout_rate"`
	ErrorRate    float64 `json:"error_rate"`

	mu sync.RWMutex
}

// NewRankingAlgorithm creates a new ranking algorithm
func NewRankingAlgorithm(config *RankingConfig) *RankingAlgorithm {
	if config == nil {
		config = &RankingConfig{
			RankingFactors: []RankingFactor{
				FactorSimilarity, FactorKeywordMatch, FactorRecency,
				FactorPopularity, FactorContext, FactorPersonal,
			},
			BaseWeights: &RankingWeights{
				Similarity:   0.30,
				KeywordMatch: 0.20,
				Recency:      0.10,
				Popularity:   0.15,
				Context:      0.10,
				Personal:     0.15,
			},
			EnableLearning:        true,
			LearningRate:          0.01,
			AdaptationSpeed:       0.1,
			MinSamples:            100,
			EnablePersonalization: true,
			UserHistoryWeight:     0.3,
			TeamPreferenceWeight:  0.2,
			GlobalTrendWeight:     0.1,
			RankingMode:           "advanced",
			MaxRankingTime:        time.Millisecond * 500,
			CacheRankings:         true,
			BatchSize:             50,
			EnableQualityMetrics:  true,
			TrackClickthrough:     true,
			TrackDwellTime:        true,
			TrackRelevance:        true,
		}
	}

	ra := &RankingAlgorithm{
		config: config,
		stats: &RankingStatistics{
			RankingsByMode: make(map[string]int64),
		},
	}

	// Initialize components
	ra.initializeComponents()

	return ra
}

// RankResults ranks search results using the configured algorithm
func (ra *RankingAlgorithm) RankResults(results []*SearchResult, request *SearchRequest, queryInfo *QueryInfo) []*SearchResult {
	if len(results) <= 1 {
		return results // No need to rank single result
	}

	start := time.Now()

	// Choose ranking strategy based on mode
	var rankedResults []*SearchResult
	var err error

	switch ra.config.RankingMode {
	case "simple":
		rankedResults = ra.simpleRanking(results, request)
	case "advanced":
		rankedResults = ra.advancedRanking(results, request, queryInfo)
	case "ml":
		rankedResults = ra.mlRanking(results, request, queryInfo)
	case "hybrid":
		rankedResults = ra.hybridRanking(results, request, queryInfo)
	default:
		rankedResults = ra.advancedRanking(results, request, queryInfo)
	}

	if err != nil {
		// Fall back to simple ranking on error
		rankedResults = ra.simpleRanking(results, request)
	}

	// Apply personalization if enabled
	if ra.config.EnablePersonalization {
		rankedResults = ra.applyPersonalization(rankedResults, request)
	}

	// Update ranking information in results
	for i, result := range rankedResults {
		result.Relevance = ra.calculateRelevanceScore(result, request, queryInfo)
		result.Confidence = ra.calculateConfidenceScore(result, request)
		result.RankingFactors = ra.extractRankingFactors(result, request)

		// Update final score based on rank position
		positionBoost := 1.0 - (float64(i) * 0.1) // Diminishing returns for position
		if positionBoost < 0.1 {
			positionBoost = 0.1
		}
		result.Score = result.Score * float32(positionBoost)
	}

	// Update statistics
	ra.updateRankingStats(ra.config.RankingMode, len(results), time.Since(start))

	return rankedResults
}

// simpleRanking performs basic ranking by similarity score
func (ra *RankingAlgorithm) simpleRanking(results []*SearchResult, request *SearchRequest) []*SearchResult {
	// Simply sort by existing score (usually similarity)
	sortedResults := make([]*SearchResult, len(results))
	copy(sortedResults, results)

	sort.Slice(sortedResults, func(i, j int) bool {
		return sortedResults[i].Score > sortedResults[j].Score
	})

	return sortedResults
}

// advancedRanking performs multi-factor ranking
func (ra *RankingAlgorithm) advancedRanking(results []*SearchResult, request *SearchRequest, queryInfo *QueryInfo) []*SearchResult {
	// Calculate composite scores for each result
	for _, result := range results {
		compositeScore := ra.calculateCompositeScore(result, request, queryInfo)
		result.Score = compositeScore
	}

	// Sort by composite score
	sortedResults := make([]*SearchResult, len(results))
	copy(sortedResults, results)

	sort.Slice(sortedResults, func(i, j int) bool {
		return sortedResults[i].Score > sortedResults[j].Score
	})

	return sortedResults
}

// mlRanking performs machine learning-based ranking
func (ra *RankingAlgorithm) mlRanking(results []*SearchResult, request *SearchRequest, queryInfo *QueryInfo) []*SearchResult {
	if ra.mlRanker == nil {
		// Fall back to advanced ranking if ML ranker not available
		return ra.advancedRanking(results, request, queryInfo)
	}

	// Use ML model to score results
	return ra.mlRanker.RankResults(results, request, queryInfo)
}

// hybridRanking combines multiple ranking approaches
func (ra *RankingAlgorithm) hybridRanking(results []*SearchResult, request *SearchRequest, queryInfo *QueryInfo) []*SearchResult {
	// Get rankings from different approaches
	simpleRanked := ra.simpleRanking(results, request)
	advancedRanked := ra.advancedRanking(results, request, queryInfo)

	// Combine rankings using ensemble method
	return ra.combineRankings([]([]*SearchResult){simpleRanked, advancedRanked}, []float64{0.3, 0.7})
}

// calculateCompositeScore calculates a multi-factor composite score
func (ra *RankingAlgorithm) calculateCompositeScore(result *SearchResult, request *SearchRequest, queryInfo *QueryInfo) float32 {
	score := float32(0.0)
	weights := ra.config.BaseWeights

	// Similarity factor
	if ra.isFactorEnabled(FactorSimilarity) {
		score += result.VectorSimilarity * weights.Similarity
	}

	// Keyword match factor
	if ra.isFactorEnabled(FactorKeywordMatch) {
		keywordScore := ra.calculateKeywordMatchScore(result, request)
		score += keywordScore * weights.KeywordMatch
	}

	// Recency factor
	if ra.isFactorEnabled(FactorRecency) {
		recencyScore := ra.calculateRecencyScore(result)
		score += recencyScore * weights.Recency
	}

	// Popularity factor
	if ra.isFactorEnabled(FactorPopularity) {
		popularityScore := ra.calculatePopularityScore(result)
		score += popularityScore * weights.Popularity
	}

	// Quality factor
	if ra.isFactorEnabled(FactorQuality) {
		qualityScore := ra.calculateQualityScore(result)
		score += qualityScore * weights.Quality
	}

	// Context factor
	if ra.isFactorEnabled(FactorContext) {
		contextScore := ra.calculateContextScore(result, request, queryInfo)
		score += contextScore * weights.Context
	}

	// Language preference factor
	if ra.isFactorEnabled(FactorLanguage) {
		languageScore := ra.calculateLanguagePreferenceScore(result, request)
		score += languageScore * weights.Language
	}

	// Project relevance factor
	if ra.isFactorEnabled(FactorProject) {
		projectScore := ra.calculateProjectRelevanceScore(result, request)
		score += projectScore * weights.Project
	}

	return score
}

// Individual factor calculation methods

func (ra *RankingAlgorithm) calculateKeywordMatchScore(result *SearchResult, request *SearchRequest) float32 {
	if request.Query == "" {
		return 0.0
	}

	queryWords := strings.Fields(strings.ToLower(request.Query))
	contentWords := strings.Fields(strings.ToLower(result.Content))

	matches := 0
	for _, queryWord := range queryWords {
		for _, contentWord := range contentWords {
			if queryWord == contentWord {
				matches++
				break
			}
		}
	}

	if len(queryWords) == 0 {
		return 0.0
	}

	return float32(matches) / float32(len(queryWords))
}

func (ra *RankingAlgorithm) calculateRecencyScore(result *SearchResult) float32 {
	if result.LastModified.IsZero() {
		return 0.5 // Neutral score for unknown modification time
	}

	daysSinceModified := time.Since(result.LastModified).Hours() / 24

	// Exponential decay: more recent = higher score
	score := math.Exp(-daysSinceModified / 30.0) // 30-day half-life

	return float32(score)
}

func (ra *RankingAlgorithm) calculatePopularityScore(result *SearchResult) float32 {
	// Use access count from metadata if available
	if accessCount, ok := result.Metadata["access_count"].(float64); ok {
		// Logarithmic scaling to prevent extremely popular items from dominating
		score := math.Log(1+accessCount) / math.Log(1000) // Normalize to ~1000 accesses
		if score > 1.0 {
			score = 1.0
		}
		return float32(score)
	}

	return 0.5 // Default neutral score
}

func (ra *RankingAlgorithm) calculateQualityScore(result *SearchResult) float32 {
	score := float32(0.5) // Base quality score

	// Factor in code complexity (lower complexity can be better for examples)
	if complexity, ok := result.Metadata["complexity"].(float64); ok {
		if complexity < 5 {
			score += 0.2 // Bonus for simple code
		} else if complexity > 15 {
			score -= 0.1 // Penalty for very complex code
		}
	}

	// Factor in documentation presence
	if result.Source.ChunkType == "function" && result.Summary != "" {
		score += 0.2 // Bonus for documented functions
	}

	// Factor in naming quality (simple heuristic)
	if len(result.Source.FunctionName) > 3 && !strings.Contains(result.Source.FunctionName, "temp") {
		score += 0.1 // Bonus for well-named functions
	}

	// Ensure score is in valid range
	if score > 1.0 {
		score = 1.0
	}
	if score < 0.0 {
		score = 0.0
	}

	return score
}

func (ra *RankingAlgorithm) calculateContextScore(result *SearchResult, request *SearchRequest, queryInfo *QueryInfo) float32 {
	score := float32(0.5)

	// Language context matching
	if queryInfo.DetectedLanguage != "" && result.Source.Language == queryInfo.DetectedLanguage {
		score += 0.3
	}

	// Query type context matching
	switch queryInfo.QueryType {
	case QueryTypeFunction:
		if result.Source.ChunkType == "function" {
			score += 0.2
		}
	case QueryTypeCode:
		if result.Source.ChunkType == "function" || result.Source.ChunkType == "class" {
			score += 0.2
		}
	case QueryTypeError:
		if strings.Contains(strings.ToLower(result.Content), "error") ||
			strings.Contains(strings.ToLower(result.Content), "exception") {
			score += 0.3
		}
	}

	// File type context
	if request.Filters != nil && len(request.Filters.FileTypes) > 0 {
		resultFileType := extractFileType(result.Source.FilePath)
		for _, preferredType := range request.Filters.FileTypes {
			if strings.EqualFold(resultFileType, preferredType) {
				score += 0.2
				break
			}
		}
	}

	// Ensure valid range
	if score > 1.0 {
		score = 1.0
	}

	return score
}

func (ra *RankingAlgorithm) calculateLanguagePreferenceScore(result *SearchResult, request *SearchRequest) float32 {
	if request.Filters == nil || len(request.Filters.Languages) == 0 {
		return 0.5 // Neutral if no preference specified
	}

	resultLanguage := result.Source.Language
	for _, preferredLang := range request.Filters.Languages {
		if strings.EqualFold(resultLanguage, preferredLang) {
			return 1.0 // Perfect match
		}
	}

	return 0.2 // Lower score for non-preferred languages
}

func (ra *RankingAlgorithm) calculateProjectRelevanceScore(result *SearchResult, request *SearchRequest) float32 {
	// This would calculate relevance based on project context
	// For now, return neutral score
	return 0.5
}

// Personalization methods

func (ra *RankingAlgorithm) applyPersonalization(results []*SearchResult, request *SearchRequest) []*SearchResult {
	if ra.personalRanker == nil || request.UserID == "" {
		return results
	}

	return ra.personalRanker.PersonalizeResults(results, request)
}

// Learning and adaptation methods

func (ra *RankingAlgorithm) RecordFeedback(searchID string, resultID string, feedback *UserFeedback) {
	if ra.config.EnableLearning && ra.learningEngine != nil {
		ra.learningEngine.RecordFeedback(searchID, resultID, feedback)
	}

	if ra.feedbackCollector != nil {
		ra.feedbackCollector.CollectFeedback(searchID, resultID, feedback)
	}
}

func (ra *RankingAlgorithm) UpdateModel(context.Context) error {
	if !ra.config.EnableLearning || ra.learningEngine == nil {
		return nil
	}

	return ra.learningEngine.UpdateModel()
}

// Helper methods

func (ra *RankingAlgorithm) isFactorEnabled(factor RankingFactor) bool {
	for _, enabledFactor := range ra.config.RankingFactors {
		if enabledFactor == factor {
			return true
		}
	}
	return false
}

func (ra *RankingAlgorithm) combineRankings(rankings []([]*SearchResult), weights []float64) []*SearchResult {
	if len(rankings) == 0 {
		return []*SearchResult{}
	}

	if len(rankings) == 1 {
		return rankings[0]
	}

	// Create position maps for each ranking
	positionMaps := make([]map[string]int, len(rankings))
	for i, ranking := range rankings {
		positionMaps[i] = make(map[string]int)
		for pos, result := range ranking {
			positionMaps[i][result.ID] = pos
		}
	}

	// Calculate combined scores
	resultScores := make(map[string]float64)
	allResults := make(map[string]*SearchResult)

	for _, result := range rankings[0] {
		allResults[result.ID] = result
		combinedScore := 0.0

		for i, posMap := range positionMaps {
			if pos, exists := posMap[result.ID]; exists {
				// Convert position to score (lower position = higher score)
				positionScore := 1.0 - (float64(pos) / float64(len(rankings[i])))
				combinedScore += positionScore * weights[i]
			}
		}

		resultScores[result.ID] = combinedScore
	}

	// Sort by combined score
	var resultIDs []string
	for id := range resultScores {
		resultIDs = append(resultIDs, id)
	}

	sort.Slice(resultIDs, func(i, j int) bool {
		return resultScores[resultIDs[i]] > resultScores[resultIDs[j]]
	})

	// Build final ranking
	var combinedRanking []*SearchResult
	for _, id := range resultIDs {
		result := allResults[id]
		result.Score = float32(resultScores[id])
		combinedRanking = append(combinedRanking, result)
	}

	return combinedRanking
}

func (ra *RankingAlgorithm) calculateRelevanceScore(result *SearchResult, request *SearchRequest, queryInfo *QueryInfo) float32 {
	// This is a simplified relevance calculation
	// In practice, this would be more sophisticated

	relevance := result.Score * 0.7 // Base from ranking score

	// Boost for exact matches
	if strings.Contains(strings.ToLower(result.Content), strings.ToLower(request.Query)) {
		relevance += 0.2
	}

	// Boost for matching chunk type
	if queryInfo.QueryType == QueryTypeFunction && result.Source.ChunkType == "function" {
		relevance += 0.1
	}

	if relevance > 1.0 {
		relevance = 1.0
	}

	return relevance
}

func (ra *RankingAlgorithm) calculateConfidenceScore(result *SearchResult, request *SearchRequest) float32 {
	// Confidence based on multiple factors
	confidence := result.Score * 0.6 // Base from ranking score

	// Higher confidence for higher similarity
	if result.VectorSimilarity > 0.8 {
		confidence += 0.2
	}

	// Higher confidence for complete functions/classes
	if result.Source.ChunkType == "function" && result.Source.FunctionName != "" {
		confidence += 0.1
	}

	// Higher confidence for documented code
	if result.Summary != "" {
		confidence += 0.1
	}

	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

func (ra *RankingAlgorithm) extractRankingFactors(result *SearchResult, request *SearchRequest) map[string]float32 {
	factors := make(map[string]float32)

	factors["similarity"] = result.VectorSimilarity
	factors["keyword_match"] = ra.calculateKeywordMatchScore(result, request)
	factors["recency"] = ra.calculateRecencyScore(result)
	factors["popularity"] = ra.calculatePopularityScore(result)
	factors["quality"] = ra.calculateQualityScore(result)

	return factors
}

// Component initialization

func (ra *RankingAlgorithm) initializeComponents() {
	// Initialize base ranker
	ra.baseRanker = NewBaseRanker(ra.config)

	// Initialize ML ranker if learning is enabled
	if ra.config.EnableLearning {
		ra.learningEngine = NewLearningEngine(ra.config)
		ra.mlRanker = NewMLRanker(ra.config, ra.learningEngine)
	}

	// Initialize personalization if enabled
	if ra.config.EnablePersonalization {
		ra.personalizer = NewPersonalizationEngine(ra.config)
		ra.personalRanker = NewPersonalRanker(ra.config, ra.personalizer)
	}

	// Initialize feedback collector if quality metrics are enabled
	if ra.config.EnableQualityMetrics {
		ra.feedbackCollector = NewFeedbackCollector(ra.config)
	}
}

// Statistics methods

func (ra *RankingAlgorithm) updateRankingStats(mode string, resultCount int, duration time.Duration) {
	ra.stats.mu.Lock()
	defer ra.stats.mu.Unlock()

	ra.stats.TotalRankings++
	ra.stats.RankingsByMode[mode]++

	// Update average ranking time
	if ra.stats.AverageRankingTime == 0 {
		ra.stats.AverageRankingTime = duration
	} else {
		ra.stats.AverageRankingTime = (ra.stats.AverageRankingTime + duration) / 2
	}
}

// Public API

func (ra *RankingAlgorithm) GetStatistics() *RankingStatistics {
	ra.stats.mu.RLock()
	defer ra.stats.mu.RUnlock()

	stats := *ra.stats
	return &stats
}

func (ra *RankingAlgorithm) GetConfig() *RankingConfig {
	ra.mu.RLock()
	defer ra.mu.RUnlock()

	return ra.config
}

func (ra *RankingAlgorithm) UpdateWeights(weights *RankingWeights) {
	ra.mu.Lock()
	defer ra.mu.Unlock()

	ra.config.BaseWeights = weights
}

// UserFeedback represents user feedback on search results
type UserFeedback struct {
	Clicked   bool          `json:"clicked"`
	DwellTime time.Duration `json:"dwell_time"`
	Relevance int           `json:"relevance"` // 1-5 scale
	Helpful   bool          `json:"helpful"`
	Timestamp time.Time     `json:"timestamp"`
	UserID    string        `json:"user_id"`
	SessionID string        `json:"session_id"`
}
