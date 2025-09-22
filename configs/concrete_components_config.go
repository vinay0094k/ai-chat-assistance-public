package configs

import (
	"time"
)

// ComponentsConfig holds configuration for all concrete component implementations
type ComponentsConfig struct {
	SearchComponents       *SearchComponentsConfig       `yaml:"search_components"`
	ContextAwareComponents *ContextAwareComponentsConfig `yaml:"context_aware_components"`
}

// SearchComponentsConfig configures SearchAgent components
type SearchComponentsConfig struct {
	TextSearchEngine *TextSearchEngineConfig `yaml:"text_search_engine"`
	QueryAnalyzer    *QueryAnalyzerConfig    `yaml:"query_analyzer"`
	ResultRanker     *ResultRankerConfig     `yaml:"result_ranker"`
	SearchCache      *SearchCacheConfig      `yaml:"search_cache"`
}

type TextSearchEngineConfig struct {
	MaxResults     int           `yaml:"max_results"`
	SearchTimeout  time.Duration `yaml:"search_timeout"`
	EnableFuzzy    bool          `yaml:"enable_fuzzy"`
	FuzzyThreshold float64       `yaml:"fuzzy_threshold"`
}

type QueryAnalyzerConfig struct {
	EnableExpansion  bool    `yaml:"enable_expansion"`
	SynonymThreshold float64 `yaml:"synonym_threshold"`
	EnableSpellCheck bool    `yaml:"enable_spell_check"`
}

type ResultRankerConfig struct {
	RelevanceWeight  float64 `yaml:"relevance_weight"`
	FreshnessWeight  float64 `yaml:"freshness_weight"`
	PopularityWeight float64 `yaml:"popularity_weight"`
}

type SearchCacheConfig struct {
	MaxSize   int           `yaml:"max_size"`
	TTL       time.Duration `yaml:"ttl"`
	EnableLRU bool          `yaml:"enable_lru"`
}

// ContextAwareComponentsConfig configures ContextAwareSearchAgent components
type ContextAwareComponentsConfig struct {
	ContextAnalyzer     *ContextAnalyzerConfig     `yaml:"context_analyzer"`
	RelevanceCalculator *RelevanceCalculatorConfig `yaml:"relevance_calculator"`
	WorkspaceAnalyzer   *WorkspaceAnalyzerConfig   `yaml:"workspace_analyzer"`
	SessionTracker      *SessionTrackerConfig      `yaml:"session_tracker"`
	IntentPredictor     *IntentPredictorConfig     `yaml:"intent_predictor"`
}

type ContextAnalyzerConfig struct {
	MaxContextDepth     int     `yaml:"max_context_depth"`
	ContextRadius       int     `yaml:"context_radius"`
	SimilarityThreshold float64 `yaml:"similarity_threshold"`
}

type RelevanceCalculatorConfig struct {
	ContextWeight   float64 `yaml:"context_weight"`
	RecencyWeight   float64 `yaml:"recency_weight"`
	FrequencyWeight float64 `yaml:"frequency_weight"`
}

type WorkspaceAnalyzerConfig struct {
	ScanDepth       int           `yaml:"scan_depth"`
	AnalysisTimeout time.Duration `yaml:"analysis_timeout"`
	EnableGitInfo   bool          `yaml:"enable_git_info"`
}

type SessionTrackerConfig struct {
	MaxSessions    int           `yaml:"max_sessions"`
	SessionTimeout time.Duration `yaml:"session_timeout"`
	EnablePersist  bool          `yaml:"enable_persist"`
}

type IntentPredictorConfig struct {
	ModelPath           string  `yaml:"model_path"`
	ConfidenceThreshold float64 `yaml:"confidence_threshold"`
	EnableLearning      bool    `yaml:"enable_learning"`
}
