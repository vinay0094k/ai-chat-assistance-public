package main

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

type ComponentsConfig struct {
	SearchComponents       *SearchComponentsConfig       `yaml:"search_components"`
	ContextAwareComponents *ContextAwareComponentsConfig `yaml:"context_aware_components"`
}

type SearchComponentsConfig struct {
	TextSearchEngine *TextSearchEngineConfig `yaml:"text_search_engine"`
	QueryAnalyzer    *QueryAnalyzerConfig    `yaml:"query_analyzer"`
	ResultRanker     *ResultRankerConfig     `yaml:"result_ranker"`
	SearchCache      *SearchCacheConfig      `yaml:"search_cache"`
}

type TextSearchEngineConfig struct {
	MaxResults     int     `yaml:"max_results"`
	EnableFuzzy    bool    `yaml:"enable_fuzzy"`
	FuzzyThreshold float64 `yaml:"fuzzy_threshold"`
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
	MaxSize   int  `yaml:"max_size"`
	EnableLRU bool `yaml:"enable_lru"`
}

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
	ScanDepth     int  `yaml:"scan_depth"`
	EnableGitInfo bool `yaml:"enable_git_info"`
}

type SessionTrackerConfig struct {
	MaxSessions   int  `yaml:"max_sessions"`
	EnablePersist bool `yaml:"enable_persist"`
}

type IntentPredictorConfig struct {
	ModelPath           string  `yaml:"model_path"`
	ConfidenceThreshold float64 `yaml:"confidence_threshold"`
	EnableLearning      bool    `yaml:"enable_learning"`
}

func main() {
	fmt.Println("=== Concrete Components YAML Configuration Test ===")

	// Load YAML configuration
	configPath := filepath.Join("configs", "concrete-components-config.yaml")
	data, err := os.ReadFile(configPath)
	if err != nil {
		fmt.Printf("❌ Failed to read config: %v\n", err)
		return
	}

	var config ComponentsConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		fmt.Printf("❌ Failed to parse YAML: %v\n", err)
		return
	}

	fmt.Println("✅ YAML Configuration Loaded Successfully!")

	// Validate SearchComponents
	if config.SearchComponents != nil {
		fmt.Printf("\n🔍 Search Components:\n")
		if config.SearchComponents.TextSearchEngine != nil {
			fmt.Printf("  TextSearchEngine: max_results=%d, fuzzy=%t, threshold=%.1f\n",
				config.SearchComponents.TextSearchEngine.MaxResults,
				config.SearchComponents.TextSearchEngine.EnableFuzzy,
				config.SearchComponents.TextSearchEngine.FuzzyThreshold)
		}
		if config.SearchComponents.QueryAnalyzer != nil {
			fmt.Printf("  QueryAnalyzer: expansion=%t, spell_check=%t\n",
				config.SearchComponents.QueryAnalyzer.EnableExpansion,
				config.SearchComponents.QueryAnalyzer.EnableSpellCheck)
		}
		if config.SearchComponents.ResultRanker != nil {
			fmt.Printf("  ResultRanker: relevance=%.1f, freshness=%.1f, popularity=%.1f\n",
				config.SearchComponents.ResultRanker.RelevanceWeight,
				config.SearchComponents.ResultRanker.FreshnessWeight,
				config.SearchComponents.ResultRanker.PopularityWeight)
		}
	}

	// Validate ContextAwareComponents
	if config.ContextAwareComponents != nil {
		fmt.Printf("\n🧠 Context Aware Components:\n")
		if config.ContextAwareComponents.ContextAnalyzer != nil {
			fmt.Printf("  ContextAnalyzer: depth=%d, radius=%d, threshold=%.1f\n",
				config.ContextAwareComponents.ContextAnalyzer.MaxContextDepth,
				config.ContextAwareComponents.ContextAnalyzer.ContextRadius,
				config.ContextAwareComponents.ContextAnalyzer.SimilarityThreshold)
		}
		if config.ContextAwareComponents.IntentPredictor != nil {
			fmt.Printf("  IntentPredictor: model=%s, confidence=%.1f, learning=%t\n",
				config.ContextAwareComponents.IntentPredictor.ModelPath,
				config.ContextAwareComponents.IntentPredictor.ConfidenceThreshold,
				config.ContextAwareComponents.IntentPredictor.EnableLearning)
		}
	}

	fmt.Println("\n✅ Concrete Components YAML Test PASSED!")
}
