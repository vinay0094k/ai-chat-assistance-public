package app

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/llm"
	"github.com/yourusername/ai-code-assistant/internal/logger"
)

// IntentClassifier uses advanced techniques to determine user intent from natural language
type IntentClassifier struct {
	// Core components
	llmProvider llm.Provider
	logger      logger.Logger

	// Classification engines
	ruleBasedClassifier *RuleBasedClassifier
	patternMatcher      *PatternMatcher
	keywordAnalyzer     *KeywordAnalyzer
	semanticAnalyzer    *SemanticAnalyzer
	contextAnalyzer     *ContextAnalyzer

	// Machine learning components
	mlClassifier      *MLClassifier
	embeddingModel    *EmbeddingModel
	similarityMatcher *SimilarityMatcher

	// Entity extraction
	entityExtractor    *EntityExtractor
	parameterExtractor *ParameterExtractor

	// Knowledge base and training data
	intentKnowledgeBase *IntentKnowledgeBase
	trainingData        []*TrainingExample
	intentDefinitions   map[string]*IntentDefinition

	// Configuration and settings
	config *IntentClassifierConfig

	// Feedback and learning
	feedbackProcessor *FeedbackProcessor
	adaptiveLearner   *AdaptiveLearner

	// Cache and optimization
	classificationCache map[string]*CachedClassification
	cacheExpiry         time.Duration
	cacheMu             sync.RWMutex

	// Metrics and monitoring
	metrics *ClassificationMetrics

	// State management
	mu            sync.RWMutex
	isInitialized bool
	lastUpdate    time.Time
}

// IntentClassifierConfig contains configuration for the intent classifier
type IntentClassifierConfig struct {
	// Classification settings
	EnableRuleBased        bool `json:"enable_rule_based"`
	EnablePatternMatching  bool `json:"enable_pattern_matching"`
	EnableKeywordAnalysis  bool `json:"enable_keyword_analysis"`
	EnableSemanticAnalysis bool `json:"enable_semantic_analysis"`
	EnableContextAnalysis  bool `json:"enable_context_analysis"`
	EnableMLClassification bool `json:"enable_ml_classification"`

	// Confidence thresholds
	MinConfidenceThreshold  float64 `json:"min_confidence_threshold"`
	HighConfidenceThreshold float64 `json:"high_confidence_threshold"`
	AmbiguityThreshold      float64 `json:"ambiguity_threshold"`

	// Language settings
	SupportedLanguages  []string `json:"supported_languages"`
	DefaultLanguage     string   `json:"default_language"`
	EnableMultiLanguage bool     `json:"enable_multi_language"`

	// Entity extraction settings
	EnableEntityExtraction bool         `json:"enable_entity_extraction"`
	EntityTypes            []EntityType `json:"entity_types"`
	MaxEntitiesPerQuery    int          `json:"max_entities_per_query"`

	// Feedback and learning
	EnableFeedbackLearning bool    `json:"enable_feedback_learning"`
	FeedbackWeight         float64 `json:"feedback_weight"`
	LearningRate           float64 `json:"learning_rate"`

	// Performance settings
	MaxProcessingTime        time.Duration `json:"max_processing_time"`
	EnableCaching            bool          `json:"enable_caching"`
	CacheExpiry              time.Duration `json:"cache_expiry"`
	EnableParallelProcessing bool          `json:"enable_parallel_processing"`

	// Model settings
	EmbeddingModelPath      string `json:"embedding_model_path"`
	ClassificationModelPath string `json:"classification_model_path"`

	// LLM settings for semantic analysis
	LLMModel       string  `json:"llm_model"`
	LLMMaxTokens   int     `json:"llm_max_tokens"`
	LLMTemperature float32 `json:"llm_temperature"`

	// Advanced settings
	EnableContextPropagation bool `json:"enable_context_propagation"`
	ContextWindowSize        int  `json:"context_window_size"`
	EnableIntentChaining     bool `json:"enable_intent_chaining"`

	// Debug and monitoring
	EnableDebugLogging     bool `json:"enable_debug_logging"`
	LogClassificationSteps bool `json:"log_classification_steps"`
	CollectMetrics         bool `json:"collect_metrics"`
}

// Core intent types
type IntentType string

const (
	// Code generation intents
	IntentCodeGeneration     IntentType = "code_generation"
	IntentFunctionGeneration IntentType = "function_generation"
	IntentClassGeneration    IntentType = "class_generation"
	IntentTestGeneration     IntentType = "test_generation"
	IntentDocGeneration      IntentType = "documentation_generation"

	// Code analysis intents
	IntentCodeExplanation  IntentType = "code_explanation"
	IntentCodeReview       IntentType = "code_review"
	IntentCodeAnalysis     IntentType = "code_analysis"
	IntentBugDetection     IntentType = "bug_detection"
	IntentSecurityAnalysis IntentType = "security_analysis"

	// Code modification intents
	IntentCodeRefactoring  IntentType = "code_refactoring"
	IntentCodeOptimization IntentType = "code_optimization"
	IntentCodeFixing       IntentType = "code_fixing"
	IntentCodeFormatting   IntentType = "code_formatting"
	IntentCodeConversion   IntentType = "code_conversion"

	// Project management intents
	IntentProjectAnalysis      IntentType = "project_analysis"
	IntentDependencyAnalysis   IntentType = "dependency_analysis"
	IntentArchitectureAnalysis IntentType = "architecture_analysis"
	IntentPerformanceAnalysis  IntentType = "performance_analysis"

	// Testing intents
	IntentTestAnalysis     IntentType = "test_analysis"
	IntentTestOptimization IntentType = "test_optimization"
	IntentCoverageAnalysis IntentType = "coverage_analysis"

	// General intents
	IntentQuestion    IntentType = "question"
	IntentHelp        IntentType = "help"
	IntentInformation IntentType = "information"
	IntentSearch      IntentType = "search"
	IntentComparison  IntentType = "comparison"

	// Unknown intent
	IntentUnknown IntentType = "unknown"
)

// Entity types for extraction
type EntityType string

const (
	EntityTypeProgrammingLanguage EntityType = "programming_language"
	EntityTypeFramework           EntityType = "framework"
	EntityTypeLibrary             EntityType = "library"
	EntityTypeFilePath            EntityType = "file_path"
	EntityTypeFunctionName        EntityType = "function_name"
	EntityTypeClassName           EntityType = "class_name"
	EntityTypeVariableName        EntityType = "variable_name"
	EntityTypeDataType            EntityType = "data_type"
	EntityTypePattern             EntityType = "pattern"
	EntityTypeKeyword             EntityType = "keyword"
	EntityTypeNumber              EntityType = "number"
	EntityTypeString              EntityType = "string"
	EntityTypeURL                 EntityType = "url"
	EntityTypeEmail               EntityType = "email"
	EntityTypeTechnology          EntityType = "technology"
	EntityTypeConcept             EntityType = "concept"
)

// Classification request and response structures

type ClassificationRequest struct {
	// Input text to classify
	Query    string `json:"query"`
	Language string `json:"language,omitempty"`

	// Context information
	Context             *ClassificationContext `json:"context,omitempty"`
	ConversationHistory []*ConversationEntry   `json:"conversation_history,omitempty"`

	// Classification options
	Options *ClassificationOptions `json:"options,omitempty"`

	// Metadata
	UserID    string    `json:"user_id,omitempty"`
	SessionID string    `json:"session_id,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

type ClassificationContext struct {
	// File and project context
	ActiveProject   string   `json:"active_project,omitempty"`
	CurrentFile     string   `json:"current_file,omitempty"`
	SelectedFiles   []string `json:"selected_files,omitempty"`
	CurrentLanguage string   `json:"current_language,omitempty"`

	// Code context
	SelectedCode   string    `json:"selected_code,omitempty"`
	CursorPosition *Position `json:"cursor_position,omitempty"`

	// Previous operations
	LastIntent       IntentType `json:"last_intent,omitempty"`
	LastOperation    string     `json:"last_operation,omitempty"`
	OperationHistory []string   `json:"operation_history,omitempty"`

	// User preferences
	PreferredStyle     string `json:"preferred_style,omitempty"`
	PreferredFramework string `json:"preferred_framework,omitempty"`

	// Environment context
	IDE           string `json:"ide,omitempty"`
	OS            string `json:"os,omitempty"`
	WorkspaceType string `json:"workspace_type,omitempty"`
}

type Position struct {
	Line   int `json:"line"`
	Column int `json:"column"`
}

type ClassificationOptions struct {
	// Classification behavior
	RequireHighConfidence bool `json:"require_high_confidence"`
	AllowMultipleIntents  bool `json:"allow_multiple_intents"`
	MaxAlternatives       int  `json:"max_alternatives"`

	// Feature flags
	UseContext             bool `json:"use_context"`
	UseConversationHistory bool `json:"use_conversation_history"`
	ExtractEntities        bool `json:"extract_entities"`
	ExtractParameters      bool `json:"extract_parameters"`

	// Performance options
	FastMode             bool `json:"fast_mode"`
	SkipSemanticAnalysis bool `json:"skip_semantic_analysis"`

	// Debug options
	IncludeDebugInfo      bool `json:"include_debug_info"`
	ExplainClassification bool `json:"explain_classification"`
}

type ClassificationResult struct {
	// Primary classification
	Intent     IntentType `json:"intent"`
	Confidence float64    `json:"confidence"`

	// Alternative classifications
	Alternatives []*IntentAlternative `json:"alternatives,omitempty"`

	// Extracted information
	Entities   []*ExtractedEntity     `json:"entities,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`

	// Analysis results
	AnalysisResults *AnalysisResults `json:"analysis_results,omitempty"`

	// Metadata
	ProcessingTime       time.Duration `json:"processing_time"`
	ClassificationMethod string        `json:"classification_method"`
	Language             string        `json:"language"`

	// Quality indicators
	IsAmbiguous           bool     `json:"is_ambiguous"`
	RequiresClarification bool     `json:"requires_clarification"`
	SuggestedQuestions    []string `json:"suggested_questions,omitempty"`

	// Debug information
	DebugInfo *ClassificationDebugInfo `json:"debug_info,omitempty"`

	// Context propagation
	ContextUpdates *ContextUpdates `json:"context_updates,omitempty"`
}

type IntentAlternative struct {
	Intent          IntentType `json:"intent"`
	Confidence      float64    `json:"confidence"`
	Reason          string     `json:"reason"`
	RequiredContext []string   `json:"required_context,omitempty"`
}

type ExtractedEntity struct {
	Type            EntityType             `json:"type"`
	Value           string                 `json:"value"`
	NormalizedValue string                 `json:"normalized_value,omitempty"`
	Confidence      float64                `json:"confidence"`
	StartIndex      int                    `json:"start_index"`
	EndIndex        int                    `json:"end_index"`
	Context         string                 `json:"context,omitempty"`
	Properties      map[string]interface{} `json:"properties,omitempty"`
}

type AnalysisResults struct {
	// Rule-based analysis
	RuleMatches    []*RuleMatch       `json:"rule_matches,omitempty"`
	PatternMatches []*PatternMatch    `json:"pattern_matches,omitempty"`
	KeywordScores  map[string]float64 `json:"keyword_scores,omitempty"`

	// Semantic analysis
	SemanticSimilarity float64  `json:"semantic_similarity,omitempty"`
	ConceptMatches     []string `json:"concept_matches,omitempty"`

	// Context analysis
	ContextRelevance float64  `json:"context_relevance,omitempty"`
	ContextFactors   []string `json:"context_factors,omitempty"`

	// ML analysis
	MLPredictions []*MLPrediction    `json:"ml_predictions,omitempty"`
	FeatureScores map[string]float64 `json:"feature_scores,omitempty"`
}

type RuleMatch struct {
	RuleID      string     `json:"rule_id"`
	RuleName    string     `json:"rule_name"`
	Score       float64    `json:"score"`
	MatchedText string     `json:"matched_text"`
	Intent      IntentType `json:"intent"`
}

type PatternMatch struct {
	PatternID     string     `json:"pattern_id"`
	Pattern       string     `json:"pattern"`
	Score         float64    `json:"score"`
	MatchedGroups []string   `json:"matched_groups"`
	Intent        IntentType `json:"intent"`
}

type MLPrediction struct {
	Model      string             `json:"model"`
	Intent     IntentType         `json:"intent"`
	Confidence float64            `json:"confidence"`
	Features   map[string]float64 `json:"features,omitempty"`
}

type ClassificationDebugInfo struct {
	ProcessingSteps   []*ProcessingStep      `json:"processing_steps"`
	FeatureExtraction *FeatureExtractionInfo `json:"feature_extraction"`
	ClassifierScores  map[string]float64     `json:"classifier_scores"`
	DecisionPath      []string               `json:"decision_path"`
	Warnings          []string               `json:"warnings,omitempty"`
}

type ProcessingStep struct {
	Step       string        `json:"step"`
	Duration   time.Duration `json:"duration"`
	Result     interface{}   `json:"result"`
	Confidence float64       `json:"confidence,omitempty"`
}

type FeatureExtractionInfo struct {
	TokenCount        int      `json:"token_count"`
	UniqueTokens      int      `json:"unique_tokens"`
	KeywordDensity    float64  `json:"keyword_density"`
	SemanticFeatures  []string `json:"semantic_features"`
	SyntacticFeatures []string `json:"syntactic_features"`
}

type ContextUpdates struct {
	UpdatedContext      map[string]interface{} `json:"updated_context,omitempty"`
	InferredPreferences map[string]interface{} `json:"inferred_preferences,omitempty"`
	ConversationState   string                 `json:"conversation_state,omitempty"`
}

// Intent definitions and knowledge base

type IntentDefinition struct {
	Intent      IntentType `json:"intent"`
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Category    string     `json:"category"`

	// Classification features
	Keywords []string              `json:"keywords"`
	Phrases  []string              `json:"phrases"`
	Patterns []*IntentPattern      `json:"patterns"`
	Rules    []*ClassificationRule `json:"rules"`

	// Requirements and constraints
	RequiredEntities []EntityType `json:"required_entities,omitempty"`
	RequiredContext  []string     `json:"required_context,omitempty"`
	MinConfidence    float64      `json:"min_confidence"`

	// Examples and training data
	PositiveExamples []string `json:"positive_examples"`
	NegativeExamples []string `json:"negative_examples,omitempty"`

	// Response templates
	ResponseTemplates      []*ResponseTemplate `json:"response_templates,omitempty"`
	ClarificationQuestions []string            `json:"clarification_questions,omitempty"`

	// Metadata
	Priority    int       `json:"priority"`
	IsActive    bool      `json:"is_active"`
	LastUpdated time.Time `json:"last_updated"`
}

type IntentPattern struct {
	Pattern       string      `json:"pattern"`
	Weight        float64     `json:"weight"`
	Type          PatternType `json:"type"`
	CaseSensitive bool        `json:"case_sensitive"`
	MatchGroups   []string    `json:"match_groups,omitempty"`
}

type PatternType string

const (
	PatternTypeRegex    PatternType = "regex"
	PatternTypeWildcard PatternType = "wildcard"
	PatternTypeExact    PatternType = "exact"
	PatternTypeFuzzy    PatternType = "fuzzy"
)

type ClassificationRule struct {
	ID        string   `json:"id"`
	Name      string   `json:"name"`
	Condition string   `json:"condition"`
	Action    string   `json:"action"`
	Weight    float64  `json:"weight"`
	Context   []string `json:"context,omitempty"`
}

type ResponseTemplate struct {
	Template   string                 `json:"template"`
	Variables  []string               `json:"variables"`
	Conditions map[string]interface{} `json:"conditions,omitempty"`
}

// Training and feedback structures

type TrainingExample struct {
	ID         string                 `json:"id"`
	Query      string                 `json:"query"`
	Intent     IntentType             `json:"intent"`
	Entities   []*ExtractedEntity     `json:"entities,omitempty"`
	Context    *ClassificationContext `json:"context,omitempty"`
	Language   string                 `json:"language"`
	Confidence float64                `json:"confidence"`
	Source     string                 `json:"source"`
	CreatedAt  time.Time              `json:"created_at"`
	Verified   bool                   `json:"verified"`
}

type FeedbackEntry struct {
	ID              string                 `json:"id"`
	OriginalQuery   string                 `json:"original_query"`
	PredictedIntent IntentType             `json:"predicted_intent"`
	ActualIntent    IntentType             `json:"actual_intent"`
	Confidence      float64                `json:"confidence"`
	UserCorrection  *UserCorrection        `json:"user_correction,omitempty"`
	Context         *ClassificationContext `json:"context,omitempty"`
	Timestamp       time.Time              `json:"timestamp"`
	UserID          string                 `json:"user_id,omitempty"`
}

type UserCorrection struct {
	CorrectedIntent   IntentType         `json:"corrected_intent"`
	CorrectedEntities []*ExtractedEntity `json:"corrected_entities,omitempty"`
	Explanation       string             `json:"explanation,omitempty"`
	Confidence        float64            `json:"confidence"`
}

// Classification engines

type RuleBasedClassifier struct {
	rules     []*ClassificationRule
	ruleIndex map[string][]*ClassificationRule
	logger    logger.Logger
}

type PatternMatcher struct {
	patterns     []*CompiledPattern
	patternIndex map[IntentType][]*CompiledPattern
	logger       logger.Logger
}

type CompiledPattern struct {
	Pattern *regexp.Regexp
	Intent  IntentType
	Weight  float64
	Groups  []string
}

type KeywordAnalyzer struct {
	keywordMaps map[IntentType]map[string]float64
	stopWords   map[string]bool
	stemmer     Stemmer
	logger      logger.Logger
}

type SemanticAnalyzer struct {
	llmProvider      llm.Provider
	embeddingModel   *EmbeddingModel
	intentEmbeddings map[IntentType][]float64
	config           *SemanticAnalyzerConfig
	logger           logger.Logger
}

type SemanticAnalyzerConfig struct {
	UseEmbeddings      bool    `json:"use_embeddings"`
	UseLLM             bool    `json:"use_llm"`
	EmbeddingThreshold float64 `json:"embedding_threshold"`
	LLMPromptTemplate  string  `json:"llm_prompt_template"`
}

type ContextAnalyzer struct {
	contextRules   []*ContextRule
	contextFactors map[string]float64
	logger         logger.Logger
}

type ContextRule struct {
	ID             string                 `json:"id"`
	Condition      string                 `json:"condition"`
	IntentModifier map[IntentType]float64 `json:"intent_modifier"`
	Weight         float64                `json:"weight"`
}

type MLClassifier struct {
	models           map[string]MLModel
	featureExtractor *FeatureExtractor
	isEnabled        bool
	logger           logger.Logger
}

type EmbeddingModel struct {
	modelPath  string
	dimensions int
	isLoaded   bool
	mu         sync.RWMutex
}

type SimilarityMatcher struct {
	threshold     float64
	algorithmType SimilarityAlgorithm
}

type SimilarityAlgorithm string

const (
	SimilarityCosineSimilarity   SimilarityAlgorithm = "cosine"
	SimilarityJaccardIndex       SimilarityAlgorithm = "jaccard"
	SimilarityLevenshtein        SimilarityAlgorithm = "levenshtein"
	SimilaritySemanticSimilarity SimilarityAlgorithm = "semantic"
)

// Entity extraction components

type EntityExtractor struct {
	extractors map[EntityType]EntityExtractorImpl
	patterns   map[EntityType][]*regexp.Regexp
	gazetteers map[EntityType]map[string]bool
	logger     logger.Logger
}

type EntityExtractorImpl interface {
	Extract(text string) ([]*ExtractedEntity, error)
}

type ParameterExtractor struct {
	extractionRules []*ParameterExtractionRule
	typeInferencers map[string]TypeInferencer
	logger          logger.Logger
}

type ParameterExtractionRule struct {
	Name          string      `json:"name"`
	Pattern       string      `json:"pattern"`
	ParameterName string      `json:"parameter_name"`
	Type          string      `json:"type"`
	Required      bool        `json:"required"`
	DefaultValue  interface{} `json:"default_value,omitempty"`
}

type TypeInferencer interface {
	InferType(value string) (interface{}, error)
}

// Cache and metrics structures

type CachedClassification struct {
	Result   *ClassificationResult `json:"result"`
	CachedAt time.Time             `json:"cached_at"`
	HitCount int                   `json:"hit_count"`
}

type ClassificationMetrics struct {
	TotalClassifications      int64                `json:"total_classifications"`
	SuccessfulClassifications int64                `json:"successful_classifications"`
	AverageConfidence         float64              `json:"average_confidence"`
	AverageProcessingTime     time.Duration        `json:"average_processing_time"`
	IntentDistribution        map[IntentType]int64 `json:"intent_distribution"`
	LanguageDistribution      map[string]int64     `json:"language_distribution"`
	MethodDistribution        map[string]int64     `json:"method_distribution"`
	CacheHitRate              float64              `json:"cache_hit_rate"`
	AmbiguousQueries          int64                `json:"ambiguous_queries"`
	FeedbackReceived          int64                `json:"feedback_received"`
	LastReset                 time.Time            `json:"last_reset"`
	mu                        sync.RWMutex
}

// Learning and feedback components

type FeedbackProcessor struct {
	feedbackQueue     []*FeedbackEntry
	processingEnabled bool
	batchSize         int
	logger            logger.Logger
}

type AdaptiveLearner struct {
	learningRate      float64
	adaptationEnabled bool
	updateThreshold   int
	logger            logger.Logger
}

// Helper interfaces and types

type Stemmer interface {
	Stem(word string) string
}

type MLModel interface {
	Predict(features map[string]float64) (*MLPrediction, error)
	Train(examples []*TrainingExample) error
	IsLoaded() bool
}

type FeatureExtractor struct {
	enabledFeatures []string
	stopWords       map[string]bool
	logger          logger.Logger
}

type IntentKnowledgeBase struct {
	intents     map[IntentType]*IntentDefinition
	categories  map[string][]*IntentDefinition
	lastUpdated time.Time
	mu          sync.RWMutex
}

// NewIntentClassifier creates a new intent classifier
func NewIntentClassifier(llmProvider llm.Provider, config *IntentClassifierConfig, logger logger.Logger) *IntentClassifier {
	if config == nil {
		config = &IntentClassifierConfig{
			EnableRuleBased:         true,
			EnablePatternMatching:   true,
			EnableKeywordAnalysis:   true,
			EnableSemanticAnalysis:  true,
			EnableContextAnalysis:   true,
			EnableMLClassification:  false, // Disabled by default
			MinConfidenceThreshold:  0.3,
			HighConfidenceThreshold: 0.8,
			AmbiguityThreshold:      0.6,
			SupportedLanguages:      []string{"en", "es", "fr", "de", "zh"},
			DefaultLanguage:         "en",
			EnableMultiLanguage:     true,
			EnableEntityExtraction:  true,
			EntityTypes: []EntityType{
				EntityTypeProgrammingLanguage,
				EntityTypeFramework,
				EntityTypeFilePath,
				EntityTypeFunctionName,
				EntityTypeClassName,
			},
			MaxEntitiesPerQuery:      20,
			EnableFeedbackLearning:   true,
			FeedbackWeight:           0.1,
			LearningRate:             0.01,
			MaxProcessingTime:        time.Second * 5,
			EnableCaching:            true,
			CacheExpiry:              time.Hour,
			EnableParallelProcessing: true,
			LLMModel:                 "gpt-3.5-turbo",
			LLMMaxTokens:             1024,
			LLMTemperature:           0.3,
			EnableContextPropagation: true,
			ContextWindowSize:        10,
			EnableIntentChaining:     true,
			EnableDebugLogging:       false,
			LogClassificationSteps:   false,
			CollectMetrics:           true,
		}
	}

	ic := &IntentClassifier{
		llmProvider:         llmProvider,
		logger:              logger,
		config:              config,
		intentDefinitions:   make(map[string]*IntentDefinition),
		trainingData:        make([]*TrainingExample, 0),
		classificationCache: make(map[string]*CachedClassification),
		cacheExpiry:         config.CacheExpiry,
		metrics: &ClassificationMetrics{
			IntentDistribution:   make(map[IntentType]int64),
			LanguageDistribution: make(map[string]int64),
			MethodDistribution:   make(map[string]int64),
			LastReset:            time.Now(),
		},
	}

	// Initialize components
	ic.initializeComponents()

	// Load default intent definitions
	ic.loadDefaultIntentDefinitions()

	// Initialize knowledge base
	ic.initializeKnowledgeBase()

	return ic
}

// Main classification method
func (ic *IntentClassifier) ClassifyIntent(ctx context.Context, request *ClassificationRequest) (*ClassificationResult, error) {
	start := time.Now()

	// Validate request
	if err := ic.validateRequest(request); err != nil {
		return nil, fmt.Errorf("invalid request: %v", err)
	}

	// Check cache first
	if ic.config.EnableCaching {
		if cached := ic.getFromCache(request.Query); cached != nil {
			ic.updateCacheMetrics(true)
			ic.logDebug("Cache hit for query", map[string]interface{}{"query": request.Query})
			return cached.Result, nil
		}
		ic.updateCacheMetrics(false)
	}

	// Apply timeout
	classifyCtx := ctx
	if ic.config.MaxProcessingTime > 0 {
		var cancel context.CancelFunc
		classifyCtx, cancel = context.WithTimeout(ctx, ic.config.MaxProcessingTime)
		defer cancel()
	}

	// Perform classification
	result, err := ic.performClassification(classifyCtx, request)
	if err != nil {
		ic.updateErrorMetrics()
		return nil, fmt.Errorf("classification failed: %v", err)
	}

	// Set processing time
	result.ProcessingTime = time.Since(start)

	// Cache result
	if ic.config.EnableCaching && result.Confidence >= ic.config.MinConfidenceThreshold {
		ic.cacheResult(request.Query, result)
	}

	// Update metrics
	ic.updateSuccessMetrics(result)

	// Log debug information
	if ic.config.EnableDebugLogging {
		ic.logClassificationResult(request, result)
	}

	return result, nil
}

// Core classification logic
func (ic *IntentClassifier) performClassification(ctx context.Context, request *ClassificationRequest) (*ClassificationResult, error) {
	if request == nil {
		return nil, fmt.Errorf("nil classification request")
	}

	result := &ClassificationResult{
		Intent:                IntentUnknown,
		Confidence:            0.0,
		Alternatives:          make([]*IntentAlternative, 0),
		Entities:              make([]*ExtractedEntity, 0),
		Parameters:            make(map[string]interface{}),
		Language:              ic.detectLanguage(request.Query),
		IsAmbiguous:           false,
		RequiresClarification: false,
	}

	// Initialize debug info if requested
	if request.Options != nil && request.Options.IncludeDebugInfo {
		result.DebugInfo = &ClassificationDebugInfo{
			ProcessingSteps:  make([]*ProcessingStep, 0),
			ClassifierScores: make(map[string]float64),
			DecisionPath:     make([]string, 0),
		}
	}

	// Preprocessing
	preprocessedQuery := ic.preprocessQuery(request.Query)
	if result.DebugInfo != nil {
		ic.addDebugStep(result, "preprocessing", preprocessedQuery, time.Millisecond*5)
	}

	// Feature extraction
	features := ic.extractFeatures(preprocessedQuery, request.Context)
	if result.DebugInfo != nil {
		ic.addDebugStep(result, "feature_extraction", features, time.Millisecond*10)
	}

	// Classification engines (run in parallel if enabled)
	var classifierResults []*ClassifierResult
	var err error

	if ic.config.EnableParallelProcessing {
		classifierResults, err = ic.runClassifiersParallel(ctx, preprocessedQuery, request, features)
	} else {
		classifierResults, err = ic.runClassifiersSequential(ctx, preprocessedQuery, request, features)
	}

	if err != nil {
		return nil, fmt.Errorf("classifier execution failed: %v", err)
	}

	// Ensemble classification
	intentScores := ic.ensembleClassification(classifierResults)
	if result.DebugInfo != nil {
		ic.addDebugStep(result, "ensemble_classification", intentScores, time.Millisecond*15)
	}

	// Apply context if available
	if request.Context != nil && ic.config.EnableContextAnalysis {
		intentScores = ic.applyContextualScoring(intentScores, request.Context)
		if result.DebugInfo != nil {
			ic.addDebugStep(result, "contextual_scoring", intentScores, time.Millisecond*8)
		}
	}

	// Determine final intent and confidence
	finalIntent, confidence := ic.selectBestIntent(intentScores)
	result.Intent = finalIntent
	result.Confidence = confidence
	result.ClassificationMethod = ic.getClassificationMethod(classifierResults)

	// Generate alternatives
	if request.Options == nil || request.Options.AllowMultipleIntents {
		result.Alternatives = ic.generateAlternatives(intentScores, finalIntent)
	}

	// Check for ambiguity
	result.IsAmbiguous = ic.isAmbiguous(intentScores, confidence)
	result.RequiresClarification = ic.requiresClarification(result)

	// Extract entities if requested
	if ic.config.EnableEntityExtraction && (request.Options == nil || request.Options.ExtractEntities) {
		entities, err := ic.extractEntities(preprocessedQuery, finalIntent)
		if err != nil {
			ic.logger.Warn("Entity extraction failed", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			result.Entities = entities
			if result.DebugInfo != nil {
				ic.addDebugStep(result, "entity_extraction", entities, time.Millisecond*12)
			}
		}
	}

	// Extract parameters if requested
	if request.Options == nil || request.Options.ExtractParameters {
		parameters, err := ic.extractParameters(preprocessedQuery, finalIntent, result.Entities)
		if err != nil {
			ic.logger.Warn("Parameter extraction failed", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			result.Parameters = parameters
			if result.DebugInfo != nil {
				ic.addDebugStep(result, "parameter_extraction", parameters, time.Millisecond*8)
			}
		}
	}

	// Generate suggested questions for ambiguous queries
	if result.IsAmbiguous || result.RequiresClarification {
		result.SuggestedQuestions = ic.generateSuggestedQuestions(finalIntent, result.Alternatives)
	}

	// Update context if enabled
	if ic.config.EnableContextPropagation {
		result.ContextUpdates = ic.updateContext(request, result)
	}

	// Store analysis results
	result.AnalysisResults = ic.buildAnalysisResults(classifierResults, features)

	return result, nil
}

// Classification engines implementation
func (ic *IntentClassifier) runClassifiersSequential(
	ctx context.Context,
	query string,
	request *ClassificationRequest,
	features map[string]interface{},
) ([]*ClassifierResult, error) {
	var results []*ClassifierResult

	// Rule-based classification
	if ic.config.EnableRuleBased {
		ruleResult := ic.ruleBasedClassifier.Classify(query, request.Context)
		results = append(results, &ClassifierResult{
			Method:     "rule_based",
			Scores:     ruleResult.Scores,
			Confidence: ruleResult.Confidence,
			Details:    ruleResult,
		})
	}

	// Pattern matching
	if ic.config.EnablePatternMatching {
		patternResult := ic.patternMatcher.Classify(query)
		results = append(results, &ClassifierResult{
			Method:     "pattern_matching",
			Scores:     patternResult.Scores,
			Confidence: patternResult.Confidence,
			Details:    patternResult,
		})
	}

	// Keyword analysis
	if ic.config.EnableKeywordAnalysis {
		keywordResult := ic.keywordAnalyzer.Classify(query)
		results = append(results, &ClassifierResult{
			Method:     "keyword_analysis",
			Scores:     keywordResult.Scores,
			Confidence: keywordResult.Confidence,
			Details:    keywordResult,
		})
	}

	// Semantic analysis
	if ic.config.EnableSemanticAnalysis && ic.semanticAnalyzer != nil {
		semanticResult, err := ic.semanticAnalyzer.Classify(ctx, query)
		if err != nil {
			ic.logger.Warn("Semantic analysis failed", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			results = append(results, &ClassifierResult{
				Method:     "semantic_analysis",
				Scores:     semanticResult.Scores,
				Confidence: semanticResult.Confidence,
				Details:    semanticResult,
			})
		}
	}

	// ML classification
	if ic.config.EnableMLClassification && ic.mlClassifier != nil && ic.mlClassifier.isEnabled {
		mlResult, err := ic.mlClassifier.Classify(features)
		if err != nil {
			ic.logger.Warn("ML classification failed", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			results = append(results, &ClassifierResult{
				Method:     "ml_classification",
				Scores:     mlResult.Scores,
				Confidence: mlResult.Confidence,
				Details:    mlResult,
			})
		}
	}

	return results, nil
}

func (ic *IntentClassifier) runClassifiersParallel(ctx context.Context, query string, request *ClassificationRequest, features map[string]interface{}) ([]*ClassifierResult, error) {
	var wg sync.WaitGroup
	results := make([]*ClassifierResult, 0)
	resultsChan := make(chan *ClassifierResult, 10)
	errorChan := make(chan error, 10)

	// Rule-based classification
	if ic.config.EnableRuleBased {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ruleResult := ic.ruleBasedClassifier.Classify(query, request.Context)
			resultsChan <- &ClassifierResult{
				Method:     "rule_based",
				Scores:     ruleResult.Scores,
				Confidence: ruleResult.Confidence,
				Details:    ruleResult,
			}
		}()
	}

	// Pattern matching
	if ic.config.EnablePatternMatching {
		wg.Add(1)
		go func() {
			defer wg.Done()
			patternResult := ic.patternMatcher.Classify(query)
			resultsChan <- &ClassifierResult{
				Method:     "pattern_matching",
				Scores:     patternResult.Scores,
				Confidence: patternResult.Confidence,
				Details:    patternResult,
			}
		}()
	}

	// Keyword analysis
	if ic.config.EnableKeywordAnalysis {
		wg.Add(1)
		go func() {
			defer wg.Done()
			keywordResult := ic.keywordAnalyzer.Classify(query)
			resultsChan <- &ClassifierResult{
				Method:     "keyword_analysis",
				Scores:     keywordResult.Scores,
				Confidence: keywordResult.Confidence,
				Details:    keywordResult,
			}
		}()
	}

	// Semantic analysis
	if ic.config.EnableSemanticAnalysis && ic.semanticAnalyzer != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			semanticResult, err := ic.semanticAnalyzer.Classify(ctx, query)
			if err != nil {
				errorChan <- err
				return
			}
			resultsChan <- &ClassifierResult{
				Method:     "semantic_analysis",
				Scores:     semanticResult.Scores,
				Confidence: semanticResult.Confidence,
				Details:    semanticResult,
			}
		}()
	}

	// ML classification
	if ic.config.EnableMLClassification && ic.mlClassifier != nil && ic.mlClassifier.isEnabled {
		wg.Add(1)
		go func() {
			defer wg.Done()
			mlResult, err := ic.mlClassifier.Classify(features)
			if err != nil {
				errorChan <- err
				return
			}
			resultsChan <- &ClassifierResult{
				Method:     "ml_classification",
				Scores:     mlResult.Scores,
				Confidence: mlResult.Confidence,
				Details:    mlResult,
			}
		}()
	}

	// Wait for all goroutines to complete
	go func() {
		wg.Wait()
		close(resultsChan)
		close(errorChan)
	}()

	// Collect results
	for result := range resultsChan {
		results = append(results, result)
	}

	// Check for errors
	for err := range errorChan {
		ic.logger.Warn("Classifier error", map[string]interface{}{
			"error": err.Error(),
		})
	}

	return results, nil
}

// Result type for individual classifiers
type ClassifierResult struct {
	Method     string                 `json:"method"`
	Scores     map[IntentType]float64 `json:"scores"`
	Confidence float64                `json:"confidence"`
	Details    interface{}            `json:"details"`
}

// Ensemble classification and result aggregation
func (ic *IntentClassifier) ensembleClassification(results []*ClassifierResult) map[IntentType]float64 {
	intentScores := make(map[IntentType]float64)
	totalWeight := 0.0

	// Weight configuration for different methods
	methodWeights := map[string]float64{
		"rule_based":        0.3,
		"pattern_matching":  0.25,
		"keyword_analysis":  0.2,
		"semantic_analysis": 0.35,
		"ml_classification": 0.4,
	}

	// Aggregate scores from all classifiers
	for _, result := range results {
		weight := methodWeights[result.Method]
		if weight == 0 {
			weight = 0.1 // Default weight
		}

		for intent, score := range result.Scores {
			intentScores[intent] += score * weight
		}
		totalWeight += weight
	}

	// Normalize scores
	if totalWeight > 0 {
		for intent := range intentScores {
			intentScores[intent] /= totalWeight
		}
	}

	return intentScores
}

func (ic *IntentClassifier) selectBestIntent(scores map[IntentType]float64) (IntentType, float64) {
	var bestIntent IntentType = IntentUnknown
	var bestScore float64 = 0.0

	for intent, score := range scores {
		if score > bestScore {
			bestIntent = intent
			bestScore = score
		}
	}

	// Check if confidence meets minimum threshold
	if bestScore < ic.config.MinConfidenceThreshold {
		return IntentUnknown, bestScore
	}

	return bestIntent, bestScore
}

func (ic *IntentClassifier) generateAlternatives(scores map[IntentType]float64, primaryIntent IntentType) []*IntentAlternative {
	type intentScore struct {
		intent IntentType
		score  float64
	}

	var sortedScores []intentScore
	for intent, score := range scores {
		if intent != primaryIntent && score > ic.config.MinConfidenceThreshold {
			sortedScores = append(sortedScores, intentScore{intent, score})
		}
	}

	// Sort by score descending
	sort.Slice(sortedScores, func(i, j int) bool {
		return sortedScores[i].score > sortedScores[j].score
	})

	alternatives := make([]*IntentAlternative, 0)
	maxAlternatives := 3 // Default limit

	for i, intentScore := range sortedScores {
		if i >= maxAlternatives {
			break
		}

		alternatives = append(alternatives, &IntentAlternative{
			Intent:     intentScore.intent,
			Confidence: intentScore.score,
			Reason:     ic.getAlternativeReason(intentScore.intent, intentScore.score),
		})
	}

	return alternatives
}

// Helper methods

func (ic *IntentClassifier) validateRequest(request *ClassificationRequest) error {
	if request == nil {
		return fmt.Errorf("request cannot be nil")
	}

	if strings.TrimSpace(request.Query) == "" {
		return fmt.Errorf("query cannot be empty")
	}

	if len(request.Query) > 10000 { // Reasonable limit
		return fmt.Errorf("query too long: maximum 10000 characters")
	}

	return nil
}

func (ic *IntentClassifier) detectLanguage(query string) string {
	// Simplified language detection - in reality would use a proper language detector
	if ic.config.EnableMultiLanguage {
		// Basic heuristics for language detection
		// This would be replaced with a proper language detection library
		return ic.config.DefaultLanguage
	}
	return ic.config.DefaultLanguage
}

func (ic *IntentClassifier) preprocessQuery(query string) string {
	// Normalize text
	query = strings.TrimSpace(query)
	query = strings.ToLower(query)

	// Remove extra whitespace
	query = regexp.MustCompile(`\s+`).ReplaceAllString(query, " ")

	// Handle contractions
	contractions := map[string]string{
		"don't":     "do not",
		"can't":     "cannot",
		"won't":     "will not",
		"shouldn't": "should not",
		"couldn't":  "could not",
		"wouldn't":  "would not",
	}

	for contraction, expansion := range contractions {
		query = strings.ReplaceAll(query, contraction, expansion)
	}

	return query
}

func (ic *IntentClassifier) extractFeatures(query string, context *ClassificationContext) map[string]interface{} {
	features := make(map[string]interface{})

	// Basic text features
	tokens := strings.Fields(query)
	features["token_count"] = len(tokens)
	features["char_count"] = len(query)
	features["avg_word_length"] = ic.calculateAverageWordLength(tokens)

	// Question indicators
	features["has_question_mark"] = strings.Contains(query, "?")
	features["starts_with_wh"] = ic.startsWithQuestionWord(query)

	// Action indicators
	actionWords := []string{"generate", "create", "make", "build", "write", "explain", "analyze", "review", "fix", "optimize"}
	for _, word := range actionWords {
		features["has_"+word] = strings.Contains(query, word)
	}

	// Programming-related features
	programmingKeywords := []string{"function", "class", "variable", "method", "API", "database", "algorithm", "code", "debug", "test"}
	programmingCount := 0
	for _, keyword := range programmingKeywords {
		if strings.Contains(query, keyword) {
			programmingCount++
		}
	}
	features["programming_keyword_count"] = programmingCount
	features["programming_density"] = float64(programmingCount) / float64(len(tokens))

	// Context features
	if context != nil {
		features["has_current_file"] = context.CurrentFile != ""
		features["has_selected_code"] = context.SelectedCode != ""
		features["current_language"] = context.CurrentLanguage
		features["file_count"] = len(context.SelectedFiles)
	}

	return features
}

func (ic *IntentClassifier) calculateAverageWordLength(tokens []string) float64 {
	if len(tokens) == 0 {
		return 0
	}

	totalLength := 0
	for _, token := range tokens {
		totalLength += len(token)
	}

	return float64(totalLength) / float64(len(tokens))
}

func (ic *IntentClassifier) startsWithQuestionWord(query string) bool {
	questionWords := []string{"what", "how", "why", "when", "where", "who", "which", "can", "could", "should", "would", "is", "are", "do", "does", "did"}

	tokens := strings.Fields(strings.ToLower(query))
	if len(tokens) == 0 {
		return false
	}

	firstWord := tokens[0]
	for _, qw := range questionWords {
		if firstWord == qw {
			return true
		}
	}

	return false
}

func (ic *IntentClassifier) applyContextualScoring(scores map[IntentType]float64, context *ClassificationContext) map[IntentType]float64 {
	// Apply context-based adjustments
	adjustedScores := make(map[IntentType]float64)
	for intent, score := range scores {
		adjustedScores[intent] = score
	}

	// Boost scores based on context
	if context.SelectedCode != "" {
		// If code is selected, boost code-related intents
		codeRelatedIntents := []IntentType{
			IntentCodeExplanation,
			IntentCodeReview,
			IntentCodeRefactoring,
			IntentCodeOptimization,
		}

		for _, intent := range codeRelatedIntents {
			if score, exists := adjustedScores[intent]; exists {
				adjustedScores[intent] = score * 1.2 // 20% boost
			}
		}
	}

	if context.CurrentLanguage != "" {
		// Language-specific adjustments could be applied here
	}

	// Previous intent influence
	if context.LastIntent != "" {
		// Apply some continuity bias
		if score, exists := adjustedScores[context.LastIntent]; exists {
			adjustedScores[context.LastIntent] = score * 1.1 // 10% boost for continuity
		}
	}

	return adjustedScores
}

func (ic *IntentClassifier) isAmbiguous(scores map[IntentType]float64, topConfidence float64) bool {
	if topConfidence < ic.config.AmbiguityThreshold {
		return true
	}

	// Check if there are multiple high-confidence intents
	highConfidenceCount := 0
	for _, score := range scores {
		if score >= ic.config.HighConfidenceThreshold {
			highConfidenceCount++
		}
	}

	return highConfidenceCount > 1
}

func (ic *IntentClassifier) requiresClarification(result *ClassificationResult) bool {
	// Requires clarification if confidence is low or if it's ambiguous
	if result.Confidence < ic.config.MinConfidenceThreshold {
		return true
	}

	if result.IsAmbiguous {
		return true
	}

	// Check if required entities are missing
	if intentDef, exists := ic.intentDefinitions[string(result.Intent)]; exists {
		requiredEntities := intentDef.RequiredEntities
		if len(requiredEntities) > 0 && len(result.Entities) == 0 {
			return true
		}
	}

	return false
}

func (ic *IntentClassifier) extractEntities(query string, intent IntentType) ([]*ExtractedEntity, error) {
	if ic.entityExtractor == nil {
		return []*ExtractedEntity{}, nil
	}

	var allEntities []*ExtractedEntity

	// Extract entities for each type
	for _, entityType := range ic.config.EntityTypes {
		if extractor, exists := ic.entityExtractor.extractors[entityType]; exists {
			entities, err := extractor.Extract(query)
			if err != nil {
				ic.logger.Warn("Entity extraction failed", map[string]interface{}{
					"type":  entityType,
					"error": err,
				})
				continue
			}
			allEntities = append(allEntities, entities...)
		}
	}

	// Limit number of entities
	if len(allEntities) > ic.config.MaxEntitiesPerQuery {
		// Sort by confidence and take top N
		sort.Slice(allEntities, func(i, j int) bool {
			return allEntities[i].Confidence > allEntities[j].Confidence
		})
		allEntities = allEntities[:ic.config.MaxEntitiesPerQuery]
	}

	return allEntities, nil
}

func (ic *IntentClassifier) extractParameters(query string, intent IntentType, entities []*ExtractedEntity) (map[string]interface{}, error) {
	if ic.parameterExtractor == nil {
		return make(map[string]interface{}), nil
	}

	parameters := make(map[string]interface{})

	// Extract parameters based on extraction rules
	for _, rule := range ic.parameterExtractor.extractionRules {
		pattern := regexp.MustCompile(rule.Pattern)
		matches := pattern.FindStringSubmatch(query)

		if len(matches) > 1 {
			value := matches[1]

			// Apply type inference
			if inferencer, exists := ic.parameterExtractor.typeInferencers[rule.Type]; exists {
				typedValue, err := inferencer.InferType(value)
				if err == nil {
					parameters[rule.ParameterName] = typedValue
				} else {
					parameters[rule.ParameterName] = value
				}
			} else {
				parameters[rule.ParameterName] = value
			}
		} else if rule.DefaultValue != nil {
			parameters[rule.ParameterName] = rule.DefaultValue
		}
	}

	// Extract parameters from entities
	for _, entity := range entities {
		paramName := string(entity.Type)
		parameters[paramName] = entity.NormalizedValue
		if parameters[paramName] == "" {
			parameters[paramName] = entity.Value
		}
	}

	return parameters, nil
}

func (ic *IntentClassifier) generateSuggestedQuestions(intent IntentType, alternatives []*IntentAlternative) []string {
	questions := make([]string, 0)

	// Get questions from intent definition
	if intentDef, exists := ic.intentDefinitions[string(intent)]; exists {
		questions = append(questions, intentDef.ClarificationQuestions...)
	}

	// Generate questions based on alternatives
	if len(alternatives) > 0 {
		questions = append(questions, "Did you mean to:")
		for _, alt := range alternatives {
			if altDef, exists := ic.intentDefinitions[string(alt.Intent)]; exists {
				questions = append(questions, fmt.Sprintf("- %s", altDef.Description))
			}
		}
	}

	// Default questions for unknown intent
	if intent == IntentUnknown {
		questions = append(questions, []string{
			"Could you provide more details about what you want to do?",
			"Are you looking to generate code, analyze existing code, or get help with something else?",
			"Which programming language are you working with?",
		}...)
	}

	return questions
}

func (ic *IntentClassifier) updateContext(request *ClassificationRequest, result *ClassificationResult) *ContextUpdates {
	updates := &ContextUpdates{
		UpdatedContext:      make(map[string]interface{}),
		InferredPreferences: make(map[string]interface{}),
	}

	// Update conversation state
	if result.Confidence >= ic.config.HighConfidenceThreshold {
		updates.ConversationState = "confident"
	} else if result.IsAmbiguous {
		updates.ConversationState = "ambiguous"
	} else {
		updates.ConversationState = "uncertain"
	}

	// Infer preferences from successful classifications
	if result.Confidence >= ic.config.HighConfidenceThreshold {
		// Infer language preference
		for _, entity := range result.Entities {
			if entity.Type == EntityTypeProgrammingLanguage {
				updates.InferredPreferences["preferred_language"] = entity.NormalizedValue
			}
			if entity.Type == EntityTypeFramework {
				updates.InferredPreferences["preferred_framework"] = entity.NormalizedValue
			}
		}
	}

	// Update context with current intent
	updates.UpdatedContext["last_intent"] = result.Intent
	updates.UpdatedContext["last_confidence"] = result.Confidence

	return updates
}

// Cache management
func (ic *IntentClassifier) getFromCache(query string) *CachedClassification {
	ic.cacheMu.RLock()
	defer ic.cacheMu.RUnlock()

	if cached, exists := ic.classificationCache[query]; exists {
		// Check if cache entry is still valid
		if time.Since(cached.CachedAt) < ic.cacheExpiry {
			cached.HitCount++
			return cached
		}
		// Remove expired entry
		delete(ic.classificationCache, query)
	}

	return nil
}

func (ic *IntentClassifier) cacheResult(query string, result *ClassificationResult) {
	ic.cacheMu.Lock()
	defer ic.cacheMu.Unlock()

	ic.classificationCache[query] = &CachedClassification{
		Result:   result,
		CachedAt: time.Now(),
		HitCount: 0,
	}
}

// Metrics and monitoring
func (ic *IntentClassifier) updateSuccessMetrics(result *ClassificationResult) {
	ic.metrics.mu.Lock()
	defer ic.metrics.mu.Unlock()

	ic.metrics.TotalClassifications++
	ic.metrics.SuccessfulClassifications++
	ic.metrics.IntentDistribution[result.Intent]++
	ic.metrics.LanguageDistribution[result.Language]++
	ic.metrics.MethodDistribution[result.ClassificationMethod]++

	// Update average confidence
	if ic.metrics.TotalClassifications == 1 {
		ic.metrics.AverageConfidence = result.Confidence
	} else {
		ic.metrics.AverageConfidence = (ic.metrics.AverageConfidence + result.Confidence) / 2
	}

	// Update average processing time
	if ic.metrics.TotalClassifications == 1 {
		ic.metrics.AverageProcessingTime = result.ProcessingTime
	} else {
		ic.metrics.AverageProcessingTime = (ic.metrics.AverageProcessingTime + result.ProcessingTime) / 2
	}

	if result.IsAmbiguous {
		ic.metrics.AmbiguousQueries++
	}
}

func (ic *IntentClassifier) updateErrorMetrics() {
	ic.metrics.mu.Lock()
	defer ic.metrics.mu.Unlock()

	ic.metrics.TotalClassifications++
}

func (ic *IntentClassifier) updateCacheMetrics(hit bool) {
	ic.metrics.mu.Lock()
	defer ic.metrics.mu.Unlock()

	if hit {
		ic.metrics.CacheHitRate = (ic.metrics.CacheHitRate + 1.0) / 2.0
	} else {
		ic.metrics.CacheHitRate = ic.metrics.CacheHitRate / 2.0
	}
}

// Component initialization
func (ic *IntentClassifier) initializeComponents() {
	// Initialize rule-based classifier
	if ic.config.EnableRuleBased {
		ic.ruleBasedClassifier = NewRuleBasedClassifier(ic.logger)
	}

	// Initialize pattern matcher
	if ic.config.EnablePatternMatching {
		ic.patternMatcher = NewPatternMatcher(ic.logger)
	}

	// Initialize keyword analyzer
	if ic.config.EnableKeywordAnalysis {
		ic.keywordAnalyzer = NewKeywordAnalyzer(ic.logger)
	}

	// Initialize semantic analyzer
	if ic.config.EnableSemanticAnalysis {
		semanticConfig := &SemanticAnalyzerConfig{
			UseEmbeddings:      true,
			UseLLM:             true,
			EmbeddingThreshold: 0.7,
			LLMPromptTemplate:  "Classify the intent of this query: {{.Query}}",
		}
		ic.semanticAnalyzer = NewSemanticAnalyzer(ic.llmProvider, semanticConfig, ic.logger)
	}

	// Initialize context analyzer
	if ic.config.EnableContextAnalysis {
		ic.contextAnalyzer = NewContextAnalyzer(ic.logger)
	}

	// Initialize ML classifier if enabled
	if ic.config.EnableMLClassification {
		ic.mlClassifier = NewMLClassifier(ic.logger)
	}

	// Initialize entity extractor
	if ic.config.EnableEntityExtraction {
		ic.entityExtractor = NewEntityExtractor(ic.config.EntityTypes, ic.logger)
	}

	// Initialize parameter extractor
	ic.parameterExtractor = NewParameterExtractor(ic.logger)

	// Initialize feedback processor
	if ic.config.EnableFeedbackLearning {
		ic.feedbackProcessor = NewFeedbackProcessor(ic.logger)
		ic.adaptiveLearner = NewAdaptiveLearner(ic.config.LearningRate, ic.logger)
	}

	ic.isInitialized = true
}

// Load default intent definitions
func (ic *IntentClassifier) loadDefaultIntentDefinitions() {
	// This would typically load from a configuration file or database
	// For now, we'll define some basic intent definitions

	definitions := []*IntentDefinition{
		{
			Intent:      IntentCodeGeneration,
			Name:        "Code Generation",
			Description: "Generate new code",
			Category:    "generation",
			Keywords:    []string{"generate", "create", "write", "make", "build"},
			Phrases:     []string{"generate code", "create function", "write class"},
			Patterns: []*IntentPattern{
				{
					Pattern: `(?i)generate|create|write|make.*(?:code|function|class|method)`,
					Weight:  0.8,
					Type:    PatternTypeRegex,
				},
			},
			PositiveExamples: []string{
				"Generate a function to calculate factorial",
				"Create a class for user authentication",
				"Write code to sort an array",
			},
			MinConfidence: 0.6,
			Priority:      10,
			IsActive:      true,
			LastUpdated:   time.Now(),
		},
		{
			Intent:      IntentCodeExplanation,
			Name:        "Code Explanation",
			Description: "Explain existing code",
			Category:    "analysis",
			Keywords:    []string{"explain", "what", "how", "understand", "describe"},
			Phrases:     []string{"explain code", "what does this do", "how does this work"},
			Patterns: []*IntentPattern{
				{
					Pattern: `(?i)explain|what.*(?:does|is)|how.*(?:does|works?)`,
					Weight:  0.8,
					Type:    PatternTypeRegex,
				},
			},
			PositiveExamples: []string{
				"Explain this function",
				"What does this code do?",
				"How does this algorithm work?",
			},
			MinConfidence: 0.6,
			Priority:      8,
			IsActive:      true,
			LastUpdated:   time.Now(),
		},
		// More intent definitions would be added here...
	}

	for _, def := range definitions {
		ic.intentDefinitions[string(def.Intent)] = def
	}
}

func (ic *IntentClassifier) initializeKnowledgeBase() {
	ic.intentKnowledgeBase = &IntentKnowledgeBase{
		intents:     make(map[IntentType]*IntentDefinition),
		categories:  make(map[string][]*IntentDefinition),
		lastUpdated: time.Now(),
	}

	// Populate knowledge base from intent definitions
	for _, def := range ic.intentDefinitions {
		ic.intentKnowledgeBase.intents[def.Intent] = def

		// Group by category
		if _, exists := ic.intentKnowledgeBase.categories[def.Category]; !exists {
			ic.intentKnowledgeBase.categories[def.Category] = make([]*IntentDefinition, 0)
		}
		ic.intentKnowledgeBase.categories[def.Category] = append(
			ic.intentKnowledgeBase.categories[def.Category], def)
	}
}

// Public API methods

func (ic *IntentClassifier) AddTrainingExample(example *TrainingExample) error {
	ic.mu.Lock()
	defer ic.mu.Unlock()

	ic.trainingData = append(ic.trainingData, example)

	// Trigger retraining if adaptive learning is enabled
	if ic.config.EnableFeedbackLearning && ic.adaptiveLearner != nil {
		return ic.adaptiveLearner.UpdateFromExample(example)
	}

	return nil
}

func (ic *IntentClassifier) ProvideFeedback(feedback *FeedbackEntry) error {
	if !ic.config.EnableFeedbackLearning {
		return fmt.Errorf("feedback learning is disabled")
	}

	ic.metrics.mu.Lock()
	ic.metrics.FeedbackReceived++
	ic.metrics.mu.Unlock()

	return ic.feedbackProcessor.ProcessFeedback(feedback)
}

func (ic *IntentClassifier) GetMetrics() *ClassificationMetrics {
	ic.metrics.mu.RLock()
	defer ic.metrics.mu.RUnlock()

	// Return a copy
	metrics := *ic.metrics
	return &metrics
}

func (ic *IntentClassifier) ResetMetrics() {
	ic.metrics.mu.Lock()
	defer ic.metrics.mu.Unlock()

	ic.metrics = &ClassificationMetrics{
		IntentDistribution:   make(map[IntentType]int64),
		LanguageDistribution: make(map[string]int64),
		MethodDistribution:   make(map[string]int64),
		LastReset:            time.Now(),
	}
}

func (ic *IntentClassifier) ClearCache() {
	ic.cacheMu.Lock()
	defer ic.cacheMu.Unlock()

	ic.classificationCache = make(map[string]*CachedClassification)
}

// Debug and utility methods

func (ic *IntentClassifier) addDebugStep(result *ClassificationResult, step string, data interface{}, duration time.Duration) {
	if result.DebugInfo != nil {
		result.DebugInfo.ProcessingSteps = append(result.DebugInfo.ProcessingSteps, &ProcessingStep{
			Step:     step,
			Duration: duration,
			Result:   data,
		})
	}
}

func (ic *IntentClassifier) logDebug(message string, data map[string]interface{}) {
	if ic.config.EnableDebugLogging {
		ic.logger.Debug(message, data)
	}
}

func (ic *IntentClassifier) logClassificationResult(request *ClassificationRequest, result *ClassificationResult) {
	ic.logger.Debug("Classification completed", map[string]interface{}{
		"query":           request.Query,
		"intent":          result.Intent,
		"confidence":      result.Confidence,
		"method":          result.ClassificationMethod,
		"processing_time": result.ProcessingTime,
		"is_ambiguous":    result.IsAmbiguous,
	})
}

func (ic *IntentClassifier) getClassificationMethod(results []*ClassifierResult) string {
	if len(results) == 0 {
		return "unknown"
	}

	// Find the method with highest confidence
	bestMethod := "ensemble"
	bestConfidence := 0.0

	for _, result := range results {
		if result.Confidence > bestConfidence {
			bestMethod = result.Method
			bestConfidence = result.Confidence
		}
	}

	return bestMethod
}

func (ic *IntentClassifier) getAlternativeReason(intent IntentType, confidence float64) string {
	if confidence >= ic.config.HighConfidenceThreshold {
		return "High confidence alternative"
	} else if confidence >= ic.config.MinConfidenceThreshold {
		return "Possible alternative interpretation"
	} else {
		return "Low confidence alternative"
	}
}

func (ic *IntentClassifier) buildAnalysisResults(classifierResults []*ClassifierResult, features map[string]interface{}) *AnalysisResults {
	analysis := &AnalysisResults{
		RuleMatches:    make([]*RuleMatch, 0),
		PatternMatches: make([]*PatternMatch, 0),
		KeywordScores:  make(map[string]float64),
		ConceptMatches: make([]string, 0),
		ContextFactors: make([]string, 0),
		MLPredictions:  make([]*MLPrediction, 0),
		FeatureScores:  make(map[string]float64),
	}

	// Extract analysis results from each classifier
	for _, result := range classifierResults {
		switch result.Method {
		case "rule_based":
			if ruleResult, ok := result.Details.(*RuleBasedResult); ok {
				analysis.RuleMatches = ruleResult.Matches
			}
		case "pattern_matching":
			if patternResult, ok := result.Details.(*PatternMatchingResult); ok {
				analysis.PatternMatches = patternResult.Matches
			}
		case "keyword_analysis":
			if keywordResult, ok := result.Details.(*KeywordAnalysisResult); ok {
				scores := make(map[string]float64)
				for k, v := range keywordResult.Scores {
					scores[string(k)] = v
				}
				analysis.KeywordScores = scores
			}
		case "semantic_analysis":
			if semanticResult, ok := result.Details.(*SemanticAnalysisResult); ok {
				analysis.SemanticSimilarity = semanticResult.Similarity
				analysis.ConceptMatches = semanticResult.Concepts
			}
		case "ml_classification":
			if mlResult, ok := result.Details.(*MLClassificationResult); ok {
				analysis.MLPredictions = mlResult.Predictions
				analysis.FeatureScores = mlResult.FeatureScores
			}
		}
	}

	return analysis
}

// Placeholder structures for classifier results
type RuleBasedResult struct {
	Scores     map[IntentType]float64
	Matches    []*RuleMatch
	Confidence float64
}

type PatternMatchingResult struct {
	Scores     map[IntentType]float64
	Matches    []*PatternMatch
	Confidence float64
}

type KeywordAnalysisResult struct {
	Scores     map[IntentType]float64
	Confidence float64
}

type SemanticAnalysisResult struct {
	Scores     map[IntentType]float64
	Similarity float64
	Concepts   []string
	Confidence float64
}

type MLClassificationResult struct {
	Scores        map[IntentType]float64
	Predictions   []*MLPrediction
	FeatureScores map[string]float64
	Confidence    float64
}

// Constructor functions for components (simplified implementations)
func NewRuleBasedClassifier(logger logger.Logger) *RuleBasedClassifier {
	return &RuleBasedClassifier{
		rules:     make([]*ClassificationRule, 0),
		ruleIndex: make(map[string][]*ClassificationRule),
		logger:    logger,
	}
}

func NewPatternMatcher(logger logger.Logger) *PatternMatcher {
	return &PatternMatcher{
		patterns:     make([]*CompiledPattern, 0),
		patternIndex: make(map[IntentType][]*CompiledPattern),
		logger:       logger,
	}
}

func NewKeywordAnalyzer(logger logger.Logger) *KeywordAnalyzer {
	return &KeywordAnalyzer{
		keywordMaps: make(map[IntentType]map[string]float64),
		stopWords:   make(map[string]bool),
		logger:      logger,
	}
}

func NewSemanticAnalyzer(llmProvider llm.Provider, config *SemanticAnalyzerConfig, logger logger.Logger) *SemanticAnalyzer {
	return &SemanticAnalyzer{
		llmProvider:      llmProvider,
		intentEmbeddings: make(map[IntentType][]float64),
		config:           config,
		logger:           logger,
	}
}

func NewContextAnalyzer(logger logger.Logger) *ContextAnalyzer {
	return &ContextAnalyzer{
		contextRules:   make([]*ContextRule, 0),
		contextFactors: make(map[string]float64),
		logger:         logger,
	}
}

func NewMLClassifier(logger logger.Logger) *MLClassifier {
	return &MLClassifier{
		models:           make(map[string]MLModel),
		featureExtractor: &FeatureExtractor{},
		isEnabled:        false,
		logger:           logger,
	}
}

func NewEntityExtractor(entityTypes []EntityType, logger logger.Logger) *EntityExtractor {
	return &EntityExtractor{
		extractors: make(map[EntityType]EntityExtractorImpl),
		patterns:   make(map[EntityType][]*regexp.Regexp),
		gazetteers: make(map[EntityType]map[string]bool),
		logger:     logger,
	}
}

func NewParameterExtractor(logger logger.Logger) *ParameterExtractor {
	return &ParameterExtractor{
		extractionRules: make([]*ParameterExtractionRule, 0),
		typeInferencers: make(map[string]TypeInferencer),
		logger:          logger,
	}
}

func NewFeedbackProcessor(logger logger.Logger) *FeedbackProcessor {
	return &FeedbackProcessor{
		feedbackQueue:     make([]*FeedbackEntry, 0),
		processingEnabled: true,
		batchSize:         10,
		logger:            logger,
	}
}

func NewAdaptiveLearner(learningRate float64, logger logger.Logger) *AdaptiveLearner {
	return &AdaptiveLearner{
		learningRate:      learningRate,
		adaptationEnabled: true,
		updateThreshold:   10,
		logger:            logger,
	}
}

// Placeholder methods for classifier implementations
func (rbc *RuleBasedClassifier) Classify(query string, context *ClassificationContext) *RuleBasedResult {
	// Implementation would apply rules to classify
	scores := make(map[IntentType]float64)
	scores[IntentCodeGeneration] = 0.7 // Simplified
	return &RuleBasedResult{
		Scores:     scores,
		Matches:    []*RuleMatch{},
		Confidence: 0.7,
	}
}

func (pm *PatternMatcher) Classify(query string) *PatternMatchingResult {
	// Implementation would match patterns
	scores := make(map[IntentType]float64)
	scores[IntentCodeGeneration] = 0.6 // Simplified
	return &PatternMatchingResult{
		Scores:     scores,
		Matches:    []*PatternMatch{},
		Confidence: 0.6,
	}
}

func (ka *KeywordAnalyzer) Classify(query string) *KeywordAnalysisResult {
	// Implementation would analyze keywords
	scores := make(map[IntentType]float64)
	scores[IntentCodeGeneration] = 0.5 // Simplified
	return &KeywordAnalysisResult{
		Scores:     scores,
		Confidence: 0.5,
	}
}

func (sa *SemanticAnalyzer) Classify(ctx context.Context, query string) (*SemanticAnalysisResult, error) {
	// Implementation would use LLM or embeddings for semantic analysis
	scores := make(map[IntentType]float64)
	scores[IntentCodeGeneration] = 0.8 // Simplified
	return &SemanticAnalysisResult{
		Scores:     scores,
		Similarity: 0.8,
		Concepts:   []string{"code", "generation"},
		Confidence: 0.8,
	}, nil
}

func (mc *MLClassifier) Classify(features map[string]interface{}) (*MLClassificationResult, error) {
	// Implementation would use ML models for classification
	scores := make(map[IntentType]float64)
	scores[IntentCodeGeneration] = 0.75 // Simplified
	return &MLClassificationResult{
		Scores:        scores,
		Predictions:   []*MLPrediction{},
		FeatureScores: make(map[string]float64),
		Confidence:    0.75,
	}, nil
}

func (al *AdaptiveLearner) UpdateFromExample(example *TrainingExample) error {
	// Implementation would update models based on new examples
	return nil
}

func (fp *FeedbackProcessor) ProcessFeedback(feedback *FeedbackEntry) error {
	// Implementation would process user feedback
	fp.feedbackQueue = append(fp.feedbackQueue, feedback)
	return nil
}
