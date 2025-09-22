package agents

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/app"
	"github.com/yourusername/ai-code-assistant/internal/logger"
)

// AgentRouter intelligently routes requests to the most appropriate AI agent
type AgentRouter struct {
	// Core components
	intentClassifier  *app.IntentClassifier
	contextManager    *app.ContextManager
	queryOptimizer    *app.QueryOptimizer
	responseFormatter *app.ResponseFormatter

	// Registered agents
	agents            map[AgentType]Agent
	agentCapabilities map[AgentType]*AgentCapabilities

	// Routing logic
	routingRules  []*RoutingRule
	fallbackAgent AgentType

	// Performance optimization
	routingCache *RoutingCache
	agentPool    *AgentPool
	loadBalancer *AgentLoadBalancer

	// Analytics and monitoring
	routingStats       *RoutingStatistics
	performanceMonitor *AgentPerformanceMonitor

	// Configuration
	config *RouterConfig
	logger logger.Logger

	// State management
	mu sync.RWMutex
}

// RouterConfig contains router configuration
type RouterConfig struct {
	// Routing behavior
	EnableIntelligentRouting  bool          `json:"enable_intelligent_routing"`
	EnableContextAwareRouting bool          `json:"enable_context_aware_routing"`
	EnableLoadBalancing       bool          `json:"enable_load_balancing"`
	MaxRoutingTime            time.Duration `json:"max_routing_time"`

	// Caching
	EnableRoutingCache bool          `json:"enable_routing_cache"`
	CacheSize          int           `json:"cache_size"`
	CacheTTL           time.Duration `json:"cache_ttl"`

	// Performance
	MaxConcurrentRequests int  `json:"max_concurrent_requests"`
	EnableAgentPooling    bool `json:"enable_agent_pooling"`
	PoolSize              int  `json:"pool_size"`

	// Fallback behavior
	DefaultAgent   AgentType `json:"default_agent"`
	EnableFallback bool      `json:"enable_fallback"`
	MaxRetries     int       `json:"max_retries"`

	// Quality assurance
	EnableConfidenceScoring bool    `json:"enable_confidence_scoring"`
	MinConfidenceThreshold  float64 `json:"min_confidence_threshold"`
	EnableAgentValidation   bool    `json:"enable_agent_validation"`
}

// AgentType represents different types of AI agents
type AgentType string

const (
	AgentTypeCoding             AgentType = "coding"
	AgentTypeSearch             AgentType = "search"
	AgentTypeContextAwareSearch AgentType = "context_aware_search"
	AgentTypeDocumentation      AgentType = "documentation"
	AgentTypeDebugging          AgentType = "debugging"
	AgentTypeReview             AgentType = "review"
	AgentTypeTesting            AgentType = "testing"
	AgentTypeTestIntelligence   AgentType = "test_intelligence"
	AgentTypeArchitectureAware  AgentType = "architecture_aware"
	AgentTypeDependencyIntel    AgentType = "dependency_intelligence"
	AgentTypeCodeIntelligence   AgentType = "code_intelligence"
)

// HealthStatusType represents the health status of an agent
type HealthStatusType string

const (
	HealthStatusHealthy   HealthStatusType = "healthy"
	HealthStatusDegraded  HealthStatusType = "degraded"
	HealthStatusUnhealthy HealthStatusType = "unhealthy"
)

// HealthStatus encapsulates an agent's health details
type HealthStatus struct {
	Status             HealthStatusType         `json:"status"`
	Message            string                   `json:"message"`
	LastCheckTime      time.Time                `json:"last_check_time"`
	Latency            time.Duration            `json:"latency"`
	ErrorCount         int64                    `json:"error_count"`
	DependenciesStatus map[string]*HealthStatus `json:"dependencies_status,omitempty"`
	Details            map[string]interface{}   `json:"details,omitempty"`
}

// HealthCheckConfig holds configurable thresholds for health checks
type HealthCheckConfig struct {
	MaxLatencyMs            int64         `yaml:"max_latency_ms" json:"max_latency_ms"`
	MaxErrorRate            float64       `yaml:"max_error_rate" json:"max_error_rate"`
	DependencyCheckInterval time.Duration `yaml:"dependency_check_interval" json:"dependency_check_interval"`
	EnableDependencyCheck   bool          `yaml:"enable_dependency_check" json:"enable_dependency_check"`
	HealthCheckTimeout      time.Duration `yaml:"health_check_timeout" json:"health_check_timeout"`
}

// Agent represents the interface that all AI agents must implement
type Agent interface {
	// Core functionality
	ProcessRequest(ctx context.Context, request *AgentRequest) (*AgentResponse, error)

	// Agent metadata
	GetCapabilities() *AgentCapabilities
	GetType() AgentType
	GetVersion() string
	GetStatus() AgentStatus

	// Lifecycle management
	Initialize(config interface{}) error
	Start() error
	Stop() error
	HealthCheck() *HealthStatus

	//Dynamic Configuration loading
	SetConfig(config interface{}) error

	// Performance monitoring
	GetMetrics() *AgentMetrics
	ResetMetrics()
}

// AgentRequest represents a request to an AI agent
type AgentRequest struct {
	// Request identification
	ID        string `json:"id"`
	SessionID string `json:"session_id"`
	UserID    string `json:"user_id"`

	// Request content
	Query      string                 `json:"query"`
	Intent     *DetectedIntent        `json:"intent"`
	Context    *RequestContext        `json:"context"`
	Parameters map[string]interface{} `json:"parameters"`

	// Request metadata
	Priority       RequestPriority `json:"priority"`
	Timeout        time.Duration   `json:"timeout"`
	RequiredAgents []AgentType     `json:"required_agents,omitempty"`
	ExcludedAgents []AgentType     `json:"excluded_agents,omitempty"`

	// Quality requirements
	MinConfidence      float64 `json:"min_confidence"`
	RequireExplanation bool    `json:"require_explanation"`
	MaxTokens          int     `json:"max_tokens,omitempty"`

	// Timestamp
	CreatedAt time.Time `json:"created_at"`
}

// AgentResponse represents a response from an AI agent
type AgentResponse struct {
	// Response identification
	RequestID    string    `json:"request_id"`
	AgentType    AgentType `json:"agent_type"`
	AgentVersion string    `json:"agent_version"`

	// Response content
	Result         interface{}   `json:"result"`
	Explanation    string        `json:"explanation,omitempty"`
	Suggestions    []string      `json:"suggestions,omitempty"`
	RelatedResults []interface{} `json:"related_results,omitempty"`

	// Quality metrics
	Confidence     float64       `json:"confidence"`
	ProcessingTime time.Duration `json:"processing_time"`
	TokensUsed     int           `json:"tokens_used,omitempty"`

	// Additional data
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	Warnings []string               `json:"warnings,omitempty"`
	Errors   []string               `json:"errors,omitempty"`

	// Timestamp
	CreatedAt time.Time `json:"created_at"`
}

// AgentCapabilities describes what an agent can do
type AgentCapabilities struct {
	AgentType          AgentType              `json:"agent_type"`
	SupportedIntents   []IntentType           `json:"supported_intents"`
	SupportedLanguages []string               `json:"supported_languages"`
	SupportedFileTypes []string               `json:"supported_file_types"`
	MaxContextSize     int                    `json:"max_context_size"`
	SupportsStreaming  bool                   `json:"supports_streaming"`
	SupportsAsync      bool                   `json:"supports_async"`
	RequiresContext    bool                   `json:"requires_context"`
	Capabilities       map[string]interface{} `json:"capabilities"`
}

// RoutingRule defines how requests should be routed
type RoutingRule struct {
	ID                 string              `json:"id"`
	Priority           int                 `json:"priority"`
	Conditions         []*RoutingCondition `json:"conditions"`
	TargetAgent        AgentType           `json:"target_agent"`
	FallbackAgents     []AgentType         `json:"fallback_agents,omitempty"`
	RequiredConfidence float64             `json:"required_confidence"`
	Enabled            bool                `json:"enabled"`
}

// RoutingCondition represents a condition for routing
type RoutingCondition struct {
	Type          ConditionType     `json:"type"`
	Field         string            `json:"field"`
	Operator      ConditionOperator `json:"operator"`
	Value         interface{}       `json:"value"`
	CaseSensitive bool              `json:"case_sensitive"`
}

type ConditionType string

const (
	ConditionTypeIntent      ConditionType = "intent"
	ConditionTypeKeyword     ConditionType = "keyword"
	ConditionTypeContext     ConditionType = "context"
	ConditionTypeFileType    ConditionType = "file_type"
	ConditionTypeLanguage    ConditionType = "language"
	ConditionTypeUserRole    ConditionType = "user_role"
	ConditionTypeProjectType ConditionType = "project_type"
)

type ConditionOperator string

const (
	OperatorEquals      ConditionOperator = "equals"
	OperatorContains    ConditionOperator = "contains"
	OperatorStartsWith  ConditionOperator = "starts_with"
	OperatorEndsWith    ConditionOperator = "ends_with"
	OperatorRegex       ConditionOperator = "regex"
	OperatorIn          ConditionOperator = "in"
	OperatorNotIn       ConditionOperator = "not_in"
	OperatorGreaterThan ConditionOperator = "greater_than"
	OperatorLessThan    ConditionOperator = "less_than"
)

// DetectedIntent represents a detected user intent
type DetectedIntent struct {
	Type       IntentType             `json:"type"`
	Confidence float64                `json:"confidence"`
	Entities   map[string]interface{} `json:"entities"`
	SubIntents []*DetectedIntent      `json:"sub_intents,omitempty"`
	Context    map[string]interface{} `json:"context,omitempty"`
}

type IntentType string

const (
	// Code-related intents
	IntentCodeGeneration   IntentType = "code_generation"
	IntentCodeModification IntentType = "code_modification"
	IntentCodeRefactoring  IntentType = "code_refactoring"
	IntentCodeReview       IntentType = "code_review"
	IntentCodeSearch       IntentType = "code_search"
	IntentCodeExplanation  IntentType = "code_explanation"

	// Testing intents
	IntentTestGeneration IntentType = "test_generation"
	IntentTestReview     IntentType = "test_review"
	IntentTestAnalysis   IntentType = "test_analysis"

	// Documentation intents
	IntentDocGeneration IntentType = "doc_generation"
	IntentDocUpdate     IntentType = "doc_update"
	IntentDocReview     IntentType = "doc_review"

	// Debugging intents
	IntentBugIdentification   IntentType = "bug_identification"
	IntentBugFix              IntentType = "bug_fix"
	IntentPerformanceAnalysis IntentType = "performance_analysis"

	// Architecture intents
	IntentArchitectureAnalysis IntentType = "architecture_analysis"
	IntentArchitectureReview   IntentType = "architecture_review"
	IntentDependencyAnalysis   IntentType = "dependency_analysis"

	// General intents
	IntentQuestion IntentType = "question"
	IntentHelp     IntentType = "help"
	IntentUnknown  IntentType = "unknown"
)

// RequestContext provides context about the request environment
type RequestContext struct {
	// Project context
	ProjectPath     string `json:"project_path,omitempty"`
	ProjectType     string `json:"project_type,omitempty"`
	ProjectLanguage string `json:"project_language,omitempty"`

	// File context
	CurrentFile    string          `json:"current_file,omitempty"`
	SelectedText   string          `json:"selected_text,omitempty"`
	CursorPosition *CursorPosition `json:"cursor_position,omitempty"`
	OpenFiles      []string        `json:"open_files,omitempty"`

	// Code context
	CurrentFunction  string   `json:"current_function,omitempty"`
	CurrentClass     string   `json:"current_class,omitempty"`
	ImportStatements []string `json:"import_statements,omitempty"`

	// User context
	UserPreferences  map[string]interface{} `json:"user_preferences,omitempty"`
	WorkingDirectory string                 `json:"working_directory,omitempty"`
	RecentActions    []string               `json:"recent_actions,omitempty"`

	// Session context
	ConversationHistory []string `json:"conversation_history,omitempty"`
	PreviousRequests    []string `json:"previous_requests,omitempty"`
}

type CursorPosition struct {
	Line   int `json:"line"`
	Column int `json:"column"`
}

type RequestPriority string

const (
	PriorityLow      RequestPriority = "low"
	PriorityNormal   RequestPriority = "normal"
	PriorityHigh     RequestPriority = "high"
	PriorityCritical RequestPriority = "critical"
)

type AgentStatus string

const (
	StatusIdle        AgentStatus = "idle"
	StatusBusy        AgentStatus = "busy"
	StatusError       AgentStatus = "error"
	StatusMaintenance AgentStatus = "maintenance"
	StatusStopped     AgentStatus = "stopped"
)

// RoutingStatistics tracks routing performance
type RoutingStatistics struct {
	TotalRequests      int64                `json:"total_requests"`
	SuccessfulRoutes   int64                `json:"successful_routes"`
	FailedRoutes       int64                `json:"failed_routes"`
	RoutesByAgent      map[AgentType]int64  `json:"routes_by_agent"`
	RoutesByIntent     map[IntentType]int64 `json:"routes_by_intent"`
	AverageRoutingTime time.Duration        `json:"average_routing_time"`
	CacheHitRate       float64              `json:"cache_hit_rate"`
	LastUpdated        time.Time            `json:"last_updated"`
	mu                 sync.RWMutex
}

// AgentMetrics represents performance metrics for an agent
type AgentMetrics struct {
	RequestsProcessed   int64         `json:"requests_processed"`
	AverageResponseTime time.Duration `json:"average_response_time"`
	SuccessRate         float64       `json:"success_rate"`
	AverageConfidence   float64       `json:"average_confidence"`
	TotalTokensUsed     int64         `json:"total_tokens_used"`
	ErrorCount          int64         `json:"error_count"`
	LastRequestAt       time.Time     `json:"last_request_at"`
}

// NewAgentRouter creates a new agent router
func NewAgentRouter(config *RouterConfig, agentsConfig *AllAgentsConfig, logger logger.Logger) *AgentRouter {
	if config == nil {
		config = &RouterConfig{
			EnableIntelligentRouting:  true,
			EnableContextAwareRouting: true,
			EnableLoadBalancing:       true,
			MaxRoutingTime:            time.Second * 5,
			EnableRoutingCache:        true,
			CacheSize:                 1000,
			CacheTTL:                  time.Minute * 10,
			MaxConcurrentRequests:     100,
			EnableAgentPooling:        true,
			PoolSize:                  10,
			DefaultAgent:              AgentTypeCodeIntelligence,
			EnableFallback:            true,
			MaxRetries:                3,
			EnableConfidenceScoring:   true,
			MinConfidenceThreshold:    0.7,
			EnableAgentValidation:     true,
		}
	}

	router := &AgentRouter{
		agents:            make(map[AgentType]Agent),
		agentCapabilities: make(map[AgentType]*AgentCapabilities),
		routingRules:      make([]*RoutingRule, 0),
		fallbackAgent:     config.DefaultAgent,
		config:            config,
		logger:            logger,
		routingStats: &RoutingStatistics{
			RoutesByAgent:  make(map[AgentType]int64),
			RoutesByIntent: make(map[IntentType]int64),
		},
	}

	// Initialize components
	router.initializeComponents()
	router.setupDefaultRoutingRules()

	// Initialize agents with configurations
	if agentsConfig != nil {
		router.initializeAgents(agentsConfig)
	}
	return router
}

// initializeAgents creates and registers all agents with their configurations
func (ar *AgentRouter) initializeAgents(agentsConfig *app.AllAgentsConfig) {
	// Initialize CodingAgent
	if agentsConfig.Coding != nil {
		if agent := NewCodingAgent(agentsConfig.Coding, ar.logger); agent != nil {
			ar.RegisterAgent(agent)
		}
	}

	// Initialize DocumentationAgent
	if agentsConfig.Documentation != nil {
		if agent := NewDocumentationAgent(agentsConfig.Documentation, ar.logger); agent != nil {
			ar.RegisterAgent(agent)
		}
	}

	// Initialize DebuggingAgent
	if agentsConfig.Debugging != nil {
		if agent := NewDebuggingAgent(agentsConfig.Debugging, ar.logger); agent != nil {
			ar.RegisterAgent(agent)
		}
	}

	// Initialize TestingAgent
	if agentsConfig.Testing != nil {
		if agent := NewTestingAgent(agentsConfig.Testing, ar.logger); agent != nil {
			ar.RegisterAgent(agent)
		}
	}

	// Initialize CodeReviewAgent
	if agentsConfig.CodeReview != nil {
		if agent := NewCodeReviewAgent(agentsConfig.CodeReview, ar.logger); agent != nil {
			ar.RegisterAgent(agent)
		}
	}

	// Initialize SearchAgent
	if agentsConfig.Search != nil {
		// Create concrete component instances
		textSearchEngine := &ConcreteTextSearchEngine{}
		queryAnalyzer := &ConcreteQueryAnalyzer{}
		resultRanker := &ConcreteResultRanker{}
		searchCache := &ConcreteSearchCache{}

		if agent := NewSearchAgent(nil, nil, agentsConfig.Search, ar.logger,
			textSearchEngine, queryAnalyzer, resultRanker, searchCache); agent != nil {
			ar.RegisterAgent(agent)
		}
	}

	// Initialize ContextAwareSearchAgent
	if agentsConfig.ContextAwareSearch != nil {
		if agent := NewContextAwareSearchAgent(agentsConfig.ContextAwareSearch, ar.logger); agent != nil {
			ar.RegisterAgent(agent)
		}
	}

	// Initialize ArchitectureAwareAgent
	if agentsConfig.ArchitectureAware != nil {
		// Create concrete component instances
		contextAnalyzer := &ConcreteContextAnalyzer{}
		relevanceCalculator := &ConcreteRelevanceCalculator{}
		workspaceAnalyzer := &ConcreteWorkspaceAnalyzer{}
		sessionTracker := &ConcreteSessionTracker{}
		intentPredictor := &ConcreteIntentPredictor{}

		if agent := NewContextAwareSearchAgent(nil, nil, nil, nil, agentsConfig.ContextAwareSearch, ar.logger,
			contextAnalyzer, relevanceCalculator, workspaceAnalyzer, sessionTracker, intentPredictor); agent != nil {
			ar.RegisterAgent(agent)
		}
	}

	// Initialize CodeIntelligenceAgent
	if agentsConfig.CodeIntelligence != nil {
		if agent := NewCodeIntelligenceAgent(agentsConfig.CodeIntelligence, ar.logger); agent != nil {
			ar.RegisterAgent(agent)
		}
	}

	// Initialize DependencyIntelligenceAgent
	if agentsConfig.DependencyIntelligence != nil {
		if agent := NewDependencyIntelligenceAgent(agentsConfig.DependencyIntelligence, ar.logger); agent != nil {
			ar.RegisterAgent(agent)
		}
	}

	// Initialize PerformanceOptimizationAgent
	if agentsConfig.PerformanceOptimization != nil {
		if agent := NewPerformanceOptimizationAgent(agentsConfig.PerformanceOptimization, ar.logger); agent != nil {
			ar.RegisterAgent(agent)
		}
	}

	// Initialize ProjectAnalysisAgent
	if agentsConfig.ProjectAnalysis != nil {
		if agent := NewProjectAnalysisAgent(agentsConfig.ProjectAnalysis, ar.logger); agent != nil {
			ar.RegisterAgent(agent)
		}
	}

	// Initialize TestIntelligenceAgent
	if agentsConfig.TestIntelligence != nil {
		if agent := NewTestIntelligenceAgent(agentsConfig.TestIntelligence, ar.logger); agent != nil {
			ar.RegisterAgent(agent)
		}
	}
}

// RegisterAgent registers a new agent with the router
func (ar *AgentRouter) RegisterAgent(agent Agent) error {
	ar.mu.Lock()
	defer ar.mu.Unlock()

	agentType := agent.GetType()
	capabilities := agent.GetCapabilities()

	// Validate agent
	if ar.config.EnableAgentValidation {
		if err := ar.validateAgent(agent); err != nil {
			return fmt.Errorf("agent validation failed: %v", err)
		}
	}

	// Register agent
	ar.agents[agentType] = agent
	ar.agentCapabilities[agentType] = capabilities

	ar.logger.Info("Registered agent",
		"type", agentType,
		"version", agent.GetVersion(),
		"capabilities", len(capabilities.SupportedIntents))

	return nil
}

// ProcessRequest routes and processes a request
func (ar *AgentRouter) ProcessRequest(ctx context.Context, request *AgentRequest) (*AgentResponse, error) {
	start := time.Now()

	// Validate request
	if err := ar.validateRequest(request); err != nil {
		return nil, fmt.Errorf("invalid request: %v", err)
	}

	// Apply timeout
	if request.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, request.Timeout)
		defer cancel()
	} else if ar.config.MaxRoutingTime > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, ar.config.MaxRoutingTime)
		defer cancel()
	}

	// Detect intent if not provided
	if request.Intent == nil {
		intent, err := ar.intentClassifier.ClassifyIntent(ctx, request.Query, request.Context)
		if err != nil {
			ar.logger.Warn("Intent classification failed", "error", err)
			// Continue with unknown intent
			request.Intent = &DetectedIntent{
				Type:       IntentUnknown,
				Confidence: 0.0,
			}
		} else {
			request.Intent = intent
		}
	}

	// Route request to appropriate agent
	targetAgent, confidence, err := ar.routeRequest(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("routing failed: %v", err)
	}

	// Check confidence threshold
	if ar.config.EnableConfidenceScoring && confidence < ar.config.MinConfidenceThreshold {
		ar.logger.Warn("Routing confidence below threshold",
			"confidence", confidence,
			"threshold", ar.config.MinConfidenceThreshold,
			"agent", targetAgent)

		// Try fallback if enabled
		if ar.config.EnableFallback {
			targetAgent = ar.fallbackAgent
		}
	}

	// Get agent instance
	agent, exists := ar.agents[targetAgent]
	if !exists {
		return nil, fmt.Errorf("target agent not available: %s", targetAgent)
	}

	// Process request with selected agent
	response, err := ar.processWithAgent(ctx, agent, request)
	if err != nil {
		// Try fallback agents if configured
		if ar.config.EnableFallback && targetAgent != ar.fallbackAgent {
			ar.logger.Warn("Primary agent failed, trying fallback",
				"primary", targetAgent,
				"fallback", ar.fallbackAgent,
				"error", err)

			if fallbackAgent, exists := ar.agents[ar.fallbackAgent]; exists {
				response, err = ar.processWithAgent(ctx, fallbackAgent, request)
			}
		}

		if err != nil {
			ar.updateRoutingStats(targetAgent, request.Intent.Type, false, time.Since(start))
			return nil, fmt.Errorf("agent processing failed: %v", err)
		}
	}

	// Update statistics
	ar.updateRoutingStats(targetAgent, request.Intent.Type, true, time.Since(start))

	return response, nil
}

// routeRequest determines the best agent for a request
func (ar *AgentRouter) routeRequest(ctx context.Context, request *AgentRequest) (AgentType, float64, error) {
	// Check cache first
	if ar.config.EnableRoutingCache {
		if cached := ar.getRoutingFromCache(request); cached != nil {
			return cached.AgentType, cached.Confidence, nil
		}
	}

	// Apply routing rules
	targetAgent, confidence := ar.applyRoutingRules(request)
	if targetAgent != "" {
		// Cache result
		if ar.config.EnableRoutingCache {
			ar.cacheRouting(request, targetAgent, confidence)
		}
		return targetAgent, confidence, nil
	}

	// Use intelligent routing
	if ar.config.EnableIntelligentRouting {
		return ar.intelligentRouting(ctx, request)
	}

	// Fall back to default agent
	return ar.fallbackAgent, 0.5, nil
}

// applyRoutingRules applies configured routing rules
func (ar *AgentRouter) applyRoutingRules(request *AgentRequest) (AgentType, float64) {
	// Sort rules by priority
	rules := make([]*RoutingRule, len(ar.routingRules))
	copy(rules, ar.routingRules)
	sort.Slice(rules, func(i, j int) bool {
		return rules[i].Priority > rules[j].Priority
	})

	// Apply rules in priority order
	for _, rule := range rules {
		if !rule.Enabled {
			continue
		}

		if ar.evaluateRoutingRule(rule, request) {
			return rule.TargetAgent, rule.RequiredConfidence
		}
	}

	return "", 0.0
}

// evaluateRoutingRule evaluates if a routing rule matches the request
func (ar *AgentRouter) evaluateRoutingRule(rule *RoutingRule, request *AgentRequest) bool {
	for _, condition := range rule.Conditions {
		if !ar.evaluateCondition(condition, request) {
			return false
		}
	}
	return true
}

// evaluateCondition evaluates a single routing condition
func (ar *AgentRouter) evaluateCondition(condition *RoutingCondition, request *AgentRequest) bool {
	var fieldValue interface{}

	// Extract field value based on condition type
	switch condition.Type {
	case ConditionTypeIntent:
		fieldValue = string(request.Intent.Type)
	case ConditionTypeKeyword:
		fieldValue = request.Query
	case ConditionTypeFileType:
		if request.Context != nil && request.Context.CurrentFile != "" {
			fieldValue = ar.extractFileExtension(request.Context.CurrentFile)
		}
	case ConditionTypeLanguage:
		if request.Context != nil {
			fieldValue = request.Context.ProjectLanguage
		}
	default:
		return false
	}

	// Apply operator
	return ar.applyConditionOperator(condition.Operator, fieldValue, condition.Value, condition.CaseSensitive)
}

// intelligentRouting uses AI to determine the best agent
func (ar *AgentRouter) intelligentRouting(ctx context.Context, request *AgentRequest) (AgentType, float64, error) {
	// Analyze request characteristics
	requestFeatures := ar.extractRequestFeatures(request)

	// Score each available agent
	agentScores := make(map[AgentType]float64)

	for agentType, capabilities := range ar.agentCapabilities {
		score := ar.calculateAgentScore(requestFeatures, capabilities, request)
		agentScores[agentType] = score
	}

	// Find best agent
	var bestAgent AgentType
	var bestScore float64

	for agentType, score := range agentScores {
		if score > bestScore {
			bestScore = score
			bestAgent = agentType
		}
	}

	if bestAgent == "" {
		return ar.fallbackAgent, 0.5, nil
	}

	return bestAgent, bestScore, nil
}

// extractRequestFeatures extracts features from the request for routing
func (ar *AgentRouter) extractRequestFeatures(request *AgentRequest) map[string]interface{} {
	features := make(map[string]interface{})

	// Intent features
	if request.Intent != nil {
		features["intent_type"] = request.Intent.Type
		features["intent_confidence"] = request.Intent.Confidence
	}

	// Query features
	features["query_length"] = len(request.Query)
	features["has_code_keywords"] = ar.hasCodeKeywords(request.Query)
	features["has_test_keywords"] = ar.hasTestKeywords(request.Query)
	features["has_doc_keywords"] = ar.hasDocKeywords(request.Query)

	// Context features
	if request.Context != nil {
		if request.Context.CurrentFile != "" {
			features["file_type"] = ar.extractFileExtension(request.Context.CurrentFile)
		}
		if request.Context.ProjectLanguage != "" {
			features["project_language"] = request.Context.ProjectLanguage
		}
		features["has_selected_text"] = request.Context.SelectedText != ""
		features["has_current_function"] = request.Context.CurrentFunction != ""
	}

	return features
}

// calculateAgentScore calculates how well an agent matches a request
func (ar *AgentRouter) calculateAgentScore(features map[string]interface{}, capabilities *AgentCapabilities, request *AgentRequest) float64 {
	score := 0.0

	// Intent matching
	if intentType, ok := features["intent_type"].(IntentType); ok {
		for _, supportedIntent := range capabilities.SupportedIntents {
			if supportedIntent == intentType {
				score += 1.0
				break
			}
		}
	}

	// Language matching
	if projectLang, ok := features["project_language"].(string); ok && projectLang != "" {
		for _, supportedLang := range capabilities.SupportedLanguages {
			if strings.EqualFold(supportedLang, projectLang) {
				score += 0.5
				break
			}
		}
	}

	// File type matching
	if fileType, ok := features["file_type"].(string); ok && fileType != "" {
		for _, supportedType := range capabilities.SupportedFileTypes {
			if strings.EqualFold(supportedType, fileType) {
				score += 0.3
				break
			}
		}
	}

	// Context requirements
	if capabilities.RequiresContext && request.Context != nil {
		score += 0.2
	}

	// Keyword matching bonuses
	if hasCode, ok := features["has_code_keywords"].(bool); ok && hasCode {
		if capabilities.AgentType == AgentTypeCoding || capabilities.AgentType == AgentTypeCodeIntelligence {
			score += 0.3
		}
	}

	if hasTest, ok := features["has_test_keywords"].(bool); ok && hasTest {
		if capabilities.AgentType == AgentTypeTesting || capabilities.AgentType == AgentTypeTestIntelligence {
			score += 0.3
		}
	}

	if hasDoc, ok := features["has_doc_keywords"].(bool); ok && hasDoc {
		if capabilities.AgentType == AgentTypeDocumentation {
			score += 0.3
		}
	}

	return score
}

// processWithAgent processes a request with a specific agent
func (ar *AgentRouter) processWithAgent(ctx context.Context, agent Agent, request *AgentRequest) (*AgentResponse, error) {
	// Check agent status
	if agent.GetStatus() != StatusIdle && agent.GetStatus() != StatusBusy {
		return nil, fmt.Errorf("agent not available: %s", agent.GetStatus())
	}

	// Optimize query if configured
	if ar.queryOptimizer != nil {
		optimizedQuery, err := ar.queryOptimizer.OptimizeQuery(ctx, request.Query, request.Context)
		if err != nil {
			ar.logger.Warn("Query optimization failed", "error", err)
		} else {
			request.Query = optimizedQuery
		}
	}

	// Process request
	response, err := agent.ProcessRequest(ctx, request)
	if err != nil {
		return nil, err
	}

	// Format response if configured
	if ar.responseFormatter != nil {
		formattedResponse, err := ar.responseFormatter.FormatResponse(response, request)
		if err != nil {
			ar.logger.Warn("Response formatting failed", "error", err)
		} else {
			response = formattedResponse
		}
	}

	return response, nil
}

// Helper methods

func (ar *AgentRouter) validateAgent(agent Agent) error {
	// Check required methods
	if agent.GetType() == "" {
		return fmt.Errorf("agent type cannot be empty")
	}

	if agent.GetCapabilities() == nil {
		return fmt.Errorf("agent capabilities cannot be nil")
	}

	healthStatus := agent.HealthCheck()
	if healthStatus.Status == HealthStatusUnhealthy {
		return fmt.Errorf("agent is unhealthy: %s", healthStatus.Message)
	}

	return nil
}

func (ar *AgentRouter) validateRequest(request *AgentRequest) error {
	if request == nil {
		return fmt.Errorf("request cannot be nil")
	}

	if request.Query == "" {
		return fmt.Errorf("query cannot be empty")
	}

	if request.ID == "" {
		request.ID = ar.generateRequestID()
	}

	if request.CreatedAt.IsZero() {
		request.CreatedAt = time.Now()
	}

	if request.Priority == "" {
		request.Priority = PriorityNormal
	}

	return nil
}

func (ar *AgentRouter) initializeComponents() {
	// Initialize caching
	if ar.config.EnableRoutingCache {
		ar.routingCache = NewRoutingCache(ar.config.CacheSize, ar.config.CacheTTL)
	}

	// Initialize agent pooling
	if ar.config.EnableAgentPooling {
		ar.agentPool = NewAgentPool(ar.config.PoolSize)
	}

	// Initialize load balancer
	if ar.config.EnableLoadBalancing {
		ar.loadBalancer = NewAgentLoadBalancer()
	}

	// Initialize performance monitor
	ar.performanceMonitor = NewAgentPerformanceMonitor()
}

func (ar *AgentRouter) setupDefaultRoutingRules() {
	// Default routing rules based on intent
	rules := []*RoutingRule{
		{
			ID:       "code_generation_rule",
			Priority: 100,
			Conditions: []*RoutingCondition{
				{
					Type:     ConditionTypeIntent,
					Operator: OperatorEquals,
					Value:    IntentCodeGeneration,
				},
			},
			TargetAgent:        AgentTypeCoding,
			RequiredConfidence: 0.8,
			Enabled:            true,
		},
		{
			ID:       "code_search_rule",
			Priority: 90,
			Conditions: []*RoutingCondition{
				{
					Type:     ConditionTypeIntent,
					Operator: OperatorEquals,
					Value:    IntentCodeSearch,
				},
			},
			TargetAgent:        AgentTypeContextAwareSearch,
			RequiredConfidence: 0.7,
			Enabled:            true,
		},
		{
			ID:       "test_generation_rule",
			Priority: 95,
			Conditions: []*RoutingCondition{
				{
					Type:     ConditionTypeIntent,
					Operator: OperatorEquals,
					Value:    IntentTestGeneration,
				},
			},
			TargetAgent:        AgentTypeTesting,
			RequiredConfidence: 0.8,
			Enabled:            true,
		},
		{
			ID:       "bug_fix_rule",
			Priority: 95,
			Conditions: []*RoutingCondition{
				{
					Type:     ConditionTypeIntent,
					Operator: OperatorIn,
					Value:    []IntentType{IntentBugIdentification, IntentBugFix},
				},
			},
			TargetAgent:        AgentTypeDebugging,
			RequiredConfidence: 0.75,
			Enabled:            true,
		},
		{
			ID:       "documentation_rule",
			Priority: 85,
			Conditions: []*RoutingCondition{
				{
					Type:     ConditionTypeIntent,
					Operator: OperatorIn,
					Value:    []IntentType{IntentDocGeneration, IntentDocUpdate},
				},
			},
			TargetAgent:        AgentTypeDocumentation,
			RequiredConfidence: 0.7,
			Enabled:            true,
		},
	}

	ar.routingRules = append(ar.routingRules, rules...)
}

// Utility methods

func (ar *AgentRouter) extractFileExtension(filename string) string {
	parts := strings.Split(filename, ".")
	if len(parts) > 1 {
		return parts[len(parts)-1]
	}
	return ""
}

func (ar *AgentRouter) hasCodeKeywords(query string) bool {
	codeKeywords := []string{"function", "class", "method", "variable", "implement", "create", "generate", "code"}
	queryLower := strings.ToLower(query)

	for _, keyword := range codeKeywords {
		if strings.Contains(queryLower, keyword) {
			return true
		}
	}
	return false
}

func (ar *AgentRouter) hasTestKeywords(query string) bool {
	testKeywords := []string{"test", "unit test", "integration test", "mock", "assert", "expect"}
	queryLower := strings.ToLower(query)

	for _, keyword := range testKeywords {
		if strings.Contains(queryLower, keyword) {
			return true
		}
	}
	return false
}

func (ar *AgentRouter) hasDocKeywords(query string) bool {
	docKeywords := []string{"document", "documentation", "comment", "docstring", "readme", "explain"}
	queryLower := strings.ToLower(query)

	for _, keyword := range docKeywords {
		if strings.Contains(queryLower, keyword) {
			return true
		}
	}
	return false
}

func (ar *AgentRouter) applyConditionOperator(operator ConditionOperator, fieldValue, conditionValue interface{}, caseSensitive bool) bool {
	fieldStr := fmt.Sprintf("%v", fieldValue)
	conditionStr := fmt.Sprintf("%v", conditionValue)

	if !caseSensitive {
		fieldStr = strings.ToLower(fieldStr)
		conditionStr = strings.ToLower(conditionStr)
	}

	switch operator {
	case OperatorEquals:
		return fieldStr == conditionStr
	case OperatorContains:
		return strings.Contains(fieldStr, conditionStr)
	case OperatorStartsWith:
		return strings.HasPrefix(fieldStr, conditionStr)
	case OperatorEndsWith:
		return strings.HasSuffix(fieldStr, conditionStr)
	case OperatorIn:
		if values, ok := conditionValue.([]interface{}); ok {
			for _, value := range values {
				if fmt.Sprintf("%v", value) == fieldStr {
					return true
				}
			}
		}
		return false
	case OperatorNotIn:
		if values, ok := conditionValue.([]interface{}); ok {
			for _, value := range values {
				if fmt.Sprintf("%v", value) == fieldStr {
					return false
				}
			}
		}
		return true
	default:
		return false
	}
}

func (ar *AgentRouter) generateRequestID() string {
	return fmt.Sprintf("req_%d", time.Now().UnixNano())
}

// Caching methods

func (ar *AgentRouter) getRoutingFromCache(request *AgentRequest) *RoutingCacheEntry {
	if ar.routingCache == nil {
		return nil
	}

	key := ar.generateRoutingCacheKey(request)
	return ar.routingCache.Get(key)
}

func (ar *AgentRouter) cacheRouting(request *AgentRequest, agentType AgentType, confidence float64) {
	if ar.routingCache == nil {
		return
	}

	key := ar.generateRoutingCacheKey(request)
	entry := &RoutingCacheEntry{
		AgentType:  agentType,
		Confidence: confidence,
		CreatedAt:  time.Now(),
	}

	ar.routingCache.Set(key, entry)
}

func (ar *AgentRouter) generateRoutingCacheKey(request *AgentRequest) string {
	// Generate a cache key based on request characteristics
	key := fmt.Sprintf("%s_%s", request.Intent.Type, request.Query)
	if request.Context != nil {
		if request.Context.CurrentFile != "" {
			key += "_" + ar.extractFileExtension(request.Context.CurrentFile)
		}
		if request.Context.ProjectLanguage != "" {
			key += "_" + request.Context.ProjectLanguage
		}
	}
	return key
}

// Statistics methods

func (ar *AgentRouter) updateRoutingStats(agentType AgentType, intentType IntentType, success bool, duration time.Duration) {
	ar.routingStats.mu.Lock()
	defer ar.routingStats.mu.Unlock()

	ar.routingStats.TotalRequests++
	if success {
		ar.routingStats.SuccessfulRoutes++
	} else {
		ar.routingStats.FailedRoutes++
	}

	ar.routingStats.RoutesByAgent[agentType]++
	ar.routingStats.RoutesByIntent[intentType]++

	// Update average routing time
	if ar.routingStats.AverageRoutingTime == 0 {
		ar.routingStats.AverageRoutingTime = duration
	} else {
		ar.routingStats.AverageRoutingTime = (ar.routingStats.AverageRoutingTime + duration) / 2
	}

	ar.routingStats.LastUpdated = time.Now()
}

// Public API methods

func (ar *AgentRouter) GetRegisteredAgents() []AgentType {
	ar.mu.RLock()
	defer ar.mu.RUnlock()

	agents := make([]AgentType, 0, len(ar.agents))
	for agentType := range ar.agents {
		agents = append(agents, agentType)
	}

	return agents
}

func (ar *AgentRouter) GetAgentCapabilities(agentType AgentType) (*AgentCapabilities, error) {
	ar.mu.RLock()
	defer ar.mu.RUnlock()

	capabilities, exists := ar.agentCapabilities[agentType]
	if !exists {
		return nil, fmt.Errorf("agent not found: %s", agentType)
	}

	return capabilities, nil
}

func (ar *AgentRouter) GetRoutingStatistics() *RoutingStatistics {
	ar.routingStats.mu.RLock()
	defer ar.routingStats.mu.RUnlock()

	stats := *ar.routingStats
	return &stats
}

func (ar *AgentRouter) AddRoutingRule(rule *RoutingRule) {
	ar.mu.Lock()
	defer ar.mu.Unlock()

	ar.routingRules = append(ar.routingRules, rule)
}

func (ar *AgentRouter) RemoveRoutingRule(ruleID string) {
	ar.mu.Lock()
	defer ar.mu.Unlock()

	for i, rule := range ar.routingRules {
		if rule.ID == ruleID {
			ar.routingRules = append(ar.routingRules[:i], ar.routingRules[i+1:]...)
			break
		}
	}
}

func (ar *AgentRouter) GetRoutingRules() []*RoutingRule {
	ar.mu.RLock()
	defer ar.mu.RUnlock()

	rules := make([]*RoutingRule, len(ar.routingRules))
	copy(rules, ar.routingRules)
	return rules
}

func (ar *AgentRouter) Start() error {
	ar.mu.Lock()
	defer ar.mu.Unlock()

	// Start all registered agents
	for agentType, agent := range ar.agents {
		if err := agent.Start(); err != nil {
			return fmt.Errorf("failed to start agent %s: %v", agentType, err)
		}
	}

	ar.logger.Info("Agent router started", "agents", len(ar.agents))
	return nil
}

func (ar *AgentRouter) Stop() error {
	ar.mu.Lock()
	defer ar.mu.Unlock()

	// Stop all registered agents
	for agentType, agent := range ar.agents {
		if err := agent.Stop(); err != nil {
			ar.logger.Warn("Failed to stop agent", "type", agentType, "error", err)
		}
	}

	ar.logger.Info("Agent router stopped")
	return nil
}

// UpdateAgentConfig updates the configuration of a specific agent
func (ar *AgentRouter) UpdateAgentConfig(agentType AgentType, newConfig interface{}) error {
	ar.mu.RLock()
	agent, exists := ar.agents[agentType]
	ar.mu.RUnlock()

	if !exists {
		return fmt.Errorf("agent not found: %s", agentType)
	}

	if err := agent.SetConfig(newConfig); err != nil {
		ar.logger.Error("Failed to update agent config",
			"agent", agentType,
			"error", err)
		return fmt.Errorf("failed to update config for agent %s: %v", agentType, err)
	}

	ar.logger.Info("Agent configuration updated successfully",
		"agent", agentType)
	return nil
}

// GetAgentHealth returns the health status of a specific agent
func (ar *AgentRouter) GetAgentHealth(agentType AgentType) (*HealthStatus, error) {
	ar.mu.RLock()
	agent, exists := ar.agents[agentType]
	ar.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("agent not found: %s", agentType)
	}

	return agent.HealthCheck(), nil
}

// GetAllAgentsHealth returns the health status of all registered agents
func (ar *AgentRouter) GetAllAgentsHealth() map[AgentType]*HealthStatus {
	ar.mu.RLock()
	defer ar.mu.RUnlock()

	healthStatuses := make(map[AgentType]*HealthStatus)
	for agentType, agent := range ar.agents {
		healthStatuses[agentType] = agent.HealthCheck()
	}

	return healthStatuses
}

// GetSystemHealth returns overall system health based on all agents
func (ar *AgentRouter) GetSystemHealth() *HealthStatus {
	allHealth := ar.GetAllAgentsHealth()

	systemHealth := &HealthStatus{
		LastCheckTime:      time.Now(),
		DependenciesStatus: allHealth,
		Details:            make(map[string]interface{}),
	}

	// Determine overall system health
	healthyCount := 0
	degradedCount := 0
	unhealthyCount := 0

	for _, health := range allHealth {
		switch health.Status {
		case HealthStatusHealthy:
			healthyCount++
		case HealthStatusDegraded:
			degradedCount++
		case HealthStatusUnhealthy:
			unhealthyCount++
		}
	}

	totalAgents := len(allHealth)
	if unhealthyCount > 0 {
		systemHealth.Status = HealthStatusUnhealthy
		systemHealth.Message = fmt.Sprintf("%d/%d agents unhealthy", unhealthyCount, totalAgents)
	} else if degradedCount > 0 {
		systemHealth.Status = HealthStatusDegraded
		systemHealth.Message = fmt.Sprintf("%d/%d agents degraded", degradedCount, totalAgents)
	} else {
		systemHealth.Status = HealthStatusHealthy
		systemHealth.Message = fmt.Sprintf("All %d agents healthy", totalAgents)
	}

	systemHealth.Details["total_agents"] = totalAgents
	systemHealth.Details["healthy_agents"] = healthyCount
	systemHealth.Details["degraded_agents"] = degradedCount
	systemHealth.Details["unhealthy_agents"] = unhealthyCount

	return systemHealth
}

// Supporting types for caching, pooling, and load balancing

type RoutingCache struct {
	cache map[string]*RoutingCacheEntry
	ttl   time.Duration
	mu    sync.RWMutex
}

type RoutingCacheEntry struct {
	AgentType  AgentType `json:"agent_type"`
	Confidence float64   `json:"confidence"`
	CreatedAt  time.Time `json:"created_at"`
}

func NewRoutingCache(size int, ttl time.Duration) *RoutingCache {
	return &RoutingCache{
		cache: make(map[string]*RoutingCacheEntry),
		ttl:   ttl,
	}
}

func (rc *RoutingCache) Get(key string) *RoutingCacheEntry {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	entry, exists := rc.cache[key]
	if !exists {
		return nil
	}

	// Check if expired
	if time.Since(entry.CreatedAt) > rc.ttl {
		delete(rc.cache, key)
		return nil
	}

	return entry
}

func (rc *RoutingCache) Set(key string, entry *RoutingCacheEntry) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	rc.cache[key] = entry
}

// AgentPool manages a pool of agent instances
type AgentPool struct {
	agents map[AgentType][]Agent
	size   int
	mu     sync.RWMutex
}

func NewAgentPool(size int) *AgentPool {
	return &AgentPool{
		agents: make(map[AgentType][]Agent),
		size:   size,
	}
}

// AgentLoadBalancer distributes requests across agent instances
type AgentLoadBalancer struct {
	roundRobinCounters map[AgentType]int
	mu                 sync.RWMutex
}

func NewAgentLoadBalancer() *AgentLoadBalancer {
	return &AgentLoadBalancer{
		roundRobinCounters: make(map[AgentType]int),
	}
}

// AgentPerformanceMonitor monitors agent performance
type AgentPerformanceMonitor struct {
	metrics map[AgentType]*AgentMetrics
	mu      sync.RWMutex
}

func NewAgentPerformanceMonitor() *AgentPerformanceMonitor {
	return &AgentPerformanceMonitor{
		metrics: make(map[AgentType]*AgentMetrics),
	}
}
