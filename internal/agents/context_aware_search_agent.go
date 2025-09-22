package agents

import (
	"context"
	"fmt"
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
type ContextAnalyzer interface {
	AnalyzeContext(ctx context.Context) (*ContextInfo, error)
}

type RelevanceCalculator interface {
	CalculateRelevance(result *SearchResult, context *ContextInfo) float64
}

type WorkspaceAnalyzer interface {
	AnalyzeWorkspace() (*WorkspaceInfo, error)
}

type SessionTracker interface {
	TrackSession(sessionID string, activity *SessionActivity)
	GetSessionHistory(sessionID string) []*SessionActivity
}

type IntentPredictor interface {
	PredictIntent(query string, context *ContextInfo) (*IntentPrediction, error)
}

// ContextAwareSearchAgent enhances search by considering current context of user's work
type ContextAwareSearchAgent struct {
	// Core components
	baseSearchAgent *SearchAgent
	contextManager  *app.ContextManager
	indexer         *indexer.UltraFastIndexer
	vectorDB        *vectordb.VectorDB

	//Injected dependencies
	contextAnalyzer     ContextAnalyzer
	relevanceCalculator RelevanceCalculator
	workspaceAnalyzer   WorkspaceAnalyzer
	sessionTracker      SessionTracker
	intentPredictor     IntentPredictor

	// Agent configuration
	config *ContextAwareSearchConfig
	logger logger.Logger

	// Capabilities
	capabilities *AgentCapabilities

	// Context analysis
	contextAnalyzer       *ContextAnalyzer
	relevanceCalculator   *RelevanceCalculator
	semanticContextEngine *SemanticContextEngine

	// Context-aware features
	workspaceAnalyzer *WorkspaceAnalyzer
	sessionTracker    *SessionTracker
	intentPredictor   *IntentPredictor

	// Advanced search features
	contextualRanker  *ContextualRanker
	adaptiveFilter    *AdaptiveFilter
	relatedCodeFinder *RelatedCodeFinder

	// Learning and personalization
	userBehaviorTracker *UserBehaviorTracker
	contextLearner      *ContextLearner

	// Statistics and monitoring
	metrics *ContextAwareSearchMetrics

	// State management
	mu     sync.RWMutex
	status AgentStatus
}

// ContextAwareSearchConfig contains context-aware search configuration
type ContextAwareSearchConfig struct {
	// Context analysis
	EnableContextAnalysis   bool `json:"enable_context_analysis"`
	EnableWorkspaceAnalysis bool `json:"enable_workspace_analysis"`
	EnableSessionTracking   bool `json:"enable_session_tracking"`
	EnableIntentPrediction  bool `json:"enable_intent_prediction"`

	// Context weights
	ContextWeights *ContextWeights `json:"context_weights"`

	// Relevance calculation
	EnableRelevanceBoost    bool    `json:"enable_relevance_boost"`
	RelevanceThreshold      float32 `json:"relevance_threshold"`
	ContextSimilarityWeight float32 `json:"context_similarity_weight"`

	// Adaptive filtering
	EnableAdaptiveFiltering bool `json:"enable_adaptive_filtering"`
	MaxContextRadius        int  `json:"max_context_radius"`     // Lines around current position
	EnableScopeFiltering    bool `json:"enable_scope_filtering"` // Filter by current scope

	// Related code discovery
	EnableRelatedCodeSearch bool    `json:"enable_related_code_search"`
	MaxRelatedResults       int     `json:"max_related_results"`
	RelatedCodeThreshold    float32 `json:"related_code_threshold"`

	// Learning and personalization
	EnableUserLearning     bool `json:"enable_user_learning"`
	EnableBehaviorTracking bool `json:"enable_behavior_tracking"`
	LearningWindowDays     int  `json:"learning_window_days"`

	// Performance optimization
	EnableResultCaching bool          `json:"enable_result_caching"`
	CacheSize           int           `json:"cache_size"`
	CacheTTL            time.Duration `json:"cache_ttl"`
	MaxProcessingTime   time.Duration `json:"max_processing_time"`

	// Context preservation
	SessionTimeout           time.Duration `json:"session_timeout"`
	MaxSessionHistory        int           `json:"max_session_history"`
	EnableContextPersistence bool          `json:"enable_context_persistence"`
}

// ContextWeights defines weights for different context factors
type ContextWeights struct {
	CurrentFile      float32 `json:"current_file"`
	OpenFiles        float32 `json:"open_files"`
	RecentFiles      float32 `json:"recent_files"`
	ProjectStructure float32 `json:"project_structure"`
	WorkingDirectory float32 `json:"working_directory"`
	CodeScope        float32 `json:"code_scope"`
	FunctionContext  float32 `json:"function_context"`
	ClassContext     float32 `json:"class_context"`
	ImportContext    float32 `json:"import_context"`
	RecentActions    float32 `json:"recent_actions"`
	UserPreferences  float32 `json:"user_preferences"`
	TimeOfDay        float32 `json:"time_of_day"`
}

// ContextAwareSearchRequest extends SearchRequest with context information
type ContextAwareSearchRequest struct {
	*SearchRequest
	WorkspaceContext *WorkspaceContext `json:"workspace_context,omitempty"`
	SessionContext   *SessionContext   `json:"session_context,omitempty"`
	CodeContext      *CodeContext      `json:"code_context,omitempty"`
	UserContext      *UserContext      `json:"user_context,omitempty"`
	TemporalContext  *TemporalContext  `json:"temporal_context,omitempty"`
}

// Context information structures

type WorkspaceContext struct {
	ProjectRoot      string            `json:"project_root"`
	ProjectType      string            `json:"project_type"`
	ProjectLanguage  string            `json:"project_language"`
	BuildSystem      string            `json:"build_system"`
	Dependencies     map[string]string `json:"dependencies"`
	ProjectStructure *ProjectStructure `json:"project_structure"`
	RecentlyModified []string          `json:"recently_modified"`
	HotspotFiles     []string          `json:"hotspot_files"` // Frequently accessed files
}

type SessionContext struct {
	SessionID         string                 `json:"session_id"`
	StartTime         time.Time              `json:"start_time"`
	Duration          time.Duration          `json:"duration"`
	RecentQueries     []string               `json:"recent_queries"`
	RecentResults     []*SearchResult        `json:"recent_results"`
	ClickedResults    []*SearchResult        `json:"clicked_results"`
	CurrentTask       string                 `json:"current_task,omitempty"`
	TaskContext       map[string]interface{} `json:"task_context,omitempty"`
	NavigationHistory []string               `json:"navigation_history"`
}

type CodeContext struct {
	CurrentPosition    *CodePosition       `json:"current_position,omitempty"`
	SelectedCode       string              `json:"selected_code,omitempty"`
	SurroundingContext *SurroundingContext `json:"surrounding_context,omitempty"`
	SymbolsInScope     []string            `json:"symbols_in_scope"`
	ImportsInScope     []string            `json:"imports_in_scope"`
	CallStack          []string            `json:"call_stack,omitempty"`
	CodeFlow           *CodeFlowContext    `json:"code_flow,omitempty"`
}

type UserContext struct {
	UserID             string                 `json:"user_id"`
	Preferences        map[string]interface{} `json:"preferences"`
	ExpertiseLevel     ExpertiseLevel         `json:"expertise_level"`
	PreferredLanguages []string               `json:"preferred_languages"`
	RecentFocus        []string               `json:"recent_focus"` // Recently focused areas
	WorkPatterns       *WorkPatterns          `json:"work_patterns"`
	LearningGoals      []string               `json:"learning_goals,omitempty"`
}

type TemporalContext struct {
	TimeOfDay          string           `json:"time_of_day"`
	DayOfWeek          string           `json:"day_of_week"`
	WorkingHours       bool             `json:"working_hours"`
	RecentActivity     *ActivitySummary `json:"recent_activity"`
	ProductivityPeriod string           `json:"productivity_period"` // high, medium, low
}

// Supporting structures

type CodePosition struct {
	FilePath string `json:"file_path"`
	Line     int    `json:"line"`
	Column   int    `json:"column"`
	Function string `json:"function,omitempty"`
	Class    string `json:"class,omitempty"`
	Scope    string `json:"scope,omitempty"`
}

type SurroundingContext struct {
	BeforeLines []string `json:"before_lines"`
	AfterLines  []string `json:"after_lines"`
	IndentLevel int      `json:"indent_level"`
	BlockType   string   `json:"block_type"` // function, class, if, loop, etc.
}

type CodeFlowContext struct {
	EntryPoints     []string `json:"entry_points"`
	DataFlow        []string `json:"data_flow"`
	ControlFlow     []string `json:"control_flow"`
	DependencyChain []string `json:"dependency_chain"`
}

type ExpertiseLevel string

const (
	ExpertiseBeginner     ExpertiseLevel = "beginner"
	ExpertiseIntermediate ExpertiseLevel = "intermediate"
	ExpertiseAdvanced     ExpertiseLevel = "advanced"
	ExpertiseExpert       ExpertiseLevel = "expert"
)

type WorkPatterns struct {
	PreferredSearchTypes []SearchType       `json:"preferred_search_types"`
	CommonQueries        map[string]int     `json:"common_queries"`
	ActivityPeaks        []string           `json:"activity_peaks"`
	FocusAreas           map[string]float32 `json:"focus_areas"`
	ProductiveTimes      []string           `json:"productive_times"`
}

type ActivitySummary struct {
	FilesAccessed    []string                 `json:"files_accessed"`
	QueriesPerformed int                      `json:"queries_performed"`
	ResultsClicked   int                      `json:"results_clicked"`
	TimeSpent        map[string]time.Duration `json:"time_spent"`
	TasksSwitched    int                      `json:"tasks_switched"`
}

// Enhanced search response with context
type ContextAwareSearchResponse struct {
	*SearchResponse
	ContextAnalysis    *ContextAnalysis     `json:"context_analysis,omitempty"`
	RelevanceFactors   *RelevanceFactors    `json:"relevance_factors,omitempty"`
	RelatedCode        []*RelatedCodeResult `json:"related_code,omitempty"`
	ContextSuggestions []*ContextSuggestion `json:"context_suggestions,omitempty"`
	LearningInsights   *LearningInsights    `json:"learning_insights,omitempty"`
}

type ContextAnalysis struct {
	ContextScore       float32  `json:"context_score"`
	RelevantFactors    []string `json:"relevant_factors"`
	ContextMatches     int      `json:"context_matches"`
	ScopeRelevance     float32  `json:"scope_relevance"`
	WorkspaceRelevance float32  `json:"workspace_relevance"`
	TemporalRelevance  float32  `json:"temporal_relevance"`
}

type RelevanceFactors struct {
	FileProximity      float32 `json:"file_proximity"`
	ScopeAlignment     float32 `json:"scope_alignment"`
	RecentUsage        float32 `json:"recent_usage"`
	UserPreference     float32 `json:"user_preference"`
	ProjectRelevance   float32 `json:"project_relevance"`
	SemanticSimilarity float32 `json:"semantic_similarity"`
}

type RelatedCodeResult struct {
	*SearchResult
	RelationshipType     RelationshipType `json:"relationship_type"`
	RelationshipStrength float32          `json:"relationship_strength"`
	ContextPath          []string         `json:"context_path"`
}

type ContextSuggestion struct {
	Type        SuggestionType         `json:"type"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Action      string                 `json:"action"`
	Context     map[string]interface{} `json:"context"`
	Priority    Priority               `json:"priority"`
	Confidence  float32                `json:"confidence"`
}

type LearningInsights struct {
	PatternDetected []string `json:"patterns_detected"`
	Recommendations []string `json:"recommendations"`
	SkillGaps       []string `json:"skill_gaps,omitempty"`
	LearningPath    []string `json:"learning_path,omitempty"`
}

// Metrics for context-aware search
type ContextAwareSearchMetrics struct {
	TotalSearches           int64         `json:"total_searches"`
	ContextEnhancedSearches int64         `json:"context_enhanced_searches"`
	AverageContextScore     float32       `json:"average_context_score"`
	AverageRelevanceBoost   float32       `json:"average_relevance_boost"`
	RelatedCodeFinds        int64         `json:"related_code_finds"`
	ContextCacheHits        int64         `json:"context_cache_hits"`
	UserSatisfactionScore   float32       `json:"user_satisfaction_score"`
	LearningAccuracy        float32       `json:"learning_accuracy"`
	AdaptationRate          float32       `json:"adaptation_rate"`
	ContextProcessingTime   time.Duration `json:"context_processing_time"`
	LastContextUpdate       time.Time     `json:"last_context_update"`
	mu                      sync.RWMutex
}

// NewContextAwareSearchAgent creates a new context-aware search agent
func NewContextAwareSearchAgent(
	baseSearchAgent *SearchAgent,
	contextManager *app.ContextManager,
	indexer *indexer.UltraFastIndexer,
	vectorDB *vectordb.VectorDB,
	config *ContextAwareSearchConfig,
	logger logger.Logger,
	contextAnalyzer ContextAnalyzer,
	relevanceCalculator RelevanceCalculator,
	workspaceAnalyzer WorkspaceAnalyzer,
	sessionTracker SessionTracker,
	intentPredictor IntentPredictor,
) *ContextAwareSearchAgent {
	if config == nil {
		config = &ContextAwareSearchConfig{
			EnableContextAnalysis:    true,
			EnableWorkspaceAnalysis:  true,
			EnableSessionTracking:    true,
			EnableIntentPrediction:   true,
			EnableRelevanceBoost:     true,
			RelevanceThreshold:       0.3,
			ContextSimilarityWeight:  0.4,
			EnableAdaptiveFiltering:  true,
			MaxContextRadius:         20,
			EnableScopeFiltering:     true,
			EnableRelatedCodeSearch:  true,
			MaxRelatedResults:        10,
			RelatedCodeThreshold:     0.5,
			EnableUserLearning:       true,
			EnableBehaviorTracking:   true,
			LearningWindowDays:       30,
			EnableResultCaching:      true,
			CacheSize:                1000,
			CacheTTL:                 time.Minute * 10,
			MaxProcessingTime:        time.Second * 5,
			SessionTimeout:           time.Minute * 30,
			MaxSessionHistory:        100,
			EnableContextPersistence: true,
			ContextWeights: &ContextWeights{
				CurrentFile:      0.15,
				OpenFiles:        0.10,
				RecentFiles:      0.08,
				ProjectStructure: 0.12,
				WorkingDirectory: 0.05,
				CodeScope:        0.20,
				FunctionContext:  0.15,
				ClassContext:     0.10,
				ImportContext:    0.05,
				RecentActions:    0.08,
				UserPreferences:  0.07,
				TimeOfDay:        0.02,
			},
		}
	}

	agent := &ContextAwareSearchAgent{
		baseSearchAgent:     baseSearchAgent,
		contextManager:      contextManager,
		indexer:             indexer,
		vectorDB:            vectorDB,
		config:              config,
		logger:              logger,
		contextAnalyzer:     contextAnalyzer,
		relevanceCalculator: relevanceCalculator,
		workspaceAnalyzer:   workspaceAnalyzer,
		intentPredictor:     intentPredictor,
		sessionTracker:      sessionTracker,
		status:              StatusIdle,
		metrics:             &ContextAwareSearchMetrics{},
	}

	// Initialize capabilities
	agent.initializeCapabilities()

	// Initialize components
	agent.initializeComponents()

	return agent
}

// ProcessRequest processes a context-aware search request
func (casa *ContextAwareSearchAgent) ProcessRequest(ctx context.Context, request *AgentRequest) (*AgentResponse, error) {
	start := time.Now()
	casa.status = StatusBusy
	defer func() { casa.status = StatusIdle }()

	// Parse context-aware search request
	searchRequest, err := casa.parseContextAwareSearchRequest(request)
	if err != nil {
		return nil, fmt.Errorf("failed to parse context-aware search request: %v", err)
	}

	// Apply processing timeout
	processCtx := ctx
	if casa.config.MaxProcessingTime > 0 {
		var cancel context.CancelFunc
		processCtx, cancel = context.WithTimeout(ctx, casa.config.MaxProcessingTime)
		defer cancel()
	}

	// Enhance request with context
	enhancedRequest, err := casa.enhanceRequestWithContext(processCtx, searchRequest)
	if err != nil {
		return nil, fmt.Errorf("context enhancement failed: %v", err)
	}

	// Perform context-aware search
	searchResponse, err := casa.performContextAwareSearch(processCtx, enhancedRequest)
	if err != nil {
		casa.updateMetrics(false, time.Since(start))
		return nil, fmt.Errorf("context-aware search failed: %v", err)
	}

	// Create agent response
	response := &AgentResponse{
		RequestID:      request.ID,
		AgentType:      casa.GetType(),
		AgentVersion:   casa.GetVersion(),
		Result:         searchResponse,
		Confidence:     casa.calculateConfidence(searchRequest, searchResponse),
		ProcessingTime: time.Since(start),
		CreatedAt:      time.Now(),
	}

	// Update user behavior tracking
	if casa.config.EnableBehaviorTracking {
		casa.userBehaviorTracker.TrackSearch(request.UserID, searchRequest, searchResponse)
	}

	// Update metrics
	casa.updateMetrics(true, time.Since(start))

	return response, nil
}

// enhanceRequestWithContext adds contextual information to the search request
func (casa *ContextAwareSearchAgent) enhanceRequestWithContext(ctx context.Context, request *ContextAwareSearchRequest) (*ContextAwareSearchRequest, error) {
	// Get current context from context manager
	currentContext := casa.contextManager.GetCurrentContext()

	// Analyze workspace context
	if casa.config.EnableWorkspaceAnalysis {
		workspaceContext, err := casa.workspaceAnalyzer.AnalyzeWorkspace(currentContext)
		if err != nil {
			casa.logger.Warn("Workspace analysis failed", "error", err)
		} else {
			request.WorkspaceContext = workspaceContext
		}
	}

	// Build session context
	if casa.config.EnableSessionTracking {
		sessionContext := casa.sessionTracker.GetSessionContext(request.Context.UserPreferences["session_id"])
		request.SessionContext = sessionContext
	}

	// Extract code context
	if request.Context != nil && request.Context.CurrentFile != "" {
		codeContext, err := casa.contextAnalyzer.AnalyzeCodeContext(request.Context)
		if err != nil {
			casa.logger.Warn("Code context analysis failed", "error", err)
		} else {
			request.CodeContext = codeContext
		}
	}

	// Build user context
	if casa.config.EnableUserLearning {
		userContext := casa.userBehaviorTracker.GetUserContext(currentContext.UserID)
		request.UserContext = userContext
	}

	// Build temporal context
	request.TemporalContext = casa.buildTemporalContext()

	return request, nil
}

// performContextAwareSearch executes search with context awareness
func (casa *ContextAwareSearchAgent) performContextAwareSearch(ctx context.Context, request *ContextAwareSearchRequest) (*ContextAwareSearchResponse, error) {
	// Perform base search first
	baseResponse, err := casa.baseSearchAgent.performSearch(ctx, request.SearchRequest)
	if err != nil {
		return nil, fmt.Errorf("base search failed: %v", err)
	}

	// Enhance results with context
	enhancedResults := casa.enhanceResultsWithContext(baseResponse.Results, request)
	baseResponse.Results = enhancedResults

	// Apply contextual ranking
	if casa.contextualRanker != nil {
		baseResponse.Results = casa.contextualRanker.RankWithContext(baseResponse.Results, request)
	}

	// Find related code
	var relatedCode []*RelatedCodeResult
	if casa.config.EnableRelatedCodeSearch && len(baseResponse.Results) > 0 {
		relatedCode = casa.findRelatedCode(ctx, baseResponse.Results[:5], request) // Top 5 results
	}

	// Analyze context
	contextAnalysis := casa.analyzeSearchContext(request, baseResponse.Results)

	// Calculate relevance factors
	relevanceFactors := casa.calculateRelevanceFactors(request, baseResponse.Results)

	// Generate context suggestions
	contextSuggestions := casa.generateContextSuggestions(request, baseResponse.Results, contextAnalysis)

	// Generate learning insights
	var learningInsights *LearningInsights
	if casa.config.EnableUserLearning {
		learningInsights = casa.contextLearner.GenerateInsights(request, baseResponse.Results)
	}

	// Create enhanced response
	response := &ContextAwareSearchResponse{
		SearchResponse:     baseResponse,
		ContextAnalysis:    contextAnalysis,
		RelevanceFactors:   relevanceFactors,
		RelatedCode:        relatedCode,
		ContextSuggestions: contextSuggestions,
		LearningInsights:   learningInsights,
	}

	return response, nil
}

// enhanceResultsWithContext adds contextual information to search results
func (casa *ContextAwareSearchAgent) enhanceResultsWithContext(results []*SearchResult, request *ContextAwareSearchRequest) []*SearchResult {
	var enhanced []*SearchResult

	for _, result := range results {
		enhancedResult := *result // Copy

		// Calculate context-based relevance boost
		contextBoost := casa.calculateContextBoost(result, request)
		enhancedResult.Relevance = result.Relevance * (1.0 + contextBoost)

		// Add context-specific metadata
		if enhancedResult.Metadata == nil {
			enhancedResult.Metadata = make(map[string]interface{})
		}

		enhancedResult.Metadata["context_boost"] = contextBoost
		enhancedResult.Metadata["context_factors"] = casa.getContextFactors(result, request)

		// Add scope information
		if request.CodeContext != nil {
			enhancedResult.Metadata["scope_relevance"] = casa.calculateScopeRelevance(result, request.CodeContext)
		}

		// Add workspace relevance
		if request.WorkspaceContext != nil {
			enhancedResult.Metadata["workspace_relevance"] = casa.calculateWorkspaceRelevance(result, request.WorkspaceContext)
		}

		enhanced = append(enhanced, &enhancedResult)
	}

	return enhanced
}

// calculateContextBoost calculates how much to boost a result based on context
func (casa *ContextAwareSearchAgent) calculateContextBoost(result *SearchResult, request *ContextAwareSearchRequest) float32 {
	var totalBoost float32 = 0.0
	weights := casa.config.ContextWeights

	// Current file boost
	if request.Context != nil && request.Context.CurrentFile != "" {
		if result.FilePath == request.Context.CurrentFile {
			totalBoost += weights.CurrentFile
		}
	}

	// Open files boost
	if request.Context != nil && len(request.Context.RecentFiles) > 0 {
		for _, openFile := range request.Context.RecentFiles {
			if result.FilePath == openFile {
				totalBoost += weights.OpenFiles
				break
			}
		}
	}

	// Working directory boost
	if request.Context != nil && request.Context.WorkingDirectory != "" {
		if strings.HasPrefix(result.FilePath, request.Context.WorkingDirectory) {
			totalBoost += weights.WorkingDirectory
		}
	}

	// Function context boost
	if request.CodeContext != nil && request.CodeContext.CurrentPosition != nil {
		if result.Metadata != nil {
			if resultFunc, ok := result.Metadata["function"].(string); ok {
				if resultFunc == request.CodeContext.CurrentPosition.Function {
					totalBoost += weights.FunctionContext
				}
			}
		}
	}

	// Class context boost
	if request.CodeContext != nil && request.CodeContext.CurrentPosition != nil {
		if result.Metadata != nil {
			if resultClass, ok := result.Metadata["class"].(string); ok {
				if resultClass == request.CodeContext.CurrentPosition.Class {
					totalBoost += weights.ClassContext
				}
			}
		}
	}

	// Recent actions boost
	if request.SessionContext != nil && len(request.SessionContext.RecentQueries) > 0 {
		for _, recentQuery := range request.SessionContext.RecentQueries {
			if strings.Contains(strings.ToLower(result.Content), strings.ToLower(recentQuery)) {
				totalBoost += weights.RecentActions * 0.5 // Partial boost
				break
			}
		}
	}

	// User preferences boost
	if request.UserContext != nil && len(request.UserContext.PreferredLanguages) > 0 {
		resultLang := casa.detectLanguageFromPath(result.FilePath)
		for _, prefLang := range request.UserContext.PreferredLanguages {
			if strings.EqualFold(resultLang, prefLang) {
				totalBoost += weights.UserPreferences
				break
			}
		}
	}

	// Cap the boost
	if totalBoost > 1.0 {
		totalBoost = 1.0
	}

	return totalBoost
}

// findRelatedCode finds code related to the search results
func (casa *ContextAwareSearchAgent) findRelatedCode(ctx context.Context, results []*SearchResult, request *ContextAwareSearchRequest) []*RelatedCodeResult {
	if casa.relatedCodeFinder == nil {
		return nil
	}

	var relatedCode []*RelatedCodeResult

	for _, result := range results {
		related := casa.relatedCodeFinder.FindRelatedCode(ctx, result, request)
		relatedCode = append(relatedCode, related...)
	}

	// Sort by relationship strength
	sort.Slice(relatedCode, func(i, j int) bool {
		return relatedCode[i].RelationshipStrength > relatedCode[j].RelationshipStrength
	})

	// Limit results
	if len(relatedCode) > casa.config.MaxRelatedResults {
		relatedCode = relatedCode[:casa.config.MaxRelatedResults]
	}

	return relatedCode
}

// analyzeSearchContext analyzes the context of the search
func (casa *ContextAwareSearchAgent) analyzeSearchContext(request *ContextAwareSearchRequest, results []*SearchResult) *ContextAnalysis {
	analysis := &ContextAnalysis{}

	// Calculate context score
	analysis.ContextScore = casa.calculateOverallContextScore(request)

	// Identify relevant factors
	analysis.RelevantFactors = casa.identifyRelevantFactors(request)

	// Count context matches
	analysis.ContextMatches = casa.countContextMatches(results, request)

	// Calculate scope relevance
	if request.CodeContext != nil {
		analysis.ScopeRelevance = casa.calculateAverageScopeRelevance(results, request.CodeContext)
	}

	// Calculate workspace relevance
	if request.WorkspaceContext != nil {
		analysis.WorkspaceRelevance = casa.calculateAverageWorkspaceRelevance(results, request.WorkspaceContext)
	}

	// Calculate temporal relevance
	if request.TemporalContext != nil {
		analysis.TemporalRelevance = casa.calculateTemporalRelevance(results, request.TemporalContext)
	}

	return analysis
}

// calculateRelevanceFactors calculates detailed relevance factors
func (casa *ContextAwareSearchAgent) calculateRelevanceFactors(request *ContextAwareSearchRequest, results []*SearchResult) *RelevanceFactors {
	factors := &RelevanceFactors{}

	if len(results) == 0 {
		return factors
	}

	// Calculate average factors across results
	var totalFileProximity, totalScopeAlignment, totalRecentUsage float32
	var totalUserPreference, totalProjectRelevance, totalSemanticSimilarity float32

	for _, result := range results {
		factors := casa.calculateResultRelevanceFactors(result, request)
		totalFileProximity += factors.FileProximity
		totalScopeAlignment += factors.ScopeAlignment
		totalRecentUsage += factors.RecentUsage
		totalUserPreference += factors.UserPreference
		totalProjectRelevance += factors.ProjectRelevance
		totalSemanticSimilarity += factors.SemanticSimilarity
	}

	count := float32(len(results))
	return &RelevanceFactors{
		FileProximity:      totalFileProximity / count,
		ScopeAlignment:     totalScopeAlignment / count,
		RecentUsage:        totalRecentUsage / count,
		UserPreference:     totalUserPreference / count,
		ProjectRelevance:   totalProjectRelevance / count,
		SemanticSimilarity: totalSemanticSimilarity / count,
	}
}

// generateContextSuggestions generates suggestions based on context
func (casa *ContextAwareSearchAgent) generateContextSuggestions(request *ContextAwareSearchRequest, results []*SearchResult, analysis *ContextAnalysis) []*ContextSuggestion {
	var suggestions []*ContextSuggestion

	// Suggest expanding search scope if few results
	if len(results) < 5 {
		suggestions = append(suggestions, &ContextSuggestion{
			Type:        SuggestionTypeBestPractice,
			Title:       "Expand Search Scope",
			Description: "Try searching in a broader context or related files",
			Action:      "expand_scope",
			Priority:    PriorityMedium,
			Confidence:  0.7,
		})
	}

	// Suggest filtering if too many results
	if len(results) > 50 {
		suggestions = append(suggestions, &ContextSuggestion{
			Type:        SuggestionTypeBestPractice,
			Title:       "Refine Search",
			Description: "Add filters to narrow down the results",
			Action:      "add_filters",
			Priority:    PriorityMedium,
			Confidence:  0.8,
		})
	}

	// Suggest related searches based on context
	if request.CodeContext != nil && request.CodeContext.CurrentPosition != nil {
		if request.CodeContext.CurrentPosition.Function != "" {
			suggestions = append(suggestions, &ContextSuggestion{
				Type:        SuggestionTypeRefactoring,
				Title:       "Search in Current Function",
				Description: fmt.Sprintf("Search within function '%s'", request.CodeContext.CurrentPosition.Function),
				Action:      "search_in_function",
				Context:     map[string]interface{}{"function": request.CodeContext.CurrentPosition.Function},
				Priority:    PriorityHigh,
				Confidence:  0.9,
			})
		}
	}

	// Learning-based suggestions
	if request.UserContext != nil && len(request.UserContext.RecentFocus) > 0 {
		for _, focus := range request.UserContext.RecentFocus {
			suggestions = append(suggestions, &ContextSuggestion{
				Type:        SuggestionTypeBestPractice,
				Title:       "Related Focus Area",
				Description: fmt.Sprintf("Consider searching in %s", focus),
				Action:      "search_focus_area",
				Context:     map[string]interface{}{"focus_area": focus},
				Priority:    PriorityLow,
				Confidence:  0.6,
			})
		}
	}

	return suggestions
}

// Utility methods

func (casa *ContextAwareSearchAgent) parseContextAwareSearchRequest(request *AgentRequest) (*ContextAwareSearchRequest, error) {
	// First parse as regular search request
	baseRequest, err := casa.baseSearchAgent.parseSearchRequest(request)
	if err != nil {
		return nil, err
	}

	// Create context-aware request
	contextRequest := &ContextAwareSearchRequest{
		SearchRequest: baseRequest,
	}

	return contextRequest, nil
}

func (casa *ContextAwareSearchAgent) calculateConfidence(request *ContextAwareSearchRequest, response *ContextAwareSearchResponse) float64 {
	baseConfidence := casa.baseSearchAgent.calculateConfidence(request.SearchRequest, response.SearchResponse)

	// Boost confidence based on context quality
	contextBoost := 0.0
	if response.ContextAnalysis != nil {
		contextBoost = float64(response.ContextAnalysis.ContextScore) * 0.2
	}

	confidence := baseConfidence + contextBoost
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

func (casa *ContextAwareSearchAgent) detectLanguageFromPath(filepath string) string {
	ext := strings.ToLower(strings.Split(filepath, ".")[len(strings.Split(filepath, "."))-1])
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

func (casa *ContextAwareSearchAgent) buildTemporalContext() *TemporalContext {
	now := time.Now()

	return &TemporalContext{
		TimeOfDay:          casa.getTimeOfDay(now),
		DayOfWeek:          now.Weekday().String(),
		WorkingHours:       casa.isWorkingHours(now),
		ProductivityPeriod: casa.getProductivityPeriod(now),
	}
}

func (casa *ContextAwareSearchAgent) getTimeOfDay(t time.Time) string {
	hour := t.Hour()
	if hour < 6 {
		return "night"
	} else if hour < 12 {
		return "morning"
	} else if hour < 18 {
		return "afternoon"
	}
	return "evening"
}

func (casa *ContextAwareSearchAgent) isWorkingHours(t time.Time) bool {
	hour := t.Hour()
	weekday := t.Weekday()
	return weekday >= time.Monday && weekday <= time.Friday && hour >= 9 && hour < 17
}

func (casa *ContextAwareSearchAgent) getProductivityPeriod(t time.Time) string {
	hour := t.Hour()
	if (hour >= 9 && hour <= 11) || (hour >= 14 && hour <= 16) {
		return "high"
	} else if (hour >= 8 && hour <= 12) || (hour >= 13 && hour <= 17) {
		return "medium"
	}
	return "low"
}

// Helper calculation methods

func (casa *ContextAwareSearchAgent) getContextFactors(result *SearchResult, request *ContextAwareSearchRequest) []string {
	var factors []string

	if request.Context != nil && result.FilePath == request.Context.CurrentFile {
		factors = append(factors, "current_file")
	}

	if request.CodeContext != nil && request.CodeContext.CurrentPosition != nil {
		if resultFunc, ok := result.Metadata["function"].(string); ok {
			if resultFunc == request.CodeContext.CurrentPosition.Function {
				factors = append(factors, "same_function")
			}
		}
	}

	return factors
}

func (casa *ContextAwareSearchAgent) calculateScopeRelevance(result *SearchResult, codeContext *CodeContext) float32 {
	relevance := float32(0.5) // Base relevance

	if codeContext.CurrentPosition != nil {
		// Same file bonus
		if result.FilePath == codeContext.CurrentPosition.FilePath {
			relevance += 0.2
		}

		// Same function bonus
		if resultFunc, ok := result.Metadata["function"].(string); ok {
			if resultFunc == codeContext.CurrentPosition.Function {
				relevance += 0.3
			}
		}
	}

	if relevance > 1.0 {
		relevance = 1.0
	}

	return relevance
}

func (casa *ContextAwareSearchAgent) calculateWorkspaceRelevance(result *SearchResult, workspaceContext *WorkspaceContext) float32 {
	relevance := float32(0.5)

	// Project root relevance
	if strings.HasPrefix(result.FilePath, workspaceContext.ProjectRoot) {
		relevance += 0.2
	}

	// Recently modified files bonus
	for _, recentFile := range workspaceContext.RecentlyModified {
		if result.FilePath == recentFile {
			relevance += 0.2
			break
		}
	}

	// Hotspot files bonus
	for _, hotspotFile := range workspaceContext.HotspotFiles {
		if result.FilePath == hotspotFile {
			relevance += 0.3
			break
		}
	}

	if relevance > 1.0 {
		relevance = 1.0
	}

	return relevance
}

func (casa *ContextAwareSearchAgent) calculateOverallContextScore(request *ContextAwareSearchRequest) float32 {
	score := float32(0.0)
	factors := 0

	if request.WorkspaceContext != nil {
		score += 0.2
		factors++
	}

	if request.SessionContext != nil {
		score += 0.2
		factors++
	}

	if request.CodeContext != nil {
		score += 0.3
		factors++
	}

	if request.UserContext != nil {
		score += 0.2
		factors++
	}

	if request.TemporalContext != nil {
		score += 0.1
		factors++
	}

	if factors > 0 {
		return score
	}

	return 0.0
}

func (casa *ContextAwareSearchAgent) identifyRelevantFactors(request *ContextAwareSearchRequest) []string {
	var factors []string

	if request.Context != nil && request.Context.CurrentFile != "" {
		factors = append(factors, "current_file_context")
	}

	if request.CodeContext != nil && request.CodeContext.CurrentPosition != nil {
		factors = append(factors, "code_position_context")
	}

	if request.SessionContext != nil && len(request.SessionContext.RecentQueries) > 0 {
		factors = append(factors, "recent_query_context")
	}

	if request.UserContext != nil && len(request.UserContext.PreferredLanguages) > 0 {
		factors = append(factors, "user_preference_context")
	}

	return factors
}

func (casa *ContextAwareSearchAgent) countContextMatches(results []*SearchResult, request *ContextAwareSearchRequest) int {
	matches := 0

	for _, result := range results {
		if casa.calculateContextBoost(result, request) > 0 {
			matches++
		}
	}

	return matches
}

func (casa *ContextAwareSearchAgent) calculateAverageScopeRelevance(results []*SearchResult, codeContext *CodeContext) float32 {
	if len(results) == 0 {
		return 0.0
	}

	var total float32
	for _, result := range results {
		total += casa.calculateScopeRelevance(result, codeContext)
	}

	return total / float32(len(results))
}

func (casa *ContextAwareSearchAgent) calculateAverageWorkspaceRelevance(results []*SearchResult, workspaceContext *WorkspaceContext) float32 {
	if len(results) == 0 {
		return 0.0
	}

	var total float32
	for _, result := range results {
		total += casa.calculateWorkspaceRelevance(result, workspaceContext)
	}

	return total / float32(len(results))
}

func (casa *ContextAwareSearchAgent) calculateTemporalRelevance(results []*SearchResult, temporalContext *TemporalContext) float32 {
	// Simple temporal relevance based on working hours
	if temporalContext.WorkingHours {
		return 0.8
	}
	return 0.5
}

func (casa *ContextAwareSearchAgent) calculateResultRelevanceFactors(result *SearchResult, request *ContextAwareSearchRequest) *RelevanceFactors {
	factors := &RelevanceFactors{}

	// File proximity
	if request.Context != nil && request.Context.CurrentFile != "" {
		if result.FilePath == request.Context.CurrentFile {
			factors.FileProximity = 1.0
		} else {
			// Calculate based on directory proximity
			factors.FileProximity = casa.calculateDirectoryProximity(result.FilePath, request.Context.CurrentFile)
		}
	}

	// Scope alignment
	if request.CodeContext != nil {
		factors.ScopeAlignment = casa.calculateScopeRelevance(result, request.CodeContext)
	}

	// Recent usage
	if request.SessionContext != nil {
		factors.RecentUsage = casa.calculateRecentUsageScore(result, request.SessionContext)
	}

	// User preference
	if request.UserContext != nil {
		factors.UserPreference = casa.calculateUserPreferenceScore(result, request.UserContext)
	}

	// Project relevance
	if request.WorkspaceContext != nil {
		factors.ProjectRelevance = casa.calculateWorkspaceRelevance(result, request.WorkspaceContext)
	}

	// Semantic similarity (placeholder)
	factors.SemanticSimilarity = result.Score // Use existing score as proxy

	return factors
}

func (casa *ContextAwareSearchAgent) calculateDirectoryProximity(filePath1, filePath2 string) float32 {
	dirs1 := strings.Split(filePath1, "/")
	dirs2 := strings.Split(filePath2, "/")

	commonDirs := 0
	maxLen := len(dirs1)
	if len(dirs2) > maxLen {
		maxLen = len(dirs2)
	}

	minLen := len(dirs1)
	if len(dirs2) < minLen {
		minLen = len(dirs2)
	}

	for i := 0; i < minLen; i++ {
		if dirs1[i] == dirs2[i] {
			commonDirs++
		} else {
			break
		}
	}

	return float32(commonDirs) / float32(maxLen)
}

func (casa *ContextAwareSearchAgent) calculateRecentUsageScore(result *SearchResult, sessionContext *SessionContext) float32 {
	score := float32(0.0)

	// Check recent results
	for _, recentResult := range sessionContext.RecentResults {
		if recentResult.FilePath == result.FilePath {
			score += 0.3
		}
	}

	// Check clicked results
	for _, clickedResult := range sessionContext.ClickedResults {
		if clickedResult.FilePath == result.FilePath {
			score += 0.5
		}
	}

	if score > 1.0 {
		score = 1.0
	}

	return score
}

func (casa *ContextAwareSearchAgent) calculateUserPreferenceScore(result *SearchResult, userContext *UserContext) float32 {
	score := float32(0.5) // Base score

	// Language preference
	resultLang := casa.detectLanguageFromPath(result.FilePath)
	for _, prefLang := range userContext.PreferredLanguages {
		if strings.EqualFold(resultLang, prefLang) {
			score += 0.3
			break
		}
	}

	// Recent focus areas
	for _, focus := range userContext.RecentFocus {
		if strings.Contains(strings.ToLower(result.FilePath), strings.ToLower(focus)) {
			score += 0.2
			break
		}
	}

	if score > 1.0 {
		score = 1.0
	}

	return score
}

// Component initialization and management

func (casa *ContextAwareSearchAgent) initializeCapabilities() {
	casa.capabilities = &AgentCapabilities{
		AgentType: AgentTypeContextAwareSearch,
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
		MaxContextSize:    8192,
		SupportsStreaming: false,
		SupportsAsync:     true,
		RequiresContext:   true,
		Capabilities: map[string]interface{}{
			"context_analysis":    casa.config.EnableContextAnalysis,
			"workspace_analysis":  casa.config.EnableWorkspaceAnalysis,
			"session_tracking":    casa.config.EnableSessionTracking,
			"intent_prediction":   casa.config.EnableIntentPrediction,
			"relevance_boost":     casa.config.EnableRelevanceBoost,
			"adaptive_filtering":  casa.config.EnableAdaptiveFiltering,
			"related_code_search": casa.config.EnableRelatedCodeSearch,
			"user_learning":       casa.config.EnableUserLearning,
			"behavior_tracking":   casa.config.EnableBehaviorTracking,
		},
	}
}

func (casa *ContextAwareSearchAgent) initializeComponents() {
	// Initialize context analyzer
	casa.contextAnalyzer = NewContextAnalyzer()

	// Initialize relevance calculator
	casa.relevanceCalculator = NewRelevanceCalculator(casa.config.ContextWeights)

	// Initialize workspace analyzer
	if casa.config.EnableWorkspaceAnalysis {
		casa.workspaceAnalyzer = NewWorkspaceAnalyzer(casa.indexer)
	}

	// Initialize session tracker
	if casa.config.EnableSessionTracking {
		casa.sessionTracker = NewSessionTracker(casa.config.SessionTimeout, casa.config.MaxSessionHistory)
	}

	// Initialize intent predictor
	if casa.config.EnableIntentPrediction {
		casa.intentPredictor = NewIntentPredictor()
	}

	// Initialize contextual ranker
	casa.contextualRanker = NewContextualRanker(casa.config.ContextWeights)

	// Initialize adaptive filter
	if casa.config.EnableAdaptiveFiltering {
		casa.adaptiveFilter = NewAdaptiveFilter(casa.config.MaxContextRadius)
	}

	// Initialize related code finder
	if casa.config.EnableRelatedCodeSearch {
		casa.relatedCodeFinder = NewRelatedCodeFinder(casa.indexer, casa.vectorDB)
	}

	// Initialize user behavior tracker
	if casa.config.EnableBehaviorTracking {
		casa.userBehaviorTracker = NewUserBehaviorTracker(casa.config.LearningWindowDays)
	}

	// Initialize context learner
	if casa.config.EnableUserLearning {
		casa.contextLearner = NewContextLearner()
	}
}

// Metrics methods

func (casa *ContextAwareSearchAgent) updateMetrics(success bool, duration time.Duration) {
	casa.metrics.mu.Lock()
	defer casa.metrics.mu.Unlock()

	casa.metrics.TotalSearches++
	if success {
		casa.metrics.ContextEnhancedSearches++
	}

	// Update average context processing time
	if casa.metrics.ContextProcessingTime == 0 {
		casa.metrics.ContextProcessingTime = duration
	} else {
		casa.metrics.ContextProcessingTime = (casa.metrics.ContextProcessingTime + duration) / 2
	}

	casa.metrics.LastContextUpdate = time.Now()
}

// Required Agent interface methods

func (casa *ContextAwareSearchAgent) GetCapabilities() *AgentCapabilities {
	return casa.capabilities
}

func (casa *ContextAwareSearchAgent) GetType() AgentType {
	return AgentTypeContextAwareSearch
}

func (casa *ContextAwareSearchAgent) GetVersion() string {
	return "1.0.0"
}

func (casa *ContextAwareSearchAgent) GetStatus() AgentStatus {
	casa.mu.RLock()
	defer casa.mu.RUnlock()
	return casa.status
}

func (casa *ContextAwareSearchAgent) Initialize(config interface{}) error {
	if cfg, ok := config.(*ContextAwareSearchConfig); ok {
		casa.config = cfg
		casa.initializeComponents()
		return nil
	}
	return fmt.Errorf("invalid config type")
}

func (casa *ContextAwareSearchAgent) Start() error {
	casa.mu.Lock()
	defer casa.mu.Unlock()

	casa.status = StatusIdle
	casa.logger.Info("Context-aware search agent started")
	return nil
}

func (casa *ContextAwareSearchAgent) Stop() error {
	casa.mu.Lock()
	defer casa.mu.Unlock()

	casa.status = StatusStopped
	casa.logger.Info("Context-aware search agent stopped")
	return nil
}

func (casa *ContextAwareSearchAgent) HealthCheck() error {
	if casa.baseSearchAgent == nil {
		return fmt.Errorf("base search agent not configured")
	}

	return casa.baseSearchAgent.HealthCheck()
}

func (casa *ContextAwareSearchAgent) GetMetrics() *AgentMetrics {
	casa.metrics.mu.RLock()
	defer casa.metrics.mu.RUnlock()

	return &AgentMetrics{
		RequestsProcessed:   casa.metrics.TotalSearches,
		AverageResponseTime: casa.metrics.ContextProcessingTime,
		SuccessRate:         float64(casa.metrics.ContextEnhancedSearches) / float64(casa.metrics.TotalSearches),
		LastRequestAt:       casa.metrics.LastContextUpdate,
	}
}

func (casa *ContextAwareSearchAgent) ResetMetrics() {
	casa.metrics.mu.Lock()
	defer casa.metrics.mu.Unlock()

	casa.metrics = &ContextAwareSearchMetrics{}
}

// Placeholder implementations for referenced components

type ContextAnalyzer struct{}

func NewContextAnalyzer() *ContextAnalyzer {
	return &ContextAnalyzer{}
}

func (ca *ContextAnalyzer) AnalyzeCodeContext(context *SearchContext) (*CodeContext, error) {
	// Placeholder implementation
	return &CodeContext{}, nil
}

type RelevanceCalculator struct {
	weights *ContextWeights
}

func NewRelevanceCalculator(weights *ContextWeights) *RelevanceCalculator {
	return &RelevanceCalculator{weights: weights}
}

type SemanticContextEngine struct{}

type WorkspaceAnalyzer struct {
	indexer *indexer.UltraFastIndexer
}

func NewWorkspaceAnalyzer(indexer *indexer.UltraFastIndexer) *WorkspaceAnalyzer {
	return &WorkspaceAnalyzer{indexer: indexer}
}

func (wa *WorkspaceAnalyzer) AnalyzeWorkspace(context interface{}) (*WorkspaceContext, error) {
	// Placeholder implementation
	return &WorkspaceContext{}, nil
}

type SessionTracker struct {
	timeout    time.Duration
	maxHistory int
	sessions   map[string]*SessionContext
	mu         sync.RWMutex
}

func NewSessionTracker(timeout time.Duration, maxHistory int) *SessionTracker {
	return &SessionTracker{
		timeout:    timeout,
		maxHistory: maxHistory,
		sessions:   make(map[string]*SessionContext),
	}
}

func (st *SessionTracker) GetSessionContext(sessionID interface{}) *SessionContext {
	// Placeholder implementation
	return &SessionContext{}
}

type IntentPredictor struct{}

func NewIntentPredictor() *IntentPredictor {
	return &IntentPredictor{}
}

type ContextualRanker struct {
	weights *ContextWeights
}

func NewContextualRanker(weights *ContextWeights) *ContextualRanker {
	return &ContextualRanker{weights: weights}
}

func (cr *ContextualRanker) RankWithContext(results []*SearchResult, request *ContextAwareSearchRequest) []*SearchResult {
	// Placeholder implementation - would apply sophisticated contextual ranking
	return results
}

type AdaptiveFilter struct {
	maxRadius int
}

func NewAdaptiveFilter(maxRadius int) *AdaptiveFilter {
	return &AdaptiveFilter{maxRadius: maxRadius}
}

type RelatedCodeFinder struct {
	indexer  *indexer.UltraFastIndexer
	vectorDB *vectordb.VectorDB
}

func NewRelatedCodeFinder(indexer *indexer.UltraFastIndexer, vectorDB *vectordb.VectorDB) *RelatedCodeFinder {
	return &RelatedCodeFinder{
		indexer:  indexer,
		vectorDB: vectorDB,
	}
}

func (rcf *RelatedCodeFinder) FindRelatedCode(ctx context.Context, result *SearchResult, request *ContextAwareSearchRequest) []*RelatedCodeResult {
	// Placeholder implementation
	return []*RelatedCodeResult{}
}

type UserBehaviorTracker struct {
	learningWindowDays int
}

func NewUserBehaviorTracker(learningWindowDays int) *UserBehaviorTracker {
	return &UserBehaviorTracker{learningWindowDays: learningWindowDays}
}

func (ubt *UserBehaviorTracker) TrackSearch(userID string, request *ContextAwareSearchRequest, response *ContextAwareSearchResponse) {
	// Placeholder implementation
}

func (ubt *UserBehaviorTracker) GetUserContext(userID string) *UserContext {
	// Placeholder implementation
	return &UserContext{}
}

type ContextLearner struct{}

func NewContextLearner() *ContextLearner {
	return &ContextLearner{}
}

func (cl *ContextLearner) GenerateInsights(request *ContextAwareSearchRequest, results []*SearchResult) *LearningInsights {
	// Placeholder implementation
	return &LearningInsights{}
}
