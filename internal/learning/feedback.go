// internal/learning/feedback.go
package learning

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

type FeedbackType int

const (
	FeedbackPositive FeedbackType = iota
	FeedbackNegative
	FeedbackCorrection
	FeedbackSuggestion
	FeedbackBug
	FeedbackFeature
	FeedbackUsability
	FeedbackPerformance
)

type FeedbackCategory int

const (
	CategoryCodeAnalysis FeedbackCategory = iota
	CategorySuggestions
	CategoryPatterns
	CategoryArchitecture
	CategorySemantic
	CategoryDisplay
	CategoryPerformance
	CategoryAccuracy
	CategoryRelevance
	CategoryUsability
)

type FeedbackEntry struct {
	ID          string                 `json:"id"`
	UserID      string                 `json:"user_id"`
	SessionID   string                 `json:"session_id"`
	Type        FeedbackType           `json:"type"`
	Category    FeedbackCategory       `json:"category"`
	Rating      int                    `json:"rating"` // 1-5 scale
	Message     string                 `json:"message"`
	Context     FeedbackContext        `json:"context"`
	Metadata    map[string]interface{} `json:"metadata"`
	Timestamp   time.Time              `json:"timestamp"`
	Processed   bool                   `json:"processed"`
	ProcessedAt time.Time              `json:"processed_at"`

	// AI Response related fields
	QueryHash    string        `json:"query_hash"`
	ResponseHash string        `json:"response_hash"`
	ModelUsed    string        `json:"model_used"`
	ResponseTime time.Duration `json:"response_time"`
	TokensUsed   int           `json:"tokens_used"`

	// Learning signals
	LearningSeed bool   `json:"learning_seed"` // Can be used for training
	Priority     int    `json:"priority"`      // Processing priority
	ValidatedBy  string `json:"validated_by"`  // Expert validation
}

type FeedbackContext struct {
	File        string            `json:"file"`
	Function    string            `json:"function"`
	LineNumber  int               `json:"line_number"`
	CodeSnippet string            `json:"code_snippet"`
	Language    string            `json:"language"`
	ProjectPath string            `json:"project_path"`
	Command     string            `json:"command"`
	Parameters  map[string]string `json:"parameters"`

	// Analysis context
	AnalysisType    string   `json:"analysis_type"`
	PatternsFound   []string `json:"patterns_found"`
	ConfidenceScore float64  `json:"confidence_score"`

	// User context
	UserExperience  string                 `json:"user_experience"` // novice, intermediate, expert
	UserPreferences map[string]interface{} `json:"user_preferences"`
}

type FeedbackMetrics struct {
	TotalFeedback      int                      `json:"total_feedback"`
	ByType             map[FeedbackType]int     `json:"by_type"`
	ByCategory         map[FeedbackCategory]int `json:"by_category"`
	AverageRating      float64                  `json:"average_rating"`
	RatingDistribution map[int]int              `json:"rating_distribution"`
	ProcessingRate     float64                  `json:"processing_rate"`

	// Trends
	DailyFeedback  map[string]int               `json:"daily_feedback"`
	WeeklyTrends   map[string]float64           `json:"weekly_trends"`
	CategoryTrends map[FeedbackCategory]float64 `json:"category_trends"`

	// Quality metrics
	UserSatisfaction float64            `json:"user_satisfaction"`
	ResponseAccuracy float64            `json:"response_accuracy"`
	FeatureUsability map[string]float64 `json:"feature_usability"`

	// Learning effectiveness
	LearningSignals  int     `json:"learning_signals"`
	ModelImprovement float64 `json:"model_improvement"`
	PatternDetection float64 `json:"pattern_detection"`
}

type FeedbackProcessor struct {
	db               *sql.DB
	dbPath           string
	processingQueue  chan *FeedbackEntry
	processedSignals chan ProcessedFeedback

	// Configuration
	batchSize          int
	processingInterval time.Duration
	maxQueueSize       int
	retentionPeriod    time.Duration
	enableMLPipeline   bool

	// State
	metrics       *FeedbackMetrics
	userProfiles  map[string]*UserProfile
	learningRules []LearningRule

	// Concurrency
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	mu     sync.RWMutex
}

type ProcessedFeedback struct {
	FeedbackID     string                 `json:"feedback_id"`
	LearningSignal LearningSignalType     `json:"learning_signal"`
	Confidence     float64                `json:"confidence"`
	Actions        []LearningAction       `json:"actions"`
	Metadata       map[string]interface{} `json:"metadata"`
	ProcessedAt    time.Time              `json:"processed_at"`
}

type LearningSignalType int

const (
	SignalModelUpdate LearningSignalType = iota
	SignalPatternUpdate
	SignalParameterTuning
	SignalFeatureToggle
	SignalDataAugmentation
	SignalErrorCorrection
	SignalPreferenceUpdate
	SignalContextEnhancement
)

type LearningAction struct {
	Type           string                 `json:"type"`
	Target         string                 `json:"target"`
	Parameters     map[string]interface{} `json:"parameters"`
	Priority       int                    `json:"priority"`
	Confidence     float64                `json:"confidence"`
	ExpectedImpact float64                `json:"expected_impact"`
}

type UserProfile struct {
	UserID           string                 `json:"user_id"`
	Experience       string                 `json:"experience"`
	Preferences      map[string]interface{} `json:"preferences"`
	FeedbackHistory  []string               `json:"feedback_history"`
	ReliabilityScore float64                `json:"reliability_score"`
	ExpertiseAreas   []string               `json:"expertise_areas"`
	CreatedAt        time.Time              `json:"created_at"`
	UpdatedAt        time.Time              `json:"updated_at"`
}

type LearningRule struct {
	ID          string           `json:"id"`
	Name        string           `json:"name"`
	Description string           `json:"description"`
	Conditions  []RuleCondition  `json:"conditions"`
	Actions     []LearningAction `json:"actions"`
	Priority    int              `json:"priority"`
	Enabled     bool             `json:"enabled"`
	CreatedAt   time.Time        `json:"created_at"`
}

type RuleCondition struct {
	Field    string      `json:"field"`
	Operator string      `json:"operator"`
	Value    interface{} `json:"value"`
}

// NewFeedbackProcessor creates a new feedback processor
func NewFeedbackProcessor(dbPath string) (*FeedbackProcessor, error) {
	ctx, cancel := context.WithCancel(context.Background())

	fp := &FeedbackProcessor{
		dbPath:             dbPath,
		processingQueue:    make(chan *FeedbackEntry, 1000),
		processedSignals:   make(chan ProcessedFeedback, 1000),
		batchSize:          10,
		processingInterval: 1 * time.Minute,
		maxQueueSize:       1000,
		retentionPeriod:    90 * 24 * time.Hour, // 90 days
		enableMLPipeline:   true,
		userProfiles:       make(map[string]*UserProfile),
		learningRules:      make([]LearningRule, 0),
		ctx:                ctx,
		cancel:             cancel,
	}

	if err := fp.initialize(); err != nil {
		cancel()
		return nil, err
	}

	// Start background processors
	fp.startProcessors()

	return fp, nil
}

// initialize sets up the database and initial state
func (fp *FeedbackProcessor) initialize() error {
	// Open database connection
	db, err := sql.Open("sqlite3", fp.dbPath+"?_journal_mode=WAL&_synchronous=NORMAL")
	if err != nil {
		return fmt.Errorf("failed to open database: %w", err)
	}
	fp.db = db

	// Create tables
	if err := fp.createTables(); err != nil {
		return fmt.Errorf("failed to create tables: %w", err)
	}

	// Load existing data
	if err := fp.loadUserProfiles(); err != nil {
		return fmt.Errorf("failed to load user profiles: %w", err)
	}

	if err := fp.loadLearningRules(); err != nil {
		return fmt.Errorf("failed to load learning rules: %w", err)
	}

	// Calculate initial metrics
	if err := fp.calculateMetrics(); err != nil {
		return fmt.Errorf("failed to calculate initial metrics: %w", err)
	}

	return nil
}

// createTables creates the necessary database tables
func (fp *FeedbackProcessor) createTables() error {
	queries := []string{
		`CREATE TABLE IF NOT EXISTS feedback (
			id TEXT PRIMARY KEY,
			user_id TEXT NOT NULL,
			session_id TEXT NOT NULL,
			type INTEGER NOT NULL,
			category INTEGER NOT NULL,
			rating INTEGER CHECK(rating >= 1 AND rating <= 5),
			message TEXT,
			context TEXT,
			metadata TEXT,
			query_hash TEXT,
			response_hash TEXT,
			model_used TEXT,
			response_time INTEGER,
			tokens_used INTEGER,
			learning_seed BOOLEAN DEFAULT FALSE,
			priority INTEGER DEFAULT 0,
			validated_by TEXT,
			processed BOOLEAN DEFAULT FALSE,
			processed_at TIMESTAMP,
			timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			INDEX(user_id),
			INDEX(session_id),
			INDEX(type),
			INDEX(category),
			INDEX(processed),
			INDEX(timestamp)
		)`,

		`CREATE TABLE IF NOT EXISTS processed_feedback (
			id TEXT PRIMARY KEY,
			feedback_id TEXT NOT NULL,
			learning_signal INTEGER NOT NULL,
			confidence REAL NOT NULL,
			actions TEXT,
			metadata TEXT,
			processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY(feedback_id) REFERENCES feedback(id),
			INDEX(feedback_id),
			INDEX(learning_signal),
			INDEX(processed_at)
		)`,

		`CREATE TABLE IF NOT EXISTS user_profiles (
			user_id TEXT PRIMARY KEY,
			experience TEXT NOT NULL,
			preferences TEXT,
			feedback_history TEXT,
			reliability_score REAL DEFAULT 0.5,
			expertise_areas TEXT,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,

		`CREATE TABLE IF NOT EXISTS learning_rules (
			id TEXT PRIMARY KEY,
			name TEXT NOT NULL,
			description TEXT,
			conditions TEXT NOT NULL,
			actions TEXT NOT NULL,
			priority INTEGER DEFAULT 0,
			enabled BOOLEAN DEFAULT TRUE,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,

		`CREATE TABLE IF NOT EXISTS feedback_metrics (
			date TEXT PRIMARY KEY,
			total_feedback INTEGER DEFAULT 0,
			by_type TEXT,
			by_category TEXT,
			average_rating REAL DEFAULT 0,
			rating_distribution TEXT,
			processing_rate REAL DEFAULT 0,
			user_satisfaction REAL DEFAULT 0,
			calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
	}

	for _, query := range queries {
		if _, err := fp.db.Exec(query); err != nil {
			return fmt.Errorf("failed to execute query %s: %w", query, err)
		}
	}

	return nil
}

// SubmitFeedback submits new feedback for processing
func (fp *FeedbackProcessor) SubmitFeedback(entry *FeedbackEntry) error {
	if entry == nil {
		return fmt.Errorf("feedback entry cannot be nil")
	}

	// Generate ID if not provided
	if entry.ID == "" {
		entry.ID = fp.generateFeedbackID(entry)
	}

	// Set timestamp if not provided
	if entry.Timestamp.IsZero() {
		entry.Timestamp = time.Now()
	}

	// Validate entry
	if err := fp.validateFeedbackEntry(entry); err != nil {
		return fmt.Errorf("invalid feedback entry: %w", err)
	}

	// Store in database
	if err := fp.storeFeedback(entry); err != nil {
		return fmt.Errorf("failed to store feedback: %w", err)
	}

	// Add to processing queue
	select {
	case fp.processingQueue <- entry:
		// Successfully queued
	default:
		// Queue is full, process immediately or log warning
		go fp.processFeedbackEntry(entry)
	}

	return nil
}

// generateFeedbackID generates a unique ID for feedback entry
func (fp *FeedbackProcessor) generateFeedbackID(entry *FeedbackEntry) string {
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("%s-%s-%s-%d-%s",
		entry.UserID, entry.SessionID, entry.Message, entry.Type, entry.Timestamp.String())))
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// validateFeedbackEntry validates a feedback entry
func (fp *FeedbackProcessor) validateFeedbackEntry(entry *FeedbackEntry) error {
	if entry.UserID == "" {
		return fmt.Errorf("user ID is required")
	}

	if entry.SessionID == "" {
		return fmt.Errorf("session ID is required")
	}

	if entry.Rating < 1 || entry.Rating > 5 {
		return fmt.Errorf("rating must be between 1 and 5")
	}

	return nil
}

// storeFeedback stores feedback in the database
func (fp *FeedbackProcessor) storeFeedback(entry *FeedbackEntry) error {
	contextJSON, _ := json.Marshal(entry.Context)
	metadataJSON, _ := json.Marshal(entry.Metadata)

	query := `
		INSERT INTO feedback (
			id, user_id, session_id, type, category, rating, message,
			context, metadata, query_hash, response_hash, model_used,
			response_time, tokens_used, learning_seed, priority,
			validated_by, timestamp
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`

	_, err := fp.db.Exec(query,
		entry.ID, entry.UserID, entry.SessionID, int(entry.Type), int(entry.Category),
		entry.Rating, entry.Message, string(contextJSON), string(metadataJSON),
		entry.QueryHash, entry.ResponseHash, entry.ModelUsed,
		int64(entry.ResponseTime), entry.TokensUsed, entry.LearningSeed,
		entry.Priority, entry.ValidatedBy, entry.Timestamp,
	)

	return err
}

// processFeedbackEntry processes a single feedback entry
func (fp *FeedbackProcessor) processFeedbackEntry(entry *FeedbackEntry) *ProcessedFeedback {
	processed := &ProcessedFeedback{
		FeedbackID:  entry.ID,
		Actions:     make([]LearningAction, 0),
		Metadata:    make(map[string]interface{}),
		ProcessedAt: time.Now(),
	}

	// Update user profile
	fp.updateUserProfile(entry)

	// Analyze feedback for learning signals
	signal, confidence := fp.analyzeLearningSignal(entry)
	processed.LearningSignal = signal
	processed.Confidence = confidence

	// Generate learning actions
	actions := fp.generateLearningActions(entry, signal, confidence)
	processed.Actions = actions

	// Apply learning rules
	ruleActions := fp.applyLearningRules(entry)
	processed.Actions = append(processed.Actions, ruleActions...)

	// Store processed feedback
	fp.storeProcessedFeedback(processed)

	// Mark original feedback as processed
	fp.markFeedbackProcessed(entry.ID)

	// Send to processed signals channel
	select {
	case fp.processedSignals <- *processed:
	default:
		// Channel full, continue processing
	}

	return processed
}

// analyzeLearningSignal analyzes feedback to determine learning signals
func (fp *FeedbackProcessor) analyzeLearningSignal(entry *FeedbackEntry) (LearningSignalType, float64) {
	confidence := 0.5

	// Analyze based on feedback type and rating
	switch entry.Type {
	case FeedbackNegative:
		if entry.Rating <= 2 {
			confidence = 0.8
			if entry.Category == CategoryAccuracy {
				return SignalModelUpdate, confidence
			} else if entry.Category == CategoryPatterns {
				return SignalPatternUpdate, confidence
			}
		}
		return SignalErrorCorrection, confidence

	case FeedbackCorrection:
		confidence = 0.9
		if entry.Category == CategoryCodeAnalysis {
			return SignalModelUpdate, confidence
		}
		return SignalErrorCorrection, confidence

	case FeedbackPositive:
		if entry.Rating >= 4 {
			confidence = 0.7
			return SignalDataAugmentation, confidence
		}

	case FeedbackSuggestion:
		confidence = 0.6
		if strings.Contains(strings.ToLower(entry.Message), "parameter") {
			return SignalParameterTuning, confidence
		} else if strings.Contains(strings.ToLower(entry.Message), "feature") {
			return SignalFeatureToggle, confidence
		}

	case FeedbackUsability:
		confidence = 0.7
		return SignalPreferenceUpdate, confidence
	}

	// Context-based analysis
	if entry.Context.ConfidenceScore < 0.5 {
		confidence += 0.2
		return SignalModelUpdate, confidence
	}

	return SignalContextEnhancement, confidence
}

// generateLearningActions generates specific learning actions
func (fp *FeedbackProcessor) generateLearningActions(entry *FeedbackEntry, signal LearningSignalType, confidence float64) []LearningAction {
	var actions []LearningAction

	switch signal {
	case SignalModelUpdate:
		actions = append(actions, LearningAction{
			Type:   "model_fine_tune",
			Target: entry.ModelUsed,
			Parameters: map[string]interface{}{
				"feedback_id":     entry.ID,
				"correction_type": entry.Category,
				"training_weight": confidence,
			},
			Priority:       fp.calculateActionPriority(signal, confidence),
			Confidence:     confidence,
			ExpectedImpact: confidence * 0.8,
		})

	case SignalPatternUpdate:
		actions = append(actions, LearningAction{
			Type:   "pattern_refinement",
			Target: entry.Context.AnalysisType,
			Parameters: map[string]interface{}{
				"patterns":        entry.Context.PatternsFound,
				"feedback_rating": entry.Rating,
				"context":         entry.Context.CodeSnippet,
			},
			Priority:       fp.calculateActionPriority(signal, confidence),
			Confidence:     confidence,
			ExpectedImpact: confidence * 0.6,
		})

	case SignalParameterTuning:
		actions = append(actions, LearningAction{
			Type:   "parameter_adjustment",
			Target: "analysis_engine",
			Parameters: map[string]interface{}{
				"suggestion":      entry.Message,
				"current_params":  entry.Context.Parameters,
				"user_experience": entry.Context.UserExperience,
			},
			Priority:       fp.calculateActionPriority(signal, confidence),
			Confidence:     confidence,
			ExpectedImpact: confidence * 0.4,
		})

	case SignalPreferenceUpdate:
		actions = append(actions, LearningAction{
			Type:   "user_preference_update",
			Target: entry.UserID,
			Parameters: map[string]interface{}{
				"preference_category": entry.Category,
				"feedback_message":    entry.Message,
				"rating":              entry.Rating,
			},
			Priority:       fp.calculateActionPriority(signal, confidence),
			Confidence:     confidence,
			ExpectedImpact: confidence * 0.3,
		})
	}

	return actions
}

// calculateActionPriority calculates priority for learning actions
func (fp *FeedbackProcessor) calculateActionPriority(signal LearningSignalType, confidence float64) int {
	basePriority := map[LearningSignalType]int{
		SignalModelUpdate:        100,
		SignalErrorCorrection:    90,
		SignalPatternUpdate:      80,
		SignalParameterTuning:    70,
		SignalDataAugmentation:   60,
		SignalFeatureToggle:      50,
		SignalPreferenceUpdate:   40,
		SignalContextEnhancement: 30,
	}

	priority := basePriority[signal]
	return priority + int(confidence*20)
}

// updateUserProfile updates user profile based on feedback
func (fp *FeedbackProcessor) updateUserProfile(entry *FeedbackEntry) {
	fp.mu.Lock()
	defer fp.mu.Unlock()

	profile, exists := fp.userProfiles[entry.UserID]
	if !exists {
		profile = &UserProfile{
			UserID:           entry.UserID,
			Experience:       entry.Context.UserExperience,
			Preferences:      make(map[string]interface{}),
			FeedbackHistory:  make([]string, 0),
			ReliabilityScore: 0.5,
			ExpertiseAreas:   make([]string, 0),
			CreatedAt:        time.Now(),
		}
		fp.userProfiles[entry.UserID] = profile
	}

	// Update feedback history
	profile.FeedbackHistory = append(profile.FeedbackHistory, entry.ID)
	if len(profile.FeedbackHistory) > 100 {
		profile.FeedbackHistory = profile.FeedbackHistory[1:]
	}

	// Update reliability score based on feedback quality
	fp.updateReliabilityScore(profile, entry)

	// Update expertise areas
	if entry.Context.Language != "" {
		fp.addExpertiseArea(profile, entry.Context.Language)
	}

	// Update preferences based on feedback
	fp.updateUserPreferences(profile, entry)

	profile.UpdatedAt = time.Now()

	// Save to database
	fp.saveUserProfile(profile)
}

// updateReliabilityScore updates user reliability score
func (fp *FeedbackProcessor) updateReliabilityScore(profile *UserProfile, entry *FeedbackEntry) {
	weight := 0.1 // Learning rate

	// Positive adjustment for detailed feedback
	if len(entry.Message) > 50 {
		profile.ReliabilityScore += weight * 0.1
	}

	// Positive adjustment for corrections
	if entry.Type == FeedbackCorrection {
		profile.ReliabilityScore += weight * 0.2
	}

	// Negative adjustment for very short feedback
	if len(entry.Message) < 10 && entry.Type != FeedbackPositive {
		profile.ReliabilityScore -= weight * 0.1
	}

	// Bound between 0 and 1
	if profile.ReliabilityScore > 1.0 {
		profile.ReliabilityScore = 1.0
	} else if profile.ReliabilityScore < 0.0 {
		profile.ReliabilityScore = 0.0
	}
}

// addExpertiseArea adds expertise area if not already present
func (fp *FeedbackProcessor) addExpertiseArea(profile *UserProfile, area string) {
	for _, existing := range profile.ExpertiseAreas {
		if existing == area {
			return
		}
	}
	profile.ExpertiseAreas = append(profile.ExpertiseAreas, area)
}

// updateUserPreferences updates user preferences
func (fp *FeedbackProcessor) updateUserPreferences(profile *UserProfile, entry *FeedbackEntry) {
	if profile.Preferences == nil {
		profile.Preferences = make(map[string]interface{})
	}

	// Update category preferences based on ratings
	categoryKey := fmt.Sprintf("category_%d", int(entry.Category))
	if current, exists := profile.Preferences[categoryKey]; exists {
		if currentRating, ok := current.(float64); ok {
			// Moving average
			profile.Preferences[categoryKey] = (currentRating*0.8 + float64(entry.Rating)*0.2)
		}
	} else {
		profile.Preferences[categoryKey] = float64(entry.Rating)
	}
}

// applyLearningRules applies configured learning rules
func (fp *FeedbackProcessor) applyLearningRules(entry *FeedbackEntry) []LearningAction {
	var actions []LearningAction

	fp.mu.RLock()
	rules := make([]LearningRule, len(fp.learningRules))
	copy(rules, fp.learningRules)
	fp.mu.RUnlock()

	// Sort rules by priority
	sort.Slice(rules, func(i, j int) bool {
		return rules[i].Priority > rules[j].Priority
	})

	for _, rule := range rules {
		if !rule.Enabled {
			continue
		}

		if fp.evaluateRuleConditions(rule.Conditions, entry) {
			actions = append(actions, rule.Actions...)
		}
	}

	return actions
}

// evaluateRuleConditions evaluates rule conditions against feedback entry
func (fp *FeedbackProcessor) evaluateRuleConditions(conditions []RuleCondition, entry *FeedbackEntry) bool {
	for _, condition := range conditions {
		if !fp.evaluateCondition(condition, entry) {
			return false
		}
	}
	return true
}

// evaluateCondition evaluates a single condition
func (fp *FeedbackProcessor) evaluateCondition(condition RuleCondition, entry *FeedbackEntry) bool {
	var fieldValue interface{}

	// Extract field value from entry
	switch condition.Field {
	case "type":
		fieldValue = int(entry.Type)
	case "category":
		fieldValue = int(entry.Category)
	case "rating":
		fieldValue = entry.Rating
	case "user_experience":
		fieldValue = entry.Context.UserExperience
	case "model_used":
		fieldValue = entry.ModelUsed
	case "confidence_score":
		fieldValue = entry.Context.ConfidenceScore
	default:
		return false
	}

	// Apply operator
	switch condition.Operator {
	case "eq":
		return fieldValue == condition.Value
	case "ne":
		return fieldValue != condition.Value
	case "gt":
		if fv, ok := fieldValue.(int); ok {
			if cv, ok := condition.Value.(float64); ok {
				return float64(fv) > cv
			}
		}
		if fv, ok := fieldValue.(float64); ok {
			if cv, ok := condition.Value.(float64); ok {
				return fv > cv
			}
		}
	case "lt":
		if fv, ok := fieldValue.(int); ok {
			if cv, ok := condition.Value.(float64); ok {
				return float64(fv) < cv
			}
		}
		if fv, ok := fieldValue.(float64); ok {
			if cv, ok := condition.Value.(float64); ok {
				return fv < cv
			}
		}
	case "contains":
		if fv, ok := fieldValue.(string); ok {
			if cv, ok := condition.Value.(string); ok {
				return strings.Contains(strings.ToLower(fv), strings.ToLower(cv))
			}
		}
	}

	return false
}

// Background processing functions

// startProcessors starts background processors
func (fp *FeedbackProcessor) startProcessors() {
	fp.wg.Add(3)

	// Feedback processing worker
	go func() {
		defer fp.wg.Done()
		fp.feedbackProcessor()
	}()

	// Metrics calculator
	go func() {
		defer fp.wg.Done()
		fp.metricsCalculator()
	}()

	// Cleanup worker
	go func() {
		defer fp.wg.Done()
		fp.cleanupWorker()
	}()
}

// feedbackProcessor processes queued feedback entries
func (fp *FeedbackProcessor) feedbackProcessor() {
	ticker := time.NewTicker(fp.processingInterval)
	defer ticker.Stop()

	batch := make([]*FeedbackEntry, 0, fp.batchSize)

	for {
		select {
		case <-fp.ctx.Done():
			// Process remaining batch
			if len(batch) > 0 {
				fp.processBatch(batch)
			}
			return

		case entry := <-fp.processingQueue:
			batch = append(batch, entry)

			if len(batch) >= fp.batchSize {
				fp.processBatch(batch)
				batch = make([]*FeedbackEntry, 0, fp.batchSize)
			}

		case <-ticker.C:
			if len(batch) > 0 {
				fp.processBatch(batch)
				batch = make([]*FeedbackEntry, 0, fp.batchSize)
			}
		}
	}
}

// processBatch processes a batch of feedback entries
func (fp *FeedbackProcessor) processBatch(batch []*FeedbackEntry) {
	for _, entry := range batch {
		fp.processFeedbackEntry(entry)
	}
}

// metricsCalculator periodically calculates metrics
func (fp *FeedbackProcessor) metricsCalculator() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-fp.ctx.Done():
			return
		case <-ticker.C:
			if err := fp.calculateMetrics(); err != nil {
				// Log error but continue
				continue
			}
		}
	}
}

// cleanupWorker performs periodic cleanup
func (fp *FeedbackProcessor) cleanupWorker() {
	ticker := time.NewTicker(24 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-fp.ctx.Done():
			return
		case <-ticker.C:
			fp.performCleanup()
		}
	}
}

// Additional utility methods

// loadUserProfiles loads user profiles from database
func (fp *FeedbackProcessor) loadUserProfiles() error {
	query := `SELECT user_id, experience, preferences, feedback_history, 
	          reliability_score, expertise_areas, created_at, updated_at FROM user_profiles`

	rows, err := fp.db.Query(query)
	if err != nil {
		return err
	}
	defer rows.Close()

	fp.mu.Lock()
	defer fp.mu.Unlock()

	for rows.Next() {
		var profile UserProfile
		var preferencesJSON, historyJSON, expertiseJSON string

		err := rows.Scan(&profile.UserID, &profile.Experience, &preferencesJSON,
			&historyJSON, &profile.ReliabilityScore, &expertiseJSON,
			&profile.CreatedAt, &profile.UpdatedAt)
		if err != nil {
			continue
		}

		// Parse JSON fields
		json.Unmarshal([]byte(preferencesJSON), &profile.Preferences)
		json.Unmarshal([]byte(historyJSON), &profile.FeedbackHistory)
		json.Unmarshal([]byte(expertiseJSON), &profile.ExpertiseAreas)

		fp.userProfiles[profile.UserID] = &profile
	}

	return nil
}

// loadLearningRules loads learning rules from database
func (fp *FeedbackProcessor) loadLearningRules() error {
	query := `SELECT id, name, description, conditions, actions, priority, enabled, created_at FROM learning_rules WHERE enabled = 1`

	rows, err := fp.db.Query(query)
	if err != nil {
		return err
	}
	defer rows.Close()

	fp.mu.Lock()
	defer fp.mu.Unlock()

	for rows.Next() {
		var rule LearningRule
		var conditionsJSON, actionsJSON string

		err := rows.Scan(&rule.ID, &rule.Name, &rule.Description, &conditionsJSON,
			&actionsJSON, &rule.Priority, &rule.Enabled, &rule.CreatedAt)
		if err != nil {
			continue
		}

		// Parse JSON fields
		json.Unmarshal([]byte(conditionsJSON), &rule.Conditions)
		json.Unmarshal([]byte(actionsJSON), &rule.Actions)

		fp.learningRules = append(fp.learningRules, rule)
	}

	return nil
}

// saveUserProfile saves user profile to database
func (fp *FeedbackProcessor) saveUserProfile(profile *UserProfile) error {
	preferencesJSON, _ := json.Marshal(profile.Preferences)
	historyJSON, _ := json.Marshal(profile.FeedbackHistory)
	expertiseJSON, _ := json.Marshal(profile.ExpertiseAreas)

	query := `INSERT OR REPLACE INTO user_profiles 
	          (user_id, experience, preferences, feedback_history, reliability_score, 
	           expertise_areas, created_at, updated_at) 
	          VALUES (?, ?, ?, ?, ?, ?, ?, ?)`

	_, err := fp.db.Exec(query, profile.UserID, profile.Experience,
		string(preferencesJSON), string(historyJSON), profile.ReliabilityScore,
		string(expertiseJSON), profile.CreatedAt, profile.UpdatedAt)

	return err
}

// storeProcessedFeedback stores processed feedback
func (fp *FeedbackProcessor) storeProcessedFeedback(processed *ProcessedFeedback) error {
	actionsJSON, _ := json.Marshal(processed.Actions)
	metadataJSON, _ := json.Marshal(processed.Metadata)

	query := `INSERT INTO processed_feedback 
	          (id, feedback_id, learning_signal, confidence, actions, metadata, processed_at)
	          VALUES (?, ?, ?, ?, ?, ?, ?)`

	_, err := fp.db.Exec(query, fp.generateID(), processed.FeedbackID,
		int(processed.LearningSignal), processed.Confidence,
		string(actionsJSON), string(metadataJSON), processed.ProcessedAt)

	return err
}

// markFeedbackProcessed marks feedback as processed
func (fp *FeedbackProcessor) markFeedbackProcessed(feedbackID string) error {
	query := `UPDATE feedback SET processed = 1, processed_at = ? WHERE id = ?`
	_, err := fp.db.Exec(query, time.Now(), feedbackID)
	return err
}

// calculateMetrics calculates feedback metrics
func (fp *FeedbackProcessor) calculateMetrics() error {
	fp.mu.Lock()
	defer fp.mu.Unlock()

	metrics := &FeedbackMetrics{
		ByType:             make(map[FeedbackType]int),
		ByCategory:         make(map[FeedbackCategory]int),
		RatingDistribution: make(map[int]int),
		DailyFeedback:      make(map[string]int),
		WeeklyTrends:       make(map[string]float64),
		CategoryTrends:     make(map[FeedbackCategory]float64),
		FeatureUsability:   make(map[string]float64),
	}

	// Query feedback data
	query := `SELECT type, category, rating, DATE(timestamp) as date, processed
	          FROM feedback WHERE timestamp >= datetime('now', '-30 days')`

	rows, err := fp.db.Query(query)
	if err != nil {
		return err
	}
	defer rows.Close()

	var totalRating int
	var processedCount int

	for rows.Next() {
		var feedbackType, category, rating int
		var date string
		var processed bool

		err := rows.Scan(&feedbackType, &category, &rating, &date, &processed)
		if err != nil {
			continue
		}

		metrics.TotalFeedback++
		metrics.ByType[FeedbackType(feedbackType)]++
		metrics.ByCategory[FeedbackCategory(category)]++
		metrics.RatingDistribution[rating]++
		metrics.DailyFeedback[date]++

		totalRating += rating
		if processed {
			processedCount++
		}
	}

	// Calculate averages and rates
	if metrics.TotalFeedback > 0 {
		metrics.AverageRating = float64(totalRating) / float64(metrics.TotalFeedback)
		metrics.ProcessingRate = float64(processedCount) / float64(metrics.TotalFeedback)

		// User satisfaction (based on ratings 4-5)
		satisfiedCount := metrics.RatingDistribution[4] + metrics.RatingDistribution[5]
		metrics.UserSatisfaction = float64(satisfiedCount) / float64(metrics.TotalFeedback) * 100
	}

	fp.metrics = metrics
	return nil
}

// generateID generates a unique ID
func (fp *FeedbackProcessor) generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

// performCleanup performs periodic cleanup
func (fp *FeedbackProcessor) performCleanup() {
	// Remove old feedback beyond retention period
	query := `DELETE FROM feedback WHERE timestamp < datetime('now', '-' || ? || ' seconds')`
	fp.db.Exec(query, int(fp.retentionPeriod.Seconds()))

	// Remove orphaned processed feedback
	query = `DELETE FROM processed_feedback WHERE feedback_id NOT IN (SELECT id FROM feedback)`
	fp.db.Exec(query)
}

// Public API methods

// GetMetrics returns current feedback metrics
func (fp *FeedbackProcessor) GetMetrics() *FeedbackMetrics {
	fp.mu.RLock()
	defer fp.mu.RUnlock()
	return fp.metrics
}

// GetUserProfile returns user profile
func (fp *FeedbackProcessor) GetUserProfile(userID string) (*UserProfile, bool) {
	fp.mu.RLock()
	defer fp.mu.RUnlock()

	profile, exists := fp.userProfiles[userID]
	return profile, exists
}

// GetProcessedSignals returns channel for processed feedback signals
func (fp *FeedbackProcessor) GetProcessedSignals() <-chan ProcessedFeedback {
	return fp.processedSignals
}

// AddLearningRule adds a new learning rule
func (fp *FeedbackProcessor) AddLearningRule(rule LearningRule) error {
	fp.mu.Lock()
	defer fp.mu.Unlock()

	fp.learningRules = append(fp.learningRules, rule)

	// Save to database
	conditionsJSON, _ := json.Marshal(rule.Conditions)
	actionsJSON, _ := json.Marshal(rule.Actions)

	query := `INSERT INTO learning_rules 
	          (id, name, description, conditions, actions, priority, enabled, created_at)
	          VALUES (?, ?, ?, ?, ?, ?, ?, ?)`

	_, err := fp.db.Exec(query, rule.ID, rule.Name, rule.Description,
		string(conditionsJSON), string(actionsJSON), rule.Priority, rule.Enabled, rule.CreatedAt)

	return err
}

// Close closes the feedback processor
func (fp *FeedbackProcessor) Close() error {
	fp.cancel()
	fp.wg.Wait()

	if fp.db != nil {
		return fp.db.Close()
	}

	return nil
}
