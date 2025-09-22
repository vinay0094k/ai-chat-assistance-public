// storage/storage_manager.go - Complete Storage System Integration
package storage

import (
	"context"
	"database/sql"
	"fmt"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/intelligence"
	"github.com/yourusername/ai-code-assistant/internal/learning"
	"github.com/yourusername/ai-code-assistant/tracking"
)

// StorageManagerV2 orchestrates all storage components with intelligence compatibility
type StorageManagerV2 struct {
	// Core database connection
	db *sql.DB

	// Storage components (updated for compatibility)
	cacheStorage        *MultiLevelCache
	graphStorage        *GraphStorageV2 // Updated to work with SemanticEntity/SemanticRelationship
	patternStorage      *PatternStorage
	performanceStorage  *PerformanceStorage
	relationshipStorage *RelationshipStorage

	// Specialized caches
	contextCache  *ContextCache
	patternCache  *PatternCache
	semanticCache *SemanticCache

	// Configuration
	config StorageConfig
	mutex  sync.RWMutex
}

// NewStorageManagerV2 creates a storage system compatible with existing intelligence types
func NewStorageManagerV2(db *sql.DB, config StorageConfig) (*StorageManagerV2, error) {
	// Initialize storage components with compatibility layer
	graphStorage, err := NewGraphStorageV2(db) // Use V2 that works with your types
	if err != nil {
		return nil, fmt.Errorf("failed to initialize graph storage: %w", err)
	}

	patternStorage, err := NewPatternStorage(db)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize pattern storage: %w", err)
	}

	performanceStorage, err := NewPerformanceStorage(db)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize performance storage: %w", err)
	}

	relationshipStorage, err := NewRelationshipStorage(db)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize relationship storage: %w", err)
	}

	// Initialize cache system
	cacheConfig := CacheConfig{
		MaxSize:         config.CacheMaxSize,
		MaxEntries:      10000,
		DefaultTTL:      config.CacheDefaultTTL,
		EvictionPolicy:  LRU,
		CleanupInterval: 5 * time.Minute,
		PersistentMode:  true,
	}

	cacheStorage := NewMultiLevelCache(cacheConfig, nil)
	contextCache := NewContextCache()
	patternCache := NewPatternCache()
	semanticCache := NewSemanticCache()

	sm := &StorageManagerV2{
		db:                  db,
		graphStorage:        graphStorage,
		patternStorage:      patternStorage,
		performanceStorage:  performanceStorage,
		relationshipStorage: relationshipStorage,
		cacheStorage:        cacheStorage,
		contextCache:        contextCache,
		patternCache:        patternCache,
		semanticCache:       semanticCache,
		config:              config,
	}

	// Start maintenance routines
	go sm.maintenanceRoutine()

	return sm, nil
}

// === Intelligence System Integration (Compatible with your types) ===

// StoreSemanticAnalysis stores complete semantic analysis using your existing types
func (sm *StorageManagerV2) StoreSemanticAnalysis(ctx context.Context, graph *intelligence.SemanticGraph, projectPath string) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	// Store the semantic graph directly (no conversion needed)
	if err := sm.graphStorage.StoreSemanticGraph(ctx, graph, projectPath); err != nil {
		return fmt.Errorf("failed to store semantic graph: %w", err)
	}

	// Cache the graph for quick access
	sm.semanticCache.SetSemanticGraph(projectPath, graph)

	// Store individual entities in relationship storage for cross-referencing
	for _, entity := range graph.Entities {
		relationshipEntity := sm.convertToRelationshipEntity(entity)
		if err := sm.relationshipStorage.StoreEntity(ctx, relationshipEntity); err != nil {
			// Log error but continue - this is supplementary storage
			fmt.Printf("Warning: failed to store entity in relationship storage: %v\n", err)
		}
	}

	// Store relationships in relationship storage
	for _, relationship := range graph.Relationships {
		relationshipRel := sm.convertToRelationshipRelationship(&relationship)
		if err := sm.relationshipStorage.StoreRelationship(ctx, relationshipRel); err != nil {
			fmt.Printf("Warning: failed to store relationship in relationship storage: %v\n", err)
		}
	}

	return nil
}

// LoadSemanticAnalysis loads semantic analysis using your types
func (sm *StorageManagerV2) LoadSemanticAnalysis(ctx context.Context, projectPath string) (*intelligence.SemanticGraph, error) {
	// Try cache first
	if graph, found := sm.semanticCache.GetSemanticGraph(projectPath); found {
		return graph, nil
	}

	// Load from storage
	graph, err := sm.graphStorage.LoadSemanticGraph(ctx, projectPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load semantic graph: %w", err)
	}

	// Cache for future use
	sm.semanticCache.SetSemanticGraph(projectPath, graph)

	return graph, nil
}

// === Pattern Analysis Integration (Compatible with your design pattern types) ===

// StoreDesignPatterns stores design patterns using your DesignPattern types
func (sm *StorageManagerV2) StoreDesignPatterns(ctx context.Context, patterns []intelligence.DesignPattern, projectPath string) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	for _, pattern := range patterns {
		// Convert your DesignPattern to learning.CodePattern for storage
		codePattern := &learning.CodePattern{
			ID:          fmt.Sprintf("pattern_%s_%d", projectPath, int(pattern.Type)),
			Name:        pattern.Name,
			Description: pattern.Description,
			Language:    "multi", // Your patterns are language-agnostic
			Confidence:  pattern.Confidence,
			ProjectPath: projectPath,
			Examples:    make([]string, 0),
			Metadata:    make(map[string]interface{}),
		}

		// Add pattern-specific metadata
		codePattern.Metadata["design_pattern_type"] = int(pattern.Type)
		codePattern.Metadata["category"] = pattern.Category
		codePattern.Metadata["suggestion_count"] = len(pattern.Suggestions)

		// Convert occurrences to examples
		for _, occurrence := range pattern.Occurrences {
			codePattern.Examples = append(codePattern.Examples, occurrence.Code)
		}

		// Store pattern
		if _, err := sm.patternStorage.StorePattern(ctx, codePattern); err != nil {
			return fmt.Errorf("failed to store design pattern %s: %w", pattern.Name, err)
		}

		// Cache pattern
		sm.patternCache.SetPattern(codePattern.ID, codePattern)
	}

	return nil
}

// StoreArchitecturePatterns stores architecture patterns
func (sm *StorageManagerV2) StoreArchitecturePatterns(ctx context.Context, patterns []intelligence.ArchitectureInfo, projectPath string) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	for _, pattern := range patterns {
		codePattern := &learning.CodePattern{
			ID:          fmt.Sprintf("arch_pattern_%s_%d", projectPath, int(pattern.Pattern)),
			Name:        pattern.Name,
			Description: pattern.Description,
			Language:    "architecture",
			Confidence:  pattern.Confidence,
			ProjectPath: projectPath,
			Examples:    pattern.Evidence, // Evidence as examples
			Metadata:    make(map[string]interface{}),
		}

		codePattern.Metadata["architecture_pattern_type"] = int(pattern.Pattern)
		codePattern.Metadata["evidence_count"] = len(pattern.Evidence)
		codePattern.Metadata["suggestion_count"] = len(pattern.Suggestions)

		if _, err := sm.patternStorage.StorePattern(ctx, codePattern); err != nil {
			return fmt.Errorf("failed to store architecture pattern %s: %w", pattern.Name, err)
		}
	}

	return nil
}

// === Query Methods Compatible with Your Types ===

// GetSemanticEntitiesByType gets entities using your EntityType enum
func (sm *StorageManagerV2) GetSemanticEntitiesByType(ctx context.Context, entityType intelligence.EntityType, projectPath string) ([]*intelligence.SemanticEntity, error) {
	return sm.graphStorage.GetSemanticEntitiesByType(ctx, entityType, projectPath)
}

// GetDesignPatternsByProject retrieves design patterns for a project
func (sm *StorageManagerV2) GetDesignPatternsByProject(ctx context.Context, projectPath string) ([]*StoredPattern, error) {
	query := &PatternQuery{
		ProjectPaths: []string{projectPath},
		SortBy:       "confidence",
		SortOrder:    "desc",
		Limit:        1000,
	}

	patterns, err := sm.patternStorage.QueryPatterns(ctx, query)
	if err != nil {
		return nil, err
	}

	// Filter for design patterns (vs architecture patterns)
	designPatterns := make([]*StoredPattern, 0)
	for _, pattern := range patterns {
		if pattern.Language != "architecture" {
			designPatterns = append(designPatterns, pattern)
		}
	}

	return designPatterns, nil
}

// GetArchitecturePattersByProject retrieves architecture patterns for a project
func (sm *StorageManagerV2) GetArchitecturePattersByProject(ctx context.Context, projectPath string) ([]*StoredPattern, error) {
	query := &PatternQuery{
		ProjectPaths: []string{projectPath},
		Languages:    []string{"architecture"},
		SortBy:       "confidence",
		SortOrder:    "desc",
		Limit:        1000,
	}

	return sm.patternStorage.QueryPatterns(ctx, query)
}

// === Performance Tracking Integration ===

// StorePerformanceMetrics stores performance data
func (sm *StorageManagerV2) StorePerformanceMetrics(ctx context.Context, metrics *tracking.PerformanceMetrics) error {
	if err := sm.performanceStorage.SaveMetrics(ctx, metrics); err != nil {
		return fmt.Errorf("failed to store performance metrics: %w", err)
	}

	// Cache recent performance data
	cacheKey := fmt.Sprintf("perf_metrics_%s", metrics.SessionID)
	sm.cacheStorage.Set(ctx, cacheKey, metrics, 30*time.Minute, "performance")

	return nil
}

// === Comprehensive Analysis Compatible with Your Types ===

// AnalyzeCodebaseV2 performs comprehensive analysis using your intelligence types
func (sm *StorageManagerV2) AnalyzeCodebaseV2(ctx context.Context, projectPath string) (*CodebaseAnalysisV2, error) {
	analysis := &CodebaseAnalysisV2{
		ProjectPath: projectPath,
		StartTime:   time.Now(),
	}

	// Get semantic analysis using your types
	semanticGraph, err := sm.LoadSemanticAnalysis(ctx, projectPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load semantic analysis: %w", err)
	}
	analysis.SemanticGraph = semanticGraph

	// Get design patterns
	designPatterns, err := sm.GetDesignPatternsByProject(ctx, projectPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get design patterns: %w", err)
	}
	analysis.DesignPatterns = designPatterns

	// Get architecture patterns
	archPatterns, err := sm.GetArchitecturePattersByProject(ctx, projectPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get architecture patterns: %w", err)
	}
	analysis.ArchitecturePatterns = archPatterns

	// Get performance insights
	insights, err := sm.GetPerformanceInsights(ctx, projectPath, 30*24*time.Hour)
	if err != nil {
		return nil, fmt.Errorf("failed to get performance insights: %w", err)
	}
	analysis.PerformanceInsights = insights

	// Calculate metrics
	analysis.Metrics = sm.calculateCodebaseMetricsV2(analysis)
	analysis.CompletedAt = time.Now()
	analysis.Duration = analysis.CompletedAt.Sub(analysis.StartTime)

	// Cache the analysis
	cacheKey := fmt.Sprintf("codebase_analysis_v2_%s", projectPath)
	sm.cacheStorage.Set(ctx, cacheKey, analysis, 2*time.Hour, "analysis")

	return analysis, nil
}

// CodebaseAnalysisV2 represents analysis results using your intelligence types
type CodebaseAnalysisV2 struct {
	ProjectPath          string                      `json:"project_path"`
	SemanticGraph        *intelligence.SemanticGraph `json:"semantic_graph"`
	DesignPatterns       []*StoredPattern            `json:"design_patterns"`
	ArchitecturePatterns []*StoredPattern            `json:"architecture_patterns"`
	PerformanceInsights  []*PerformanceInsight       `json:"performance_insights"`
	Metrics              *CodebaseMetricsV2          `json:"metrics"`
	StartTime            time.Time                   `json:"start_time"`
	CompletedAt          time.Time                   `json:"completed_at"`
	Duration             time.Duration               `json:"duration"`
}

// CodebaseMetricsV2 provides metrics based on your intelligence types
type CodebaseMetricsV2 struct {
	TotalEntities       int                             `json:"total_entities"`
	TotalRelationships  int                             `json:"total_relationships"`
	TotalDesignPatterns int                             `json:"total_design_patterns"`
	TotalArchPatterns   int                             `json:"total_architecture_patterns"`
	AverageComplexity   float64                         `json:"average_complexity"`
	TechnicalDebt       float64                         `json:"technical_debt"`
	Maintainability     float64                         `json:"maintainability"`
	ArchitectureHealth  float64                         `json:"architecture_health"`
	EntityTypeBreakdown map[intelligence.EntityType]int `json:"entity_type_breakdown"`
	LanguageBreakdown   map[string]int                  `json:"language_breakdown"`
	PatternBreakdown    map[string]int                  `json:"pattern_breakdown"`
	Recommendations     []string                        `json:"recommendations"`
}

// === Conversion Helper Methods ===

// convertToRelationshipEntity converts SemanticEntity to relationship storage format
func (sm *StorageManagerV2) convertToRelationshipEntity(entity *intelligence.SemanticEntity) *CodeEntity {
	return &CodeEntity{
		ID:            entity.ID,
		Type:          EntityType(entity.Type.String()), // Convert enum to string then back
		Name:          entity.Name,
		FullName:      entity.Package + "::" + entity.Name,
		FilePath:      entity.File,
		LineStart:     entity.Line,
		LineEnd:       entity.Line + 1, // Estimate
		Language:      entity.Language,
		Signature:     entity.Signature,
		Visibility:    VisibilityType(entity.Visibility.String()),
		IsStatic:      entity.IsStatic,
		IsAbstract:    entity.IsAbstract,
		Complexity:    entity.Complexity,
		Documentation: entity.Documentation,
		ProjectPath:   entity.Package, // Use package as project path
		CreatedAt:     entity.LastModified,
		UpdatedAt:     time.Now(),
	}
}

// convertToRelationshipRelationship converts SemanticRelationship to relationship storage format
func (sm *StorageManagerV2) convertToRelationshipRelationship(relationship *intelligence.SemanticRelationship) *EntityRelationship {
	return &EntityRelationship{
		ID:           fmt.Sprintf("%s->%s:%s", relationship.From, relationship.To, relationship.Type.String()),
		SourceID:     relationship.From,
		TargetID:     relationship.To,
		RelationType: RelationshipType(relationship.Type.String()),
		Strength:     relationship.Strength,
		Confidence:   0.9, // Default confidence
		Context:      relationship.Context,
		LineNumber:   relationship.Line,
		Properties:   make(map[string]interface{}),
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}
}

// calculateCodebaseMetricsV2 calculates metrics using your intelligence types
func (sm *StorageManagerV2) calculateCodebaseMetricsV2(analysis *CodebaseAnalysisV2) *CodebaseMetricsV2 {
	metrics := &CodebaseMetricsV2{
		TotalEntities:       len(analysis.SemanticGraph.Entities),
		TotalRelationships:  len(analysis.SemanticGraph.Relationships),
		TotalDesignPatterns: len(analysis.DesignPatterns),
		TotalArchPatterns:   len(analysis.ArchitecturePatterns),
		EntityTypeBreakdown: make(map[intelligence.EntityType]int),
		LanguageBreakdown:   make(map[string]int),
		PatternBreakdown:    make(map[string]int),
		Recommendations:     make([]string, 0),
	}

	// Calculate entity metrics
	totalComplexity := 0
	for _, entity := range analysis.SemanticGraph.Entities {
		totalComplexity += entity.Complexity
		metrics.EntityTypeBreakdown[entity.Type]++
		metrics.LanguageBreakdown[entity.Language]++
	}

	if len(analysis.SemanticGraph.Entities) > 0 {
		metrics.AverageComplexity = float64(totalComplexity) / float64(len(analysis.SemanticGraph.Entities))
	}

	// Pattern analysis
	for _, pattern := range analysis.DesignPatterns {
		metrics.PatternBreakdown[pattern.Category]++
	}
	for _, pattern := range analysis.ArchitecturePatterns {
		metrics.PatternBreakdown["Architecture: "+pattern.Category]++
	}

	// Calculate quality scores (simplified)
	if metrics.AverageComplexity < 10 {
		metrics.Maintainability = 0.8
	} else {
		metrics.Maintainability = 0.6
	}

	// Generate recommendations based on your intelligence analysis
	if metrics.AverageComplexity > 20 {
		metrics.Recommendations = append(metrics.Recommendations, "Consider refactoring high-complexity functions")
	}

	if metrics.TotalDesignPatterns == 0 {
		metrics.Recommendations = append(metrics.Recommendations, "Consider implementing design patterns to improve code organization")
	}

	if len(metrics.Recommendations) == 0 {
		metrics.Recommendations = append(metrics.Recommendations, "Codebase analysis looks good")
	}

	return metrics
}

// === Additional Helper Methods ===

func (sm *StorageManagerV2) GetPerformanceInsights(ctx context.Context, projectPath string, timeRange time.Duration) ([]*PerformanceInsight, error) {
	cacheKey := fmt.Sprintf("insights_%s_%v", projectPath, timeRange)
	if cached, found := sm.cacheStorage.Get(ctx, cacheKey); found {
		if insights, ok := cached.([]*PerformanceInsight); ok {
			return insights, nil
		}
	}

	insights, err := sm.performanceStorage.GenerateInsights(ctx, timeRange)
	if err != nil {
		return nil, fmt.Errorf("failed to generate insights: %w", err)
	}

	sm.cacheStorage.Set(ctx, cacheKey, insights, time.Hour, "insights", "performance")
	return insights, nil
}

// maintenanceRoutine runs periodic maintenance
func (sm *StorageManagerV2) maintenanceRoutine() {
	ticker := time.NewTicker(sm.config.MaintenanceInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			sm.performMaintenance()
		}
	}
}

func (sm *StorageManagerV2) performMaintenance() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	// Vacuum databases
	sm.db.ExecContext(ctx, "VACUUM")
	sm.db.ExecContext(ctx, "ANALYZE")

	// Clear analysis cache
	sm.mutex.Lock()
	// Clear any internal caches here
	sm.mutex.Unlock()
}

// Close gracefully closes all storage components
func (sm *StorageManagerV2) Close() error {
	sm.cacheStorage.Stop()
	return sm.db.Close()
}
