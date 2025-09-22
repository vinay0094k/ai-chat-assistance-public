// storage/relationship_storage.go - Manages storage of relationships between code entities
package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// CodeEntity represents a code entity (function, class, module, etc.)
type CodeEntity struct {
	ID            string                 `json:"id" db:"id"`
	Type          EntityType             `json:"type" db:"type"`
	Name          string                 `json:"name" db:"name"`
	FullName      string                 `json:"full_name" db:"full_name"` // package.class.method
	FilePath      string                 `json:"file_path" db:"file_path"`
	LineStart     int                    `json:"line_start" db:"line_start"`
	LineEnd       int                    `json:"line_end" db:"line_end"`
	Language      string                 `json:"language" db:"language"`
	Signature     string                 `json:"signature" db:"signature"`
	Visibility    VisibilityType         `json:"visibility" db:"visibility"` // public, private, protected
	IsStatic      bool                   `json:"is_static" db:"is_static"`
	IsAbstract    bool                   `json:"is_abstract" db:"is_abstract"`
	Complexity    int                    `json:"complexity" db:"complexity"`
	LinesOfCode   int                    `json:"lines_of_code" db:"lines_of_code"`
	Documentation string                 `json:"documentation" db:"documentation"`
	Tags          []string               `json:"tags" db:"tags"`         // JSON array
	Metadata      map[string]interface{} `json:"metadata" db:"metadata"` // JSON object
	CreatedAt     time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at" db:"updated_at"`
	ProjectPath   string                 `json:"project_path" db:"project_path"`
	Hash          string                 `json:"hash" db:"hash"` // Content hash for change detection
}

// EntityRelationship represents a relationship between two code entities
type EntityRelationship struct {
	ID           string                 `json:"id" db:"id"`
	SourceID     string                 `json:"source_id" db:"source_id"`
	TargetID     string                 `json:"target_id" db:"target_id"`
	RelationType RelationshipType       `json:"relation_type" db:"relation_type"`
	Direction    DirectionType          `json:"direction" db:"direction"`     // bidirectional, source_to_target, target_to_source
	Strength     float64                `json:"strength" db:"strength"`       // Relationship strength (0-1)
	Confidence   float64                `json:"confidence" db:"confidence"`   // Detection confidence (0-1)
	Context      string                 `json:"context" db:"context"`         // Where/how the relationship occurs
	LineNumber   int                    `json:"line_number" db:"line_number"` // Line where relationship occurs
	Occurrences  int                    `json:"occurrences" db:"occurrences"` // How many times this relationship occurs
	Properties   map[string]interface{} `json:"properties" db:"properties"`   // Additional properties
	CreatedAt    time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at" db:"updated_at"`
	LastSeen     time.Time              `json:"last_seen" db:"last_seen"`
}

// EntityType represents the type of code entity
type EntityType string

const (
	EntityFunction   EntityType = "function"
	EntityMethod     EntityType = "method"
	EntityClass      EntityType = "class"
	EntityInterface  EntityType = "interface"
	EntityStruct     EntityType = "struct"
	EntityModule     EntityType = "module"
	EntityPackage    EntityType = "package"
	EntityVariable   EntityType = "variable"
	EntityConstant   EntityType = "constant"
	EntityEnum       EntityType = "enum"
	EntityAnnotation EntityType = "annotation"
	EntityTrait      EntityType = "trait"
)

// RelationshipType represents the type of relationship between entities
type RelationshipType string

const (
	RelationshipCalls       RelationshipType = "calls"
	RelationshipInherits    RelationshipType = "inherits"
	RelationshipImplements  RelationshipType = "implements"
	RelationshipImports     RelationshipType = "imports"
	RelationshipDependsOn   RelationshipType = "depends_on"
	RelationshipContains    RelationshipType = "contains"
	RelationshipUses        RelationshipType = "uses"
	RelationshipOverrides   RelationshipType = "overrides"
	RelationshipDecorates   RelationshipType = "decorates"
	RelationshipComposition RelationshipType = "composition"
	RelationshipAggregation RelationshipType = "aggregation"
	RelationshipAssociation RelationshipType = "association"
	RelationshipReturns     RelationshipType = "returns"
	RelationshipThrows      RelationshipType = "throws"
	RelationshipAccesses    RelationshipType = "accesses"
)

// VisibilityType represents the visibility/access level of an entity
type VisibilityType string

const (
	VisibilityPublic    VisibilityType = "public"
	VisibilityPrivate   VisibilityType = "private"
	VisibilityProtected VisibilityType = "protected"
	VisibilityInternal  VisibilityType = "internal"
	VisibilityPackage   VisibilityType = "package"
)

// DirectionType represents the direction of a relationship
type DirectionType string

const (
	DirectionBidirectional  DirectionType = "bidirectional"
	DirectionSourceToTarget DirectionType = "source_to_target"
	DirectionTargetToSource DirectionType = "target_to_source"
)

// RelationshipQuery represents search criteria for relationships
type RelationshipQuery struct {
	EntityIDs         []string           `json:"entity_ids,omitempty"`
	EntityTypes       []EntityType       `json:"entity_types,omitempty"`
	RelationshipTypes []RelationshipType `json:"relationship_types,omitempty"`
	Languages         []string           `json:"languages,omitempty"`
	ProjectPaths      []string           `json:"project_paths,omitempty"`
	MinStrength       float64            `json:"min_strength,omitempty"`
	MinConfidence     float64            `json:"min_confidence,omitempty"`
	MaxDepth          int                `json:"max_depth,omitempty"`
	IncludeIndirect   bool               `json:"include_indirect,omitempty"`
	TimeRange         *TimeRange         `json:"time_range,omitempty"`
	SearchText        string             `json:"search_text,omitempty"`
	SortBy            string             `json:"sort_by,omitempty"`
	SortOrder         string             `json:"sort_order,omitempty"`
	Limit             int                `json:"limit,omitempty"`
	Offset            int                `json:"offset,omitempty"`
}

// RelationshipGraph represents a graph of entity relationships
type RelationshipGraph struct {
	Entities      map[string]*CodeEntity `json:"entities"`
	Relationships []*EntityRelationship  `json:"relationships"`
	Metrics       *GraphMetrics          `json:"metrics"`
	CreatedAt     time.Time              `json:"created_at"`
	ProjectPath   string                 `json:"project_path"`
}

// GraphMetrics provides statistics about the relationship graph
type GraphMetrics struct {
	EntityCount           int                      `json:"entity_count"`
	RelationshipCount     int                      `json:"relationship_count"`
	AverageConnections    float64                  `json:"average_connections"`
	MaxConnections        int                      `json:"max_connections"`
	MinConnections        int                      `json:"min_connections"`
	MostConnectedEntity   string                   `json:"most_connected_entity"`
	CouplingIndex         float64                  `json:"coupling_index"` // Overall coupling strength
	CohesionIndex         float64                  `json:"cohesion_index"` // Internal cohesion
	CyclomaticComplexity  int                      `json:"cyclomatic_complexity"`
	RelationshipBreakdown map[RelationshipType]int `json:"relationship_breakdown"`
	EntityTypeBreakdown   map[EntityType]int       `json:"entity_type_breakdown"`
	LanguageBreakdown     map[string]int           `json:"language_breakdown"`
	HotspotEntities       []*EntityHotspot         `json:"hotspot_entities"` // Most connected/complex entities
	PotentialBottlenecks  []*EntityHotspot         `json:"potential_bottlenecks"`
}

// EntityHotspot represents a highly connected or complex entity
type EntityHotspot struct {
	EntityID        string   `json:"entity_id"`
	EntityName      string   `json:"entity_name"`
	ConnectionCount int      `json:"connection_count"`
	Complexity      int      `json:"complexity"`
	ImpactScore     float64  `json:"impact_score"` // Calculated impact based on connections and complexity
	Reasons         []string `json:"reasons"`      // Why this is considered a hotspot
}

// RelationshipStorage provides storage operations for code entity relationships
type RelationshipStorage struct {
	db            *sql.DB
	cache         *SemanticCache
	indexer       *RelationshipIndexer
	mutex         sync.RWMutex
	batchSize     int
	analysisCache map[string]*GraphMetrics // Cache for expensive graph analysis
}

// RelationshipIndexer provides fast lookup capabilities
type RelationshipIndexer struct {
	entityByName      map[string][]*CodeEntity     // Name -> entities
	entityByType      map[EntityType][]*CodeEntity // Type -> entities
	relationsByType   map[RelationshipType][]*EntityRelationship
	relationsByEntity map[string][]*EntityRelationship // EntityID -> relationships
	mutex             sync.RWMutex
}

// NewRelationshipStorage creates a new relationship storage instance
func NewRelationshipStorage(db *sql.DB) (*RelationshipStorage, error) {
	rs := &RelationshipStorage{
		db:            db,
		cache:         NewSemanticCache(),
		batchSize:     1000,
		analysisCache: make(map[string]*GraphMetrics),
		indexer: &RelationshipIndexer{
			entityByName:      make(map[string][]*CodeEntity),
			entityByType:      make(map[EntityType][]*CodeEntity),
			relationsByType:   make(map[RelationshipType][]*EntityRelationship),
			relationsByEntity: make(map[string][]*EntityRelationship),
		},
	}

	// Initialize database schema
	if err := rs.initializeSchema(); err != nil {
		return nil, fmt.Errorf("failed to initialize relationship storage schema: %w", err)
	}

	// Build indexes
	go rs.buildIndexes()

	return rs, nil
}

// initializeSchema creates the necessary database tables
func (rs *RelationshipStorage) initializeSchema() error {
	queries := []string{
		// Code entities table
		`CREATE TABLE IF NOT EXISTS code_entities (
			id TEXT PRIMARY KEY,
			type TEXT NOT NULL,
			name TEXT NOT NULL,
			full_name TEXT,
			file_path TEXT NOT NULL,
			line_start INTEGER,
			line_end INTEGER,
			language TEXT,
			signature TEXT,
			visibility TEXT DEFAULT 'public',
			is_static BOOLEAN DEFAULT FALSE,
			is_abstract BOOLEAN DEFAULT FALSE,
			complexity INTEGER DEFAULT 0,
			lines_of_code INTEGER DEFAULT 0,
			documentation TEXT,
			tags TEXT, -- JSON array
			metadata TEXT, -- JSON object
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			project_path TEXT,
			hash TEXT
		)`,

		// Entity relationships table
		`CREATE TABLE IF NOT EXISTS entity_relationships (
			id TEXT PRIMARY KEY,
			source_id TEXT NOT NULL,
			target_id TEXT NOT NULL,
			relation_type TEXT NOT NULL,
			direction TEXT DEFAULT 'source_to_target',
			strength REAL DEFAULT 1.0,
			confidence REAL DEFAULT 1.0,
			context TEXT,
			line_number INTEGER,
			occurrences INTEGER DEFAULT 1,
			properties TEXT, -- JSON object
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY (source_id) REFERENCES code_entities(id) ON DELETE CASCADE,
			FOREIGN KEY (target_id) REFERENCES code_entities(id) ON DELETE CASCADE
		)`,

		// Relationship analysis cache table
		`CREATE TABLE IF NOT EXISTS relationship_analysis_cache (
			project_path TEXT PRIMARY KEY,
			analysis_data TEXT, -- JSON serialized GraphMetrics
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			expires_at DATETIME
		)`,

		// Indexes for performance
		`CREATE INDEX IF NOT EXISTS idx_entities_type ON code_entities(type)`,
		`CREATE INDEX IF NOT EXISTS idx_entities_name ON code_entities(name)`,
		`CREATE INDEX IF NOT EXISTS idx_entities_full_name ON code_entities(full_name)`,
		`CREATE INDEX IF NOT EXISTS idx_entities_file_path ON code_entities(file_path)`,
		`CREATE INDEX IF NOT EXISTS idx_entities_language ON code_entities(language)`,
		`CREATE INDEX IF NOT EXISTS idx_entities_project ON code_entities(project_path)`,
		`CREATE INDEX IF NOT EXISTS idx_entities_visibility ON code_entities(visibility)`,
		`CREATE INDEX IF NOT EXISTS idx_entities_complexity ON code_entities(complexity)`,

		`CREATE INDEX IF NOT EXISTS idx_relationships_source ON entity_relationships(source_id)`,
		`CREATE INDEX IF NOT EXISTS idx_relationships_target ON entity_relationships(target_id)`,
		`CREATE INDEX IF NOT EXISTS idx_relationships_type ON entity_relationships(relation_type)`,
		`CREATE INDEX IF NOT EXISTS idx_relationships_strength ON entity_relationships(strength)`,
		`CREATE INDEX IF NOT EXISTS idx_relationships_confidence ON entity_relationships(confidence)`,
		`CREATE INDEX IF NOT EXISTS idx_relationships_updated ON entity_relationships(updated_at)`,

		// Composite indexes for common queries
		`CREATE INDEX IF NOT EXISTS idx_relationships_source_type ON entity_relationships(source_id, relation_type)`,
		`CREATE INDEX IF NOT EXISTS idx_relationships_target_type ON entity_relationships(target_id, relation_type)`,

		// Full-text search for entities
		`CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
			id UNINDEXED,
			name,
			full_name,
			signature,
			documentation,
			content='code_entities',
			content_rowid='rowid'
		)`,

		// FTS triggers
		`CREATE TRIGGER IF NOT EXISTS entities_fts_insert AFTER INSERT ON code_entities BEGIN
			INSERT INTO entities_fts(id, name, full_name, signature, documentation) 
			VALUES (NEW.id, NEW.name, NEW.full_name, NEW.signature, NEW.documentation);
		END`,

		`CREATE TRIGGER IF NOT EXISTS entities_fts_update AFTER UPDATE ON code_entities BEGIN
			UPDATE entities_fts SET name=NEW.name, full_name=NEW.full_name, signature=NEW.signature, documentation=NEW.documentation 
			WHERE id=NEW.id;
		END`,

		`CREATE TRIGGER IF NOT EXISTS entities_fts_delete AFTER DELETE ON code_entities BEGIN
			DELETE FROM entities_fts WHERE id=OLD.id;
		END`,
	}

	for _, query := range queries {
		if _, err := rs.db.Exec(query); err != nil {
			return fmt.Errorf("failed to execute query: %s, error: %w", query, err)
		}
	}

	return nil
}

// StoreEntity saves a code entity
func (rs *RelationshipStorage) StoreEntity(ctx context.Context, entity *CodeEntity) error {
	rs.mutex.Lock()
	defer rs.mutex.Unlock()

	tagsJSON, _ := json.Marshal(entity.Tags)
	metadataJSON, _ := json.Marshal(entity.Metadata)

	query := `INSERT OR REPLACE INTO code_entities 
			(id, type, name, full_name, file_path, line_start, line_end, language, signature,
			 visibility, is_static, is_abstract, complexity, lines_of_code, documentation,
			 tags, metadata, created_at, updated_at, project_path, hash)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	now := time.Now()
	if entity.CreatedAt.IsZero() {
		entity.CreatedAt = now
	}
	entity.UpdatedAt = now

	_, err := rs.db.ExecContext(ctx, query,
		entity.ID,
		entity.Type,
		entity.Name,
		entity.FullName,
		entity.FilePath,
		entity.LineStart,
		entity.LineEnd,
		entity.Language,
		entity.Signature,
		entity.Visibility,
		entity.IsStatic,
		entity.IsAbstract,
		entity.Complexity,
		entity.LinesOfCode,
		entity.Documentation,
		string(tagsJSON),
		string(metadataJSON),
		entity.CreatedAt,
		entity.UpdatedAt,
		entity.ProjectPath,
		entity.Hash,
	)

	if err != nil {
		return fmt.Errorf("failed to store entity: %w", err)
	}

	// Update indexes
	rs.updateEntityIndex(entity)

	return nil
}

// StoreRelationship saves an entity relationship
func (rs *RelationshipStorage) StoreRelationship(ctx context.Context, relationship *EntityRelationship) error {
	rs.mutex.Lock()
	defer rs.mutex.Unlock()

	// Check if relationship already exists and merge if so
	existing, err := rs.getRelationshipByEntities(ctx, relationship.SourceID, relationship.TargetID, relationship.RelationType)
	if err == nil && existing != nil {
		// Update existing relationship
		existing.Occurrences++
		existing.Strength = (existing.Strength + relationship.Strength) / 2       // Average strength
		existing.Confidence = (existing.Confidence + relationship.Confidence) / 2 // Average confidence
		existing.LastSeen = time.Now()
		existing.UpdatedAt = time.Now()
		relationship = existing
	}

	propertiesJSON, _ := json.Marshal(relationship.Properties)

	query := `INSERT OR REPLACE INTO entity_relationships 
			(id, source_id, target_id, relation_type, direction, strength, confidence,
			 context, line_number, occurrences, properties, created_at, updated_at, last_seen)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	now := time.Now()
	if relationship.CreatedAt.IsZero() {
		relationship.CreatedAt = now
	}
	relationship.UpdatedAt = now
	relationship.LastSeen = now

	_, err = rs.db.ExecContext(ctx, query,
		relationship.ID,
		relationship.SourceID,
		relationship.TargetID,
		relationship.RelationType,
		relationship.Direction,
		relationship.Strength,
		relationship.Confidence,
		relationship.Context,
		relationship.LineNumber,
		relationship.Occurrences,
		string(propertiesJSON),
		relationship.CreatedAt,
		relationship.UpdatedAt,
		relationship.LastSeen,
	)

	if err != nil {
		return fmt.Errorf("failed to store relationship: %w", err)
	}

	// Update indexes
	rs.updateRelationshipIndex(relationship)

	// Invalidate analysis cache
	rs.analysisCache = make(map[string]*GraphMetrics)

	return nil
}

// GetEntity retrieves an entity by ID
func (rs *RelationshipStorage) GetEntity(ctx context.Context, entityID string) (*CodeEntity, error) {
	rs.mutex.RLock()
	defer rs.mutex.RUnlock()

	query := `SELECT id, type, name, full_name, file_path, line_start, line_end, language, signature,
			visibility, is_static, is_abstract, complexity, lines_of_code, documentation,
			tags, metadata, created_at, updated_at, project_path, hash
			FROM code_entities WHERE id = ?`

	return rs.scanEntity(rs.db.QueryRowContext(ctx, query, entityID))
}

// QueryEntities searches for entities based on criteria
func (rs *RelationshipStorage) QueryEntities(ctx context.Context, query *RelationshipQuery) ([]*CodeEntity, error) {
	rs.mutex.RLock()
	defer rs.mutex.RUnlock()

	sqlQuery := `SELECT id, type, name, full_name, file_path, line_start, line_end, language, signature,
				visibility, is_static, is_abstract, complexity, lines_of_code, documentation,
				tags, metadata, created_at, updated_at, project_path, hash
				FROM code_entities`

	args := make([]interface{}, 0)
	conditions := make([]string, 0)

	// Build WHERE conditions
	if len(query.EntityTypes) > 0 {
		placeholders := make([]string, len(query.EntityTypes))
		for i, entityType := range query.EntityTypes {
			placeholders[i] = "?"
			args = append(args, entityType)
		}
		conditions = append(conditions, fmt.Sprintf("type IN (%s)", strings.Join(placeholders, ",")))
	}

	if len(query.Languages) > 0 {
		placeholders := make([]string, len(query.Languages))
		for i, lang := range query.Languages {
			placeholders[i] = "?"
			args = append(args, lang)
		}
		conditions = append(conditions, fmt.Sprintf("language IN (%s)", strings.Join(placeholders, ",")))
	}

	if len(query.ProjectPaths) > 0 {
		placeholders := make([]string, len(query.ProjectPaths))
		for i, path := range query.ProjectPaths {
			placeholders[i] = "?"
			args = append(args, path+"%")
		}
		conditions = append(conditions, fmt.Sprintf("project_path LIKE ANY(%s)", strings.Join(placeholders, ",")))
	}

	// Full-text search
	if query.SearchText != "" {
		sqlQuery = `SELECT e.id, e.type, e.name, e.full_name, e.file_path, e.line_start, e.line_end, 
					e.language, e.signature, e.visibility, e.is_static, e.is_abstract, e.complexity, 
					e.lines_of_code, e.documentation, e.tags, e.metadata, e.created_at, e.updated_at, 
					e.project_path, e.hash
					FROM code_entities e
					JOIN entities_fts fts ON e.id = fts.id
					WHERE entities_fts MATCH ?`
		args = append([]interface{}{query.SearchText}, args...)
	}

	if len(conditions) > 0 {
		if query.SearchText != "" {
			sqlQuery += " AND " + strings.Join(conditions, " AND ")
		} else {
			sqlQuery += " WHERE " + strings.Join(conditions, " AND ")
		}
	}

	// Add sorting
	sortBy := "updated_at"
	if query.SortBy != "" {
		sortBy = query.SortBy
	}
	sortOrder := "DESC"
	if query.SortOrder == "asc" {
		sortOrder = "ASC"
	}
	sqlQuery += fmt.Sprintf(" ORDER BY %s %s", sortBy, sortOrder)

	// Add pagination
	if query.Limit > 0 {
		sqlQuery += " LIMIT ?"
		args = append(args, query.Limit)
		if query.Offset > 0 {
			sqlQuery += " OFFSET ?"
			args = append(args, query.Offset)
		}
	}

	rows, err := rs.db.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query entities: %w", err)
	}
	defer rows.Close()

	return rs.scanEntities(rows)
}

// QueryRelationships searches for relationships based on criteria
func (rs *RelationshipStorage) QueryRelationships(ctx context.Context, query *RelationshipQuery) ([]*EntityRelationship, error) {
	rs.mutex.RLock()
	defer rs.mutex.RUnlock()

	sqlQuery := `SELECT r.id, r.source_id, r.target_id, r.relation_type, r.direction, r.strength, 
				r.confidence, r.context, r.line_number, r.occurrences, r.properties, 
				r.created_at, r.updated_at, r.last_seen
				FROM entity_relationships r`

	args := make([]interface{}, 0)
	conditions := make([]string, 0)

	// Build WHERE conditions
	if len(query.EntityIDs) > 0 {
		placeholders := make([]string, len(query.EntityIDs))
		for i, entityID := range query.EntityIDs {
			placeholders[i] = "?"
			args = append(args, entityID)
		}
		entityCondition := fmt.Sprintf("(source_id IN (%s) OR target_id IN (%s))",
			strings.Join(placeholders, ","), strings.Join(placeholders, ","))
		conditions = append(conditions, entityCondition)
		// Duplicate args for both IN clauses
		args = append(args, query.EntityIDs...)
	}

	if len(query.RelationshipTypes) > 0 {
		placeholders := make([]string, len(query.RelationshipTypes))
		for i, relType := range query.RelationshipTypes {
			placeholders[i] = "?"
			args = append(args, relType)
		}
		conditions = append(conditions, fmt.Sprintf("relation_type IN (%s)", strings.Join(placeholders, ",")))
	}

	if query.MinStrength > 0 {
		conditions = append(conditions, "strength >= ?")
		args = append(args, query.MinStrength)
	}

	if query.MinConfidence > 0 {
		conditions = append(conditions, "confidence >= ?")
		args = append(args, query.MinConfidence)
	}

	if len(conditions) > 0 {
		sqlQuery += " WHERE " + strings.Join(conditions, " AND ")
	}

	// Add sorting
	sortBy := "updated_at"
	if query.SortBy != "" {
		sortBy = query.SortBy
	}
	sortOrder := "DESC"
	if query.SortOrder == "asc" {
		sortOrder = "ASC"
	}
	sqlQuery += fmt.Sprintf(" ORDER BY %s %s", sortBy, sortOrder)

	// Add pagination
	if query.Limit > 0 {
		sqlQuery += " LIMIT ?"
		args = append(args, query.Limit)
		if query.Offset > 0 {
			sqlQuery += " OFFSET ?"
			args = append(args, query.Offset)
		}
	}

	rows, err := rs.db.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query relationships: %w", err)
	}
	defer rows.Close()

	return rs.scanRelationships(rows)
}

// GetRelationshipGraph builds a relationship graph for entities
func (rs *RelationshipStorage) GetRelationshipGraph(ctx context.Context, query *RelationshipQuery) (*RelationshipGraph, error) {
	// Get entities matching the query
	entities, err := rs.QueryEntities(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get entities: %w", err)
	}

	// Get relationships for these entities
	entityIDs := make([]string, len(entities))
	for i, entity := range entities {
		entityIDs[i] = entity.ID
	}

	relationshipQuery := &RelationshipQuery{
		EntityIDs:         entityIDs,
		RelationshipTypes: query.RelationshipTypes,
		MinStrength:       query.MinStrength,
		MinConfidence:     query.MinConfidence,
	}

	relationships, err := rs.QueryRelationships(ctx, relationshipQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to get relationships: %w", err)
	}

	// Build entity map
	entityMap := make(map[string]*CodeEntity)
	for _, entity := range entities {
		entityMap[entity.ID] = entity
	}

	// Calculate metrics
	metrics := rs.calculateGraphMetrics(entities, relationships)

	return &RelationshipGraph{
		Entities:      entityMap,
		Relationships: relationships,
		Metrics:       metrics,
		CreatedAt:     time.Now(),
		ProjectPath:   query.ProjectPaths[0], // Assuming single project for now
	}, nil
}

// GetEntityConnections gets all connections for a specific entity
func (rs *RelationshipStorage) GetEntityConnections(ctx context.Context, entityID string, maxDepth int) (*RelationshipGraph, error) {
	visited := make(map[string]bool)
	entities := make(map[string]*CodeEntity)
	relationships := make([]*EntityRelationship, 0)

	// BFS to find connected entities
	queue := []struct {
		entityID string
		depth    int
	}{{entityID, 0}}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if visited[current.entityID] || current.depth > maxDepth {
			continue
		}
		visited[current.entityID] = true

		// Get the entity
		entity, err := rs.GetEntity(ctx, current.entityID)
		if err != nil {
			continue
		}
		entities[current.entityID] = entity

		// Get related relationships
		relQuery := &RelationshipQuery{
			EntityIDs: []string{current.entityID},
		}
		entityRelationships, err := rs.QueryRelationships(ctx, relQuery)
		if err != nil {
			continue
		}

		for _, rel := range entityRelationships {
			relationships = append(relationships, rel)

			// Add connected entities to queue
			if rel.SourceID == current.entityID && !visited[rel.TargetID] {
				queue = append(queue, struct {
					entityID string
					depth    int
				}{rel.TargetID, current.depth + 1})
			}
			if rel.TargetID == current.entityID && !visited[rel.SourceID] {
				queue = append(queue, struct {
					entityID string
					depth    int
				}{rel.SourceID, current.depth + 1})
			}
		}
	}

	// Convert entities map to slice for metrics calculation
	entitySlice := make([]*CodeEntity, 0, len(entities))
	for _, entity := range entities {
		entitySlice = append(entitySlice, entity)
	}

	metrics := rs.calculateGraphMetrics(entitySlice, relationships)

	return &RelationshipGraph{
		Entities:      entities,
		Relationships: relationships,
		Metrics:       metrics,
		CreatedAt:     time.Now(),
	}, nil
}

// Helper methods

func (rs *RelationshipStorage) scanEntity(row *sql.Row) (*CodeEntity, error) {
	var entity CodeEntity
	var tagsJSON, metadataJSON string

	err := row.Scan(
		&entity.ID,
		&entity.Type,
		&entity.Name,
		&entity.FullName,
		&entity.FilePath,
		&entity.LineStart,
		&entity.LineEnd,
		&entity.Language,
		&entity.Signature,
		&entity.Visibility,
		&entity.IsStatic,
		&entity.IsAbstract,
		&entity.Complexity,
		&entity.LinesOfCode,
		&entity.Documentation,
		&tagsJSON,
		&metadataJSON,
		&entity.CreatedAt,
		&entity.UpdatedAt,
		&entity.ProjectPath,
		&entity.Hash,
	)

	if err != nil {
		return nil, fmt.Errorf("failed to scan entity: %w", err)
	}

	// Parse JSON fields
	if tagsJSON != "" {
		json.Unmarshal([]byte(tagsJSON), &entity.Tags)
	}
	if metadataJSON != "" {
		json.Unmarshal([]byte(metadataJSON), &entity.Metadata)
	}

	if entity.Tags == nil {
		entity.Tags = make([]string, 0)
	}
	if entity.Metadata == nil {
		entity.Metadata = make(map[string]interface{})
	}

	return &entity, nil
}

func (rs *RelationshipStorage) scanEntities(rows *sql.Rows) ([]*CodeEntity, error) {
	var entities []*CodeEntity

	for rows.Next() {
		var entity CodeEntity
		var tagsJSON, metadataJSON string

		err := rows.Scan(
			&entity.ID,
			&entity.Type,
			&entity.Name,
			&entity.FullName,
			&entity.FilePath,
			&entity.LineStart,
			&entity.LineEnd,
			&entity.Language,
			&entity.Signature,
			&entity.Visibility,
			&entity.IsStatic,
			&entity.IsAbstract,
			&entity.Complexity,
			&entity.LinesOfCode,
			&entity.Documentation,
			&tagsJSON,
			&metadataJSON,
			&entity.CreatedAt,
			&entity.UpdatedAt,
			&entity.ProjectPath,
			&entity.Hash,
		)

		if err != nil {
			return nil, fmt.Errorf("failed to scan entity: %w", err)
		}

		// Parse JSON fields
		if tagsJSON != "" {
			json.Unmarshal([]byte(tagsJSON), &entity.Tags)
		}
		if metadataJSON != "" {
			json.Unmarshal([]byte(metadataJSON), &entity.Metadata)
		}

		if entity.Tags == nil {
			entity.Tags = make([]string, 0)
		}
		if entity.Metadata == nil {
			entity.Metadata = make(map[string]interface{})
		}

		entities = append(entities, &entity)
	}

	return entities, rows.Err()
}

func (rs *RelationshipStorage) scanRelationships(rows *sql.Rows) ([]*EntityRelationship, error) {
	var relationships []*EntityRelationship

	for rows.Next() {
		var relationship EntityRelationship
		var propertiesJSON string

		err := rows.Scan(
			&relationship.ID,
			&relationship.SourceID,
			&relationship.TargetID,
			&relationship.RelationType,
			&relationship.Direction,
			&relationship.Strength,
			&relationship.Confidence,
			&relationship.Context,
			&relationship.LineNumber,
			&relationship.Occurrences,
			&propertiesJSON,
			&relationship.CreatedAt,
			&relationship.UpdatedAt,
			&relationship.LastSeen,
		)

		if err != nil {
			return nil, fmt.Errorf("failed to scan relationship: %w", err)
		}

		// Parse JSON fields
		if propertiesJSON != "" {
			json.Unmarshal([]byte(propertiesJSON), &relationship.Properties)
		}

		if relationship.Properties == nil {
			relationship.Properties = make(map[string]interface{})
		}

		relationships = append(relationships, &relationship)
	}

	return relationships, rows.Err()
}

func (rs *RelationshipStorage) getRelationshipByEntities(ctx context.Context, sourceID, targetID string, relationType RelationshipType) (*EntityRelationship, error) {
	query := `SELECT id, source_id, target_id, relation_type, direction, strength, confidence,
			context, line_number, occurrences, properties, created_at, updated_at, last_seen
			FROM entity_relationships 
			WHERE source_id = ? AND target_id = ? AND relation_type = ?`

	var relationship EntityRelationship
	var propertiesJSON string

	err := rs.db.QueryRowContext(ctx, query, sourceID, targetID, relationType).Scan(
		&relationship.ID,
		&relationship.SourceID,
		&relationship.TargetID,
		&relationship.RelationType,
		&relationship.Direction,
		&relationship.Strength,
		&relationship.Confidence,
		&relationship.Context,
		&relationship.LineNumber,
		&relationship.Occurrences,
		&propertiesJSON,
		&relationship.CreatedAt,
		&relationship.UpdatedAt,
		&relationship.LastSeen,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("relationship not found")
		}
		return nil, fmt.Errorf("failed to get relationship: %w", err)
	}

	// Parse JSON fields
	if propertiesJSON != "" {
		json.Unmarshal([]byte(propertiesJSON), &relationship.Properties)
	}

	if relationship.Properties == nil {
		relationship.Properties = make(map[string]interface{})
	}

	return &relationship, nil
}

func (rs *RelationshipStorage) calculateGraphMetrics(entities []*CodeEntity, relationships []*EntityRelationship) *GraphMetrics {
	metrics := &GraphMetrics{
		EntityCount:           len(entities),
		RelationshipCount:     len(relationships),
		RelationshipBreakdown: make(map[RelationshipType]int),
		EntityTypeBreakdown:   make(map[EntityType]int),
		LanguageBreakdown:     make(map[string]int),
	}

	// Connection counts per entity
	connectionCounts := make(map[string]int)
	complexities := make(map[string]int)

	// Count connections and analyze entities
	for _, entity := range entities {
		metrics.EntityTypeBreakdown[entity.Type]++
		metrics.LanguageBreakdown[entity.Language]++
		complexities[entity.ID] = entity.Complexity
	}

	// Analyze relationships
	for _, rel := range relationships {
		metrics.RelationshipBreakdown[rel.RelationType]++
		connectionCounts[rel.SourceID]++
		connectionCounts[rel.TargetID]++
	}

	// Calculate connection statistics
	if len(connectionCounts) > 0 {
		totalConnections := 0
		maxConnections := 0
		minConnections := int(^uint(0) >> 1) // Max int
		mostConnectedEntity := ""

		for entityID, count := range connectionCounts {
			totalConnections += count
			if count > maxConnections {
				maxConnections = count
				mostConnectedEntity = entityID
			}
			if count < minConnections {
				minConnections = count
			}
		}

		metrics.AverageConnections = float64(totalConnections) / float64(len(connectionCounts))
		metrics.MaxConnections = maxConnections
		metrics.MinConnections = minConnections
		metrics.MostConnectedEntity = mostConnectedEntity
	}

	// Calculate coupling and cohesion indexes (simplified)
	metrics.CouplingIndex = rs.calculateCouplingIndex(entities, relationships)
	metrics.CohesionIndex = rs.calculateCohesionIndex(entities, relationships)

	// Find hotspots and bottlenecks
	metrics.HotspotEntities = rs.findHotspots(entities, connectionCounts, complexities)
	metrics.PotentialBottlenecks = rs.findBottlenecks(entities, connectionCounts, complexities)

	return metrics
}

func (rs *RelationshipStorage) calculateCouplingIndex(entities []*CodeEntity, relationships []*EntityRelationship) float64 {
	if len(entities) <= 1 {
		return 0.0
	}

	// Simple coupling calculation based on external dependencies
	externalConnections := 0
	for _, rel := range relationships {
		if rel.RelationType == RelationshipDependsOn || rel.RelationType == RelationshipImports {
			externalConnections++
		}
	}

	maxPossibleConnections := len(entities) * (len(entities) - 1)
	if maxPossibleConnections == 0 {
		return 0.0
	}

	return float64(externalConnections) / float64(maxPossibleConnections)
}

func (rs *RelationshipStorage) calculateCohesionIndex(entities []*CodeEntity, relationships []*EntityRelationship) float64 {
	if len(entities) <= 1 {
		return 1.0
	}

	// Simple cohesion calculation based on internal connections
	internalConnections := 0
	for _, rel := range relationships {
		if rel.RelationType == RelationshipCalls || rel.RelationType == RelationshipUses {
			internalConnections++
		}
	}

	maxPossibleConnections := len(entities) * (len(entities) - 1)
	if maxPossibleConnections == 0 {
		return 1.0
	}

	return float64(internalConnections) / float64(maxPossibleConnections)
}

func (rs *RelationshipStorage) findHotspots(entities []*CodeEntity, connectionCounts, complexities map[string]int) []*EntityHotspot {
	type hotspotCandidate struct {
		entity      *CodeEntity
		connections int
		complexity  int
		impactScore float64
	}

	candidates := make([]hotspotCandidate, 0)

	for _, entity := range entities {
		connections := connectionCounts[entity.ID]
		complexity := complexities[entity.ID]

		// Simple impact score calculation
		impactScore := float64(connections)*0.6 + float64(complexity)*0.4

		if connections > 10 || complexity > 20 || impactScore > 15 {
			candidates = append(candidates, hotspotCandidate{
				entity:      entity,
				connections: connections,
				complexity:  complexity,
				impactScore: impactScore,
			})
		}
	}

	// Sort by impact score
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].impactScore > candidates[j].impactScore
	})

	// Take top 10
	limit := 10
	if len(candidates) < limit {
		limit = len(candidates)
	}

	hotspots := make([]*EntityHotspot, limit)
	for i := 0; i < limit; i++ {
		candidate := candidates[i]
		reasons := make([]string, 0)

		if candidate.connections > 10 {
			reasons = append(reasons, fmt.Sprintf("High connectivity: %d connections", candidate.connections))
		}
		if candidate.complexity > 20 {
			reasons = append(reasons, fmt.Sprintf("High complexity: %d", candidate.complexity))
		}
		if candidate.impactScore > 20 {
			reasons = append(reasons, "High overall impact score")
		}

		hotspots[i] = &EntityHotspot{
			EntityID:        candidate.entity.ID,
			EntityName:      candidate.entity.Name,
			ConnectionCount: candidate.connections,
			Complexity:      candidate.complexity,
			ImpactScore:     candidate.impactScore,
			Reasons:         reasons,
		}
	}

	return hotspots
}

func (rs *RelationshipStorage) findBottlenecks(entities []*CodeEntity, connectionCounts, complexities map[string]int) []*EntityHotspot {
	// Bottlenecks are entities that many others depend on
	// This is a simplified implementation
	return rs.findHotspots(entities, connectionCounts, complexities) // Reuse hotspot logic for now
}

func (rs *RelationshipStorage) updateEntityIndex(entity *CodeEntity) {
	rs.indexer.mutex.Lock()
	defer rs.indexer.mutex.Unlock()

	// Update name index
	if rs.indexer.entityByName[entity.Name] == nil {
		rs.indexer.entityByName[entity.Name] = make([]*CodeEntity, 0)
	}
	rs.indexer.entityByName[entity.Name] = append(rs.indexer.entityByName[entity.Name], entity)

	// Update type index
	if rs.indexer.entityByType[entity.Type] == nil {
		rs.indexer.entityByType[entity.Type] = make([]*CodeEntity, 0)
	}
	rs.indexer.entityByType[entity.Type] = append(rs.indexer.entityByType[entity.Type], entity)
}

func (rs *RelationshipStorage) updateRelationshipIndex(relationship *EntityRelationship) {
	rs.indexer.mutex.Lock()
	defer rs.indexer.mutex.Unlock()

	// Update type index
	if rs.indexer.relationsByType[relationship.RelationType] == nil {
		rs.indexer.relationsByType[relationship.RelationType] = make([]*EntityRelationship, 0)
	}
	rs.indexer.relationsByType[relationship.RelationType] = append(rs.indexer.relationsByType[relationship.RelationType], relationship)

	// Update entity index
	if rs.indexer.relationsByEntity[relationship.SourceID] == nil {
		rs.indexer.relationsByEntity[relationship.SourceID] = make([]*EntityRelationship, 0)
	}
	rs.indexer.relationsByEntity[relationship.SourceID] = append(rs.indexer.relationsByEntity[relationship.SourceID], relationship)

	if rs.indexer.relationsByEntity[relationship.TargetID] == nil {
		rs.indexer.relationsByEntity[relationship.TargetID] = make([]*EntityRelationship, 0)
	}
	rs.indexer.relationsByEntity[relationship.TargetID] = append(rs.indexer.relationsByEntity[relationship.TargetID], relationship)
}

func (rs *RelationshipStorage) buildIndexes() {
	// Build indexes from existing data - this would be called on startup
	ctx := context.Background()

	// Load all entities
	entities, err := rs.QueryEntities(ctx, &RelationshipQuery{Limit: 100000})
	if err != nil {
		return
	}

	for _, entity := range entities {
		rs.updateEntityIndex(entity)
	}

	// Load all relationships
	relationships, err := rs.QueryRelationships(ctx, &RelationshipQuery{Limit: 100000})
	if err != nil {
		return
	}

	for _, relationship := range relationships {
		rs.updateRelationshipIndex(relationship)
	}
}

// DeleteProject removes all entities and relationships for a project
func (rs *RelationshipStorage) DeleteProject(ctx context.Context, projectPath string) error {
	rs.mutex.Lock()
	defer rs.mutex.Unlock()

	tx, err := rs.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to start transaction: %w", err)
	}
	defer tx.Rollback()

	// Delete relationships first (foreign key constraints)
	_, err = tx.ExecContext(ctx, `DELETE FROM entity_relationships 
		WHERE source_id IN (SELECT id FROM code_entities WHERE project_path LIKE ?) 
		   OR target_id IN (SELECT id FROM code_entities WHERE project_path LIKE ?)`,
		projectPath+"%", projectPath+"%")
	if err != nil {
		return fmt.Errorf("failed to delete relationships: %w", err)
	}

	// Delete entities
	_, err = tx.ExecContext(ctx, "DELETE FROM code_entities WHERE project_path LIKE ?", projectPath+"%")
	if err != nil {
		return fmt.Errorf("failed to delete entities: %w", err)
	}

	// Delete analysis cache
	_, err = tx.ExecContext(ctx, "DELETE FROM relationship_analysis_cache WHERE project_path = ?", projectPath)
	if err != nil {
		return fmt.Errorf("failed to delete analysis cache: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	// Clear in-memory caches
	rs.analysisCache = make(map[string]*GraphMetrics)

	return nil
}
