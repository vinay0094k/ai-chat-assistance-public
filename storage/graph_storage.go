// storage/graph_storage.go - Stores and manages code relationship graph data
package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/intelligence"
)

// Updated GraphStorage to work with existing SemanticEntity and SemanticRelationship types
type GraphStorageV2 struct {
	db    *sql.DB
	mutex sync.RWMutex
	cache *SemanticCache
}

// NewGraphStorageV2 creates a new graph storage that works with existing intelligence types
func NewGraphStorageV2(db *sql.DB) (*GraphStorageV2, error) {
	gs := &GraphStorageV2{
		db:    db,
		cache: NewSemanticCache(),
	}

	// Initialize database schema for semantic entities
	if err := gs.initializeSemanticSchema(); err != nil {
		return nil, fmt.Errorf("failed to initialize semantic storage schema: %w", err)
	}

	return gs, nil
}

// initializeSemanticSchema creates tables for SemanticEntity and SemanticRelationship
func (gs *GraphStorageV2) initializeSemanticSchema() error {
	queries := []string{
		// Semantic entities table - matches your SemanticEntity struct
		`CREATE TABLE IF NOT EXISTS semantic_entities (
			id TEXT PRIMARY KEY,
			name TEXT NOT NULL,
			type INTEGER NOT NULL,  -- EntityType enum value
			scope TEXT,
			file TEXT NOT NULL,
			line INTEGER,
			column_pos INTEGER,
			package TEXT,
			signature TEXT,
			documentation TEXT,
			visibility INTEGER,  -- Visibility enum value
			language TEXT,
			is_abstract BOOLEAN DEFAULT FALSE,
			is_interface BOOLEAN DEFAULT FALSE,
			is_generic BOOLEAN DEFAULT FALSE,
			is_static BOOLEAN DEFAULT FALSE,
			is_async BOOLEAN DEFAULT FALSE,
			is_recursive BOOLEAN DEFAULT FALSE,
			is_deprecated BOOLEAN DEFAULT FALSE,
			complexity INTEGER DEFAULT 0,
			importance REAL DEFAULT 0.0,
			change_risk REAL DEFAULT 0.0,
			last_modified DATETIME,
			uses TEXT,        -- JSON array of strings
			used_by TEXT,     -- JSON array of strings
			calls TEXT,       -- JSON array of strings
			called_by TEXT,   -- JSON array of strings
			implements TEXT,  -- JSON array of strings
			extends TEXT,     -- JSON array of strings
			contains TEXT,    -- JSON array of strings
			contained_by TEXT,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)`,

		// Semantic relationships table - matches your SemanticRelationship struct
		`CREATE TABLE IF NOT EXISTS semantic_relationships (
			id TEXT PRIMARY KEY,
			from_entity TEXT NOT NULL,
			to_entity TEXT NOT NULL,
			relationship_type INTEGER NOT NULL,  -- RelationshipType enum value
			strength REAL DEFAULT 1.0,
			context TEXT,
			file TEXT,
			line INTEGER,
			is_transitive BOOLEAN DEFAULT FALSE,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY (from_entity) REFERENCES semantic_entities(id) ON DELETE CASCADE,
			FOREIGN KEY (to_entity) REFERENCES semantic_entities(id) ON DELETE CASCADE
		)`,

		// Indexes for performance
		`CREATE INDEX IF NOT EXISTS idx_semantic_entities_type ON semantic_entities(type)`,
		`CREATE INDEX IF NOT EXISTS idx_semantic_entities_language ON semantic_entities(language)`,
		`CREATE INDEX IF NOT EXISTS idx_semantic_entities_file ON semantic_entities(file)`,
		`CREATE INDEX IF NOT EXISTS idx_semantic_entities_package ON semantic_entities(package)`,
		`CREATE INDEX IF NOT EXISTS idx_semantic_entities_importance ON semantic_entities(importance)`,

		`CREATE INDEX IF NOT EXISTS idx_semantic_relationships_from ON semantic_relationships(from_entity)`,
		`CREATE INDEX IF NOT EXISTS idx_semantic_relationships_to ON semantic_relationships(to_entity)`,
		`CREATE INDEX IF NOT EXISTS idx_semantic_relationships_type ON semantic_relationships(relationship_type)`,
		`CREATE INDEX IF NOT EXISTS idx_semantic_relationships_strength ON semantic_relationships(strength)`,
	}

	for _, query := range queries {
		if _, err := gs.db.Exec(query); err != nil {
			return fmt.Errorf("failed to execute query: %s, error: %w", query, err)
		}
	}

	return nil
}

// StoreSemanticGraph stores a complete semantic graph using your existing types
func (gs *GraphStorageV2) StoreSemanticGraph(ctx context.Context, graph *intelligence.SemanticGraph, projectPath string) error {
	gs.mutex.Lock()
	defer gs.mutex.Unlock()

	tx, err := gs.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to start transaction: %w", err)
	}
	defer tx.Rollback()

	// Clear existing data for this project
	if err := gs.clearProjectSemanticData(tx, projectPath); err != nil {
		return fmt.Errorf("failed to clear existing data: %w", err)
	}

	// Store entities
	for _, entity := range graph.Entities {
		if err := gs.storeSemanticEntity(tx, entity); err != nil {
			return fmt.Errorf("failed to store entity %s: %w", entity.ID, err)
		}
	}

	// Store relationships
	for _, relationship := range graph.Relationships {
		if err := gs.storeSemanticRelationship(tx, &relationship); err != nil {
			return fmt.Errorf("failed to store relationship: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	// Cache the graph
	gs.cache.SetSemanticGraph(projectPath, graph)

	return nil
}

// storeSemanticEntity stores a single semantic entity
func (gs *GraphStorageV2) storeSemanticEntity(tx *sql.Tx, entity *intelligence.SemanticEntity) error {
	// Convert string slices to JSON
	usesJSON, _ := json.Marshal(entity.Uses)
	usedByJSON, _ := json.Marshal(entity.UsedBy)
	callsJSON, _ := json.Marshal(entity.Calls)
	calledByJSON, _ := json.Marshal(entity.CalledBy)
	implementsJSON, _ := json.Marshal(entity.Implements)
	extendsJSON, _ := json.Marshal(entity.Extends)
	containsJSON, _ := json.Marshal(entity.Contains)

	query := `INSERT OR REPLACE INTO semantic_entities 
			(id, name, type, scope, file, line, column_pos, package, signature, documentation,
			 visibility, language, is_abstract, is_interface, is_generic, is_static, is_async,
			 is_recursive, is_deprecated, complexity, importance, change_risk, last_modified,
			 uses, used_by, calls, called_by, implements, extends, contains, contained_by,
			 updated_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	_, err := tx.Exec(query,
		entity.ID,
		entity.Name,
		int(entity.Type),
		entity.Scope,
		entity.File,
		entity.Line,
		entity.Column,
		entity.Package,
		entity.Signature,
		entity.Documentation,
		int(entity.Visibility),
		entity.Language,
		entity.IsAbstract,
		entity.IsInterface,
		entity.IsGeneric,
		entity.IsStatic,
		entity.IsAsync,
		entity.IsRecursive,
		entity.IsDeprecated,
		entity.Complexity,
		entity.Importance,
		entity.ChangeRisk,
		entity.LastModified,
		string(usesJSON),
		string(usedByJSON),
		string(callsJSON),
		string(calledByJSON),
		string(implementsJSON),
		string(extendsJSON),
		string(containsJSON),
		entity.ContainedBy,
		time.Now(),
	)

	return err
}

// storeSemanticRelationship stores a single semantic relationship
func (gs *GraphStorageV2) storeSemanticRelationship(tx *sql.Tx, relationship *intelligence.SemanticRelationship) error {
	relationshipID := fmt.Sprintf("%s->%s:%d", relationship.From, relationship.To, int(relationship.Type))

	query := `INSERT OR REPLACE INTO semantic_relationships 
			(id, from_entity, to_entity, relationship_type, strength, context, file, line,
			 is_transitive, updated_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	_, err := tx.Exec(query,
		relationshipID,
		relationship.From,
		relationship.To,
		int(relationship.Type),
		relationship.Strength,
		relationship.Context,
		relationship.File,
		relationship.Line,
		relationship.IsTransitive,
		time.Now(),
	)

	return err
}

// LoadSemanticGraph loads a semantic graph using your existing types
func (gs *GraphStorageV2) LoadSemanticGraph(ctx context.Context, projectPath string) (*intelligence.SemanticGraph, error) {
	// Check cache first
	if graph, found := gs.cache.GetSemanticGraph(projectPath); found {
		return graph, nil
	}

	gs.mutex.RLock()
	defer gs.mutex.RUnlock()

	// Load entities
	entities, err := gs.loadSemanticEntities(ctx, projectPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load semantic entities: %w", err)
	}

	// Load relationships
	relationships, err := gs.loadSemanticRelationships(ctx, projectPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load semantic relationships: %w", err)
	}

	// Build graph
	graph := &intelligence.SemanticGraph{
		Entities:      entities,
		Relationships: relationships,
	}

	// Cache for future use
	gs.cache.SetSemanticGraph(projectPath, graph)

	return graph, nil
}

// loadSemanticEntities loads semantic entities from database
func (gs *GraphStorageV2) loadSemanticEntities(ctx context.Context, projectPath string) (map[string]*intelligence.SemanticEntity, error) {
	query := `SELECT id, name, type, scope, file, line, column_pos, package, signature, documentation,
			visibility, language, is_abstract, is_interface, is_generic, is_static, is_async,
			is_recursive, is_deprecated, complexity, importance, change_risk, last_modified,
			uses, used_by, calls, called_by, implements, extends, contains, contained_by
			FROM semantic_entities 
			WHERE file LIKE ?`

	rows, err := gs.db.QueryContext(ctx, query, projectPath+"%")
	if err != nil {
		return nil, fmt.Errorf("failed to query semantic entities: %w", err)
	}
	defer rows.Close()

	entities := make(map[string]*intelligence.SemanticEntity)

	for rows.Next() {
		var entity intelligence.SemanticEntity
		var entityType, visibility int
		var usesJSON, usedByJSON, callsJSON, calledByJSON, implementsJSON, extendsJSON, containsJSON string

		err := rows.Scan(
			&entity.ID,
			&entity.Name,
			&entityType,
			&entity.Scope,
			&entity.File,
			&entity.Line,
			&entity.Column,
			&entity.Package,
			&entity.Signature,
			&entity.Documentation,
			&visibility,
			&entity.Language,
			&entity.IsAbstract,
			&entity.IsInterface,
			&entity.IsGeneric,
			&entity.IsStatic,
			&entity.IsAsync,
			&entity.IsRecursive,
			&entity.IsDeprecated,
			&entity.Complexity,
			&entity.Importance,
			&entity.ChangeRisk,
			&entity.LastModified,
			&usesJSON,
			&usedByJSON,
			&callsJSON,
			&calledByJSON,
			&implementsJSON,
			&extendsJSON,
			&containsJSON,
			&entity.ContainedBy,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan semantic entity: %w", err)
		}

		// Convert enum values back to your types
		entity.Type = intelligence.EntityType(entityType)
		entity.Visibility = intelligence.Visibility(visibility)

		// Parse JSON arrays back to string slices
		json.Unmarshal([]byte(usesJSON), &entity.Uses)
		json.Unmarshal([]byte(usedByJSON), &entity.UsedBy)
		json.Unmarshal([]byte(callsJSON), &entity.Calls)
		json.Unmarshal([]byte(calledByJSON), &entity.CalledBy)
		json.Unmarshal([]byte(implementsJSON), &entity.Implements)
		json.Unmarshal([]byte(extendsJSON), &entity.Extends)
		json.Unmarshal([]byte(containsJSON), &entity.Contains)

		// Initialize slices if they're nil
		if entity.Uses == nil {
			entity.Uses = make([]string, 0)
		}
		if entity.UsedBy == nil {
			entity.UsedBy = make([]string, 0)
		}
		if entity.Calls == nil {
			entity.Calls = make([]string, 0)
		}
		if entity.CalledBy == nil {
			entity.CalledBy = make([]string, 0)
		}
		if entity.Implements == nil {
			entity.Implements = make([]string, 0)
		}
		if entity.Extends == nil {
			entity.Extends = make([]string, 0)
		}
		if entity.Contains == nil {
			entity.Contains = make([]string, 0)
		}

		entities[entity.ID] = &entity
	}

	return entities, rows.Err()
}

// loadSemanticRelationships loads semantic relationships from database
func (gs *GraphStorageV2) loadSemanticRelationships(ctx context.Context, projectPath string) ([]intelligence.SemanticRelationship, error) {
	query := `SELECT r.from_entity, r.to_entity, r.relationship_type, r.strength, r.context, r.file, r.line, r.is_transitive
			FROM semantic_relationships r
			JOIN semantic_entities e1 ON r.from_entity = e1.id
			JOIN semantic_entities e2 ON r.to_entity = e2.id
			WHERE e1.file LIKE ? OR e2.file LIKE ?`

	rows, err := gs.db.QueryContext(ctx, query, projectPath+"%", projectPath+"%")
	if err != nil {
		return nil, fmt.Errorf("failed to query semantic relationships: %w", err)
	}
	defer rows.Close()

	var relationships []intelligence.SemanticRelationship

	for rows.Next() {
		var relationship intelligence.SemanticRelationship
		var relationshipType int

		err := rows.Scan(
			&relationship.From,
			&relationship.To,
			&relationshipType,
			&relationship.Strength,
			&relationship.Context,
			&relationship.File,
			&relationship.Line,
			&relationship.IsTransitive,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan semantic relationship: %w", err)
		}

		// Convert enum value back to your type
		relationship.Type = intelligence.RelationshipType(relationshipType)

		relationships = append(relationships, relationship)
	}

	return relationships, rows.Err()
}

// GetSemanticEntitiesByType returns entities of a specific type
func (gs *GraphStorageV2) GetSemanticEntitiesByType(ctx context.Context, entityType intelligence.EntityType, projectPath string) ([]*intelligence.SemanticEntity, error) {
	gs.mutex.RLock()
	defer gs.mutex.RUnlock()

	query := `SELECT id, name, type, scope, file, line, column_pos, package, signature, documentation,
			visibility, language, is_abstract, is_interface, is_generic, is_static, is_async,
			is_recursive, is_deprecated, complexity, importance, change_risk, last_modified,
			uses, used_by, calls, called_by, implements, extends, contains, contained_by
			FROM semantic_entities 
			WHERE type = ? AND file LIKE ?
			ORDER BY importance DESC`

	rows, err := gs.db.QueryContext(ctx, query, int(entityType), projectPath+"%")
	if err != nil {
		return nil, fmt.Errorf("failed to query entities by type: %w", err)
	}
	defer rows.Close()

	var entities []*intelligence.SemanticEntity

	for rows.Next() {
		var entity intelligence.SemanticEntity
		var entityTypeInt, visibility int
		var usesJSON, usedByJSON, callsJSON, calledByJSON, implementsJSON, extendsJSON, containsJSON string

		err := rows.Scan(
			&entity.ID, &entity.Name, &entityTypeInt, &entity.Scope, &entity.File, &entity.Line, &entity.Column,
			&entity.Package, &entity.Signature, &entity.Documentation, &visibility, &entity.Language,
			&entity.IsAbstract, &entity.IsInterface, &entity.IsGeneric, &entity.IsStatic, &entity.IsAsync,
			&entity.IsRecursive, &entity.IsDeprecated, &entity.Complexity, &entity.Importance, &entity.ChangeRisk,
			&entity.LastModified, &usesJSON, &usedByJSON, &callsJSON, &calledByJSON, &implementsJSON, &extendsJSON,
			&containsJSON, &entity.ContainedBy,
		)
		if err != nil {
			continue
		}

		entity.Type = intelligence.EntityType(entityTypeInt)
		entity.Visibility = intelligence.Visibility(visibility)

		// Parse JSON arrays
		json.Unmarshal([]byte(usesJSON), &entity.Uses)
		json.Unmarshal([]byte(usedByJSON), &entity.UsedBy)
		json.Unmarshal([]byte(callsJSON), &entity.Calls)
		json.Unmarshal([]byte(calledByJSON), &entity.CalledBy)
		json.Unmarshal([]byte(implementsJSON), &entity.Implements)
		json.Unmarshal([]byte(extendsJSON), &entity.Extends)
		json.Unmarshal([]byte(containsJSON), &entity.Contains)

		entities = append(entities, &entity)
	}

	return entities, nil
}

// Helper methods

func (gs *GraphStorageV2) clearProjectSemanticData(tx *sql.Tx, projectPath string) error {
	// Delete relationships first due to foreign key constraints
	_, err := tx.Exec("DELETE FROM semantic_relationships WHERE id IN (SELECT r.id FROM semantic_relationships r JOIN semantic_entities e ON r.from_entity = e.id WHERE e.file LIKE ?)", projectPath+"%")
	if err != nil {
		return fmt.Errorf("failed to delete relationships: %w", err)
	}

	// Delete entities
	_, err = tx.Exec("DELETE FROM semantic_entities WHERE file LIKE ?", projectPath+"%")
	if err != nil {
		return fmt.Errorf("failed to delete entities: %w", err)
	}

	return nil
}

// DeleteProjectSemanticData removes all semantic data for a project
func (gs *GraphStorageV2) DeleteProjectSemanticData(ctx context.Context, projectPath string) error {
	gs.mutex.Lock()
	defer gs.mutex.Unlock()

	tx, err := gs.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to start transaction: %w", err)
	}
	defer tx.Rollback()

	if err := gs.clearProjectSemanticData(tx, projectPath); err != nil {
		return err
	}

	return tx.Commit()
}
