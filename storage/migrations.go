package storage

import (
	"database/sql"
	"fmt"
	"sort"
	"strings"
	"time"
)

// Migration represents a database migration
type Migration struct {
	Version     int       `json:"version"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	UpSQL       string    `json:"up_sql"`
	DownSQL     string    `json:"down_sql"`
	AppliedAt   time.Time `json:"applied_at,omitempty"`
}

// MigrationManager manages database migrations
type MigrationManager struct {
	db         *SQLiteDB
	migrations []Migration
}

// NewMigrationManager creates a new migration manager
func NewMigrationManager(db *SQLiteDB) *MigrationManager {
	manager := &MigrationManager{
		db:         db,
		migrations: make([]Migration, 0),
	}

	// Register all migrations
	manager.registerMigrations()

	return manager
}

// registerMigrations registers all database migrations
func (mm *MigrationManager) registerMigrations() {
	// Migration 001: Initial schema
	mm.migrations = append(mm.migrations, Migration{
		Version:     1,
		Name:        "initial_schema",
		Description: "Create initial database schema for code indexing",
		UpSQL: `
			-- Create projects table
			CREATE TABLE IF NOT EXISTS projects (
				id TEXT PRIMARY KEY,
				name TEXT NOT NULL,
				path TEXT NOT NULL UNIQUE,
				language TEXT,
				framework TEXT,
				file_count INTEGER DEFAULT 0,
				chunk_count INTEGER DEFAULT 0,
				total_lines INTEGER DEFAULT 0,
				last_indexed DATETIME,
				version TEXT,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);

			-- Create code_files table
			CREATE TABLE IF NOT EXISTS code_files (
				id TEXT PRIMARY KEY,
				path TEXT NOT NULL,
				relative_path TEXT NOT NULL,
				language TEXT,
				size INTEGER,
				hash TEXT NOT NULL,
				line_count INTEGER DEFAULT 0,
				chunk_count INTEGER DEFAULT 0,
				last_modified DATETIME,
				indexed_at DATETIME,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);

			-- Create code_chunks table
			CREATE TABLE IF NOT EXISTS code_chunks (
				id TEXT PRIMARY KEY,
				file_id TEXT NOT NULL,
				type TEXT NOT NULL,
				name TEXT NOT NULL,
				code TEXT NOT NULL,
				language TEXT,
				start_line INTEGER,
				end_line INTEGER,
				hash TEXT NOT NULL,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (file_id) REFERENCES code_files(id) ON DELETE CASCADE
			);

			-- Create code_relationships table
			CREATE TABLE IF NOT EXISTS code_relationships (
				id TEXT PRIMARY KEY,
				from_chunk_id TEXT NOT NULL,
				to_chunk_id TEXT NOT NULL,
				type TEXT NOT NULL,
				weight REAL DEFAULT 1.0,
				context TEXT,
				line_number INTEGER,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (from_chunk_id) REFERENCES code_chunks(id) ON DELETE CASCADE,
				FOREIGN KEY (to_chunk_id) REFERENCES code_chunks(id) ON DELETE CASCADE
			);

			-- Create indexes for performance
			CREATE INDEX IF NOT EXISTS idx_code_files_path ON code_files(path);
			CREATE INDEX IF NOT EXISTS idx_code_files_hash ON code_files(hash);
			CREATE INDEX IF NOT EXISTS idx_code_files_language ON code_files(language);
			CREATE INDEX IF NOT EXISTS idx_code_chunks_file_id ON code_chunks(file_id);
			CREATE INDEX IF NOT EXISTS idx_code_chunks_type ON code_chunks(type);
			CREATE INDEX IF NOT EXISTS idx_code_chunks_name ON code_chunks(name);
			CREATE INDEX IF NOT EXISTS idx_code_chunks_language ON code_chunks(language);
			CREATE INDEX IF NOT EXISTS idx_code_relationships_from ON code_relationships(from_chunk_id);
			CREATE INDEX IF NOT EXISTS idx_code_relationships_to ON code_relationships(to_chunk_id);
			CREATE INDEX IF NOT EXISTS idx_code_relationships_type ON code_relationships(type);
		`,
		DownSQL: `
			DROP INDEX IF EXISTS idx_code_relationships_type;
			DROP INDEX IF EXISTS idx_code_relationships_to;
			DROP INDEX IF EXISTS idx_code_relationships_from;
			DROP INDEX IF EXISTS idx_code_chunks_language;
			DROP INDEX IF EXISTS idx_code_chunks_name;
			DROP INDEX IF EXISTS idx_code_chunks_type;
			DROP INDEX IF EXISTS idx_code_chunks_file_id;
			DROP INDEX IF EXISTS idx_code_files_language;
			DROP INDEX IF EXISTS idx_code_files_hash;
			DROP INDEX IF EXISTS idx_code_files_path;
			DROP TABLE IF EXISTS code_relationships;
			DROP TABLE IF EXISTS code_chunks;
			DROP TABLE IF EXISTS code_files;
			DROP TABLE IF EXISTS projects;
		`,
	})

	// Migration 002: Session and query tracking
	mm.migrations = append(mm.migrations, Migration{
		Version:     2,
		Name:        "session_tracking",
		Description: "Add tables for session and query tracking",
		UpSQL: `
			-- Create sessions table
			CREATE TABLE IF NOT EXISTS sessions (
				id TEXT PRIMARY KEY,
				start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
				end_time DATETIME,
				total_queries INTEGER DEFAULT 0,
				total_tokens INTEGER DEFAULT 0,
				total_cost REAL DEFAULT 0.0,
				avg_response_time INTEGER DEFAULT 0,
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);

			-- Create query_logs table
			CREATE TABLE IF NOT EXISTS query_logs (
				id TEXT PRIMARY KEY,
				session_id TEXT NOT NULL,
				raw_input TEXT NOT NULL,
				processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				duration INTEGER,
				status TEXT DEFAULT 'pending',
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
			);

			-- Create query_intents table
			CREATE TABLE IF NOT EXISTS query_intents (
				id TEXT PRIMARY KEY,
				query_id TEXT NOT NULL,
				type TEXT NOT NULL,
				confidence REAL,
				entities TEXT DEFAULT '{}',
				parameters TEXT DEFAULT '{}',
				context TEXT,
				agent_type TEXT,
				priority INTEGER DEFAULT 0,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (query_id) REFERENCES query_logs(id) ON DELETE CASCADE
			);

			-- Create indexes
			CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time);
			CREATE INDEX IF NOT EXISTS idx_query_logs_session_id ON query_logs(session_id);
			CREATE INDEX IF NOT EXISTS idx_query_logs_status ON query_logs(status);
			CREATE INDEX IF NOT EXISTS idx_query_intents_query_id ON query_intents(query_id);
			CREATE INDEX IF NOT EXISTS idx_query_intents_type ON query_intents(type);
		`,
		DownSQL: `
			DROP INDEX IF EXISTS idx_query_intents_type;
			DROP INDEX IF EXISTS idx_query_intents_query_id;
			DROP INDEX IF EXISTS idx_query_logs_status;
			DROP INDEX IF EXISTS idx_query_logs_session_id;
			DROP INDEX IF EXISTS idx_sessions_start_time;
			DROP TABLE IF EXISTS query_intents;
			DROP TABLE IF EXISTS query_logs;
			DROP TABLE IF EXISTS sessions;
		`,
	})

	// Migration 003: Token tracking
	mm.migrations = append(mm.migrations, Migration{
		Version:     3,
		Name:        "token_tracking",
		Description: "Add tables for comprehensive token tracking",
		UpSQL: `
			-- Create session_token_usage table
			CREATE TABLE IF NOT EXISTS session_token_usage (
				session_id TEXT PRIMARY KEY,
				start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
				end_time DATETIME,
				total_queries INTEGER DEFAULT 0,
				total_tokens INTEGER DEFAULT 0,
				total_cost REAL DEFAULT 0.0,
				avg_response_time INTEGER DEFAULT 0,
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
			);

			-- Create provider_usage table
			CREATE TABLE IF NOT EXISTS provider_usage (
				provider TEXT NOT NULL,
				session_id TEXT NOT NULL,
				request_count INTEGER DEFAULT 0,
				input_tokens INTEGER DEFAULT 0,
				output_tokens INTEGER DEFAULT 0,
				total_tokens INTEGER DEFAULT 0,
				total_cost REAL DEFAULT 0.0,
				avg_tokens_per_request REAL DEFAULT 0.0,
				avg_cost_per_request REAL DEFAULT 0.0,
				avg_response_time INTEGER DEFAULT 0,
				success_rate REAL DEFAULT 1.0,
				last_used DATETIME,
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				PRIMARY KEY (provider, session_id),
				FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
			);

			-- Create token_transactions table
			CREATE TABLE IF NOT EXISTS token_transactions (
				id TEXT PRIMARY KEY,
				session_id TEXT NOT NULL,
				query_id TEXT,
				provider TEXT NOT NULL,
				model TEXT NOT NULL,
				transaction_type TEXT DEFAULT 'request',
				input_tokens INTEGER DEFAULT 0,
				output_tokens INTEGER DEFAULT 0,
				total_tokens INTEGER DEFAULT 0,
				cost REAL DEFAULT 0.0,
				status TEXT DEFAULT 'pending',
				response_time INTEGER DEFAULT 0,
				quality REAL,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
				FOREIGN KEY (query_id) REFERENCES query_logs(id) ON DELETE SET NULL
			);

			-- Create token_budgets table
			CREATE TABLE IF NOT EXISTS token_budgets (
				id TEXT PRIMARY KEY,
				user_id TEXT,
				session_id TEXT,
				daily_limit INTEGER DEFAULT 10000,
				monthly_limit INTEGER DEFAULT 300000,
				daily_used INTEGER DEFAULT 0,
				monthly_used INTEGER DEFAULT 0,
				cost_limit REAL DEFAULT 100.0,
				cost_used REAL DEFAULT 0.0,
				warning_threshold REAL DEFAULT 0.8,
				alert_threshold REAL DEFAULT 0.95,
				reset_date DATETIME,
				is_active BOOLEAN DEFAULT 1,
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);

			-- Create token_alerts table
			CREATE TABLE IF NOT EXISTS token_alerts (
				id TEXT PRIMARY KEY,
				type TEXT NOT NULL,
				level TEXT NOT NULL,
				title TEXT NOT NULL,
				message TEXT NOT NULL,
				threshold REAL,
				current_usage REAL,
				period TEXT,
				session_id TEXT,
				acknowledged BOOLEAN DEFAULT 0,
				acknowledged_at DATETIME,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
			);

			-- Create indexes
			CREATE INDEX IF NOT EXISTS idx_provider_usage_session ON provider_usage(session_id);
			CREATE INDEX IF NOT EXISTS idx_provider_usage_provider ON provider_usage(provider);
			CREATE INDEX IF NOT EXISTS idx_token_transactions_session ON token_transactions(session_id);
			CREATE INDEX IF NOT EXISTS idx_token_transactions_provider ON token_transactions(provider);
			CREATE INDEX IF NOT EXISTS idx_token_transactions_status ON token_transactions(status);
			CREATE INDEX IF NOT EXISTS idx_token_budgets_session ON token_budgets(session_id);
			CREATE INDEX IF NOT EXISTS idx_token_alerts_type ON token_alerts(type);
			CREATE INDEX IF NOT EXISTS idx_token_alerts_acknowledged ON token_alerts(acknowledged);
		`,
		DownSQL: `
			DROP INDEX IF EXISTS idx_token_alerts_acknowledged;
			DROP INDEX IF EXISTS idx_token_alerts_type;
			DROP INDEX IF EXISTS idx_token_budgets_session;
			DROP INDEX IF EXISTS idx_token_transactions_status;
			DROP INDEX IF EXISTS idx_token_transactions_provider;
			DROP INDEX IF EXISTS idx_token_transactions_session;
			DROP INDEX IF EXISTS idx_provider_usage_provider;
			DROP INDEX IF EXISTS idx_provider_usage_session;
			DROP TABLE IF EXISTS token_alerts;
			DROP TABLE IF EXISTS token_budgets;
			DROP TABLE IF EXISTS token_transactions;
			DROP TABLE IF EXISTS provider_usage;
			DROP TABLE IF EXISTS session_token_usage;
		`,
	})

	// Migration 004: MCP integration
	mm.migrations = append(mm.migrations, Migration{
		Version:     4,
		Name:        "mcp_integration",
		Description: "Add tables for MCP server and tool management",
		UpSQL: `
			-- Create mcp_servers table
			CREATE TABLE IF NOT EXISTS mcp_servers (
				id TEXT PRIMARY KEY,
				name TEXT NOT NULL UNIQUE,
				description TEXT,
				version TEXT,
				command TEXT NOT NULL,
				args TEXT DEFAULT '[]',
				transport TEXT DEFAULT 'stdio',
				enabled BOOLEAN DEFAULT 1,
				auto_install BOOLEAN DEFAULT 0,
				auto_restart BOOLEAN DEFAULT 1,
				timeout INTEGER DEFAULT 30,
				environment TEXT DEFAULT '{}',
				status TEXT DEFAULT 'stopped',
				last_ping DATETIME,
				error_count INTEGER DEFAULT 0,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);

			-- Create mcp_tools table
			CREATE TABLE IF NOT EXISTS mcp_tools (
				id TEXT PRIMARY KEY,
				server_id TEXT NOT NULL,
				name TEXT NOT NULL,
				description TEXT,
				input_schema TEXT DEFAULT '{}',
				output_schema TEXT DEFAULT '{}',
				category TEXT,
				usage_count INTEGER DEFAULT 0,
				last_used DATETIME,
				enabled BOOLEAN DEFAULT 1,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (server_id) REFERENCES mcp_servers(id) ON DELETE CASCADE
			);

			-- Create mcp_resources table
			CREATE TABLE IF NOT EXISTS mcp_resources (
				id TEXT PRIMARY KEY,
				server_id TEXT NOT NULL,
				uri TEXT NOT NULL,
				name TEXT NOT NULL,
				description TEXT,
				mime_type TEXT,
				category TEXT,
				access_count INTEGER DEFAULT 0,
				last_accessed DATETIME,
				enabled BOOLEAN DEFAULT 1,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (server_id) REFERENCES mcp_servers(id) ON DELETE CASCADE
			);

			-- Create mcp_tool_calls table
			CREATE TABLE IF NOT EXISTS mcp_tool_calls (
				id TEXT PRIMARY KEY,
				session_id TEXT NOT NULL,
				query_id TEXT,
				server_id TEXT NOT NULL,
				tool_name TEXT NOT NULL,
				arguments TEXT DEFAULT '{}',
				result TEXT,
				status TEXT DEFAULT 'pending',
				started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				completed_at DATETIME,
				duration INTEGER DEFAULT 0,
				error TEXT,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
				FOREIGN KEY (query_id) REFERENCES query_logs(id) ON DELETE SET NULL,
				FOREIGN KEY (server_id) REFERENCES mcp_servers(id) ON DELETE CASCADE
			);

			-- Create mcp_integrations table
			CREATE TABLE IF NOT EXISTS mcp_integrations (
				id TEXT PRIMARY KEY,
				query_type TEXT NOT NULL,
				mcp_tools TEXT DEFAULT '[]',
				priority INTEGER DEFAULT 0,
				enabled BOOLEAN DEFAULT 1,
				conditions TEXT DEFAULT '{}',
				parameters TEXT DEFAULT '{}',
				timeout INTEGER DEFAULT 30,
				retry_count INTEGER DEFAULT 3,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);

			-- Create indexes
			CREATE INDEX IF NOT EXISTS idx_mcp_servers_name ON mcp_servers(name);
			CREATE INDEX IF NOT EXISTS idx_mcp_servers_status ON mcp_servers(status);
			CREATE INDEX IF NOT EXISTS idx_mcp_tools_server_id ON mcp_tools(server_id);
			CREATE INDEX IF NOT EXISTS idx_mcp_tools_name ON mcp_tools(name);
			CREATE INDEX IF NOT EXISTS idx_mcp_tools_category ON mcp_tools(category);
			CREATE INDEX IF NOT EXISTS idx_mcp_resources_server_id ON mcp_resources(server_id);
			CREATE INDEX IF NOT EXISTS idx_mcp_resources_uri ON mcp_resources(uri);
			CREATE INDEX IF NOT EXISTS idx_mcp_tool_calls_session ON mcp_tool_calls(session_id);
			CREATE INDEX IF NOT EXISTS idx_mcp_tool_calls_server ON mcp_tool_calls(server_id);
			CREATE INDEX IF NOT EXISTS idx_mcp_tool_calls_status ON mcp_tool_calls(status);
			CREATE INDEX IF NOT EXISTS idx_mcp_integrations_query_type ON mcp_integrations(query_type);
		`,
		DownSQL: `
			DROP INDEX IF EXISTS idx_mcp_integrations_query_type;
			DROP INDEX IF EXISTS idx_mcp_tool_calls_status;
			DROP INDEX IF EXISTS idx_mcp_tool_calls_server;
			DROP INDEX IF EXISTS idx_mcp_tool_calls_session;
			DROP INDEX IF EXISTS idx_mcp_resources_uri;
			DROP INDEX IF EXISTS idx_mcp_resources_server_id;
			DROP INDEX IF EXISTS idx_mcp_tools_category;
			DROP INDEX IF EXISTS idx_mcp_tools_name;
			DROP INDEX IF EXISTS idx_mcp_tools_server_id;
			DROP INDEX IF EXISTS idx_mcp_servers_status;
			DROP INDEX IF EXISTS idx_mcp_servers_name;
			DROP TABLE IF EXISTS mcp_integrations;
			DROP TABLE IF EXISTS mcp_tool_calls;
			DROP TABLE IF EXISTS mcp_resources;
			DROP TABLE IF EXISTS mcp_tools;
			DROP TABLE IF EXISTS mcp_servers;
		`,
	})

	// Migration 005: Learning and feedback system
	mm.migrations = append(mm.migrations, Migration{
		Version:     5,
		Name:        "learning_system",
		Description: "Add tables for learning and feedback system",
		UpSQL: `
			-- Create code_patterns table
			CREATE TABLE IF NOT EXISTS code_patterns (
				id TEXT PRIMARY KEY,
				type TEXT NOT NULL,
				name TEXT NOT NULL,
				description TEXT,
				pattern TEXT NOT NULL,
				examples TEXT DEFAULT '[]',
				frequency INTEGER DEFAULT 1,
				confidence REAL DEFAULT 1.0,
				context TEXT,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);

			-- Create code_architectures table
			CREATE TABLE IF NOT EXISTS code_architectures (
				id TEXT PRIMARY KEY,
				project_id TEXT NOT NULL,
				type TEXT NOT NULL,
				patterns TEXT DEFAULT '[]',
				layers TEXT DEFAULT '[]',
				dependencies TEXT DEFAULT '[]',
				test_strategy TEXT,
				confidence REAL DEFAULT 1.0,
				detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
			);

			-- Create code_usage table
			CREATE TABLE IF NOT EXISTS code_usage (
				id TEXT PRIMARY KEY,
				chunk_id TEXT NOT NULL,
				usage_type TEXT NOT NULL,
				used_by TEXT NOT NULL,
				frequency INTEGER DEFAULT 1,
				context TEXT,
				location TEXT,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (chunk_id) REFERENCES code_chunks(id) ON DELETE CASCADE
			);

			-- Create user_feedback table
			CREATE TABLE IF NOT EXISTS user_feedback (
				id TEXT PRIMARY KEY,
				suggestion_id TEXT NOT NULL,
				session_id TEXT NOT NULL,
				query_id TEXT,
				action TEXT NOT NULL,
				modified_code TEXT,
				rating INTEGER,
				comments TEXT,
				issues TEXT DEFAULT '[]',
				timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
				user_id TEXT,
				metadata TEXT DEFAULT '{}',
				FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
				FOREIGN KEY (query_id) REFERENCES query_logs(id) ON DELETE SET NULL
			);

			-- Create accuracy_metrics table
			CREATE TABLE IF NOT EXISTS accuracy_metrics (
				provider TEXT NOT NULL,
				suggestion_type TEXT NOT NULL,
				total_count INTEGER DEFAULT 0,
				accepted_count INTEGER DEFAULT 0,
				rejected_count INTEGER DEFAULT 0,
				modified_count INTEGER DEFAULT 0,
				avg_rating REAL DEFAULT 0.0,
				accuracy_rate REAL DEFAULT 0.0,
				last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
				PRIMARY KEY (provider, suggestion_type)
			);

			-- Create learning_patterns table
			CREATE TABLE IF NOT EXISTS learning_patterns (
				id TEXT PRIMARY KEY,
				pattern TEXT NOT NULL,
				context TEXT DEFAULT '{}',
				success_rate REAL DEFAULT 0.0,
				usage_count INTEGER DEFAULT 0,
				last_used DATETIME,
				category TEXT,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);

			-- Create indexes
			CREATE INDEX IF NOT EXISTS idx_code_patterns_type ON code_patterns(type);
			CREATE INDEX IF NOT EXISTS idx_code_patterns_frequency ON code_patterns(frequency);
			CREATE INDEX IF NOT EXISTS idx_code_architectures_project ON code_architectures(project_id);
			CREATE INDEX IF NOT EXISTS idx_code_architectures_type ON code_architectures(type);
			CREATE INDEX IF NOT EXISTS idx_code_usage_chunk ON code_usage(chunk_id);
			CREATE INDEX IF NOT EXISTS idx_code_usage_type ON code_usage(usage_type);
			CREATE INDEX IF NOT EXISTS idx_user_feedback_session ON user_feedback(session_id);
			CREATE INDEX IF NOT EXISTS idx_user_feedback_action ON user_feedback(action);
			CREATE INDEX IF NOT EXISTS idx_user_feedback_rating ON user_feedback(rating);
			CREATE INDEX IF NOT EXISTS idx_accuracy_metrics_provider ON accuracy_metrics(provider);
			CREATE INDEX IF NOT EXISTS idx_learning_patterns_category ON learning_patterns(category);
			CREATE INDEX IF NOT EXISTS idx_learning_patterns_success_rate ON learning_patterns(success_rate);
		`,
		DownSQL: `
			DROP INDEX IF EXISTS idx_learning_patterns_success_rate;
			DROP INDEX IF EXISTS idx_learning_patterns_category;
			DROP INDEX IF EXISTS idx_accuracy_metrics_provider;
			DROP INDEX IF EXISTS idx_user_feedback_rating;
			DROP INDEX IF EXISTS idx_user_feedback_action;
			DROP INDEX IF EXISTS idx_user_feedback_session;
			DROP INDEX IF EXISTS idx_code_usage_type;
			DROP INDEX IF EXISTS idx_code_usage_chunk;
			DROP INDEX IF EXISTS idx_code_architectures_type;
			DROP INDEX IF EXISTS idx_code_architectures_project;
			DROP INDEX IF EXISTS idx_code_patterns_frequency;
			DROP INDEX IF EXISTS idx_code_patterns_type;
			DROP TABLE IF EXISTS learning_patterns;
			DROP TABLE IF EXISTS accuracy_metrics;
			DROP TABLE IF EXISTS user_feedback;
			DROP TABLE IF EXISTS code_usage;
			DROP TABLE IF EXISTS code_architectures;
			DROP TABLE IF EXISTS code_patterns;
		`,
	})

	// Migration 006: Indexing jobs and performance
	mm.migrations = append(mm.migrations, Migration{
		Version:     6,
		Name:        "indexing_performance",
		Description: "Add tables for indexing jobs and performance tracking",
		UpSQL: `
			-- Create indexing_jobs table
			CREATE TABLE IF NOT EXISTS indexing_jobs (
				id TEXT PRIMARY KEY,
				type TEXT NOT NULL,
				status TEXT DEFAULT 'pending',
				project_path TEXT NOT NULL,
				file_paths TEXT DEFAULT '[]',
				priority INTEGER DEFAULT 0,
				progress REAL DEFAULT 0.0,
				message TEXT,
				error TEXT,
				started_at DATETIME,
				completed_at DATETIME,
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);

			-- Create performance_metrics table
			CREATE TABLE IF NOT EXISTS performance_metrics (
				id TEXT PRIMARY KEY,
				component TEXT NOT NULL,
				operation TEXT NOT NULL,
				duration INTEGER NOT NULL,
				memory_used INTEGER,
				cpu_used REAL,
				success BOOLEAN DEFAULT 1,
				error TEXT,
				metadata TEXT DEFAULT '{}',
				recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);

			-- Create system_health table
			CREATE TABLE IF NOT EXISTS system_health (
				id TEXT PRIMARY KEY,
				component TEXT NOT NULL,
				status TEXT NOT NULL,
				health_score REAL DEFAULT 1.0,
				last_check DATETIME DEFAULT CURRENT_TIMESTAMP,
				response_time INTEGER,
				error_count INTEGER DEFAULT 0,
				uptime INTEGER DEFAULT 0,
				metadata TEXT DEFAULT '{}',
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);

			-- Create cache_entries table (for performance optimization)
			CREATE TABLE IF NOT EXISTS cache_entries (
				key TEXT PRIMARY KEY,
				value TEXT NOT NULL,
				expires_at DATETIME,
				category TEXT,
				size INTEGER DEFAULT 0,
				hit_count INTEGER DEFAULT 0,
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);

			-- Create search_history table
			CREATE TABLE IF NOT EXISTS search_history (
				id TEXT PRIMARY KEY,
				session_id TEXT NOT NULL,
				query TEXT NOT NULL,
				results_count INTEGER DEFAULT 0,
				response_time INTEGER DEFAULT 0,
				search_type TEXT,
				confidence REAL DEFAULT 0.0,
				clicked_results TEXT DEFAULT '[]',
				metadata TEXT DEFAULT '{}',
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
				FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
			);

			-- Create indexes
			CREATE INDEX IF NOT EXISTS idx_indexing_jobs_status ON indexing_jobs(status);
			CREATE INDEX IF NOT EXISTS idx_indexing_jobs_type ON indexing_jobs(type);
			CREATE INDEX IF NOT EXISTS idx_indexing_jobs_priority ON indexing_jobs(priority);
			CREATE INDEX IF NOT EXISTS idx_performance_metrics_component ON performance_metrics(component);
			CREATE INDEX IF NOT EXISTS idx_performance_metrics_operation ON performance_metrics(operation);
			CREATE INDEX IF NOT EXISTS idx_performance_metrics_recorded_at ON performance_metrics(recorded_at);
			CREATE INDEX IF NOT EXISTS idx_system_health_component ON system_health(component);
			CREATE INDEX IF NOT EXISTS idx_system_health_status ON system_health(status);
			CREATE INDEX IF NOT EXISTS idx_cache_entries_expires_at ON cache_entries(expires_at);
			CREATE INDEX IF NOT EXISTS idx_cache_entries_category ON cache_entries(category);
			CREATE INDEX IF NOT EXISTS idx_search_history_session ON search_history(session_id);
			CREATE INDEX IF NOT EXISTS idx_search_history_created_at ON search_history(created_at);
		`,
		DownSQL: `
			DROP INDEX IF EXISTS idx_search_history_created_at;
			DROP INDEX IF EXISTS idx_search_history_session;
			DROP INDEX IF EXISTS idx_cache_entries_category;
			DROP INDEX IF EXISTS idx_cache_entries_expires_at;
			DROP INDEX IF EXISTS idx_system_health_status;
			DROP INDEX IF EXISTS idx_system_health_component;
			DROP INDEX IF EXISTS idx_performance_metrics_recorded_at;
			DROP INDEX IF EXISTS idx_performance_metrics_operation;
			DROP INDEX IF EXISTS idx_performance_metrics_component;
			DROP INDEX IF EXISTS idx_indexing_jobs_priority;
			DROP INDEX IF EXISTS idx_indexing_jobs_type;
			DROP INDEX IF EXISTS idx_indexing_jobs_status;
			DROP TABLE IF EXISTS search_history;
			DROP TABLE IF EXISTS cache_entries;
			DROP TABLE IF EXISTS system_health;
			DROP TABLE IF EXISTS performance_metrics;
			DROP TABLE IF EXISTS indexing_jobs;
		`,
	})

	// Sort migrations by version
	sort.Slice(mm.migrations, func(i, j int) bool {
		return mm.migrations[i].Version < mm.migrations[j].Version
	})
}

// createMigrationsTable creates the migrations tracking table
func (mm *MigrationManager) createMigrationsTable() error {
	query := `
		CREATE TABLE IF NOT EXISTS schema_migrations (
			version INTEGER PRIMARY KEY,
			name TEXT NOT NULL,
			description TEXT,
			applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			checksum TEXT
		)
	`
	_, err := mm.db.Execute(query)
	return err
}

// RunMigrations runs all pending migrations
func (mm *MigrationManager) RunMigrations() error {
	if !mm.db.IsConnected() {
		return fmt.Errorf("database not connected")
	}

	// Create migrations table if it doesn't exist
	if err := mm.createMigrationsTable(); err != nil {
		return fmt.Errorf("failed to create migrations table: %v", err)
	}

	// Get applied migrations
	appliedVersions, err := mm.getAppliedVersions()
	if err != nil {
		return fmt.Errorf("failed to get applied migrations: %v", err)
	}

	// Apply pending migrations
	for _, migration := range mm.migrations {
		if _, applied := appliedVersions[migration.Version]; !applied {
			if err := mm.applyMigration(migration); err != nil {
				return fmt.Errorf("failed to apply migration %d: %v", migration.Version, err)
			}
		}
	}

	return nil
}

// getAppliedVersions returns a set of applied migration versions
func (mm *MigrationManager) getAppliedVersions() (map[int]bool, error) {
	query := "SELECT version FROM schema_migrations"
	result, err := mm.db.Query(query)
	if err != nil {
		return nil, err
	}

	applied := make(map[int]bool)
	for _, row := range result.Rows {
		if version, ok := row["version"].(int64); ok {
			applied[int(version)] = true
		}
	}

	return applied, nil
}

// applyMigration applies a single migration
func (mm *MigrationManager) applyMigration(migration Migration) error {
	return mm.db.Transaction(func(tx *sql.Tx) error {
		// Split SQL by semicolon and execute each statement
		statements := strings.Split(migration.UpSQL, ";")
		for _, stmt := range statements {
			stmt = strings.TrimSpace(stmt)
			if stmt == "" {
				continue
			}

			if _, err := tx.Exec(stmt); err != nil {
				return fmt.Errorf("failed to execute statement: %v", err)
			}
		}

		// Record migration as applied
		insertQuery := `
			INSERT INTO schema_migrations (version, name, description, applied_at) 
			VALUES (?, ?, ?, ?)
		`
		if _, err := tx.Exec(insertQuery, migration.Version, migration.Name, migration.Description, time.Now()); err != nil {
			return fmt.Errorf("failed to record migration: %v", err)
		}

		return nil
	})
}

// RollbackMigration rolls back a specific migration
func (mm *MigrationManager) RollbackMigration(version int) error {
	if !mm.db.IsConnected() {
		return fmt.Errorf("database not connected")
	}

	// Find the migration
	var migration *Migration
	for _, m := range mm.migrations {
		if m.Version == version {
			migration = &m
			break
		}
	}

	if migration == nil {
		return fmt.Errorf("migration version %d not found", version)
	}

	// Check if migration is applied
	appliedVersions, err := mm.getAppliedVersions()
	if err != nil {
		return err
	}

	if _, applied := appliedVersions[version]; !applied {
		return fmt.Errorf("migration %d is not applied", version)
	}

	// Apply rollback
	return mm.db.Transaction(func(tx *sql.Tx) error {
		// Execute rollback SQL
		statements := strings.Split(migration.DownSQL, ";")
		for _, stmt := range statements {
			stmt = strings.TrimSpace(stmt)
			if stmt == "" {
				continue
			}

			if _, err := tx.Exec(stmt); err != nil {
				return fmt.Errorf("failed to execute rollback statement: %v", err)
			}
		}

		// Remove migration record
		deleteQuery := "DELETE FROM schema_migrations WHERE version = ?"
		if _, err := tx.Exec(deleteQuery, version); err != nil {
			return fmt.Errorf("failed to remove migration record: %v", err)
		}

		return nil
	})
}

// GetMigrationStatus returns the status of all migrations
func (mm *MigrationManager) GetMigrationStatus() ([]MigrationStatus, error) {
	appliedVersions, err := mm.getAppliedVersions()
	if err != nil {
		return nil, err
	}

	status := make([]MigrationStatus, len(mm.migrations))
	for i, migration := range mm.migrations {
		_, applied := appliedVersions[migration.Version]
		status[i] = MigrationStatus{
			Version:     migration.Version,
			Name:        migration.Name,
			Description: migration.Description,
			Applied:     applied,
		}

		if applied {
			// Get applied time
			query := "SELECT applied_at FROM schema_migrations WHERE version = ?"
			if row, err := mm.db.QueryRow(query, migration.Version); err == nil {
				if appliedAt, ok := row["applied_at"].(string); ok {
					if parsed, err := time.Parse("2006-01-02 15:04:05", appliedAt); err == nil {
						status[i].AppliedAt = &parsed
					}
				}
			}
		}
	}

	return status, nil
}

// MigrationStatus represents the status of a migration
type MigrationStatus struct {
	Version     int        `json:"version"`
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Applied     bool       `json:"applied"`
	AppliedAt   *time.Time `json:"applied_at,omitempty"`
}

// ResetDatabase drops all tables and re-runs migrations (DANGEROUS)
func (mm *MigrationManager) ResetDatabase() error {
	if !mm.db.IsConnected() {
		return fmt.Errorf("database not connected")
	}

	// Get all table names
	query := "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
	result, err := mm.db.Query(query)
	if err != nil {
		return err
	}

	// Drop all tables
	return mm.db.Transaction(func(tx *sql.Tx) error {
		// Disable foreign key constraints temporarily
		if _, err := tx.Exec("PRAGMA foreign_keys = OFF"); err != nil {
			return err
		}

		// Drop all tables
		for _, row := range result.Rows {
			tableName := row["name"].(string)
			dropQuery := fmt.Sprintf("DROP TABLE IF EXISTS %s", tableName)
			if _, err := tx.Exec(dropQuery); err != nil {
				return fmt.Errorf("failed to drop table %s: %v", tableName, err)
			}
		}

		// Re-enable foreign key constraints
		if _, err := tx.Exec("PRAGMA foreign_keys = ON"); err != nil {
			return err
		}

		return nil
	})
}

// GetLatestVersion returns the latest migration version
func (mm *MigrationManager) GetLatestVersion() int {
	if len(mm.migrations) == 0 {
		return 0
	}
	return mm.migrations[len(mm.migrations)-1].Version
}

// GetCurrentVersion returns the current applied migration version
func (mm *MigrationManager) GetCurrentVersion() (int, error) {
	query := "SELECT MAX(version) as version FROM schema_migrations"
	row, err := mm.db.QueryRow(query)
	if err != nil {
		return 0, err
	}

	if version, ok := row["version"].(int64); ok {
		return int(version), nil
	}

	return 0, nil
}
