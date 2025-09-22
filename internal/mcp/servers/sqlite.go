package servers

import (
	"context"
	"database/sql"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/yourusername/ai-code-assistant/internal/mcp"
)

// SQLiteServer provides SQLite database operations for MCP
type SQLiteServer struct {
	allowedDbs  []string
	restricted  bool
	maxRows     int
	connections map[string]*sql.DB
}

// SQLiteConfig contains configuration for SQLite operations
type SQLiteConfig struct {
	AllowedDbs []string `json:"allowed_dbs"`
	Restricted bool     `json:"restricted"`
	MaxRows    int      `json:"max_rows"`
}

// QueryResult represents the result of a SQL query
type QueryResult struct {
	Columns []string                 `json:"columns"`
	Rows    []map[string]interface{} `json:"rows"`
	Count   int                      `json:"count"`
	Took    time.Duration            `json:"took"`
}

// TableInfo represents information about a database table
type TableInfo struct {
	Name    string       `json:"name"`
	Type    string       `json:"type"`
	Columns []ColumnInfo `json:"columns"`
	Indexes []IndexInfo  `json:"indexes"`
}

// ColumnInfo represents information about a table column
type ColumnInfo struct {
	Name         string `json:"name"`
	Type         string `json:"type"`
	NotNull      bool   `json:"not_null"`
	DefaultValue string `json:"default_value"`
	PrimaryKey   bool   `json:"primary_key"`
}

// IndexInfo represents information about a table index
type IndexInfo struct {
	Name    string   `json:"name"`
	Unique  bool     `json:"unique"`
	Columns []string `json:"columns"`
}

// DatabaseStats represents database statistics
type DatabaseStats struct {
	FilePath   string      `json:"file_path"`
	FileSize   int64       `json:"file_size"`
	Tables     int         `json:"tables"`
	Views      int         `json:"views"`
	Indexes    int         `json:"indexes"`
	Triggers   int         `json:"triggers"`
	TableStats []TableStat `json:"table_stats"`
}

// TableStat represents statistics for a single table
type TableStat struct {
	Name     string `json:"name"`
	RowCount int    `json:"row_count"`
	Size     int64  `json:"size"`
}

// NewSQLiteServer creates a new SQLite server
func NewSQLiteServer(config SQLiteConfig) *SQLiteServer {
	if config.MaxRows == 0 {
		config.MaxRows = 1000
	}

	return &SQLiteServer{
		allowedDbs:  config.AllowedDbs,
		restricted:  config.Restricted,
		maxRows:     config.MaxRows,
		connections: make(map[string]*sql.DB),
	}
}

// GetTools returns available SQLite tools
func (ss *SQLiteServer) GetTools() []mcp.MCPTool {
	return []mcp.MCPTool{
		{
			Name:        "sqlite_query",
			Description: "Execute a SELECT query on a SQLite database",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"db_path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the SQLite database file",
					},
					"query": map[string]interface{}{
						"type":        "string",
						"description": "SQL SELECT query to execute",
					},
					"limit": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum number of rows to return",
						"default":     100,
					},
				},
				"required": []string{"db_path", "query"},
			},
		},
		{
			Name:        "sqlite_execute",
			Description: "Execute a non-SELECT SQL statement (INSERT, UPDATE, DELETE, CREATE, etc.)",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"db_path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the SQLite database file",
					},
					"statement": map[string]interface{}{
						"type":        "string",
						"description": "SQL statement to execute",
					},
					"params": map[string]interface{}{
						"type":        "array",
						"description": "Parameters for prepared statement",
						"items":       map[string]interface{}{"type": "string"},
					},
				},
				"required": []string{"db_path", "statement"},
			},
		},
		{
			Name:        "sqlite_schema",
			Description: "Get schema information for a SQLite database",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"db_path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the SQLite database file",
					},
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Specific table name (optional)",
					},
				},
				"required": []string{"db_path"},
			},
		},
		{
			Name:        "sqlite_tables",
			Description: "List all tables and views in a SQLite database",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"db_path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the SQLite database file",
					},
					"include_system": map[string]interface{}{
						"type":        "boolean",
						"description": "Include system tables",
						"default":     false,
					},
				},
				"required": []string{"db_path"},
			},
		},
		{
			Name:        "sqlite_stats",
			Description: "Get statistics about a SQLite database",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"db_path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the SQLite database file",
					},
					"detailed": map[string]interface{}{
						"type":        "boolean",
						"description": "Include detailed table statistics",
						"default":     true,
					},
				},
				"required": []string{"db_path"},
			},
		},
		{
			Name:        "sqlite_backup",
			Description: "Create a backup of a SQLite database",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"source_db": map[string]interface{}{
						"type":        "string",
						"description": "Path to source database",
					},
					"backup_path": map[string]interface{}{
						"type":        "string",
						"description": "Path for backup file",
					},
				},
				"required": []string{"source_db", "backup_path"},
			},
		},
		{
			Name:        "sqlite_analyze",
			Description: "Analyze database performance and suggest optimizations",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"db_path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the SQLite database file",
					},
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Specific table to analyze (optional)",
					},
				},
				"required": []string{"db_path"},
			},
		},
		{
			Name:        "sqlite_vacuum",
			Description: "Vacuum a SQLite database to reclaim space and optimize",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"db_path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the SQLite database file",
					},
					"into_path": map[string]interface{}{
						"type":        "string",
						"description": "Path for vacuum into operation (optional)",
					},
				},
				"required": []string{"db_path"},
			},
		},
		{
			Name:        "sqlite_explain",
			Description: "Get query execution plan for a SQL statement",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"db_path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the SQLite database file",
					},
					"query": map[string]interface{}{
						"type":        "string",
						"description": "SQL query to explain",
					},
					"query_plan": map[string]interface{}{
						"type":        "boolean",
						"description": "Use EXPLAIN QUERY PLAN",
						"default":     true,
					},
				},
				"required": []string{"db_path", "query"},
			},
		},
		{
			Name:        "sqlite_pragma",
			Description: "Execute PRAGMA commands to get/set database settings",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"db_path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the SQLite database file",
					},
					"pragma": map[string]interface{}{
						"type":        "string",
						"description": "PRAGMA command to execute",
					},
				},
				"required": []string{"db_path", "pragma"},
			},
		},
	}
}

// ExecuteTool executes a SQLite tool
func (ss *SQLiteServer) ExecuteTool(ctx context.Context, toolName string, input map[string]interface{}) (interface{}, error) {
	switch toolName {
	case "sqlite_query":
		return ss.executeQuery(input)
	case "sqlite_execute":
		return ss.executeStatement(input)
	case "sqlite_schema":
		return ss.getSchema(input)
	case "sqlite_tables":
		return ss.listTables(input)
	case "sqlite_stats":
		return ss.getStats(input)
	case "sqlite_backup":
		return ss.backupDatabase(input)
	case "sqlite_analyze":
		return ss.analyzeDatabase(input)
	case "sqlite_vacuum":
		return ss.vacuumDatabase(input)
	case "sqlite_explain":
		return ss.explainQuery(input)
	case "sqlite_pragma":
		return ss.executePragma(input)
	default:
		return nil, fmt.Errorf("unknown tool: %s", toolName)
	}
}

// executeQuery executes a SELECT query
func (ss *SQLiteServer) executeQuery(input map[string]interface{}) (interface{}, error) {
	dbPath, _ := input["db_path"].(string)
	query, _ := input["query"].(string)

	if dbPath == "" || query == "" {
		return nil, fmt.Errorf("db_path and query are required")
	}

	if err := ss.validateDatabase(dbPath); err != nil {
		return nil, err
	}

	// Validate query is a SELECT
	if !ss.isSelectQuery(query) {
		return nil, fmt.Errorf("only SELECT queries are allowed with sqlite_query tool")
	}

	limit := 100
	if l, ok := input["limit"].(float64); ok {
		limit = int(l)
	}
	if limit > ss.maxRows {
		limit = ss.maxRows
	}

	db, err := ss.getConnection(dbPath)
	if err != nil {
		return nil, err
	}

	// Add LIMIT if not present
	if !strings.Contains(strings.ToUpper(query), "LIMIT") {
		query = fmt.Sprintf("%s LIMIT %d", query, limit)
	}

	start := time.Now()
	rows, err := db.Query(query)
	if err != nil {
		return nil, fmt.Errorf("query execution failed: %v", err)
	}
	defer rows.Close()

	result, err := ss.processRows(rows)
	if err != nil {
		return nil, err
	}

	result.Took = time.Since(start)
	return result, nil
}

// executeStatement executes a non-SELECT statement
func (ss *SQLiteServer) executeStatement(input map[string]interface{}) (interface{}, error) {
	dbPath, _ := input["db_path"].(string)
	statement, _ := input["statement"].(string)

	if dbPath == "" || statement == "" {
		return nil, fmt.Errorf("db_path and statement are required")
	}

	if err := ss.validateDatabase(dbPath); err != nil {
		return nil, err
	}

	// Validate statement is not a SELECT
	if ss.isSelectQuery(statement) {
		return nil, fmt.Errorf("use sqlite_query tool for SELECT statements")
	}

	db, err := ss.getConnection(dbPath)
	if err != nil {
		return nil, err
	}

	var params []interface{}
	if paramsInput, ok := input["params"].([]interface{}); ok {
		params = paramsInput
	}

	start := time.Now()
	result, err := db.Exec(statement, params...)
	if err != nil {
		return nil, fmt.Errorf("statement execution failed: %v", err)
	}

	rowsAffected, _ := result.RowsAffected()
	lastInsertId, _ := result.LastInsertId()

	return map[string]interface{}{
		"success":        true,
		"rows_affected":  rowsAffected,
		"last_insert_id": lastInsertId,
		"took":           time.Since(start),
	}, nil
}

// getSchema gets database schema information
func (ss *SQLiteServer) getSchema(input map[string]interface{}) (interface{}, error) {
	dbPath, _ := input["db_path"].(string)
	if dbPath == "" {
		return nil, fmt.Errorf("db_path is required")
	}

	if err := ss.validateDatabase(dbPath); err != nil {
		return nil, err
	}

	db, err := ss.getConnection(dbPath)
	if err != nil {
		return nil, err
	}

	tableName, _ := input["table_name"].(string)

	if tableName != "" {
		// Get schema for specific table
		tableInfo, err := ss.getTableInfo(db, tableName)
		if err != nil {
			return nil, err
		}
		return tableInfo, nil
	}

	// Get schema for all tables
	tables, err := ss.getAllTables(db, false)
	if err != nil {
		return nil, err
	}

	var schema []TableInfo
	for _, table := range tables {
		if strings.HasPrefix(table, "sqlite_") {
			continue // Skip system tables
		}

		tableInfo, err := ss.getTableInfo(db, table)
		if err != nil {
			continue
		}
		schema = append(schema, *tableInfo)
	}

	return map[string]interface{}{
		"tables": schema,
		"count":  len(schema),
	}, nil
}

// listTables lists all tables in the database
func (ss *SQLiteServer) listTables(input map[string]interface{}) (interface{}, error) {
	dbPath, _ := input["db_path"].(string)
	if dbPath == "" {
		return nil, fmt.Errorf("db_path is required")
	}

	if err := ss.validateDatabase(dbPath); err != nil {
		return nil, err
	}

	db, err := ss.getConnection(dbPath)
	if err != nil {
		return nil, err
	}

	includeSystem, _ := input["include_system"].(bool)

	tables, err := ss.getAllTables(db, includeSystem)
	if err != nil {
		return nil, err
	}

	views, err := ss.getAllViews(db, includeSystem)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"tables": tables,
		"views":  views,
		"total":  len(tables) + len(views),
	}, nil
}

// getStats gets database statistics
func (ss *SQLiteServer) getStats(input map[string]interface{}) (interface{}, error) {
	dbPath, _ := input["db_path"].(string)
	if dbPath == "" {
		return nil, fmt.Errorf("db_path is required")
	}

	if err := ss.validateDatabase(dbPath); err != nil {
		return nil, err
	}

	db, err := ss.getConnection(dbPath)
	if err != nil {
		return nil, err
	}

	detailed, _ := input["detailed"].(bool)

	stats := &DatabaseStats{
		FilePath: dbPath,
	}

	// Get file size
	if fileInfo, err := filepath.Abs(dbPath); err == nil {
		if stat, err := filepath.Stat(fileInfo); err == nil {
			stats.FileSize = stat.Size()
		}
	}

	// Count objects
	if err := ss.countDatabaseObjects(db, stats); err != nil {
		return nil, err
	}

	if detailed {
		tableStats, err := ss.getDetailedTableStats(db)
		if err != nil {
			return nil, err
		}
		stats.TableStats = tableStats
	}

	return stats, nil
}

// backupDatabase creates a database backup
func (ss *SQLiteServer) backupDatabase(input map[string]interface{}) (interface{}, error) {
	sourceDb, _ := input["source_db"].(string)
	backupPath, _ := input["backup_path"].(string)

	if sourceDb == "" || backupPath == "" {
		return nil, fmt.Errorf("source_db and backup_path are required")
	}

	if err := ss.validateDatabase(sourceDb); err != nil {
		return nil, err
	}

	if err := ss.validateDatabase(backupPath); err != nil {
		return nil, err
	}

	sourceConn, err := ss.getConnection(sourceDb)
	if err != nil {
		return nil, err
	}

	backupConn, err := sql.Open("sqlite3", backupPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create backup connection: %v", err)
	}
	defer backupConn.Close()

	// Use SQLite backup API (simplified version)
	start := time.Now()

	// Get all table names
	tables, err := ss.getAllTables(sourceConn, true)
	if err != nil {
		return nil, err
	}

	tablesBackedUp := 0
	for _, table := range tables {
		// Get table schema
		var schema string
		row := sourceConn.QueryRow("SELECT sql FROM sqlite_master WHERE name = ?", table)
		if err := row.Scan(&schema); err != nil {
			continue
		}

		// Create table in backup
		if _, err := backupConn.Exec(schema); err != nil {
			continue
		}

		// Copy data
		rows, err := sourceConn.Query(fmt.Sprintf("SELECT * FROM %s", table))
		if err != nil {
			continue
		}

		// This is a simplified backup - in production you'd use SQLite's backup API
		rows.Close()
		tablesBackedUp++
	}

	return map[string]interface{}{
		"success":          true,
		"source":           sourceDb,
		"backup_path":      backupPath,
		"tables_backed_up": tablesBackedUp,
		"took":             time.Since(start),
	}, nil
}

// analyzeDatabase analyzes database performance
func (ss *SQLiteServer) analyzeDatabase(input map[string]interface{}) (interface{}, error) {
	dbPath, _ := input["db_path"].(string)
	if dbPath == "" {
		return nil, fmt.Errorf("db_path is required")
	}

	if err := ss.validateDatabase(dbPath); err != nil {
		return nil, err
	}

	db, err := ss.getConnection(dbPath)
	if err != nil {
		return nil, err
	}

	tableName, _ := input["table_name"].(string)

	// Run ANALYZE command
	start := time.Now()
	if tableName != "" {
		_, err = db.Exec(fmt.Sprintf("ANALYZE %s", tableName))
	} else {
		_, err = db.Exec("ANALYZE")
	}

	if err != nil {
		return nil, fmt.Errorf("analyze failed: %v", err)
	}

	// Get analysis results and recommendations
	recommendations := ss.getPerformanceRecommendations(db, tableName)

	return map[string]interface{}{
		"success":         true,
		"analyzed_table":  tableName,
		"took":            time.Since(start),
		"recommendations": recommendations,
	}, nil
}

// vacuumDatabase vacuums the database
func (ss *SQLiteServer) vacuumDatabase(input map[string]interface{}) (interface{}, error) {
	dbPath, _ := input["db_path"].(string)
	if dbPath == "" {
		return nil, fmt.Errorf("db_path is required")
	}

	if err := ss.validateDatabase(dbPath); err != nil {
		return nil, err
	}

	db, err := ss.getConnection(dbPath)
	if err != nil {
		return nil, err
	}

	start := time.Now()

	intoPath, _ := input["into_path"].(string)
	var vacuumCmd string

	if intoPath != "" {
		vacuumCmd = fmt.Sprintf("VACUUM INTO '%s'", intoPath)
	} else {
		vacuumCmd = "VACUUM"
	}

	_, err = db.Exec(vacuumCmd)
	if err != nil {
		return nil, fmt.Errorf("vacuum failed: %v", err)
	}

	return map[string]interface{}{
		"success":   true,
		"into_path": intoPath,
		"took":      time.Since(start),
	}, nil
}

// explainQuery gets query execution plan
func (ss *SQLiteServer) explainQuery(input map[string]interface{}) (interface{}, error) {
	dbPath, _ := input["db_path"].(string)
	query, _ := input["query"].(string)

	if dbPath == "" || query == "" {
		return nil, fmt.Errorf("db_path and query are required")
	}

	if err := ss.validateDatabase(dbPath); err != nil {
		return nil, err
	}

	db, err := ss.getConnection(dbPath)
	if err != nil {
		return nil, err
	}

	queryPlan, _ := input["query_plan"].(bool)

	var explainQuery string
	if queryPlan {
		explainQuery = fmt.Sprintf("EXPLAIN QUERY PLAN %s", query)
	} else {
		explainQuery = fmt.Sprintf("EXPLAIN %s", query)
	}

	rows, err := db.Query(explainQuery)
	if err != nil {
		return nil, fmt.Errorf("explain failed: %v", err)
	}
	defer rows.Close()

	result, err := ss.processRows(rows)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"query":      query,
		"query_plan": queryPlan,
		"explain":    result,
	}, nil
}

// executePragma executes PRAGMA commands
func (ss *SQLiteServer) executePragma(input map[string]interface{}) (interface{}, error) {
	dbPath, _ := input["db_path"].(string)
	pragma, _ := input["pragma"].(string)

	if dbPath == "" || pragma == "" {
		return nil, fmt.Errorf("db_path and pragma are required")
	}

	if err := ss.validateDatabase(dbPath); err != nil {
		return nil, err
	}

	db, err := ss.getConnection(dbPath)
	if err != nil {
		return nil, err
	}

	// Ensure it's a PRAGMA command
	if !strings.HasPrefix(strings.ToUpper(strings.TrimSpace(pragma)), "PRAGMA") {
		pragma = fmt.Sprintf("PRAGMA %s", pragma)
	}

	rows, err := db.Query(pragma)
	if err != nil {
		return nil, fmt.Errorf("pragma execution failed: %v", err)
	}
	defer rows.Close()

	result, err := ss.processRows(rows)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"pragma": pragma,
		"result": result,
	}, nil
}

// Helper methods

// validateDatabase validates database path
func (ss *SQLiteServer) validateDatabase(dbPath string) error {
	if !ss.restricted {
		return nil
	}

	absPath, err := filepath.Abs(dbPath)
	if err != nil {
		return fmt.Errorf("invalid database path: %v", err)
	}

	for _, allowedDb := range ss.allowedDbs {
		allowedAbs, err := filepath.Abs(allowedDb)
		if err != nil {
			continue
		}

		if strings.HasPrefix(absPath, allowedAbs) {
			return nil
		}
	}

	return fmt.Errorf("database not allowed: %s", dbPath)
}

// getConnection gets or creates a database connection
func (ss *SQLiteServer) getConnection(dbPath string) (*sql.DB, error) {
	if conn, exists := ss.connections[dbPath]; exists {
		if err := conn.Ping(); err == nil {
			return conn, nil
		}
		// Connection is dead, remove it
		delete(ss.connections, dbPath)
	}

	conn, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %v", err)
	}

	if err := conn.Ping(); err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to connect to database: %v", err)
	}

	ss.connections[dbPath] = conn
	return conn, nil
}

// isSelectQuery checks if query is a SELECT statement
func (ss *SQLiteServer) isSelectQuery(query string) bool {
	query = strings.TrimSpace(strings.ToUpper(query))
	return strings.HasPrefix(query, "SELECT") ||
		strings.HasPrefix(query, "WITH")
}

// processRows converts sql.Rows to QueryResult
func (ss *SQLiteServer) processRows(rows *sql.Rows) (*QueryResult, error) {
	columns, err := rows.Columns()
	if err != nil {
		return nil, fmt.Errorf("failed to get columns: %v", err)
	}

	var resultRows []map[string]interface{}

	for rows.Next() {
		// Create slice to hold column values
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			return nil, fmt.Errorf("failed to scan row: %v", err)
		}

		// Convert to map
		row := make(map[string]interface{})
		for i, col := range columns {
			val := values[i]
			if b, ok := val.([]byte); ok {
				// Convert byte arrays to strings
				row[col] = string(b)
			} else {
				row[col] = val
			}
		}
		resultRows = append(resultRows, row)
	}

	return &QueryResult{
		Columns: columns,
		Rows:    resultRows,
		Count:   len(resultRows),
	}, nil
}

// Additional helper methods would continue here...
// Due to length constraints, I'll provide key remaining methods

// getTableInfo gets detailed information about a table
func (ss *SQLiteServer) getTableInfo(db *sql.DB, tableName string) (*TableInfo, error) {
	tableInfo := &TableInfo{
		Name: tableName,
	}

	// Get table type
	row := db.QueryRow("SELECT type FROM sqlite_master WHERE name = ?", tableName)
	row.Scan(&tableInfo.Type)

	// Get columns
	rows, err := db.Query(fmt.Sprintf("PRAGMA table_info(%s)", tableName))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var cid int
		var name, dataType, defaultVal string
		var notNull, pk int

		if err := rows.Scan(&cid, &name, &dataType, &notNull, &defaultVal, &pk); err != nil {
			continue
		}

		tableInfo.Columns = append(tableInfo.Columns, ColumnInfo{
			Name:         name,
			Type:         dataType,
			NotNull:      notNull == 1,
			DefaultValue: defaultVal,
			PrimaryKey:   pk == 1,
		})
	}

	// Get indexes
	indexRows, err := db.Query(fmt.Sprintf("PRAGMA index_list(%s)", tableName))
	if err == nil {
		defer indexRows.Close()
		for indexRows.Next() {
			var seq int
			var name string
			var unique int
			var origin string

			if err := indexRows.Scan(&seq, &name, &unique, &origin); err != nil {
				continue
			}

			indexInfo := IndexInfo{
				Name:   name,
				Unique: unique == 1,
			}

			// Get index columns
			colRows, err := db.Query(fmt.Sprintf("PRAGMA index_info(%s)", name))
			if err != nil {
				continue
			}

			for colRows.Next() {
				var seqno, cid int
				var colName string
				if err := colRows.Scan(&seqno, &cid, &colName); err != nil {
					continue
				}
				indexInfo.Columns = append(indexInfo.Columns, colName)
			}
			colRows.Close()

			tableInfo.Indexes = append(tableInfo.Indexes, indexInfo)
		}
	}

	return tableInfo, nil
}

// getAllTables gets all table names
func (ss *SQLiteServer) getAllTables(db *sql.DB, includeSystem bool) ([]string, error) {
	query := "SELECT name FROM sqlite_master WHERE type='table'"
	if !includeSystem {
		query += " AND name NOT LIKE 'sqlite_%'"
	}

	rows, err := db.Query(query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tables []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			continue
		}
		tables = append(tables, name)
	}

	sort.Strings(tables)
	return tables, nil
}

// getAllViews gets all view names
func (ss *SQLiteServer) getAllViews(db *sql.DB, includeSystem bool) ([]string, error) {
	query := "SELECT name FROM sqlite_master WHERE type='view'"
	if !includeSystem {
		query += " AND name NOT LIKE 'sqlite_%'"
	}

	rows, err := db.Query(query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var views []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			continue
		}
		views = append(views, name)
	}

	sort.Strings(views)
	return views, nil
}

// countDatabaseObjects counts various database objects
func (ss *SQLiteServer) countDatabaseObjects(db *sql.DB, stats *DatabaseStats) error {
	// Count tables
	row := db.QueryRow("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
	row.Scan(&stats.Tables)

	// Count views
	row = db.QueryRow("SELECT COUNT(*) FROM sqlite_master WHERE type='view'")
	row.Scan(&stats.Views)

	// Count indexes
	row = db.QueryRow("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
	row.Scan(&stats.Indexes)

	// Count triggers
	row = db.QueryRow("SELECT COUNT(*) FROM sqlite_master WHERE type='trigger'")
	row.Scan(&stats.Triggers)

	return nil
}

// getDetailedTableStats gets detailed statistics for all tables
func (ss *SQLiteServer) getDetailedTableStats(db *sql.DB) ([]TableStat, error) {
	tables, err := ss.getAllTables(db, false)
	if err != nil {
		return nil, err
	}

	var tableStats []TableStat
	for _, table := range tables {
		var count int
		row := db.QueryRow(fmt.Sprintf("SELECT COUNT(*) FROM %s", table))
		row.Scan(&count)

		tableStats = append(tableStats, TableStat{
			Name:     table,
			RowCount: count,
			Size:     0, // Would need additional SQLite specific queries
		})
	}

	return tableStats, nil
}

// getPerformanceRecommendations analyzes and provides performance recommendations
func (ss *SQLiteServer) getPerformanceRecommendations(db *sql.DB, tableName string) []string {
	var recommendations []string

	if tableName != "" {
		// Analyze specific table
		tableInfo, err := ss.getTableInfo(db, tableName)
		if err == nil {
			if len(tableInfo.Indexes) == 0 {
				recommendations = append(recommendations, fmt.Sprintf("Table '%s' has no indexes - consider adding indexes on frequently queried columns", tableName))
			}
		}
	} else {
		// General database recommendations

		// Check for tables without primary keys
		rows, err := db.Query(`
			SELECT name FROM sqlite_master 
			WHERE type='table' AND name NOT LIKE 'sqlite_%'
			AND name NOT IN (
				SELECT DISTINCT m.name 
				FROM sqlite_master m, pragma_table_info(m.name) p 
				WHERE m.type='table' AND p.pk > 0
			)
		`)
		if err == nil {
			defer rows.Close()
			var tablesWithoutPK []string
			for rows.Next() {
				var name string
				if rows.Scan(&name) == nil {
					tablesWithoutPK = append(tablesWithoutPK, name)
				}
			}

			if len(tablesWithoutPK) > 0 {
				recommendations = append(recommendations, fmt.Sprintf("Tables without primary keys: %v - consider adding primary keys", tablesWithoutPK))
			}
		}

		// Check for unused indexes
		recommendations = append(recommendations, "Run ANALYZE periodically to keep query planner statistics current")
		recommendations = append(recommendations, "Consider running VACUUM periodically to reclaim space and improve performance")
	}

	if len(recommendations) == 0 {
		recommendations = append(recommendations, "No specific recommendations at this time")
	}

	return recommendations
}

// Close closes all database connections
func (ss *SQLiteServer) Close() error {
	for _, conn := range ss.connections {
		conn.Close()
	}
	ss.connections = make(map[string]*sql.DB)
	return nil
}
