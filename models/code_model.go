package models

import (
	"time"
)

// CodeChunk represents a semantic chunk of code
type CodeChunk struct {
	ID        string                 `json:"id" db:"id"`
	FileID    string                 `json:"file_id" db:"file_id"`
	Type      string                 `json:"type" db:"type"` // function, struct, interface, variable, import
	Name      string                 `json:"name" db:"name"`
	Code      string                 `json:"code" db:"code"`
	Language  string                 `json:"language" db:"language"`
	StartLine int                    `json:"start_line" db:"start_line"`
	EndLine   int                    `json:"end_line" db:"end_line"`
	Hash      string                 `json:"hash" db:"hash"`             // Content hash for change detection
	Embedding []float32              `json:"embedding,omitempty" db:"-"` // Vector embedding
	Metadata  map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt time.Time              `json:"updated_at" db:"updated_at"`
}

// CodeFile represents a source code file
type CodeFile struct {
	ID           string                 `json:"id" db:"id"`
	Path         string                 `json:"path" db:"path"`
	RelativePath string                 `json:"relative_path" db:"relative_path"`
	Language     string                 `json:"language" db:"language"`
	Size         int64                  `json:"size" db:"size"`
	Hash         string                 `json:"hash" db:"hash"` // File content hash
	LineCount    int                    `json:"line_count" db:"line_count"`
	ChunkCount   int                    `json:"chunk_count" db:"chunk_count"`
	LastModified time.Time              `json:"last_modified" db:"last_modified"`
	IndexedAt    time.Time              `json:"indexed_at" db:"indexed_at"`
	Metadata     map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt    time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at" db:"updated_at"`
}

// CodeProject represents the entire project being indexed
type CodeProject struct {
	ID          string                 `json:"id" db:"id"`
	Name        string                 `json:"name" db:"name"`
	Path        string                 `json:"path" db:"path"`
	Language    string                 `json:"language" db:"language"`   // Primary language
	Framework   string                 `json:"framework" db:"framework"` // Primary framework
	FileCount   int                    `json:"file_count" db:"file_count"`
	ChunkCount  int                    `json:"chunk_count" db:"chunk_count"`
	TotalLines  int                    `json:"total_lines" db:"total_lines"`
	LastIndexed time.Time              `json:"last_indexed" db:"last_indexed"`
	Version     string                 `json:"version" db:"version"` // Git commit hash or version
	Metadata    map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt   time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" db:"updated_at"`
}

// CodeRelationship represents relationships between code elements
type CodeRelationship struct {
	ID          string                 `json:"id" db:"id"`
	FromChunkID string                 `json:"from_chunk_id" db:"from_chunk_id"`
	ToChunkID   string                 `json:"to_chunk_id" db:"to_chunk_id"`
	Type        string                 `json:"type" db:"type"`               // calls, imports, implements, extends, uses
	Weight      float64                `json:"weight" db:"weight"`           // Relationship strength
	Context     string                 `json:"context" db:"context"`         // Additional context
	LineNumber  int                    `json:"line_number" db:"line_number"` // Where relationship occurs
	Metadata    map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt   time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" db:"updated_at"`
}

// CodePattern represents identified patterns in the codebase
type CodePattern struct {
	ID          string                 `json:"id" db:"id"`
	Type        string                 `json:"type" db:"type"` // architectural, design, naming, etc.
	Name        string                 `json:"name" db:"name"`
	Description string                 `json:"description" db:"description"`
	Pattern     string                 `json:"pattern" db:"pattern"` // Regex or description
	Examples    []string               `json:"examples" db:"examples"`
	Frequency   int                    `json:"frequency" db:"frequency"`   // How often seen
	Confidence  float64                `json:"confidence" db:"confidence"` // Pattern confidence
	Context     string                 `json:"context" db:"context"`       // Where it applies
	Metadata    map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt   time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" db:"updated_at"`
}

// CodeArchitecture represents the detected architecture of the project
type CodeArchitecture struct {
	ID           string                 `json:"id" db:"id"`
	ProjectID    string                 `json:"project_id" db:"project_id"`
	Type         string                 `json:"type" db:"type"`         // monolith, microservices, layered, etc.
	Patterns     []string               `json:"patterns" db:"patterns"` // MVC, Repository, Factory, etc.
	Layers       []ArchitectureLayer    `json:"layers" db:"layers"`
	Dependencies []string               `json:"dependencies" db:"dependencies"`
	TestStrategy string                 `json:"test_strategy" db:"test_strategy"`
	Confidence   float64                `json:"confidence" db:"confidence"`
	DetectedAt   time.Time              `json:"detected_at" db:"detected_at"`
	Metadata     map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt    time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at" db:"updated_at"`
}

// ArchitectureLayer represents a layer in the architecture
type ArchitectureLayer struct {
	Name        string   `json:"name"`        // controller, service, repository, model
	Directory   string   `json:"directory"`   // Path to layer files
	Files       []string `json:"files"`       // Files in this layer
	Patterns    []string `json:"patterns"`    // Patterns used in layer
	Description string   `json:"description"` // Layer description
}

// CodeUsage represents how code elements are used
type CodeUsage struct {
	ID        string                 `json:"id" db:"id"`
	ChunkID   string                 `json:"chunk_id" db:"chunk_id"`
	UsageType string                 `json:"usage_type" db:"usage_type"` // called_by, imports, implements
	UsedBy    string                 `json:"used_by" db:"used_by"`       // Which chunk uses this
	Frequency int                    `json:"frequency" db:"frequency"`   // How often used
	Context   string                 `json:"context" db:"context"`       // Usage context
	Location  string                 `json:"location" db:"location"`     // File:line where used
	Metadata  map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt time.Time              `json:"updated_at" db:"updated_at"`
}

// IndexingJob represents a background indexing task
type IndexingJob struct {
	ID          string                 `json:"id" db:"id"`
	Type        string                 `json:"type" db:"type"`     // full, incremental, file
	Status      string                 `json:"status" db:"status"` // pending, running, completed, failed
	ProjectPath string                 `json:"project_path" db:"project_path"`
	FilePaths   []string               `json:"file_paths" db:"file_paths"` // Specific files to index
	Priority    int                    `json:"priority" db:"priority"`     // Job priority
	Progress    float64                `json:"progress" db:"progress"`     // 0.0 to 1.0
	Message     string                 `json:"message" db:"message"`       // Status message
	Error       string                 `json:"error" db:"error"`           // Error message if failed
	StartedAt   *time.Time             `json:"started_at" db:"started_at"`
	CompletedAt *time.Time             `json:"completed_at" db:"completed_at"`
	Metadata    map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt   time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" db:"updated_at"`
}

// SearchResult represents a search result
type SearchResult struct {
	ChunkID    string                 `json:"chunk_id"`
	Chunk      *CodeChunk             `json:"chunk"`
	Score      float64                `json:"score"`                // Relevance score
	Type       string                 `json:"type"`                 // semantic, keyword, graph
	Highlights []string               `json:"highlights"`           // Highlighted text
	Context    string                 `json:"context"`              // Surrounding context
	UsageInfo  *CodeUsage             `json:"usage_info,omitempty"` // Usage information
	Metadata   map[string]interface{} `json:"metadata"`
}

// ProjectStatistics represents project-wide statistics
type ProjectStatistics struct {
	ProjectID      string         `json:"project_id"`
	TotalFiles     int            `json:"total_files"`
	TotalLines     int            `json:"total_lines"`
	TotalChunks    int            `json:"total_chunks"`
	LanguageStats  map[string]int `json:"language_stats"`  // Language -> line count
	FileTypeStats  map[string]int `json:"file_type_stats"` // Extension -> file count
	FunctionCount  int            `json:"function_count"`
	StructCount    int            `json:"struct_count"`
	InterfaceCount int            `json:"interface_count"`
	TestCoverage   float64        `json:"test_coverage"`
	LastUpdated    time.Time      `json:"last_updated"`
}
