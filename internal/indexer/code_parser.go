package indexer

import (
	"crypto/md5"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/yourusername/ai-code-assistant/utils"
)

// CodeParser handles parsing of source code files into indexable chunks
type CodeParser struct {
	languageParsers map[string]LanguageParser
	config          *ParserConfig
}

// ParserConfig contains configuration for code parsing
type ParserConfig struct {
	MaxFileSize     int64    `json:"max_file_size"`
	ChunkSize       int      `json:"chunk_size"`
	ChunkOverlap    int      `json:"chunk_overlap"`
	IgnorePatterns  []string `json:"ignore_patterns"`
	IncludeComments bool     `json:"include_comments"`
	MinChunkSize    int      `json:"min_chunk_size"`
}

// LanguageParser defines interface for language-specific parsers
type LanguageParser interface {
	ParseFile(filePath string, content []byte) ([]*CodeChunk, error)
	GetLanguage() string
	GetFileExtensions() []string
	SupportsFile(filePath string) bool
}

// CodeChunk represents a parsed chunk of code with metadata
type CodeChunk struct {
	ID           string                 `json:"id"`
	FilePath     string                 `json:"file_path"`
	Language     string                 `json:"language"`
	ChunkType    string                 `json:"chunk_type"` // function, class, module, comment, etc.
	Name         string                 `json:"name"`       // function/class name
	Content      string                 `json:"content"`
	StartLine    int                    `json:"start_line"`
	EndLine      int                    `json:"end_line"`
	Signature    string                 `json:"signature,omitempty"`
	DocString    string                 `json:"doc_string,omitempty"`
	Dependencies []string               `json:"dependencies"`
	Imports      []string               `json:"imports"`
	Exports      []string               `json:"exports"`
	Complexity   int                    `json:"complexity"`
	Hash         string                 `json:"hash"`
	Metadata     map[string]interface{} `json:"metadata"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
}

// ParseResult contains the results of parsing a file
type ParseResult struct {
	FilePath  string        `json:"file_path"`
	Language  string        `json:"language"`
	Chunks    []*CodeChunk  `json:"chunks"`
	Imports   []string      `json:"imports"`
	Exports   []string      `json:"exports"`
	FileHash  string        `json:"file_hash"`
	ParsedAt  time.Time     `json:"parsed_at"`
	ParseTime time.Duration `json:"parse_time"`
	LineCount int           `json:"line_count"`
	Size      int64         `json:"size"`
	Errors    []ParseError  `json:"errors,omitempty"`
}

// ParseError represents an error that occurred during parsing
type ParseError struct {
	Line    int    `json:"line"`
	Column  int    `json:"column"`
	Message string `json:"message"`
	Type    string `json:"type"`
}

// NewCodeParser creates a new code parser with language parsers
func NewCodeParser(config *ParserConfig) *CodeParser {
	if config == nil {
		config = &ParserConfig{
			MaxFileSize:     10 * 1024 * 1024, // 10MB
			ChunkSize:       1000,
			ChunkOverlap:    100,
			IncludeComments: true,
			MinChunkSize:    50,
		}
	}

	parser := &CodeParser{
		languageParsers: make(map[string]LanguageParser),
		config:          config,
	}

	// Register language parsers
	parser.RegisterLanguageParser(NewGoParser())
	// We'll add more language parsers as we implement them

	return parser
}

// RegisterLanguageParser registers a language-specific parser
func (cp *CodeParser) RegisterLanguageParser(parser LanguageParser) {
	cp.languageParsers[parser.GetLanguage()] = parser
}

// ParseFile parses a single file and returns chunks
func (cp *CodeParser) ParseFile(filePath string) (*ParseResult, error) {
	start := time.Now()

	// Check file size
	stat, err := os.Stat(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %v", err)
	}

	if stat.Size() > cp.config.MaxFileSize {
		return nil, fmt.Errorf("file too large: %d bytes", stat.Size())
	}

	// Read file content
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %v", err)
	}

	// Detect language
	language := cp.detectLanguage(filePath)
	if language == "" {
		return nil, fmt.Errorf("unsupported file type: %s", filePath)
	}

	// Get appropriate parser
	parser, exists := cp.languageParsers[language]
	if !exists {
		// Fall back to generic parsing
		return cp.parseGeneric(filePath, content, language)
	}

	// Parse with language-specific parser
	chunks, err := parser.ParseFile(filePath, content)
	if err != nil {
		return nil, fmt.Errorf("failed to parse file: %v", err)
	}

	// Calculate file hash
	fileHash := cp.calculateHash(content)

	// Extract file-level imports/exports
	imports, exports := cp.extractFileMetadata(content, language)

	// Count lines
	lineCount := strings.Count(string(content), "\n") + 1

	result := &ParseResult{
		FilePath:  filePath,
		Language:  language,
		Chunks:    chunks,
		Imports:   imports,
		Exports:   exports,
		FileHash:  fileHash,
		ParsedAt:  time.Now(),
		ParseTime: time.Since(start),
		LineCount: lineCount,
		Size:      stat.Size(),
	}

	return result, nil
}

// ParseDirectory recursively parses all supported files in a directory
func (cp *CodeParser) ParseDirectory(dirPath string) ([]*ParseResult, error) {
	var results []*ParseResult
	var errors []error

	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		// Check if file should be ignored
		if cp.shouldIgnoreFile(path) {
			return nil
		}

		// Check if file is supported
		if !cp.isSupported(path) {
			return nil
		}

		// Parse file
		result, parseErr := cp.ParseFile(path)
		if parseErr != nil {
			errors = append(errors, fmt.Errorf("failed to parse %s: %v", path, parseErr))
			return nil // Continue with other files
		}

		results = append(results, result)
		return nil
	})

	if err != nil {
		return results, err
	}

	if len(errors) > 0 {
		// Return results with first error
		return results, errors[0]
	}

	return results, nil
}

// detectLanguage detects the programming language of a file
func (cp *CodeParser) detectLanguage(filePath string) string {
	// First try by file extension
	if lang := utils.GetFileLanguage(filePath); lang != "unknown" {
		return lang
	}

	// Check if any parser supports this file
	for _, parser := range cp.languageParsers {
		if parser.SupportsFile(filePath) {
			return parser.GetLanguage()
		}
	}

	return ""
}

// parseGeneric provides basic parsing for unsupported languages
func (cp *CodeParser) parseGeneric(filePath string, content []byte, language string) (*ParseResult, error) {
	lines := strings.Split(string(content), "\n")

	var chunks []*CodeChunk
	chunkSize := cp.config.ChunkSize
	overlap := cp.config.ChunkOverlap

	for i := 0; i < len(lines); i += chunkSize - overlap {
		end := i + chunkSize
		if end > len(lines) {
			end = len(lines)
		}

		chunkLines := lines[i:end]
		chunkContent := strings.Join(chunkLines, "\n")

		if len(strings.TrimSpace(chunkContent)) < cp.config.MinChunkSize {
			continue
		}

		chunk := &CodeChunk{
			ID:        cp.generateChunkID(filePath, i),
			FilePath:  filePath,
			Language:  language,
			ChunkType: "generic",
			Content:   chunkContent,
			StartLine: i + 1,
			EndLine:   end,
			Hash:      cp.calculateHash([]byte(chunkContent)),
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}

		chunks = append(chunks, chunk)

		// Break if we've reached the end
		if end >= len(lines) {
			break
		}
	}

	result := &ParseResult{
		FilePath:  filePath,
		Language:  language,
		Chunks:    chunks,
		FileHash:  cp.calculateHash(content),
		ParsedAt:  time.Now(),
		LineCount: len(lines),
		Size:      int64(len(content)),
	}

	return result, nil
}

// shouldIgnoreFile checks if a file should be ignored
func (cp *CodeParser) shouldIgnoreFile(filePath string) bool {
	// Check against ignore patterns
	for _, pattern := range cp.config.IgnorePatterns {
		if strings.Contains(filePath, pattern) {
			return true
		}
	}

	// Use utility function for common ignore patterns
	return utils.ShouldIgnoreFile(filePath, []string{})
}

// isSupported checks if a file type is supported
func (cp *CodeParser) isSupported(filePath string) bool {
	// Check if it's a text file
	if !utils.IsTextFile(filePath) {
		return false
	}

	// Check if any parser supports it
	language := cp.detectLanguage(filePath)
	return language != ""
}

// extractFileMetadata extracts imports and exports from file content
func (cp *CodeParser) extractFileMetadata(content []byte, language string) ([]string, []string) {
	contentStr := string(content)

	imports := utils.ExtractImports(contentStr, language)

	// For exports, we'll need language-specific logic
	// This is a simplified version
	var exports []string

	return imports, exports
}

// calculateHash calculates MD5 hash of content
func (cp *CodeParser) calculateHash(content []byte) string {
	hash := md5.Sum(content)
	return fmt.Sprintf("%x", hash)
}

// generateChunkID generates a unique ID for a chunk
func (cp *CodeParser) generateChunkID(filePath string, startLine int) string {
	return fmt.Sprintf("%s:%d", cp.calculateHash([]byte(filePath)), startLine)
}

// GetSupportedLanguages returns list of supported languages
func (cp *CodeParser) GetSupportedLanguages() []string {
	var languages []string
	for lang := range cp.languageParsers {
		languages = append(languages, lang)
	}
	return languages
}

// GetSupportedExtensions returns list of supported file extensions
func (cp *CodeParser) GetSupportedExtensions() []string {
	var extensions []string
	for _, parser := range cp.languageParsers {
		extensions = append(extensions, parser.GetFileExtensions()...)
	}
	return extensions
}

// GetParserStats returns statistics about parsing
func (cp *CodeParser) GetParserStats() map[string]interface{} {
	return map[string]interface{}{
		"supported_languages":  len(cp.languageParsers),
		"supported_extensions": len(cp.GetSupportedExtensions()),
		"config":               cp.config,
	}
}
