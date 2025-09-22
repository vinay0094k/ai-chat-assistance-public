// internal/intelligence/context_builder.go
package intelligence

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type ContextType int

const (
	ContextFile ContextType = iota
	ContextFunction
	ContextClass
	ContextPackage
	ContextModule
	ContextProject
	ContextDependency
	ContextDocumentation
	ContextTest
	ContextConfiguration
)

type ContextItem struct {
	Type         ContextType
	ID           string
	Name         string
	Path         string
	Content      string
	Summary      string
	Relevance    float64
	LastModified time.Time
	Dependencies []string
	References   []string
	Metadata     map[string]interface{}
}

type ContextRequest struct {
	Query         string
	FocusFile     string
	FocusFunction string
	IncludeTypes  []ContextType
	MaxTokens     int
	MaxItems      int
	MinRelevance  float64
}

type ContextResponse struct {
	Items      []ContextItem
	Summary    string
	TokenCount int
	TotalItems int
	Truncated  bool
	BuildTime  time.Duration
}

type ContextBuilder struct {
	projectPath  string
	indexedItems map[string]*ContextItem
	dependencies map[string][]string
	references   map[string][]string
	lastIndexed  time.Time
	mu           sync.RWMutex

	// Configuration
	maxFileSize  int64
	excludePaths []string
	includeExts  []string
	maxDepth     int
}

// NewContextBuilder creates a new context builder
func NewContextBuilder(projectPath string) *ContextBuilder {
	return &ContextBuilder{
		projectPath:  projectPath,
		indexedItems: make(map[string]*ContextItem),
		dependencies: make(map[string][]string),
		references:   make(map[string][]string),
		maxFileSize:  1024 * 1024, // 1MB
		excludePaths: []string{
			".git", "node_modules", "vendor", "target", "build", "dist",
			".vscode", ".idea", "__pycache__", ".pytest_cache",
		},
		includeExts: []string{
			".go", ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h",
			".rs", ".rb", ".php", ".cs", ".kt", ".swift", ".scala",
			".md", ".txt", ".yml", ".yaml", ".json", ".toml",
		},
		maxDepth: 10,
	}
}

// BuildContext builds context for AI agents based on request
func (cb *ContextBuilder) BuildContext(request ContextRequest) (*ContextResponse, error) {
	startTime := time.Now()

	cb.mu.RLock()
	defer cb.mu.RUnlock()

	// Ensure index is up to date
	if time.Since(cb.lastIndexed) > 5*time.Minute {
		cb.mu.RUnlock()
		if err := cb.IndexProject(); err != nil {
			cb.mu.RLock()
			return nil, fmt.Errorf("failed to index project: %w", err)
		}
		cb.mu.RLock()
	}

	// Find relevant items
	relevantItems := cb.findRelevantItems(request)

	// Sort by relevance
	sort.Slice(relevantItems, func(i, j int) bool {
		return relevantItems[i].Relevance > relevantItems[j].Relevance
	})

	// Apply limits
	truncated := false
	if request.MaxItems > 0 && len(relevantItems) > request.MaxItems {
		relevantItems = relevantItems[:request.MaxItems]
		truncated = true
	}

	// Calculate token count and apply token limit
	tokenCount := 0
	finalItems := make([]ContextItem, 0)

	for _, item := range relevantItems {
		itemTokens := cb.estimateTokens(item.Content)
		if request.MaxTokens > 0 && tokenCount+itemTokens > request.MaxTokens {
			truncated = true
			break
		}

		finalItems = append(finalItems, *item)
		tokenCount += itemTokens
	}

	// Generate summary
	summary := cb.generateContextSummary(finalItems, request)

	response := &ContextResponse{
		Items:      finalItems,
		Summary:    summary,
		TokenCount: tokenCount,
		TotalItems: len(relevantItems),
		Truncated:  truncated,
		BuildTime:  time.Since(startTime),
	}

	return response, nil
}

// IndexProject indexes the entire project
func (cb *ContextBuilder) IndexProject() error {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	// Clear existing index
	cb.indexedItems = make(map[string]*ContextItem)
	cb.dependencies = make(map[string][]string)
	cb.references = make(map[string][]string)

	// Walk through project directory
	err := filepath.Walk(cb.projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip excluded paths
		if cb.shouldExcludePath(path) {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// Check depth
		relPath, _ := filepath.Rel(cb.projectPath, path)
		if strings.Count(relPath, string(filepath.Separator)) > cb.maxDepth {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if !info.IsDir() {
			return cb.indexFile(path, info)
		}

		return cb.indexDirectory(path, info)
	})

	if err != nil {
		return err
	}

	// Build dependency and reference maps
	cb.buildDependencyMaps()

	cb.lastIndexed = time.Now()
	return nil
}

// indexFile indexes a single file
func (cb *ContextBuilder) indexFile(path string, info os.FileInfo) error {
	// Check file size
	if info.Size() > cb.maxFileSize {
		return nil
	}

	// Check file extension
	ext := filepath.Ext(path)
	if !cb.isIncludedExtension(ext) {
		return nil
	}

	// Read file content
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	relPath, _ := filepath.Rel(cb.projectPath, path)
	contentStr := string(content)

	// Create base file context item
	item := &ContextItem{
		Type:         ContextFile,
		ID:           relPath,
		Name:         filepath.Base(path),
		Path:         relPath,
		Content:      contentStr,
		Summary:      cb.generateFileSummary(contentStr, ext),
		Relevance:    0.0,
		LastModified: info.ModTime(),
		Dependencies: make([]string, 0),
		References:   make([]string, 0),
		Metadata: map[string]interface{}{
			"size":      info.Size(),
			"extension": ext,
			"lines":     strings.Count(contentStr, "\n") + 1,
		},
	}

	cb.indexedItems[item.ID] = item

	// Extract functions, classes, etc. based on file type
	switch ext {
	case ".go":
		cb.extractGoStructures(item, contentStr)
	case ".py":
		cb.extractPythonStructures(item, contentStr)
	case ".js", ".ts":
		cb.extractJavaScriptStructures(item, contentStr)
	case ".java":
		cb.extractJavaStructures(item, contentStr)
	}

	return nil
}

// indexDirectory indexes a directory
func (cb *ContextBuilder) indexDirectory(path string, info os.FileInfo) error {
	relPath, _ := filepath.Rel(cb.projectPath, path)
	if relPath == "." {
		return nil
	}

	item := &ContextItem{
		Type:         ContextPackage,
		ID:           relPath,
		Name:         filepath.Base(path),
		Path:         relPath,
		Content:      "",
		Summary:      fmt.Sprintf("Package/Directory: %s", filepath.Base(path)),
		Relevance:    0.0,
		LastModified: info.ModTime(),
		Dependencies: make([]string, 0),
		References:   make([]string, 0),
		Metadata: map[string]interface{}{
			"type": "directory",
		},
	}

	cb.indexedItems[item.ID] = item
	return nil
}

// extractGoStructures extracts Go language structures
func (cb *ContextBuilder) extractGoStructures(fileItem *ContextItem, content string) {
	lines := strings.Split(content, "\n")

	for i, line := range lines {
		line = strings.TrimSpace(line)

		// Extract functions
		if strings.HasPrefix(line, "func ") {
			funcName := cb.extractGoFunctionName(line)
			if funcName != "" {
				funcItem := &ContextItem{
					Type:         ContextFunction,
					ID:           fmt.Sprintf("%s::%s", fileItem.ID, funcName),
					Name:         funcName,
					Path:         fileItem.Path,
					Content:      cb.extractGoFunctionBody(lines, i),
					Summary:      fmt.Sprintf("Function: %s", funcName),
					Relevance:    0.0,
					LastModified: fileItem.LastModified,
					Metadata: map[string]interface{}{
						"language": "go",
						"type":     "function",
						"line":     i + 1,
					},
				}
				cb.indexedItems[funcItem.ID] = funcItem
			}
		}

		// Extract types/structs
		if strings.HasPrefix(line, "type ") && (strings.Contains(line, "struct") || strings.Contains(line, "interface")) {
			typeName := cb.extractGoTypeName(line)
			if typeName != "" {
				typeItem := &ContextItem{
					Type:         ContextClass,
					ID:           fmt.Sprintf("%s::%s", fileItem.ID, typeName),
					Name:         typeName,
					Path:         fileItem.Path,
					Content:      cb.extractGoTypeBody(lines, i),
					Summary:      fmt.Sprintf("Type: %s", typeName),
					Relevance:    0.0,
					LastModified: fileItem.LastModified,
					Metadata: map[string]interface{}{
						"language": "go",
						"type":     "struct/interface",
						"line":     i + 1,
					},
				}
				cb.indexedItems[typeItem.ID] = typeItem
			}
		}
	}
}

// extractPythonStructures extracts Python language structures
func (cb *ContextBuilder) extractPythonStructures(fileItem *ContextItem, content string) {
	lines := strings.Split(content, "\n")

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Extract functions
		if strings.HasPrefix(trimmed, "def ") {
			funcName := cb.extractPythonFunctionName(trimmed)
			if funcName != "" {
				funcItem := &ContextItem{
					Type:         ContextFunction,
					ID:           fmt.Sprintf("%s::%s", fileItem.ID, funcName),
					Name:         funcName,
					Path:         fileItem.Path,
					Content:      cb.extractPythonFunctionBody(lines, i),
					Summary:      fmt.Sprintf("Function: %s", funcName),
					Relevance:    0.0,
					LastModified: fileItem.LastModified,
					Metadata: map[string]interface{}{
						"language": "python",
						"type":     "function",
						"line":     i + 1,
					},
				}
				cb.indexedItems[funcItem.ID] = funcItem
			}
		}

		// Extract classes
		if strings.HasPrefix(trimmed, "class ") {
			className := cb.extractPythonClassName(trimmed)
			if className != "" {
				classItem := &ContextItem{
					Type:         ContextClass,
					ID:           fmt.Sprintf("%s::%s", fileItem.ID, className),
					Name:         className,
					Path:         fileItem.Path,
					Content:      cb.extractPythonClassBody(lines, i),
					Summary:      fmt.Sprintf("Class: %s", className),
					Relevance:    0.0,
					LastModified: fileItem.LastModified,
					Metadata: map[string]interface{}{
						"language": "python",
						"type":     "class",
						"line":     i + 1,
					},
				}
				cb.indexedItems[classItem.ID] = classItem
			}
		}
	}
}

// extractJavaScriptStructures extracts JavaScript/TypeScript structures
func (cb *ContextBuilder) extractJavaScriptStructures(fileItem *ContextItem, content string) {
	lines := strings.Split(content, "\n")

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Extract functions
		if strings.Contains(trimmed, "function ") || strings.Contains(trimmed, "=> {") || strings.Contains(trimmed, "() => {") {
			funcName := cb.extractJavaScriptFunctionName(trimmed)
			if funcName != "" {
				funcItem := &ContextItem{
					Type:         ContextFunction,
					ID:           fmt.Sprintf("%s::%s", fileItem.ID, funcName),
					Name:         funcName,
					Path:         fileItem.Path,
					Content:      cb.extractJavaScriptFunctionBody(lines, i),
					Summary:      fmt.Sprintf("Function: %s", funcName),
					Relevance:    0.0,
					LastModified: fileItem.LastModified,
					Metadata: map[string]interface{}{
						"language": "javascript",
						"type":     "function",
						"line":     i + 1,
					},
				}
				cb.indexedItems[funcItem.ID] = funcItem
			}
		}

		// Extract classes
		if strings.HasPrefix(trimmed, "class ") {
			className := cb.extractJavaScriptClassName(trimmed)
			if className != "" {
				classItem := &ContextItem{
					Type:         ContextClass,
					ID:           fmt.Sprintf("%s::%s", fileItem.ID, className),
					Name:         className,
					Path:         fileItem.Path,
					Content:      cb.extractJavaScriptClassBody(lines, i),
					Summary:      fmt.Sprintf("Class: %s", className),
					Relevance:    0.0,
					LastModified: fileItem.LastModified,
					Metadata: map[string]interface{}{
						"language": "javascript",
						"type":     "class",
						"line":     i + 1,
					},
				}
				cb.indexedItems[classItem.ID] = classItem
			}
		}
	}
}

// extractJavaStructures extracts Java language structures
func (cb *ContextBuilder) extractJavaStructures(fileItem *ContextItem, content string) {
	lines := strings.Split(content, "\n")

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Extract methods
		if strings.Contains(trimmed, "public ") || strings.Contains(trimmed, "private ") || strings.Contains(trimmed, "protected ") {
			if strings.Contains(trimmed, "(") && strings.Contains(trimmed, ")") && !strings.Contains(trimmed, "class ") {
				methodName := cb.extractJavaMethodName(trimmed)
				if methodName != "" {
					methodItem := &ContextItem{
						Type:         ContextFunction,
						ID:           fmt.Sprintf("%s::%s", fileItem.ID, methodName),
						Name:         methodName,
						Path:         fileItem.Path,
						Content:      cb.extractJavaMethodBody(lines, i),
						Summary:      fmt.Sprintf("Method: %s", methodName),
						Relevance:    0.0,
						LastModified: fileItem.LastModified,
						Metadata: map[string]interface{}{
							"language": "java",
							"type":     "method",
							"line":     i + 1,
						},
					}
					cb.indexedItems[methodItem.ID] = methodItem
				}
			}
		}

		// Extract classes
		if strings.Contains(trimmed, "class ") && (strings.Contains(trimmed, "public ") || strings.Contains(trimmed, "private ")) {
			className := cb.extractJavaClassName(trimmed)
			if className != "" {
				classItem := &ContextItem{
					Type:         ContextClass,
					ID:           fmt.Sprintf("%s::%s", fileItem.ID, className),
					Name:         className,
					Path:         fileItem.Path,
					Content:      cb.extractJavaClassBody(lines, i),
					Summary:      fmt.Sprintf("Class: %s", className),
					Relevance:    0.0,
					LastModified: fileItem.LastModified,
					Metadata: map[string]interface{}{
						"language": "java",
						"type":     "class",
						"line":     i + 1,
					},
				}
				cb.indexedItems[classItem.ID] = classItem
			}
		}
	}
}

// Helper methods for extracting names and bodies

func (cb *ContextBuilder) extractGoFunctionName(line string) string {
	// Extract function name from "func FunctionName(" or "func (receiver) FunctionName("
	parts := strings.Fields(line)
	if len(parts) < 2 {
		return ""
	}

	if strings.HasPrefix(parts[1], "(") {
		// Method with receiver
		if len(parts) < 3 {
			return ""
		}
		nameWithParen := parts[2]
		if parenIndex := strings.Index(nameWithParen, "("); parenIndex > 0 {
			return nameWithParen[:parenIndex]
		}
	} else {
		// Regular function
		nameWithParen := parts[1]
		if parenIndex := strings.Index(nameWithParen, "("); parenIndex > 0 {
			return nameWithParen[:parenIndex]
		}
	}

	return ""
}

func (cb *ContextBuilder) extractGoTypeName(line string) string {
	// Extract type name from "type TypeName struct" or "type TypeName interface"
	parts := strings.Fields(line)
	if len(parts) >= 3 && parts[0] == "type" {
		return parts[1]
	}
	return ""
}

func (cb *ContextBuilder) extractPythonFunctionName(line string) string {
	// Extract function name from "def function_name("
	if strings.HasPrefix(line, "def ") {
		remaining := strings.TrimPrefix(line, "def ")
		if parenIndex := strings.Index(remaining, "("); parenIndex > 0 {
			return strings.TrimSpace(remaining[:parenIndex])
		}
	}
	return ""
}

func (cb *ContextBuilder) extractPythonClassName(line string) string {
	// Extract class name from "class ClassName:" or "class ClassName(Parent):"
	if strings.HasPrefix(line, "class ") {
		remaining := strings.TrimPrefix(line, "class ")
		if colonIndex := strings.Index(remaining, ":"); colonIndex > 0 {
			nameWithParent := remaining[:colonIndex]
			if parenIndex := strings.Index(nameWithParent, "("); parenIndex > 0 {
				return strings.TrimSpace(nameWithParent[:parenIndex])
			}
			return strings.TrimSpace(nameWithParent)
		}
	}
	return ""
}

func (cb *ContextBuilder) extractJavaScriptFunctionName(line string) string {
	// Various JavaScript function patterns
	if strings.Contains(line, "function ") {
		start := strings.Index(line, "function ") + 9
		remaining := line[start:]
		if parenIndex := strings.Index(remaining, "("); parenIndex > 0 {
			return strings.TrimSpace(remaining[:parenIndex])
		}
	}

	// Arrow functions: const name = () => {
	if strings.Contains(line, "= (") && strings.Contains(line, "=> {") {
		parts := strings.Split(line, "=")
		if len(parts) >= 2 {
			name := strings.TrimSpace(parts[0])
			// Remove const/let/var
			name = strings.TrimPrefix(name, "const ")
			name = strings.TrimPrefix(name, "let ")
			name = strings.TrimPrefix(name, "var ")
			return strings.TrimSpace(name)
		}
	}

	return ""
}

func (cb *ContextBuilder) extractJavaScriptClassName(line string) string {
	if strings.HasPrefix(strings.TrimSpace(line), "class ") {
		remaining := strings.TrimPrefix(strings.TrimSpace(line), "class ")
		parts := strings.Fields(remaining)
		if len(parts) > 0 {
			name := parts[0]
			// Remove any { or extends
			if braceIndex := strings.Index(name, "{"); braceIndex > 0 {
				name = name[:braceIndex]
			}
			return strings.TrimSpace(name)
		}
	}
	return ""
}

func (cb *ContextBuilder) extractJavaMethodName(line string) string {
	// Extract method name from Java method declaration
	parts := strings.Fields(line)
	for i, part := range parts {
		if strings.Contains(part, "(") {
			nameWithParen := part
			if parenIndex := strings.Index(nameWithParen, "("); parenIndex > 0 {
				return nameWithParen[:parenIndex]
			}
		}
	}
	return ""
}

func (cb *ContextBuilder) extractJavaClassName(line string) string {
	parts := strings.Fields(line)
	for i, part := range parts {
		if part == "class" && i+1 < len(parts) {
			className := parts[i+1]
			// Remove any { or extends
			if braceIndex := strings.Index(className, "{"); braceIndex > 0 {
				className = className[:braceIndex]
			}
			return strings.TrimSpace(className)
		}
	}
	return ""
}

// Body extraction methods (simplified - extract a few lines for context)

func (cb *ContextBuilder) extractGoFunctionBody(lines []string, startLine int) string {
	return cb.extractCodeBlock(lines, startLine, "{", "}", 20)
}

func (cb *ContextBuilder) extractGoTypeBody(lines []string, startLine int) string {
	return cb.extractCodeBlock(lines, startLine, "{", "}", 15)
}

func (cb *ContextBuilder) extractPythonFunctionBody(lines []string, startLine int) string {
	return cb.extractIndentedBlock(lines, startLine, 15)
}

func (cb *ContextBuilder) extractPythonClassBody(lines []string, startLine int) string {
	return cb.extractIndentedBlock(lines, startLine, 20)
}

func (cb *ContextBuilder) extractJavaScriptFunctionBody(lines []string, startLine int) string {
	return cb.extractCodeBlock(lines, startLine, "{", "}", 15)
}

func (cb *ContextBuilder) extractJavaScriptClassBody(lines []string, startLine int) string {
	return cb.extractCodeBlock(lines, startLine, "{", "}", 20)
}

func (cb *ContextBuilder) extractJavaMethodBody(lines []string, startLine int) string {
	return cb.extractCodeBlock(lines, startLine, "{", "}", 15)
}

func (cb *ContextBuilder) extractJavaClassBody(lines []string, startLine int) string {
	return cb.extractCodeBlock(lines, startLine, "{", "}", 25)
}

// Generic code block extraction
func (cb *ContextBuilder) extractCodeBlock(lines []string, startLine int, openBrace, closeBrace string, maxLines int) string {
	if startLine >= len(lines) {
		return ""
	}

	var result []string
	braceCount := 0
	started := false

	for i := startLine; i < len(lines) && len(result) < maxLines; i++ {
		line := lines[i]
		result = append(result, line)

		if strings.Contains(line, openBrace) {
			braceCount += strings.Count(line, openBrace)
			started = true
		}

		if strings.Contains(line, closeBrace) {
			braceCount -= strings.Count(line, closeBrace)
		}

		if started && braceCount <= 0 {
			break
		}
	}

	return strings.Join(result, "\n")
}

// Extract indented block (for Python)
func (cb *ContextBuilder) extractIndentedBlock(lines []string, startLine int, maxLines int) string {
	if startLine >= len(lines) {
		return ""
	}

	var result []string
	baseIndent := len(lines[startLine]) - len(strings.TrimLeft(lines[startLine], " \t"))

	for i := startLine; i < len(lines) && len(result) < maxLines; i++ {
		line := lines[i]

		// Stop if we hit a line with less or equal indentation (except empty lines)
		if strings.TrimSpace(line) != "" {
			currentIndent := len(line) - len(strings.TrimLeft(line, " \t"))
			if i > startLine && currentIndent <= baseIndent {
				break
			}
		}

		result = append(result, line)
	}

	return strings.Join(result, "\n")
}

// Additional helper methods

func (cb *ContextBuilder) shouldExcludePath(path string) bool {
	for _, exclude := range cb.excludePaths {
		if strings.Contains(path, exclude) {
			return true
		}
	}
	return false
}

func (cb *ContextBuilder) isIncludedExtension(ext string) bool {
	for _, include := range cb.includeExts {
		if ext == include {
			return true
		}
	}
	return false
}

func (cb *ContextBuilder) findRelevantItems(request ContextRequest) []*ContextItem {
	var relevant []*ContextItem

	for _, item := range cb.indexedItems {
		// Filter by type if specified
		if len(request.IncludeTypes) > 0 {
			typeMatch := false
			for _, t := range request.IncludeTypes {
				if item.Type == t {
					typeMatch = true
					break
				}
			}
			if !typeMatch {
				continue
			}
		}

		// Calculate relevance
		relevance := cb.calculateRelevance(item, request)

		if relevance >= request.MinRelevance {
			itemCopy := *item
			itemCopy.Relevance = relevance
			relevant = append(relevant, &itemCopy)
		}
	}

	return relevant
}

func (cb *ContextBuilder) calculateRelevance(item *ContextItem, request ContextRequest) float64 {
	relevance := 0.0

	// Exact path match
	if request.FocusFile != "" && item.Path == request.FocusFile {
		relevance += 1.0
	}

	// Exact function match
	if request.FocusFunction != "" && strings.Contains(item.ID, request.FocusFunction) {
		relevance += 1.0
	}

	// Query relevance (simple text matching)
	if request.Query != "" {
		queryLower := strings.ToLower(request.Query)
		contentLower := strings.ToLower(item.Content)
		nameLower := strings.ToLower(item.Name)

		if strings.Contains(nameLower, queryLower) {
			relevance += 0.8
		}

		if strings.Contains(contentLower, queryLower) {
			relevance += 0.3
		}
	}

	// Type-based relevance
	switch item.Type {
	case ContextFunction:
		relevance += 0.1
	case ContextClass:
		relevance += 0.15
	case ContextFile:
		relevance += 0.05
	}

	// Recent modification bonus
	if time.Since(item.LastModified) < 24*time.Hour {
		relevance += 0.1
	}

	return relevance
}

func (cb *ContextBuilder) buildDependencyMaps() {
	// Build dependency and reference maps by analyzing imports/includes
	for _, item := range cb.indexedItems {
		if item.Type == ContextFile {
			deps := cb.extractDependencies(item.Content, filepath.Ext(item.Path))
			cb.dependencies[item.ID] = deps

			// Add reverse references
			for _, dep := range deps {
				if cb.references[dep] == nil {
					cb.references[dep] = make([]string, 0)
				}
				cb.references[dep] = append(cb.references[dep], item.ID)
			}
		}
	}
}

func (cb *ContextBuilder) extractDependencies(content, ext string) []string {
	var deps []string

	lines := strings.Split(content, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)

		switch ext {
		case ".go":
			if strings.HasPrefix(line, "import ") {
				// Extract Go imports
				if dep := cb.extractGoImport(line); dep != "" {
					deps = append(deps, dep)
				}
			}
		case ".py":
			if strings.HasPrefix(line, "import ") || strings.HasPrefix(line, "from ") {
				// Extract Python imports
				if dep := cb.extractPythonImport(line); dep != "" {
					deps = append(deps, dep)
				}
			}
		case ".js", ".ts":
			if strings.Contains(line, "import ") || strings.Contains(line, "require(") {
				// Extract JavaScript imports
				if dep := cb.extractJavaScriptImport(line); dep != "" {
					deps = append(deps, dep)
				}
			}
		}
	}

	return deps
}

func (cb *ContextBuilder) extractGoImport(line string) string {
	// Simplified Go import extraction
	if strings.Contains(line, `"`) {
		start := strings.Index(line, `"`) + 1
		end := strings.LastIndex(line, `"`)
		if start < end {
			return line[start:end]
		}
	}
	return ""
}

func (cb *ContextBuilder) extractPythonImport(line string) string {
	// Simplified Python import extraction
	if strings.HasPrefix(line, "import ") {
		parts := strings.Fields(line)
		if len(parts) >= 2 {
			return parts[1]
		}
	} else if strings.HasPrefix(line, "from ") {
		parts := strings.Fields(line)
		if len(parts) >= 2 {
			return parts[1]
		}
	}
	return ""
}

func (cb *ContextBuilder) extractJavaScriptImport(line string) string {
	// Simplified JavaScript import extraction
	if strings.Contains(line, `"`) {
		start := strings.LastIndex(line, `"`)
		beforeStart := strings.LastIndex(line[:start], `"`)
		if beforeStart >= 0 && beforeStart < start {
			return line[beforeStart+1 : start]
		}
	}
	return ""
}

func (cb *ContextBuilder) generateFileSummary(content, ext string) string {
	lines := strings.Split(content, "\n")
	lineCount := len(lines)

	// Count different types of content
	var funcCount, classCount, commentLines int

	for _, line := range lines {
		line = strings.TrimSpace(line)

		// Count comments
		if strings.HasPrefix(line, "//") || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "/*") {
			commentLines++
		}

		// Count functions/methods based on language
		switch ext {
		case ".go":
			if strings.HasPrefix(line, "func ") {
				funcCount++
			}
			if strings.HasPrefix(line, "type ") && (strings.Contains(line, "struct") || strings.Contains(line, "interface")) {
				classCount++
			}
		case ".py":
			if strings.HasPrefix(line, "def ") {
				funcCount++
			}
			if strings.HasPrefix(line, "class ") {
				classCount++
			}
		case ".js", ".ts":
			if strings.Contains(line, "function ") || strings.Contains(line, "=> {") {
				funcCount++
			}
			if strings.HasPrefix(line, "class ") {
				classCount++
			}
		}
	}

	return fmt.Sprintf("%d lines, %d functions, %d classes, %d comments",
		lineCount, funcCount, classCount, commentLines)
}

func (cb *ContextBuilder) generateContextSummary(items []ContextItem, request ContextRequest) string {
	if len(items) == 0 {
		return "No relevant context found."
	}

	typeCount := make(map[ContextType]int)
	for _, item := range items {
		typeCount[item.Type]++
	}

	var summary strings.Builder
	summary.WriteString(fmt.Sprintf("Context includes %d items: ", len(items)))

	var parts []string
	if typeCount[ContextFile] > 0 {
		parts = append(parts, fmt.Sprintf("%d files", typeCount[ContextFile]))
	}
	if typeCount[ContextFunction] > 0 {
		parts = append(parts, fmt.Sprintf("%d functions", typeCount[ContextFunction]))
	}
	if typeCount[ContextClass] > 0 {
		parts = append(parts, fmt.Sprintf("%d classes", typeCount[ContextClass]))
	}
	if typeCount[ContextPackage] > 0 {
		parts = append(parts, fmt.Sprintf("%d packages", typeCount[ContextPackage]))
	}

	summary.WriteString(strings.Join(parts, ", "))

	if request.FocusFile != "" {
		summary.WriteString(fmt.Sprintf(". Focused on: %s", request.FocusFile))
	}

	return summary.String()
}

func (cb *ContextBuilder) estimateTokens(content string) int {
	// Rough token estimation (1 token ≈ 4 characters for code)
	return len(content) / 4
}

// Public methods for configuration

func (cb *ContextBuilder) SetMaxFileSize(size int64) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.maxFileSize = size
}

func (cb *ContextBuilder) SetExcludePaths(paths []string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.excludePaths = paths
}

func (cb *ContextBuilder) SetIncludeExtensions(exts []string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.includeExts = exts
}

func (cb *ContextBuilder) GetIndexStats() map[string]interface{} {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	typeCount := make(map[ContextType]int)
	for _, item := range cb.indexedItems {
		typeCount[item.Type]++
	}

	return map[string]interface{}{
		"total_items":  len(cb.indexedItems),
		"files":        typeCount[ContextFile],
		"functions":    typeCount[ContextFunction],
		"classes":      typeCount[ContextClass],
		"packages":     typeCount[ContextPackage],
		"last_indexed": cb.lastIndexed,
		"dependencies": len(cb.dependencies),
		"references":   len(cb.references),
	}
}
