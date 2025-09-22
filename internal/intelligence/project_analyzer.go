// internal/intelligence/project_analyzer.go
package intelligence

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

type ProjectMetrics struct {
	// Size metrics
	TotalFiles   int
	TotalLines   int
	CodeLines    int
	CommentLines int
	BlankLines   int

	// Language breakdown
	LanguageStats map[string]LanguageStat

	// Complexity metrics
	CyclomaticComplexity int
	CognitiveComplexity  int
	TechnicalDebt        time.Duration

	// Quality metrics
	TestCoverage     float64
	DuplicationRatio float64
	CodeSmells       int

	// Dependency metrics
	TotalDependencies      int
	DirectDependencies     int
	TransitiveDependencies int
	DependencyDepth        int
	CircularDependencies   []string

	// Maintenance metrics
	LastCommit      time.Time
	CommitFrequency float64 // commits per day
	FileChanges     map[string]int
	Hotspots        []Hotspot

	// Security metrics
	SecurityIssues     []SecurityIssue
	VulnerablePackages []VulnerablePackage
}

type LanguageStat struct {
	Name       string
	Files      int
	Lines      int
	Percentage float64
	Extensions []string
}

type Hotspot struct {
	File        string
	Changes     int
	Complexity  int
	LastChanged time.Time
	Risk        float64
}

type SecurityIssue struct {
	Type        string
	Severity    string
	File        string
	Line        int
	Description string
	Suggestion  string
}

type VulnerablePackage struct {
	Name        string
	Version     string
	Severity    string
	CVE         string
	Description string
}

type ProjectHealth struct {
	OverallScore    float64
	Scores          map[string]float64
	Issues          []HealthIssue
	Recommendations []string
	Trends          []Trend
}

type HealthIssue struct {
	Category    string
	Severity    string
	Title       string
	Description string
	File        string
	Line        int
	Suggestion  string
}

type Trend struct {
	Metric    string
	Direction string // "improving", "declining", "stable"
	Change    float64
	Period    string
}

type ProjectAnalyzer struct {
	projectPath string
	metrics     *ProjectMetrics
	health      *ProjectHealth
	mu          sync.RWMutex

	// Configuration
	excludePaths []string
	includeTests bool
	maxFileSize  int64

	// Analysis components
	architectureDetector *ArchitectureDetector
	patternAnalyzer      *PatternAnalyzer
	contextBuilder       *ContextBuilder

	// Cache
	lastAnalysis   time.Time
	fileStatsCache map[string]FileStats
}

type FileStats struct {
	Path         string
	Language     string
	Lines        int
	CodeLines    int
	CommentLines int
	BlankLines   int
	Complexity   int
	LastModified time.Time
	Functions    int
	Classes      int
	TestFile     bool
}

// NewProjectAnalyzer creates a new project analyzer
func NewProjectAnalyzer(projectPath string) *ProjectAnalyzer {
	pa := &ProjectAnalyzer{
		projectPath:    projectPath,
		excludePaths:   []string{".git", "node_modules", "vendor", "target", "build", "dist", "__pycache__", ".vscode", ".idea"},
		includeTests:   true,
		maxFileSize:    1024 * 1024, // 1MB
		fileStatsCache: make(map[string]FileStats),
	}

	// Initialize analysis components
	pa.architectureDetector = NewArchitectureDetector(projectPath)
	pa.patternAnalyzer = NewPatternAnalyzer(projectPath)
	pa.contextBuilder = NewContextBuilder(projectPath)

	return pa
}

// AnalyzeProject performs comprehensive project analysis
func (pa *ProjectAnalyzer) AnalyzeProject() (*ProjectMetrics, *ProjectHealth, error) {
	pa.mu.Lock()
	defer pa.mu.Unlock()

	startTime := time.Now()

	// Initialize metrics
	pa.metrics = &ProjectMetrics{
		LanguageStats:        make(map[string]LanguageStat),
		FileChanges:          make(map[string]int),
		Hotspots:             make([]Hotspot, 0),
		SecurityIssues:       make([]SecurityIssue, 0),
		VulnerablePackages:   make([]VulnerablePackage, 0),
		CircularDependencies: make([]string, 0),
	}

	pa.health = &ProjectHealth{
		Scores:          make(map[string]float64),
		Issues:          make([]HealthIssue, 0),
		Recommendations: make([]string, 0),
		Trends:          make([]Trend, 0),
	}

	// Analyze file structure
	if err := pa.analyzeFileStructure(); err != nil {
		return nil, nil, fmt.Errorf("file structure analysis failed: %w", err)
	}

	// Analyze code metrics
	if err := pa.analyzeCodeMetrics(); err != nil {
		return nil, nil, fmt.Errorf("code metrics analysis failed: %w", err)
	}

	// Analyze dependencies
	if err := pa.analyzeDependencies(); err != nil {
		return nil, nil, fmt.Errorf("dependency analysis failed: %w", err)
	}

	// Analyze version control history
	if err := pa.analyzeVersionControl(); err != nil {
		// Don't fail if git is not available
		fmt.Printf("Warning: version control analysis failed: %v\n", err)
	}

	// Analyze security
	if err := pa.analyzeSecurityIssues(); err != nil {
		fmt.Printf("Warning: security analysis failed: %v\n", err)
	}

	// Calculate health scores
	pa.calculateHealthScores()

	// Generate recommendations
	pa.generateRecommendations()

	pa.lastAnalysis = time.Now()

	fmt.Printf("Project analysis completed in %v\n", time.Since(startTime))

	return pa.metrics, pa.health, nil
}

// analyzeFileStructure analyzes the project file structure
func (pa *ProjectAnalyzer) analyzeFileStructure() error {
	return filepath.Walk(pa.projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if pa.shouldExcludePath(path) {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if !info.IsDir() && info.Size() <= pa.maxFileSize {
			return pa.analyzeFile(path, info)
		}

		return nil
	})
}

// analyzeFile analyzes a single file
func (pa *ProjectAnalyzer) analyzeFile(path string, info os.FileInfo) error {
	ext := filepath.Ext(path)
	language := pa.detectLanguage(ext, path)

	if language == "" {
		return nil // Skip non-source files
	}

	// Read file content
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	stats := FileStats{
		Path:         path,
		Language:     language,
		LastModified: info.ModTime(),
		TestFile:     pa.isTestFile(path),
	}

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		stats.Lines++

		if line == "" {
			stats.BlankLines++
		} else if pa.isCommentLine(line, language) {
			stats.CommentLines++
		} else {
			stats.CodeLines++

			// Count functions and classes
			if pa.isFunctionDeclaration(line, language) {
				stats.Functions++
			}
			if pa.isClassDeclaration(line, language) {
				stats.Classes++
			}

			// Calculate complexity
			stats.Complexity += pa.calculateLineComplexity(line, language)
		}
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	// Update project metrics
	pa.updateMetricsFromFile(stats)

	// Cache file stats
	relPath, _ := filepath.Rel(pa.projectPath, path)
	pa.fileStatsCache[relPath] = stats

	return nil
}

// detectLanguage detects programming language from file extension
func (pa *ProjectAnalyzer) detectLanguage(ext, path string) string {
	langMap := map[string]string{
		".go":    "Go",
		".py":    "Python",
		".js":    "JavaScript",
		".ts":    "TypeScript",
		".java":  "Java",
		".cpp":   "C++",
		".c":     "C",
		".h":     "C/C++ Header",
		".hpp":   "C++ Header",
		".cs":    "C#",
		".rb":    "Ruby",
		".php":   "PHP",
		".kt":    "Kotlin",
		".swift": "Swift",
		".rs":    "Rust",
		".scala": "Scala",
		".sh":    "Shell",
		".sql":   "SQL",
		".html":  "HTML",
		".css":   "CSS",
		".scss":  "SCSS",
		".less":  "LESS",
		".json":  "JSON",
		".xml":   "XML",
		".yml":   "YAML",
		".yaml":  "YAML",
		".toml":  "TOML",
		".md":    "Markdown",
		".txt":   "Text",
	}

	if lang, exists := langMap[ext]; exists {
		return lang
	}

	// Special cases
	if strings.Contains(path, "Dockerfile") {
		return "Docker"
	}
	if strings.Contains(path, "Makefile") {
		return "Makefile"
	}

	return ""
}

// isTestFile determines if a file is a test file
func (pa *ProjectAnalyzer) isTestFile(path string) bool {
	name := strings.ToLower(filepath.Base(path))

	// Common test file patterns
	testPatterns := []string{
		"_test.", "test_", "spec_", "_spec.", ".test.", ".spec.",
		"tests/", "test/", "__tests__/", "spec/",
	}

	for _, pattern := range testPatterns {
		if strings.Contains(strings.ToLower(path), pattern) {
			return true
		}
	}

	return false
}

// isCommentLine determines if a line is a comment
func (pa *ProjectAnalyzer) isCommentLine(line, language string) bool {
	commentPrefixes := map[string][]string{
		"Go":         {"//", "/*"},
		"Python":     {"#", `"""`},
		"JavaScript": {"//", "/*"},
		"TypeScript": {"//", "/*"},
		"Java":       {"//", "/*"},
		"C++":        {"//", "/*"},
		"C":          {"/*"},
		"C#":         {"//", "/*"},
		"Ruby":       {"#"},
		"PHP":        {"//", "#", "/*"},
		"Rust":       {"//", "/*"},
		"Shell":      {"#"},
	}

	if prefixes, exists := commentPrefixes[language]; exists {
		for _, prefix := range prefixes {
			if strings.HasPrefix(line, prefix) {
				return true
			}
		}
	}

	return false
}

// isFunctionDeclaration determines if a line contains a function declaration
func (pa *ProjectAnalyzer) isFunctionDeclaration(line, language string) bool {
	patterns := map[string][]string{
		"Go":         {`func\s+\w+\s*\(`},
		"Python":     {`def\s+\w+\s*\(`},
		"JavaScript": {`function\s+\w+\s*\(`, `\w+\s*=\s*function\s*\(`, `\w+\s*=\s*\([^)]*\)\s*=>`},
		"TypeScript": {`function\s+\w+\s*\(`, `\w+\s*=\s*function\s*\(`, `\w+\s*=\s*\([^)]*\)\s*=>`},
		"Java":       {`(public|private|protected).*\w+\s*\([^)]*\)\s*\{`},
		"C++":        {`\w+\s+\w+\s*\([^)]*\)\s*\{`},
		"C":          {`\w+\s+\w+\s*\([^)]*\)\s*\{`},
		"Rust":       {`fn\s+\w+\s*\(`},
	}

	if regexes, exists := patterns[language]; exists {
		for _, pattern := range regexes {
			if matched, _ := regexp.MatchString(pattern, line); matched {
				return true
			}
		}
	}

	return false
}

// isClassDeclaration determines if a line contains a class declaration
func (pa *ProjectAnalyzer) isClassDeclaration(line, language string) bool {
	patterns := map[string][]string{
		"Go":         {`type\s+\w+\s+struct`},
		"Python":     {`class\s+\w+`},
		"JavaScript": {`class\s+\w+`},
		"TypeScript": {`class\s+\w+`, `interface\s+\w+`},
		"Java":       {`(public|private)?\s*class\s+\w+`},
		"C++":        {`class\s+\w+`},
		"C#":         {`(public|private)?\s*class\s+\w+`},
		"Rust":       {`struct\s+\w+`, `enum\s+\w+`, `trait\s+\w+`},
	}

	if regexes, exists := patterns[language]; exists {
		for _, pattern := range regexes {
			if matched, _ := regexp.MatchString(pattern, line); matched {
				return true
			}
		}
	}

	return false
}

// calculateLineComplexity calculates cyclomatic complexity for a line
func (pa *ProjectAnalyzer) calculateLineComplexity(line, language string) int {
	complexity := 0

	// Keywords that increase complexity
	complexityKeywords := []string{
		"if", "else", "for", "while", "switch", "case", "catch", "&&", "||", "?",
	}

	for _, keyword := range complexityKeywords {
		if strings.Contains(line, keyword) {
			complexity++
		}
	}

	return complexity
}

// updateMetricsFromFile updates project metrics from file stats
func (pa *ProjectAnalyzer) updateMetricsFromFile(stats FileStats) {
	pa.metrics.TotalFiles++
	pa.metrics.TotalLines += stats.Lines
	pa.metrics.CodeLines += stats.CodeLines
	pa.metrics.CommentLines += stats.CommentLines
	pa.metrics.BlankLines += stats.BlankLines
	pa.metrics.CyclomaticComplexity += stats.Complexity

	// Update language statistics
	if langStat, exists := pa.metrics.LanguageStats[stats.Language]; exists {
		langStat.Files++
		langStat.Lines += stats.Lines
		pa.metrics.LanguageStats[stats.Language] = langStat
	} else {
		pa.metrics.LanguageStats[stats.Language] = LanguageStat{
			Name:  stats.Language,
			Files: 1,
			Lines: stats.Lines,
		}
	}
}

// analyzeCodeMetrics analyzes advanced code metrics
func (pa *ProjectAnalyzer) analyzeCodeMetrics() error {
	// Calculate language percentages
	for lang, stat := range pa.metrics.LanguageStats {
		stat.Percentage = float64(stat.Lines) / float64(pa.metrics.TotalLines) * 100
		pa.metrics.LanguageStats[lang] = stat
	}

	// Calculate cognitive complexity
	pa.metrics.CognitiveComplexity = pa.calculateCognitiveComplexity()

	// Estimate technical debt
	pa.metrics.TechnicalDebt = pa.estimateTechnicalDebt()

	// Analyze code duplication
	pa.metrics.DuplicationRatio = pa.analyzeDuplication()

	// Count code smells
	pa.metrics.CodeSmells = pa.countCodeSmells()

	// Calculate test coverage
	pa.metrics.TestCoverage = pa.calculateTestCoverage()

	return nil
}

// calculateCognitiveComplexity calculates cognitive complexity
func (pa *ProjectAnalyzer) calculateCognitiveComplexity() int {
	// Simplified cognitive complexity calculation
	// In a real implementation, this would be more sophisticated
	return pa.metrics.CyclomaticComplexity * 2
}

// estimateTechnicalDebt estimates technical debt
func (pa *ProjectAnalyzer) estimateTechnicalDebt() time.Duration {
	// Simplified technical debt estimation
	// Based on code smells, complexity, and duplication

	baseMinutes := 0

	// Add time based on complexity
	if pa.metrics.CyclomaticComplexity > 1000 {
		baseMinutes += 60 // 1 hour for high complexity
	}

	// Add time based on duplication
	if pa.metrics.DuplicationRatio > 0.1 {
		baseMinutes += 30 // 30 minutes for high duplication
	}

	// Add time based on code smells
	baseMinutes += pa.metrics.CodeSmells * 5 // 5 minutes per code smell

	return time.Duration(baseMinutes) * time.Minute
}

// analyzeDuplication analyzes code duplication
func (pa *ProjectAnalyzer) analyzeDuplication() float64 {
	// Simplified duplication analysis
	// In a real implementation, this would analyze actual code blocks

	if pa.metrics.TotalLines == 0 {
		return 0.0
	}

	// Estimate based on file patterns and common code structures
	estimatedDuplicateLines := 0

	// Look for similar file names (potential duplication)
	fileNames := make(map[string]int)
	for _, stats := range pa.fileStatsCache {
		baseName := strings.TrimSuffix(filepath.Base(stats.Path), filepath.Ext(stats.Path))
		fileNames[baseName]++
	}

	for _, count := range fileNames {
		if count > 1 {
			estimatedDuplicateLines += 50 // Estimate 50 lines per duplicate file
		}
	}

	return float64(estimatedDuplicateLines) / float64(pa.metrics.CodeLines)
}

// countCodeSmells counts various code smells
func (pa *ProjectAnalyzer) countCodeSmells() int {
	smells := 0

	for _, stats := range pa.fileStatsCache {
		// Large file smell
		if stats.Lines > 500 {
			smells++
		}

		// High complexity smell
		if stats.Complexity > 50 {
			smells++
		}

		// Low comment ratio smell
		if stats.CodeLines > 0 {
			commentRatio := float64(stats.CommentLines) / float64(stats.CodeLines)
			if commentRatio < 0.1 {
				smells++
			}
		}
	}

	return smells
}

// calculateTestCoverage calculates test coverage estimate
func (pa *ProjectAnalyzer) calculateTestCoverage() float64 {
	testLines := 0
	codeLines := 0

	for _, stats := range pa.fileStatsCache {
		if stats.TestFile {
			testLines += stats.CodeLines
		} else {
			codeLines += stats.CodeLines
		}
	}

	if codeLines == 0 {
		return 0.0
	}

	// Simplified coverage calculation
	// Assumes 1 line of test covers 2 lines of code
	coverage := float64(testLines*2) / float64(codeLines)
	if coverage > 1.0 {
		coverage = 1.0
	}

	return coverage * 100
}

// analyzeDependencies analyzes project dependencies
func (pa *ProjectAnalyzer) analyzeDependencies() error {
	// Analyze different dependency files
	dependencyFiles := []string{
		"go.mod", "package.json", "requirements.txt", "pom.xml", "Cargo.toml", "composer.json",
	}

	for _, file := range dependencyFiles {
		path := filepath.Join(pa.projectPath, file)
		if _, err := os.Stat(path); err == nil {
			if err := pa.analyzeDependencyFile(path); err != nil {
				fmt.Printf("Warning: failed to analyze %s: %v\n", file, err)
			}
		}
	}

	return nil
}

// analyzeDependencyFile analyzes a specific dependency file
func (pa *ProjectAnalyzer) analyzeDependencyFile(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	filename := filepath.Base(path)
	contentStr := string(content)

	switch filename {
	case "go.mod":
		return pa.analyzeGoMod(contentStr)
	case "package.json":
		return pa.analyzePackageJson(contentStr)
	case "requirements.txt":
		return pa.analyzeRequirementsTxt(contentStr)
	}

	return nil
}

// analyzeGoMod analyzes Go module dependencies
func (pa *ProjectAnalyzer) analyzeGoMod(content string) error {
	lines := strings.Split(content, "\n")
	inRequire := false

	for _, line := range lines {
		line = strings.TrimSpace(line)

		if strings.HasPrefix(line, "require") {
			if strings.Contains(line, "(") {
				inRequire = true
				continue
			} else {
				// Single line require
				pa.metrics.DirectDependencies++
			}
		}

		if inRequire {
			if line == ")" {
				inRequire = false
			} else if line != "" && !strings.HasPrefix(line, "//") {
				pa.metrics.DirectDependencies++
			}
		}
	}

	pa.metrics.TotalDependencies = pa.metrics.DirectDependencies
	return nil
}

// analyzePackageJson analyzes Node.js dependencies
func (pa *ProjectAnalyzer) analyzePackageJson(content string) error {
	// Simplified JSON parsing - count dependencies and devDependencies
	dependenciesCount := strings.Count(content, `"dependencies"`)
	devDependenciesCount := strings.Count(content, `"devDependencies"`)

	// Rough estimate based on typical package.json structure
	pa.metrics.DirectDependencies = dependenciesCount*10 + devDependenciesCount*5
	pa.metrics.TotalDependencies = pa.metrics.DirectDependencies

	return nil
}

// analyzeRequirementsTxt analyzes Python requirements
func (pa *ProjectAnalyzer) analyzeRequirementsTxt(content string) error {
	lines := strings.Split(content, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" && !strings.HasPrefix(line, "#") {
			pa.metrics.DirectDependencies++
		}
	}

	pa.metrics.TotalDependencies = pa.metrics.DirectDependencies
	return nil
}

// analyzeVersionControl analyzes git history
func (pa *ProjectAnalyzer) analyzeVersionControl() error {
	// This would require git integration
	// For now, just return nil
	return nil
}

// analyzeSecurityIssues analyzes potential security issues
func (pa *ProjectAnalyzer) analyzeSecurityIssues() error {
	// Simplified security analysis
	for path, stats := range pa.fileStatsCache {
		issues := pa.findSecurityIssuesInFile(path, stats)
		pa.metrics.SecurityIssues = append(pa.metrics.SecurityIssues, issues...)
	}

	return nil
}

// findSecurityIssuesInFile finds security issues in a file
func (pa *ProjectAnalyzer) findSecurityIssuesInFile(path string, stats FileStats) []SecurityIssue {
	var issues []SecurityIssue

	// Read file content for analysis
	fullPath := filepath.Join(pa.projectPath, path)
	content, err := os.ReadFile(fullPath)
	if err != nil {
		return issues
	}

	lines := strings.Split(string(content), "\n")

	// Common security patterns to look for
	securityPatterns := map[string]SecurityIssue{
		`password\s*=\s*["'][^"']*["']`: {
			Type:        "Hardcoded Password",
			Severity:    "High",
			Description: "Hardcoded password found in source code",
			Suggestion:  "Use environment variables or secure credential storage",
		},
		`api[_-]?key\s*=\s*["'][^"']*["']`: {
			Type:        "Hardcoded API Key",
			Severity:    "High",
			Description: "Hardcoded API key found in source code",
			Suggestion:  "Use environment variables or secure configuration",
		},
		`eval\s*\(`: {
			Type:        "Code Injection",
			Severity:    "High",
			Description: "Use of eval() function detected",
			Suggestion:  "Avoid eval() and use safer alternatives",
		},
		`sql.*\+.*["']`: {
			Type:        "SQL Injection",
			Severity:    "High",
			Description: "Potential SQL injection vulnerability",
			Suggestion:  "Use parameterized queries or prepared statements",
		},
	}

	for i, line := range lines {
		for pattern, issue := range securityPatterns {
			if matched, _ := regexp.MatchString("(?i)"+pattern, line); matched {
				securityIssue := issue
				securityIssue.File = path
				securityIssue.Line = i + 1
				issues = append(issues, securityIssue)
			}
		}
	}

	return issues
}

// calculateHealthScores calculates various health scores
func (pa *ProjectAnalyzer) calculateHealthScores() {
	scores := make(map[string]float64)

	// Code Quality Score (0-100)
	codeQuality := 100.0

	// Penalize high complexity
	if pa.metrics.CyclomaticComplexity > pa.metrics.TotalFiles*10 {
		codeQuality -= 20
	}

	// Penalize high duplication
	if pa.metrics.DuplicationRatio > 0.1 {
		codeQuality -= 15
	}

	// Penalize code smells
	if pa.metrics.CodeSmells > pa.metrics.TotalFiles/10 {
		codeQuality -= 10
	}

	scores["code_quality"] = max(0, codeQuality)

	// Test Coverage Score
	scores["test_coverage"] = pa.metrics.TestCoverage

	// Documentation Score
	commentRatio := float64(pa.metrics.CommentLines) / float64(pa.metrics.CodeLines) * 100
	scores["documentation"] = min(100, commentRatio*5) // Max out at 20% comment ratio

	// Security Score
	securityScore := 100.0
	if len(pa.metrics.SecurityIssues) > 0 {
		securityScore -= float64(len(pa.metrics.SecurityIssues)) * 10
	}
	scores["security"] = max(0, securityScore)

	// Maintainability Score
	maintainability := 100.0
	if pa.metrics.TechnicalDebt > time.Hour {
		maintainability -= 30
	} else if pa.metrics.TechnicalDebt > 30*time.Minute {
		maintainability -= 15
	}
	scores["maintainability"] = max(0, maintainability)

	// Calculate overall score as weighted average
	weights := map[string]float64{
		"code_quality":    0.25,
		"test_coverage":   0.20,
		"documentation":   0.15,
		"security":        0.25,
		"maintainability": 0.15,
	}

	overall := 0.0
	for metric, score := range scores {
		if weight, exists := weights[metric]; exists {
			overall += score * weight
		}
	}

	pa.health.Scores = scores
	pa.health.OverallScore = overall
}

// generateRecommendations generates improvement recommendations
func (pa *ProjectAnalyzer) generateRecommendations() {
	var recommendations []string

	// Code quality recommendations
	if pa.health.Scores["code_quality"] < 70 {
		recommendations = append(recommendations, "Improve code quality by reducing complexity and eliminating code smells")
	}

	// Test coverage recommendations
	if pa.health.Scores["test_coverage"] < 60 {
		recommendations = append(recommendations, "Increase test coverage to at least 60-80%")
	}

	// Documentation recommendations
	if pa.health.Scores["documentation"] < 50 {
		recommendations = append(recommendations, "Add more code comments and documentation")
	}

	// Security recommendations
	if len(pa.metrics.SecurityIssues) > 0 {
		recommendations = append(recommendations, fmt.Sprintf("Address %d security issues found in the codebase", len(pa.metrics.SecurityIssues)))
	}

	// Technical debt recommendations
	if pa.metrics.TechnicalDebt > time.Hour {
		recommendations = append(recommendations, "Significant technical debt detected - plan refactoring sessions")
	}

	// Dependency recommendations
	if pa.metrics.DirectDependencies > 50 {
		recommendations = append(recommendations, "Consider reducing the number of dependencies")
	}

	pa.health.Recommendations = recommendations
}

// Helper methods
func (pa *ProjectAnalyzer) shouldExcludePath(path string) bool {
	for _, exclude := range pa.excludePaths {
		if strings.Contains(path, exclude) {
			return true
		}
	}
	return false
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// GetTopLanguages returns the top programming languages by usage
func (pa *ProjectAnalyzer) GetTopLanguages(limit int) []LanguageStat {
	pa.mu.RLock()
	defer pa.mu.RUnlock()

	var languages []LanguageStat
	for _, lang := range pa.metrics.LanguageStats {
		languages = append(languages, lang)
	}

	sort.Slice(languages, func(i, j int) bool {
		return languages[i].Percentage > languages[j].Percentage
	})

	if limit > 0 && len(languages) > limit {
		languages = languages[:limit]
	}

	return languages
}

// GetComplexityHotspots returns files with highest complexity
func (pa *ProjectAnalyzer) GetComplexityHotspots(limit int) []FileStats {
	pa.mu.RLock()
	defer pa.mu.RUnlock()

	var files []FileStats
	for _, stats := range pa.fileStatsCache {
		files = append(files, stats)
	}

	sort.Slice(files, func(i, j int) bool {
		return files[i].Complexity > files[j].Complexity
	})

	if limit > 0 && len(files) > limit {
		files = files[:limit]
	}

	return files
}

// GenerateReport generates a comprehensive project analysis report
func (pa *ProjectAnalyzer) GenerateReport() string {
	pa.mu.RLock()
	defer pa.mu.RUnlock()

	var report strings.Builder

	report.WriteString("📊 Project Analysis Report\n")
	report.WriteString("==========================\n\n")

	// Overall health
	report.WriteString(fmt.Sprintf("Overall Health Score: %.1f/100\n\n", pa.health.OverallScore))

	// Project overview
	report.WriteString("📁 Project Overview\n")
	report.WriteString("------------------\n")
	report.WriteString(fmt.Sprintf("Total Files: %d\n", pa.metrics.TotalFiles))
	report.WriteString(fmt.Sprintf("Total Lines: %d\n", pa.metrics.TotalLines))
	report.WriteString(fmt.Sprintf("Code Lines: %d (%.1f%%)\n", pa.metrics.CodeLines, float64(pa.metrics.CodeLines)/float64(pa.metrics.TotalLines)*100))
	report.WriteString(fmt.Sprintf("Comment Lines: %d (%.1f%%)\n", pa.metrics.CommentLines, float64(pa.metrics.CommentLines)/float64(pa.metrics.TotalLines)*100))
	report.WriteString(fmt.Sprintf("Blank Lines: %d (%.1f%%)\n\n", pa.metrics.BlankLines, float64(pa.metrics.BlankLines)/float64(pa.metrics.TotalLines)*100))

	// Language breakdown
	report.WriteString("🔤 Language Breakdown\n")
	report.WriteString("---------------------\n")
	topLangs := pa.GetTopLanguages(5)
	for _, lang := range topLangs {
		report.WriteString(fmt.Sprintf("%-12s: %d files, %d lines (%.1f%%)\n", lang.Name, lang.Files, lang.Lines, lang.Percentage))
	}
	report.WriteString("\n")

	// Health scores
	report.WriteString("🏥 Health Scores\n")
	report.WriteString("---------------\n")
	for metric, score := range pa.health.Scores {
		report.WriteString(fmt.Sprintf("%-15s: %.1f/100\n", strings.Title(strings.ReplaceAll(metric, "_", " ")), score))
	}
	report.WriteString("\n")

	// Quality metrics
	report.WriteString("📈 Quality Metrics\n")
	report.WriteString("------------------\n")
	report.WriteString(fmt.Sprintf("Cyclomatic Complexity: %d\n", pa.metrics.CyclomaticComplexity))
	report.WriteString(fmt.Sprintf("Technical Debt: %v\n", pa.metrics.TechnicalDebt))
	report.WriteString(fmt.Sprintf("Code Duplication: %.1f%%\n", pa.metrics.DuplicationRatio*100))
	report.WriteString(fmt.Sprintf("Code Smells: %d\n", pa.metrics.CodeSmells))
	report.WriteString(fmt.Sprintf("Test Coverage: %.1f%%\n\n", pa.metrics.TestCoverage))

	// Dependencies
	if pa.metrics.TotalDependencies > 0 {
		report.WriteString("📦 Dependencies\n")
		report.WriteString("---------------\n")
		report.WriteString(fmt.Sprintf("Direct Dependencies: %d\n", pa.metrics.DirectDependencies))
		report.WriteString(fmt.Sprintf("Total Dependencies: %d\n\n", pa.metrics.TotalDependencies))
	}

	// Security issues
	if len(pa.metrics.SecurityIssues) > 0 {
		report.WriteString("🔒 Security Issues\n")
		report.WriteString("------------------\n")
		for _, issue := range pa.metrics.SecurityIssues {
			report.WriteString(fmt.Sprintf("• %s (%s) in %s:%d\n", issue.Type, issue.Severity, issue.File, issue.Line))
		}
		report.WriteString("\n")
	}

	// Complexity hotspots
	hotspots := pa.GetComplexityHotspots(5)
	if len(hotspots) > 0 {
		report.WriteString("🔥 Complexity Hotspots\n")
		report.WriteString("----------------------\n")
		for i, file := range hotspots {
			if i >= 5 {
				break
			}
			relPath, _ := filepath.Rel(pa.projectPath, file.Path)
			report.WriteString(fmt.Sprintf("• %s (complexity: %d, lines: %d)\n", relPath, file.Complexity, file.Lines))
		}
		report.WriteString("\n")
	}

	// Recommendations
	if len(pa.health.Recommendations) > 0 {
		report.WriteString("💡 Recommendations\n")
		report.WriteString("------------------\n")
		for _, rec := range pa.health.Recommendations {
			report.WriteString(fmt.Sprintf("• %s\n", rec))
		}
		report.WriteString("\n")
	}

	return report.String()
}

// GetMetrics returns the current project metrics
func (pa *ProjectAnalyzer) GetMetrics() *ProjectMetrics {
	pa.mu.RLock()
	defer pa.mu.RUnlock()
	return pa.metrics
}

// GetHealth returns the current project health
func (pa *ProjectAnalyzer) GetHealth() *ProjectHealth {
	pa.mu.RLock()
	defer pa.mu.RUnlock()
	return pa.health
}
