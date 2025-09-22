// internal/intelligence/pattern_analyzer.go
package intelligence

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

type PatternType int

const (
	PatternSingleton PatternType = iota
	PatternFactory
	PatternBuilder
	PatternObserver
	PatternStrategy
	PatternCommand
	PatternDecorator
	PatternAdapter
	PatternFacade
	PatternProxy
	PatternTemplate
	PatternIterator
	PatternComposite
	PatternState
	PatternChainOfResponsibility
	PatternMediator
	PatternVisitor
	PatternMemento
	PatternPrototype
	PatternBridge
	PatternFlyweight
	PatternInterpreter
	PatternNullObject
	PatternDependencyInjection
	PatternRepository
	PatternUnitOfWork
	PatternSpecification
	PatternCQRS
	PatternEventSourcing
	PatternSaga
	PatternCircuitBreaker
	PatternRetry
	PatternBulkhead
	PatternTimeout
)

type Pattern struct {
	Type        PatternType
	Name        string
	Description string
	Category    string
	Confidence  float64
	Occurrences []PatternOccurrence
	Suggestions []string
}

type PatternOccurrence struct {
	File       string
	LineStart  int
	LineEnd    int
	Code       string
	Context    string
	Confidence float64
	Evidence   []string
}

type PatternRule struct {
	Type        PatternType
	Name        string
	Description string
	Category    string
	Matchers    []PatternMatcher
	MinMatches  int
	BaseScore   float64
}

type PatternMatcher struct {
	Type        MatcherType
	Pattern     string
	Weight      float64
	Description string
}

type MatcherType int

const (
	MatcherRegex MatcherType = iota
	MatcherStructural
	MatcherSemantic
	MatcherNaming
	MatcherAnnotation
)

type PatternAnalyzer struct {
	projectPath string
	rules       map[PatternType]*PatternRule
	findings    []Pattern
	mu          sync.RWMutex

	// Analysis state
	analyzedFiles map[string]time.Time
	codeCache     map[string]string
}

// Pattern rules definition
var patternRules = map[PatternType]*PatternRule{
	PatternSingleton: {
		Type:        PatternSingleton,
		Name:        "Singleton Pattern",
		Description: "Ensures a class has only one instance and provides global access",
		Category:    "Creational",
		BaseScore:   0.8,
		MinMatches:  2,
		Matchers: []PatternMatcher{
			{MatcherRegex, `private\s+static\s+\w+\s+instance`, 0.9, "Private static instance"},
			{MatcherRegex, `private\s+\w+\s*\(\s*\)`, 0.8, "Private constructor"},
			{MatcherRegex, `public\s+static\s+\w+\s+getInstance\s*\(`, 0.9, "GetInstance method"},
			{MatcherRegex, `var\s+\w+\s+\*\w+\s*$`, 0.7, "Go singleton variable"},
			{MatcherRegex, `sync\.Once`, 0.8, "Go sync.Once usage"},
			{MatcherNaming, `getInstance|GetInstance|instance`, 0.6, "Singleton naming convention"},
		},
	},
	PatternFactory: {
		Type:        PatternFactory,
		Name:        "Factory Pattern",
		Description: "Creates objects without specifying their concrete classes",
		Category:    "Creational",
		BaseScore:   0.7,
		MinMatches:  2,
		Matchers: []PatternMatcher{
			{MatcherRegex, `func\s+\w*[Ff]actory\w*\s*\(`, 0.8, "Factory function"},
			{MatcherRegex, `class\s+\w*Factory\w*`, 0.9, "Factory class"},
			{MatcherRegex, `interface\s+\w*Factory\w*`, 0.8, "Factory interface"},
			{MatcherRegex, `func\s+[Nn]ew\w+\s*\(`, 0.6, "Constructor function"},
			{MatcherRegex, `func\s+[Cc]reate\w+\s*\(`, 0.7, "Create function"},
			{MatcherNaming, `Factory|Create|New|Build`, 0.5, "Factory naming"},
		},
	},
	PatternBuilder: {
		Type:        PatternBuilder,
		Name:        "Builder Pattern",
		Description: "Constructs complex objects step by step",
		Category:    "Creational",
		BaseScore:   0.7,
		MinMatches:  3,
		Matchers: []PatternMatcher{
			{MatcherRegex, `class\s+\w*Builder\w*`, 0.9, "Builder class"},
			{MatcherRegex, `func\s+\(\w*\s+\*?\w*Builder\w*\)\s+\w+\s*\(`, 0.8, "Builder method"},
			{MatcherRegex, `func\s+\(\w*\s+\*?\w*\)\s+Build\s*\(`, 0.9, "Build method"},
			{MatcherRegex, `return\s+\w*\.Build\(\)`, 0.7, "Build call"},
			{MatcherRegex, `\.With\w+\(`, 0.6, "Fluent interface"},
			{MatcherNaming, `Builder|Build|With`, 0.5, "Builder naming"},
		},
	},
	PatternObserver: {
		Type:        PatternObserver,
		Name:        "Observer Pattern",
		Description: "Defines one-to-many dependency between objects",
		Category:    "Behavioral",
		BaseScore:   0.7,
		MinMatches:  2,
		Matchers: []PatternMatcher{
			{MatcherRegex, `interface\s+\w*Observer\w*`, 0.9, "Observer interface"},
			{MatcherRegex, `func\s+\w*[Nn]otify\w*\s*\(`, 0.8, "Notify method"},
			{MatcherRegex, `func\s+\w*[Ss]ubscribe\w*\s*\(`, 0.7, "Subscribe method"},
			{MatcherRegex, `func\s+\w*[Uu]nsubscribe\w*\s*\(`, 0.7, "Unsubscribe method"},
			{MatcherRegex, `func\s+\w*[Uu]pdate\w*\s*\(`, 0.6, "Update method"},
			{MatcherNaming, `Observer|Subject|Notify|Subscribe|Update`, 0.5, "Observer naming"},
		},
	},
	PatternStrategy: {
		Type:        PatternStrategy,
		Name:        "Strategy Pattern",
		Description: "Defines family of algorithms and makes them interchangeable",
		Category:    "Behavioral",
		BaseScore:   0.7,
		MinMatches:  2,
		Matchers: []PatternMatcher{
			{MatcherRegex, `interface\s+\w*Strategy\w*`, 0.9, "Strategy interface"},
			{MatcherRegex, `class\s+\w*Strategy\w*`, 0.8, "Strategy implementation"},
			{MatcherRegex, `func\s+\w*[Ee]xecute\w*\s*\(`, 0.6, "Execute method"},
			{MatcherRegex, `func\s+\w*[Aa]lgorithm\w*\s*\(`, 0.7, "Algorithm method"},
			{MatcherRegex, `func\s+SetStrategy\s*\(`, 0.8, "Strategy setter"},
			{MatcherNaming, `Strategy|Algorithm|Execute`, 0.5, "Strategy naming"},
		},
	},
	PatternRepository: {
		Type:        PatternRepository,
		Name:        "Repository Pattern",
		Description: "Encapsulates data access logic and provides a uniform interface",
		Category:    "Data Access",
		BaseScore:   0.8,
		MinMatches:  2,
		Matchers: []PatternMatcher{
			{MatcherRegex, `interface\s+\w*Repository\w*`, 0.9, "Repository interface"},
			{MatcherRegex, `class\s+\w*Repository\w*`, 0.8, "Repository class"},
			{MatcherRegex, `func\s+\w*[Ff]ind\w*\s*\(`, 0.7, "Find method"},
			{MatcherRegex, `func\s+\w*[Ss]ave\w*\s*\(`, 0.7, "Save method"},
			{MatcherRegex, `func\s+\w*[Dd]elete\w*\s*\(`, 0.7, "Delete method"},
			{MatcherRegex, `func\s+\w*[Gg]et\w*[Bb]y\w*\s*\(`, 0.6, "GetBy method"},
			{MatcherNaming, `Repository|Find|Save|Delete|GetBy`, 0.5, "Repository naming"},
		},
	},
	PatternCircuitBreaker: {
		Type:        PatternCircuitBreaker,
		Name:        "Circuit Breaker Pattern",
		Description: "Prevents cascading failures in distributed systems",
		Category:    "Resilience",
		BaseScore:   0.8,
		MinMatches:  2,
		Matchers: []PatternMatcher{
			{MatcherRegex, `type\s+CircuitBreaker\s+struct`, 0.9, "CircuitBreaker type"},
			{MatcherRegex, `CircuitBreakerState|State\s+(Open|Closed|HalfOpen)`, 0.8, "Circuit breaker states"},
			{MatcherRegex, `func\s+\w*[Cc]all\w*\s*\(`, 0.6, "Call method"},
			{MatcherRegex, `failureCount|successCount|threshold`, 0.7, "Failure tracking"},
			{MatcherRegex, `timeout|lastFailureTime`, 0.6, "Timeout handling"},
			{MatcherNaming, `CircuitBreaker|Breaker|Failure|Threshold`, 0.5, "Circuit breaker naming"},
		},
	},
	PatternRetry: {
		Type:        PatternRetry,
		Name:        "Retry Pattern",
		Description: "Automatically retries failed operations",
		Category:    "Resilience",
		BaseScore:   0.7,
		MinMatches:  2,
		Matchers: []PatternMatcher{
			{MatcherRegex, `for\s+\w+\s*:=\s*0;\s*\w+\s*<\s*\w*[Rr]etries?\w*`, 0.8, "Retry loop"},
			{MatcherRegex, `func\s+\w*[Rr]etry\w*\s*\(`, 0.9, "Retry function"},
			{MatcherRegex, `time\.Sleep\(.*\*\s*\w+\)`, 0.7, "Backoff delay"},
			{MatcherRegex, `maxRetries|retryCount|attempts`, 0.6, "Retry configuration"},
			{MatcherRegex, `backoff|exponential|linear`, 0.6, "Backoff strategy"},
			{MatcherNaming, `Retry|Attempt|Backoff|MaxRetries`, 0.5, "Retry naming"},
		},
	},
	PatternDependencyInjection: {
		Type:        PatternDependencyInjection,
		Name:        "Dependency Injection Pattern",
		Description: "Provides dependencies from external sources",
		Category:    "Structural",
		BaseScore:   0.7,
		MinMatches:  2,
		Matchers: []PatternMatcher{
			{MatcherRegex, `func\s+[Nn]ew\w+\s*\([^)]*\w+\s+\w+Interface`, 0.8, "Constructor injection"},
			{MatcherRegex, `type\s+\w+\s+struct\s*{[^}]*\w+\s+\w+Interface`, 0.7, "Interface dependency"},
			{MatcherRegex, `@Inject|@Autowired|@Component`, 0.9, "DI annotations"},
			{MatcherRegex, `container\.Register|container\.Resolve`, 0.8, "DI container"},
			{MatcherRegex, `wire\.Build|wire\.Bind`, 0.9, "Google Wire"},
			{MatcherNaming, `Inject|Container|Wire|Dependency`, 0.5, "DI naming"},
		},
	},
}

// NewPatternAnalyzer creates a new pattern analyzer
func NewPatternAnalyzer(projectPath string) *PatternAnalyzer {
	return &PatternAnalyzer{
		projectPath:   projectPath,
		rules:         patternRules,
		findings:      make([]Pattern, 0),
		analyzedFiles: make(map[string]time.Time),
		codeCache:     make(map[string]string),
	}
}

// AnalyzePatterns analyzes the project for design patterns
func (pa *PatternAnalyzer) AnalyzePatterns() ([]Pattern, error) {
	pa.mu.Lock()
	defer pa.mu.Unlock()

	// Clear previous findings
	pa.findings = make([]Pattern, 0)

	// Analyze all source files
	err := filepath.Walk(pa.projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if pa.shouldAnalyzeFile(path, info) {
			return pa.analyzeFile(path)
		}

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to analyze patterns: %w", err)
	}

	// Aggregate patterns by type
	pa.aggregatePatterns()

	// Sort by confidence
	sort.Slice(pa.findings, func(i, j int) bool {
		return pa.findings[i].Confidence > pa.findings[j].Confidence
	})

	return pa.findings, nil
}

// shouldAnalyzeFile determines if a file should be analyzed
func (pa *PatternAnalyzer) shouldAnalyzeFile(path string, info os.FileInfo) bool {
	if info.IsDir() {
		return false
	}

	// Skip hidden files and common build directories
	if strings.HasPrefix(info.Name(), ".") {
		return false
	}

	skipDirs := []string{"node_modules", "vendor", "target", "build", "dist", "__pycache__"}
	for _, skip := range skipDirs {
		if strings.Contains(path, skip) {
			return false
		}
	}

	// Only analyze source code files
	ext := filepath.Ext(path)
	sourceExts := []string{".go", ".java", ".py", ".js", ".ts", ".cpp", ".c", ".cs", ".rb", ".php", ".kt", ".swift", ".scala", ".rs"}

	for _, sourceExt := range sourceExts {
		if ext == sourceExt {
			return true
		}
	}

	return false
}

// analyzeFile analyzes a single file for patterns
func (pa *PatternAnalyzer) analyzeFile(path string) error {
	// Check if file needs re-analysis
	info, err := os.Stat(path)
	if err != nil {
		return err
	}

	if lastAnalyzed, exists := pa.analyzedFiles[path]; exists {
		if info.ModTime().Before(lastAnalyzed) {
			return nil // File hasn't changed, skip analysis
		}
	}

	// Read file content
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	contentStr := string(content)
	pa.codeCache[path] = contentStr

	// Analyze patterns in this file
	for patternType, rule := range pa.rules {
		occurrences := pa.findPatternOccurrences(path, contentStr, rule)
		if len(occurrences) >= rule.MinMatches {
			pa.addPatternFinding(patternType, rule, occurrences)
		}
	}

	pa.analyzedFiles[path] = time.Now()
	return nil
}

// findPatternOccurrences finds pattern occurrences in file content
func (pa *PatternAnalyzer) findPatternOccurrences(filePath, content string, rule *PatternRule) []PatternOccurrence {
	var occurrences []PatternOccurrence
	lines := strings.Split(content, "\n")

	for _, matcher := range rule.Matchers {
		matches := pa.findMatches(lines, matcher)
		for _, match := range matches {
			occurrence := PatternOccurrence{
				File:       filePath,
				LineStart:  match.LineStart,
				LineEnd:    match.LineEnd,
				Code:       match.Code,
				Context:    match.Context,
				Confidence: matcher.Weight,
				Evidence:   []string{matcher.Description},
			}
			occurrences = append(occurrences, occurrence)
		}
	}

	return occurrences
}

type Match struct {
	LineStart int
	LineEnd   int
	Code      string
	Context   string
}

// findMatches finds matches for a specific matcher
func (pa *PatternAnalyzer) findMatches(lines []string, matcher PatternMatcher) []Match {
	var matches []Match

	switch matcher.Type {
	case MatcherRegex:
		regex, err := regexp.Compile(matcher.Pattern)
		if err != nil {
			return matches
		}

		for i, line := range lines {
			if regex.MatchString(line) {
				match := Match{
					LineStart: i + 1,
					LineEnd:   i + 1,
					Code:      strings.TrimSpace(line),
					Context:   pa.getContext(lines, i, 2),
				}
				matches = append(matches, match)
			}
		}

	case MatcherNaming:
		nameRegex, err := regexp.Compile("(?i)" + matcher.Pattern)
		if err != nil {
			return matches
		}

		for i, line := range lines {
			if nameRegex.MatchString(line) {
				match := Match{
					LineStart: i + 1,
					LineEnd:   i + 1,
					Code:      strings.TrimSpace(line),
					Context:   pa.getContext(lines, i, 1),
				}
				matches = append(matches, match)
			}
		}

	case MatcherStructural:
		// More complex structural analysis could be implemented here
		matches = pa.findStructuralMatches(lines, matcher)

	case MatcherSemantic:
		// Semantic analysis could be implemented here
		matches = pa.findSemanticMatches(lines, matcher)

	case MatcherAnnotation:
		// Annotation-based matching
		matches = pa.findAnnotationMatches(lines, matcher)
	}

	return matches
}

// getContext extracts context lines around a match
func (pa *PatternAnalyzer) getContext(lines []string, lineIndex, contextSize int) string {
	start := lineIndex - contextSize
	if start < 0 {
		start = 0
	}

	end := lineIndex + contextSize + 1
	if end > len(lines) {
		end = len(lines)
	}

	contextLines := lines[start:end]
	return strings.Join(contextLines, "\n")
}

// findStructuralMatches finds matches based on code structure
func (pa *PatternAnalyzer) findStructuralMatches(lines []string, matcher PatternMatcher) []Match {
	var matches []Match

	// Example: Find class definitions followed by specific methods
	if strings.Contains(matcher.Pattern, "class+method") {
		classLine := -1
		for i, line := range lines {
			line = strings.TrimSpace(line)

			if strings.HasPrefix(line, "class ") || strings.HasPrefix(line, "type ") {
				classLine = i
			} else if classLine >= 0 && strings.Contains(line, "func ") {
				// Found method after class, check if it matches pattern
				if strings.Contains(matcher.Pattern, "getInstance") && strings.Contains(line, "getInstance") {
					match := Match{
						LineStart: classLine + 1,
						LineEnd:   i + 1,
						Code:      pa.getCodeBlock(lines, classLine, i),
						Context:   pa.getContext(lines, classLine, 3),
					}
					matches = append(matches, match)
					classLine = -1
				}
			}
		}
	}

	return matches
}

// findSemanticMatches finds matches based on semantic meaning
func (pa *PatternAnalyzer) findSemanticMatches(lines []string, matcher PatternMatcher) []Match {
	var matches []Match

	// Example: Find patterns where multiple classes implement the same interface
	// This would require more sophisticated analysis

	return matches
}

// findAnnotationMatches finds matches based on annotations/decorators
func (pa *PatternAnalyzer) findAnnotationMatches(lines []string, matcher PatternMatcher) []Match {
	var matches []Match

	regex, err := regexp.Compile(matcher.Pattern)
	if err != nil {
		return matches
	}

	for i, line := range lines {
		line = strings.TrimSpace(line)
		if regex.MatchString(line) {
			match := Match{
				LineStart: i + 1,
				LineEnd:   i + 1,
				Code:      line,
				Context:   pa.getContext(lines, i, 2),
			}
			matches = append(matches, match)
		}
	}

	return matches
}

// getCodeBlock extracts a block of code between two lines
func (pa *PatternAnalyzer) getCodeBlock(lines []string, start, end int) string {
	if start < 0 || end >= len(lines) || start > end {
		return ""
	}

	return strings.Join(lines[start:end+1], "\n")
}

// addPatternFinding adds a pattern finding
func (pa *PatternAnalyzer) addPatternFinding(patternType PatternType, rule *PatternRule, occurrences []PatternOccurrence) {
	// Calculate overall confidence
	totalWeight := 0.0
	matchedWeight := 0.0

	for _, matcher := range rule.Matchers {
		totalWeight += matcher.Weight

		// Check if this matcher has occurrences
		for _, occurrence := range occurrences {
			for _, evidence := range occurrence.Evidence {
				if evidence == matcher.Description {
					matchedWeight += matcher.Weight
					break
				}
			}
		}
	}

	confidence := (matchedWeight / totalWeight) * rule.BaseScore
	if confidence > 1.0 {
		confidence = 1.0
	}

	pattern := Pattern{
		Type:        patternType,
		Name:        rule.Name,
		Description: rule.Description,
		Category:    rule.Category,
		Confidence:  confidence,
		Occurrences: occurrences,
		Suggestions: pa.generatePatternSuggestions(patternType, confidence, occurrences),
	}

	pa.findings = append(pa.findings, pattern)
}

// aggregatePatterns combines similar pattern findings
func (pa *PatternAnalyzer) aggregatePatterns() {
	patternMap := make(map[PatternType]*Pattern)

	for _, finding := range pa.findings {
		if existing, exists := patternMap[finding.Type]; exists {
			// Merge occurrences
			existing.Occurrences = append(existing.Occurrences, finding.Occurrences...)

			// Update confidence (take max)
			if finding.Confidence > existing.Confidence {
				existing.Confidence = finding.Confidence
			}

			// Merge suggestions
			existing.Suggestions = pa.mergeSuggestions(existing.Suggestions, finding.Suggestions)
		} else {
			findingCopy := finding
			patternMap[finding.Type] = &findingCopy
		}
	}

	// Replace findings with aggregated results
	pa.findings = make([]Pattern, 0, len(patternMap))
	for _, pattern := range patternMap {
		pa.findings = append(pa.findings, *pattern)
	}
}

// mergeSuggestions merges two suggestion lists, removing duplicates
func (pa *PatternAnalyzer) mergeSuggestions(a, b []string) []string {
	seen := make(map[string]bool)
	var result []string

	for _, suggestion := range a {
		if !seen[suggestion] {
			result = append(result, suggestion)
			seen[suggestion] = true
		}
	}

	for _, suggestion := range b {
		if !seen[suggestion] {
			result = append(result, suggestion)
			seen[suggestion] = true
		}
	}

	return result
}

// generatePatternSuggestions generates improvement suggestions for patterns
func (pa *PatternAnalyzer) generatePatternSuggestions(patternType PatternType, confidence float64, occurrences []PatternOccurrence) []string {
	var suggestions []string

	switch patternType {
	case PatternSingleton:
		if confidence < 0.8 {
			suggestions = append(suggestions, "Consider using sync.Once for thread-safe singleton initialization")
			suggestions = append(suggestions, "Ensure the constructor is private to prevent multiple instances")
		}
		suggestions = append(suggestions, "Consider dependency injection instead of singleton for better testability")

	case PatternFactory:
		if confidence < 0.8 {
			suggestions = append(suggestions, "Consider using interfaces to make the factory more flexible")
			suggestions = append(suggestions, "Add error handling for unsupported types")
		}
		suggestions = append(suggestions, "Consider abstract factory for families of related objects")

	case PatternBuilder:
		if confidence < 0.8 {
			suggestions = append(suggestions, "Ensure builder validates required fields before building")
			suggestions = append(suggestions, "Consider making builder methods return the builder for fluent interface")
		}

	case PatternObserver:
		if confidence < 0.8 {
			suggestions = append(suggestions, "Add error handling for observer notifications")
			suggestions = append(suggestions, "Consider using channels for Go observer pattern")
		}
		suggestions = append(suggestions, "Ensure observers are properly unregistered to prevent memory leaks")

	case PatternRepository:
		if confidence < 0.8 {
			suggestions = append(suggestions, "Use interfaces to define repository contracts")
			suggestions = append(suggestions, "Implement proper error handling and transaction support")
		}
		suggestions = append(suggestions, "Consider using specification pattern for complex queries")

	case PatternCircuitBreaker:
		if confidence < 0.8 {
			suggestions = append(suggestions, "Implement proper state transitions (Closed -> Open -> Half-Open)")
			suggestions = append(suggestions, "Add metrics and monitoring for circuit breaker state")
		}
		suggestions = append(suggestions, "Consider configurable thresholds and timeout values")

	case PatternRetry:
		if confidence < 0.8 {
			suggestions = append(suggestions, "Implement exponential backoff with jitter")
			suggestions = append(suggestions, "Add maximum retry limits to prevent infinite loops")
		}
		suggestions = append(suggestions, "Consider different retry strategies for different error types")
	}

	// Add generic suggestions based on number of occurrences
	if len(occurrences) == 1 {
		suggestions = append(suggestions, "Pattern implementation found - consider applying consistently across the codebase")
	} else if len(occurrences) > 5 {
		suggestions = append(suggestions, "Multiple implementations found - ensure consistency across all instances")
	}

	return suggestions
}

// GetPatternsByCategory returns patterns grouped by category
func (pa *PatternAnalyzer) GetPatternsByCategory() map[string][]Pattern {
	pa.mu.RLock()
	defer pa.mu.RUnlock()

	categories := make(map[string][]Pattern)

	for _, pattern := range pa.findings {
		if categories[pattern.Category] == nil {
			categories[pattern.Category] = make([]Pattern, 0)
		}
		categories[pattern.Category] = append(categories[pattern.Category], pattern)
	}

	return categories
}

// GetPatternDetails returns detailed information about a specific pattern
func (pa *PatternAnalyzer) GetPatternDetails(patternType PatternType) (*Pattern, bool) {
	pa.mu.RLock()
	defer pa.mu.RUnlock()

	for _, pattern := range pa.findings {
		if pattern.Type == patternType {
			return &pattern, true
		}
	}

	return nil, false
}

// GeneratePatternReport generates a comprehensive pattern analysis report
func (pa *PatternAnalyzer) GeneratePatternReport() string {
	pa.mu.RLock()
	defer pa.mu.RUnlock()

	var report strings.Builder

	report.WriteString("🔍 Design Pattern Analysis Report\n")
	report.WriteString("=================================\n\n")

	if len(pa.findings) == 0 {
		report.WriteString("No design patterns detected in the codebase.\n")
		report.WriteString("Consider implementing established design patterns to improve code organization and maintainability.\n")
		return report.String()
	}

	// Group by category
	categories := pa.GetPatternsByCategory()

	report.WriteString(fmt.Sprintf("Found %d design pattern(s) across %d categories:\n\n", len(pa.findings), len(categories)))

	for category, patterns := range categories {
		report.WriteString(fmt.Sprintf("📂 %s Patterns\n", category))
		report.WriteString(strings.Repeat("-", len(category)+10) + "\n")

		for _, pattern := range patterns {
			report.WriteString(fmt.Sprintf("• %s (%.1f%% confidence)\n", pattern.Name, pattern.Confidence*100))
			report.WriteString(fmt.Sprintf("  %s\n", pattern.Description))
			report.WriteString(fmt.Sprintf("  Found %d occurrence(s) across:\n", len(pattern.Occurrences)))

			// Show unique files
			fileMap := make(map[string]bool)
			for _, occurrence := range pattern.Occurrences {
				relPath, _ := filepath.Rel(pa.projectPath, occurrence.File)
				fileMap[relPath] = true
			}

			for file := range fileMap {
				report.WriteString(fmt.Sprintf("    - %s\n", file))
			}

			if len(pattern.Suggestions) > 0 {
				report.WriteString("  Suggestions:\n")
				for _, suggestion := range pattern.Suggestions {
					report.WriteString(fmt.Sprintf("    → %s\n", suggestion))
				}
			}

			report.WriteString("\n")
		}

		report.WriteString("\n")
	}

	return report.String()
}

// GetAnalysisStats returns analysis statistics
func (pa *PatternAnalyzer) GetAnalysisStats() map[string]interface{} {
	pa.mu.RLock()
	defer pa.mu.RUnlock()

	totalOccurrences := 0
	for _, pattern := range pa.findings {
		totalOccurrences += len(pattern.Occurrences)
	}

	return map[string]interface{}{
		"patterns_found":    len(pa.findings),
		"total_occurrences": totalOccurrences,
		"analyzed_files":    len(pa.analyzedFiles),
		"cached_files":      len(pa.codeCache),
		"available_rules":   len(pa.rules),
	}
}
