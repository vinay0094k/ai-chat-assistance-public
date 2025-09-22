// internal/intelligence/architecture_detector.go
package intelligence

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

type ArchitecturePattern int

const (
	PatternMVC ArchitecturePattern = iota
	PatternMVP
	PatternMVVM
	PatternCleanArchitecture
	PatternHexagonal
	PatternLayered
	PatternMicroservices
	PatternServerless
	PatternEventDriven
	PatternCQRS
	PatternDomainDriven
	PatternRepository
	PatternFactory
	PatternSingleton
	PatternObserver
	PatternStrategy
	PatternMonolithic
	PatternModular
)

type ArchitectureInfo struct {
	Pattern     ArchitecturePattern
	Name        string
	Description string
	Confidence  float64
	Evidence    []string
	Suggestions []string
}

type DirectoryStructure struct {
	Path      string
	Children  map[string]*DirectoryStructure
	Files     []string
	IsPackage bool
}

type ArchitectureDetector struct {
	projectPath      string
	structure        *DirectoryStructure
	filePatterns     map[string][]string
	detectedPatterns []ArchitectureInfo
	mu               sync.RWMutex
}

// Pattern detection rules
var architectureRules = map[ArchitecturePattern]ArchitectureRule{
	PatternMVC: {
		Name:        "Model-View-Controller (MVC)",
		Description: "Separates application logic into three interconnected components",
		DirectoryPatterns: []string{
			"models?", "views?", "controllers?",
			"model", "view", "controller",
		},
		FilePatterns: []string{
			`.*[Mm]odel.*`, `.*[Vv]iew.*`, `.*[Cc]ontroller.*`,
		},
		ContentPatterns: []string{
			`class.*Controller`, `class.*Model`, `class.*View`,
			`type.*Controller`, `type.*Model`, `type.*View`,
		},
		MinConfidence: 0.6,
	},
	PatternCleanArchitecture: {
		Name:        "Clean Architecture",
		Description: "Layered architecture with dependency inversion",
		DirectoryPatterns: []string{
			"domain", "usecase", "interface", "infrastructure",
			"entities", "usecases", "interfaces", "frameworks",
			"core", "application", "adapter",
		},
		FilePatterns: []string{
			`.*[Uu]secase.*`, `.*[Ee]ntity.*`, `.*[Rr]epository.*`,
			`.*[Aa]dapter.*`, `.*[Gg]ateway.*`,
		},
		ContentPatterns: []string{
			`interface.*Repository`, `type.*UseCase`, `type.*Entity`,
			`interface.*Gateway`, `dependency injection`,
		},
		MinConfidence: 0.7,
	},
	PatternMicroservices: {
		Name:        "Microservices",
		Description: "Distributed architecture with loosely coupled services",
		DirectoryPatterns: []string{
			"services", "service-*", "*-service", "microservices",
		},
		FilePatterns: []string{
			`docker-compose\.ya?ml`, `Dockerfile.*`, `.*[Ss]ervice.*`,
			`k8s/.*`, `kubernetes/.*`,
		},
		ContentPatterns: []string{
			`apiVersion.*apps/v1`, `kind.*Service`, `kind.*Deployment`,
			`FROM.*`, `EXPOSE.*`, `grpc`, `REST API`,
		},
		MinConfidence: 0.8,
	},
	PatternDomainDriven: {
		Name:        "Domain-Driven Design (DDD)",
		Description: "Domain-centric architecture with bounded contexts",
		DirectoryPatterns: []string{
			"domain", "aggregate", "bounded-context", "context",
			"domains", "aggregates",
		},
		FilePatterns: []string{
			`.*[Aa]ggregate.*`, `.*[Dd]omain.*`, `.*[Vv]alue[Oo]bject.*`,
			`.*[Ee]ntity.*`, `.*[Ss]ervice.*`,
		},
		ContentPatterns: []string{
			`type.*Aggregate`, `type.*ValueObject`, `type.*DomainService`,
			`bounded context`, `domain event`, `aggregate root`,
		},
		MinConfidence: 0.7,
	},
	PatternHexagonal: {
		Name:        "Hexagonal Architecture (Ports & Adapters)",
		Description: "Isolates core logic from external concerns",
		DirectoryPatterns: []string{
			"ports", "adapters", "hexagon", "core", "application",
		},
		FilePatterns: []string{
			`.*[Pp]ort.*`, `.*[Aa]dapter.*`, `.*[Cc]ore.*`,
		},
		ContentPatterns: []string{
			`interface.*Port`, `type.*Adapter`, `primary port`, `secondary port`,
		},
		MinConfidence: 0.7,
	},
	PatternLayered: {
		Name:        "Layered Architecture",
		Description: "Organized into horizontal layers with clear dependencies",
		DirectoryPatterns: []string{
			"presentation", "business", "data", "persistence",
			"ui", "logic", "dal", "service", "repository",
		},
		FilePatterns: []string{
			`.*[Ll]ayer.*`, `.*[Tt]ier.*`,
		},
		ContentPatterns: []string{
			`presentation layer`, `business layer`, `data layer`,
		},
		MinConfidence: 0.6,
	},
	PatternRepository: {
		Name:        "Repository Pattern",
		Description: "Encapsulates data access logic",
		DirectoryPatterns: []string{
			"repository", "repositories", "repo",
		},
		FilePatterns: []string{
			`.*[Rr]epository.*`, `.*[Rr]epo.*`,
		},
		ContentPatterns: []string{
			`interface.*Repository`, `type.*Repository`, `func.*FindBy`,
			`func.*Save`, `func.*Delete`,
		},
		MinConfidence: 0.8,
	},
	PatternFactory: {
		Name:        "Factory Pattern",
		Description: "Creates objects without specifying exact classes",
		FilePatterns: []string{
			`.*[Ff]actory.*`, `.*[Bb]uilder.*`,
		},
		ContentPatterns: []string{
			`func.*Factory`, `func.*Create`, `func.*New.*Factory`,
			`type.*Factory`, `Builder.*Build`,
		},
		MinConfidence: 0.7,
	},
}

type ArchitectureRule struct {
	Name              string
	Description       string
	DirectoryPatterns []string
	FilePatterns      []string
	ContentPatterns   []string
	MinConfidence     float64
}

// NewArchitectureDetector creates a new architecture detector
func NewArchitectureDetector(projectPath string) *ArchitectureDetector {
	return &ArchitectureDetector{
		projectPath:      projectPath,
		filePatterns:     make(map[string][]string),
		detectedPatterns: make([]ArchitectureInfo, 0),
	}
}

// AnalyzeArchitecture analyzes the project architecture
func (ad *ArchitectureDetector) AnalyzeArchitecture() ([]ArchitectureInfo, error) {
	ad.mu.Lock()
	defer ad.mu.Unlock()

	// Build directory structure
	if err := ad.buildDirectoryStructure(); err != nil {
		return nil, fmt.Errorf("failed to build directory structure: %w", err)
	}

	// Detect patterns
	ad.detectPatterns()

	// Sort by confidence
	ad.sortByConfidence()

	return ad.detectedPatterns, nil
}

// buildDirectoryStructure builds the project directory structure
func (ad *ArchitectureDetector) buildDirectoryStructure() error {
	ad.structure = &DirectoryStructure{
		Path:     ad.projectPath,
		Children: make(map[string]*DirectoryStructure),
		Files:    make([]string, 0),
	}

	return filepath.Walk(ad.projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip hidden files and directories
		if strings.HasPrefix(info.Name(), ".") {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// Skip common build/cache directories
		skipDirs := []string{"node_modules", "vendor", "target", "build", "dist", ".git"}
		for _, skipDir := range skipDirs {
			if strings.Contains(path, skipDir) {
				if info.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
		}

		relPath, err := filepath.Rel(ad.projectPath, path)
		if err != nil {
			return err
		}

		if info.IsDir() {
			ad.addDirectory(relPath)
		} else {
			ad.addFile(relPath)
		}

		return nil
	})
}

// addDirectory adds a directory to the structure
func (ad *ArchitectureDetector) addDirectory(path string) {
	parts := strings.Split(filepath.Clean(path), string(filepath.Separator))
	current := ad.structure

	for _, part := range parts {
		if part == "." {
			continue
		}

		if current.Children[part] == nil {
			current.Children[part] = &DirectoryStructure{
				Path:     filepath.Join(current.Path, part),
				Children: make(map[string]*DirectoryStructure),
				Files:    make([]string, 0),
			}
		}
		current = current.Children[part]
	}

	current.IsPackage = true
}

// addFile adds a file to the structure
func (ad *ArchitectureDetector) addFile(path string) {
	dir := filepath.Dir(path)
	filename := filepath.Base(path)

	if dir == "." {
		ad.structure.Files = append(ad.structure.Files, filename)
		return
	}

	parts := strings.Split(dir, string(filepath.Separator))
	current := ad.structure

	for _, part := range parts {
		if current.Children[part] == nil {
			current.Children[part] = &DirectoryStructure{
				Path:     filepath.Join(current.Path, part),
				Children: make(map[string]*DirectoryStructure),
				Files:    make([]string, 0),
			}
		}
		current = current.Children[part]
	}

	current.Files = append(current.Files, filename)

	// Store file patterns for content analysis
	ext := filepath.Ext(filename)
	if ad.filePatterns[ext] == nil {
		ad.filePatterns[ext] = make([]string, 0)
	}
	ad.filePatterns[ext] = append(ad.filePatterns[ext], path)
}

// detectPatterns detects architecture patterns
func (ad *ArchitectureDetector) detectPatterns() {
	ad.detectedPatterns = make([]ArchitectureInfo, 0)

	for pattern, rule := range architectureRules {
		confidence, evidence := ad.analyzePattern(pattern, rule)

		if confidence >= rule.MinConfidence {
			info := ArchitectureInfo{
				Pattern:     pattern,
				Name:        rule.Name,
				Description: rule.Description,
				Confidence:  confidence,
				Evidence:    evidence,
				Suggestions: ad.generateSuggestions(pattern, confidence),
			}

			ad.detectedPatterns = append(ad.detectedPatterns, info)
		}
	}
}

// analyzePattern analyzes a specific pattern
func (ad *ArchitectureDetector) analyzePattern(pattern ArchitecturePattern, rule ArchitectureRule) (float64, []string) {
	var evidence []string
	var score float64
	totalChecks := 0

	// Check directory patterns
	for _, dirPattern := range rule.DirectoryPatterns {
		totalChecks++
		if ad.checkDirectoryPattern(dirPattern) {
			score += 1.0
			evidence = append(evidence, fmt.Sprintf("Found directory matching pattern: %s", dirPattern))
		}
	}

	// Check file patterns
	for _, filePattern := range rule.FilePatterns {
		totalChecks++
		if matches := ad.checkFilePattern(filePattern); len(matches) > 0 {
			score += 1.0
			evidence = append(evidence, fmt.Sprintf("Found files matching pattern: %s (%d files)", filePattern, len(matches)))
		}
	}

	// Check content patterns
	for _, contentPattern := range rule.ContentPatterns {
		totalChecks++
		if matches := ad.checkContentPattern(contentPattern); len(matches) > 0 {
			score += 1.0
			evidence = append(evidence, fmt.Sprintf("Found content matching pattern: %s (%d occurrences)", contentPattern, len(matches)))
		}
	}

	// Calculate confidence
	confidence := 0.0
	if totalChecks > 0 {
		confidence = score / float64(totalChecks)
	}

	return confidence, evidence
}

// checkDirectoryPattern checks if directory pattern exists
func (ad *ArchitectureDetector) checkDirectoryPattern(pattern string) bool {
	regex, err := regexp.Compile("(?i)" + pattern)
	if err != nil {
		return false
	}

	return ad.searchDirectories(ad.structure, regex)
}

// searchDirectories recursively searches directories
func (ad *ArchitectureDetector) searchDirectories(dir *DirectoryStructure, regex *regexp.Regexp) bool {
	// Check current directory name
	dirName := filepath.Base(dir.Path)
	if regex.MatchString(dirName) {
		return true
	}

	// Check children
	for _, child := range dir.Children {
		if ad.searchDirectories(child, regex) {
			return true
		}
	}

	return false
}

// checkFilePattern checks if file pattern exists
func (ad *ArchitectureDetector) checkFilePattern(pattern string) []string {
	regex, err := regexp.Compile("(?i)" + pattern)
	if err != nil {
		return nil
	}

	var matches []string

	for _, files := range ad.filePatterns {
		for _, file := range files {
			if regex.MatchString(filepath.Base(file)) {
				matches = append(matches, file)
			}
		}
	}

	return matches
}

// checkContentPattern checks if content pattern exists in files
func (ad *ArchitectureDetector) checkContentPattern(pattern string) []string {
	regex, err := regexp.Compile("(?i)" + pattern)
	if err != nil {
		return nil
	}

	var matches []string

	// Search in Go, Python, JavaScript, and other code files
	searchExtensions := []string{".go", ".py", ".js", ".ts", ".java", ".cpp", ".c", ".rs"}

	for _, ext := range searchExtensions {
		if files, exists := ad.filePatterns[ext]; exists {
			for _, file := range files {
				if ad.searchFileContent(filepath.Join(ad.projectPath, file), regex) {
					matches = append(matches, file)
				}
			}
		}
	}

	return matches
}

// searchFileContent searches for pattern in file content
func (ad *ArchitectureDetector) searchFileContent(filepath string, regex *regexp.Regexp) bool {
	content, err := os.ReadFile(filepath)
	if err != nil {
		return false
	}

	return regex.Match(content)
}

// generateSuggestions generates improvement suggestions
func (ad *ArchitectureDetector) generateSuggestions(pattern ArchitecturePattern, confidence float64) []string {
	suggestions := make([]string, 0)

	switch pattern {
	case PatternMVC:
		if confidence < 0.8 {
			suggestions = append(suggestions, "Consider organizing code into clear Model, View, and Controller directories")
			suggestions = append(suggestions, "Ensure clear separation of concerns between components")
		}
	case PatternCleanArchitecture:
		if confidence < 0.9 {
			suggestions = append(suggestions, "Consider implementing stricter layer separation")
			suggestions = append(suggestions, "Add interfaces to decouple dependencies")
		}
	case PatternMicroservices:
		if confidence < 0.9 {
			suggestions = append(suggestions, "Consider adding service discovery mechanisms")
			suggestions = append(suggestions, "Implement proper API versioning")
			suggestions = append(suggestions, "Add health checks and monitoring")
		}
	case PatternRepository:
		if confidence < 0.9 {
			suggestions = append(suggestions, "Consider using interfaces for repository contracts")
			suggestions = append(suggestions, "Implement proper error handling in repositories")
		}
	}

	return suggestions
}

// sortByConfidence sorts patterns by confidence level
func (ad *ArchitectureDetector) sortByConfidence() {
	// Simple bubble sort by confidence (descending)
	n := len(ad.detectedPatterns)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if ad.detectedPatterns[j].Confidence < ad.detectedPatterns[j+1].Confidence {
				ad.detectedPatterns[j], ad.detectedPatterns[j+1] = ad.detectedPatterns[j+1], ad.detectedPatterns[j]
			}
		}
	}
}

// GetDirectoryStructure returns the analyzed directory structure
func (ad *ArchitectureDetector) GetDirectoryStructure() *DirectoryStructure {
	ad.mu.RLock()
	defer ad.mu.RUnlock()
	return ad.structure
}

// GetPatternDetails returns detailed information about a specific pattern
func (ad *ArchitectureDetector) GetPatternDetails(pattern ArchitecturePattern) (*ArchitectureInfo, bool) {
	ad.mu.RLock()
	defer ad.mu.RUnlock()

	for _, info := range ad.detectedPatterns {
		if info.Pattern == pattern {
			return &info, true
		}
	}

	return nil, false
}

// GenerateArchitectureReport generates a comprehensive architecture report
func (ad *ArchitectureDetector) GenerateArchitectureReport() string {
	ad.mu.RLock()
	defer ad.mu.RUnlock()

	var report strings.Builder

	report.WriteString("🏗️  Architecture Analysis Report\n")
	report.WriteString("================================\n\n")

	if len(ad.detectedPatterns) == 0 {
		report.WriteString("No recognizable architecture patterns detected.\n")
		report.WriteString("Consider implementing established architectural patterns for better maintainability.\n")
		return report.String()
	}

	report.WriteString(fmt.Sprintf("Detected %d architecture pattern(s):\n\n", len(ad.detectedPatterns)))

	for i, pattern := range ad.detectedPatterns {
		report.WriteString(fmt.Sprintf("%d. %s (%.1f%% confidence)\n", i+1, pattern.Name, pattern.Confidence*100))
		report.WriteString(fmt.Sprintf("   %s\n\n", pattern.Description))

		if len(pattern.Evidence) > 0 {
			report.WriteString("   Evidence:\n")
			for _, evidence := range pattern.Evidence {
				report.WriteString(fmt.Sprintf("   • %s\n", evidence))
			}
			report.WriteString("\n")
		}

		if len(pattern.Suggestions) > 0 {
			report.WriteString("   Suggestions:\n")
			for _, suggestion := range pattern.Suggestions {
				report.WriteString(fmt.Sprintf("   → %s\n", suggestion))
			}
			report.WriteString("\n")
		}
	}

	return report.String()
}
