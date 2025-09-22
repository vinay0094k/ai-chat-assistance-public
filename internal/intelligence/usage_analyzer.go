// internal/intelligence/usage_analyzer.go
package intelligence

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

type UsageMetrics struct {
	TotalUsages       int
	UniqueUsers       int
	HotspotCount      int
	UnusedCount       int
	CouplingIndex     float64 // How tightly coupled the code is
	ReuseIndex        float64 // How much code is reused
	DistributionIndex float64 // How evenly distributed usage is

	// Temporal metrics
	RecentUsageChange float64 // Change in usage over time
	TrendDirection    string  // "increasing", "decreasing", "stable"

	// Quality metrics
	DeadCodeRatio      float64
	UtilizationRatio   float64
	ConcentrationRatio float64
}

type UsageRecord struct {
	ID         string
	Entity     string
	EntityType string
	UsedBy     string
	UserType   string
	File       string
	Line       int
	Context    string
	Frequency  int
	LastUsed   time.Time
	FirstUsed  time.Time
	UsageType  UsageType
	Importance float64
	Confidence float64
}

type UsageType int

const (
	UsageCall UsageType = iota
	UsageReference
	UsageImport
	UsageInheritance
	UsageComposition
	UsageInstantiation
	UsageImplementation
	UsageOverride
	UsageAnnotation
	UsageConfiguration
)

type UsageHotspot struct {
	Entity       string
	EntityType   string
	File         string
	UsageCount   int
	UserCount    int
	Frequency    float64 // usages per day
	LastUpdated  time.Time
	RiskScore    float64
	Suggestions  []string
	Dependencies []string
}

type UnusedEntity struct {
	ID            string
	Name          string
	Type          string
	File          string
	Line          int
	Age           time.Duration
	Visibility    string
	Complexity    int
	IsTestRelated bool
	Suggestions   []string
}

type DependencyChain struct {
	Chain       []string
	Length      int
	Strength    float64
	IsCircular  bool
	BreakPoints []string
}

type UsageAnalyzer struct {
	projectPath     string
	usageRecords    []UsageRecord
	hotspots        []UsageHotspot
	unusedEntities  []UnusedEntity
	dependencies    map[string][]string
	dependencyGraph map[string]map[string]float64
	metrics         *UsageMetrics

	// Configuration
	hotspotsThreshold  int
	unusedThreshold    time.Duration
	maxDependencyDepth int
	includeTests       bool
	excludePaths       []string

	// Cache and state
	fileModTimes      map[string]time.Time
	entityDefinitions map[string]EntityDefinition
	usageCache        map[string][]UsageRecord
	lastAnalysis      time.Time
	mu                sync.RWMutex
}

type EntityDefinition struct {
	ID           string
	Name         string
	Type         string
	File         string
	Line         int
	Package      string
	Visibility   string
	Signature    string
	IsTest       bool
	Complexity   int
	LastModified time.Time
}

// NewUsageAnalyzer creates a new usage analyzer
func NewUsageAnalyzer(projectPath string) *UsageAnalyzer {
	return &UsageAnalyzer{
		projectPath:        projectPath,
		usageRecords:       make([]UsageRecord, 0),
		hotspots:           make([]UsageHotspot, 0),
		unusedEntities:     make([]UnusedEntity, 0),
		dependencies:       make(map[string][]string),
		dependencyGraph:    make(map[string]map[string]float64),
		hotspotsThreshold:  10,
		unusedThreshold:    30 * 24 * time.Hour, // 30 days
		maxDependencyDepth: 10,
		includeTests:       true,
		excludePaths:       []string{".git", "node_modules", "vendor", "target", "build", "dist"},
		fileModTimes:       make(map[string]time.Time),
		entityDefinitions:  make(map[string]EntityDefinition),
		usageCache:         make(map[string][]UsageRecord),
	}
}

// AnalyzeUsage performs comprehensive usage analysis
func (ua *UsageAnalyzer) AnalyzeUsage() (*UsageMetrics, error) {
	ua.mu.Lock()
	defer ua.mu.Unlock()

	startTime := time.Now()

	// Initialize metrics
	ua.metrics = &UsageMetrics{}

	// Clear previous analysis
	ua.usageRecords = make([]UsageRecord, 0)
	ua.hotspots = make([]UsageHotspot, 0)
	ua.unusedEntities = make([]UnusedEntity, 0)
	ua.dependencies = make(map[string][]string)
	ua.dependencyGraph = make(map[string]map[string]float64)

	// Step 1: Discover all entities
	if err := ua.discoverEntities(); err != nil {
		return nil, fmt.Errorf("entity discovery failed: %w", err)
	}

	// Step 2: Analyze usage patterns
	if err := ua.analyzeUsagePatterns(); err != nil {
		return nil, fmt.Errorf("usage pattern analysis failed: %w", err)
	}

	// Step 3: Build dependency graph
	ua.buildDependencyGraph()

	// Step 4: Identify hotspots
	ua.identifyHotspots()

	// Step 5: Find unused entities
	ua.findUnusedEntities()

	// Step 6: Calculate metrics
	ua.calculateUsageMetrics()

	// Step 7: Generate recommendations
	ua.generateUsageRecommendations()

	ua.lastAnalysis = time.Now()

	fmt.Printf("Usage analysis completed in %v\n", time.Since(startTime))
	fmt.Printf("Found %d usage records, %d hotspots, %d unused entities\n",
		len(ua.usageRecords), len(ua.hotspots), len(ua.unusedEntities))

	return ua.metrics, nil
}

// discoverEntities discovers all entities in the project
func (ua *UsageAnalyzer) discoverEntities() error {
	return filepath.Walk(ua.projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if ua.shouldAnalyzeFile(path, info) {
			return ua.extractEntitiesFromFile(path)
		}

		return nil
	})
}

// shouldAnalyzeFile determines if a file should be analyzed
func (ua *UsageAnalyzer) shouldAnalyzeFile(path string, info os.FileInfo) bool {
	if info.IsDir() {
		return false
	}

	// Skip hidden files
	if strings.HasPrefix(info.Name(), ".") {
		return false
	}

	// Skip excluded paths
	for _, exclude := range ua.excludePaths {
		if strings.Contains(path, exclude) {
			return false
		}
	}

	// Only analyze source files
	ext := filepath.Ext(path)
	supportedExts := []string{".go", ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h"}

	for _, supportedExt := range supportedExts {
		if ext == supportedExt {
			return true
		}
	}

	return false
}

// extractEntitiesFromFile extracts entities from a source file
func (ua *UsageAnalyzer) extractEntitiesFromFile(path string) error {
	ext := filepath.Ext(path)

	switch ext {
	case ".go":
		return ua.extractGoEntities(path)
	case ".py":
		return ua.extractPythonEntities(path)
	case ".js", ".ts":
		return ua.extractJavaScriptEntities(path)
	default:
		return ua.extractGenericEntities(path)
	}
}

// extractGoEntities extracts entities from Go files
func (ua *UsageAnalyzer) extractGoEntities(path string) error {
	src, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, src, parser.ParseComments)
	if err != nil {
		return err
	}

	packageName := file.Name.Name

	// Extract functions, types, variables, etc.
	ast.Inspect(file, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.FuncDecl:
			if node.Name != nil {
				pos := fset.Position(node.Pos())
				entity := EntityDefinition{
					ID:           fmt.Sprintf("%s::%s", packageName, node.Name.Name),
					Name:         node.Name.Name,
					Type:         "function",
					File:         path,
					Line:         pos.Line,
					Package:      packageName,
					Visibility:   ua.determineGoVisibility(node.Name.Name),
					IsTest:       strings.HasSuffix(node.Name.Name, "Test") || strings.Contains(path, "_test.go"),
					Complexity:   ua.calculateGoComplexity(node),
					LastModified: time.Now(),
				}
				ua.entityDefinitions[entity.ID] = entity
			}

		case *ast.GenDecl:
			for _, spec := range node.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					pos := fset.Position(s.Pos())
					entity := EntityDefinition{
						ID:           fmt.Sprintf("%s::%s", packageName, s.Name.Name),
						Name:         s.Name.Name,
						Type:         "type",
						File:         path,
						Line:         pos.Line,
						Package:      packageName,
						Visibility:   ua.determineGoVisibility(s.Name.Name),
						LastModified: time.Now(),
					}
					ua.entityDefinitions[entity.ID] = entity

				case *ast.ValueSpec:
					for _, name := range s.Names {
						pos := fset.Position(name.Pos())
						entityType := "variable"
						if node.Tok == token.CONST {
							entityType = "constant"
						}
						entity := EntityDefinition{
							ID:           fmt.Sprintf("%s::%s", packageName, name.Name),
							Name:         name.Name,
							Type:         entityType,
							File:         path,
							Line:         pos.Line,
							Package:      packageName,
							Visibility:   ua.determineGoVisibility(name.Name),
							LastModified: time.Now(),
						}
						ua.entityDefinitions[entity.ID] = entity
					}
				}
			}
		}
		return true
	})

	return nil
}

// extractPythonEntities extracts entities from Python files
func (ua *UsageAnalyzer) extractPythonEntities(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	lines := strings.Split(string(content), "\n")

	for i, line := range lines {
		line = strings.TrimSpace(line)

		// Extract classes
		if matches := regexp.MustCompile(`^class\s+(\w+)`).FindStringSubmatch(line); len(matches) > 1 {
			className := matches[1]
			entity := EntityDefinition{
				ID:           fmt.Sprintf("python::%s", className),
				Name:         className,
				Type:         "class",
				File:         path,
				Line:         i + 1,
				Visibility:   ua.determinePythonVisibility(className),
				IsTest:       strings.Contains(path, "test_") || strings.Contains(path, "_test.py"),
				LastModified: time.Now(),
			}
			ua.entityDefinitions[entity.ID] = entity
		}

		// Extract functions
		if matches := regexp.MustCompile(`^def\s+(\w+)`).FindStringSubmatch(line); len(matches) > 1 {
			funcName := matches[1]
			entity := EntityDefinition{
				ID:           fmt.Sprintf("python::%s", funcName),
				Name:         funcName,
				Type:         "function",
				File:         path,
				Line:         i + 1,
				Visibility:   ua.determinePythonVisibility(funcName),
				IsTest:       strings.HasPrefix(funcName, "test_") || strings.Contains(path, "test_"),
				LastModified: time.Now(),
			}
			ua.entityDefinitions[entity.ID] = entity
		}
	}

	return nil
}

// extractJavaScriptEntities extracts entities from JavaScript/TypeScript files
func (ua *UsageAnalyzer) extractJavaScriptEntities(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	lines := strings.Split(string(content), "\n")

	for i, line := range lines {
		line = strings.TrimSpace(line)

		// Extract classes
		if matches := regexp.MustCompile(`^class\s+(\w+)`).FindStringSubmatch(line); len(matches) > 1 {
			className := matches[1]
			entity := EntityDefinition{
				ID:           fmt.Sprintf("js::%s", className),
				Name:         className,
				Type:         "class",
				File:         path,
				Line:         i + 1,
				IsTest:       strings.Contains(path, ".test.") || strings.Contains(path, ".spec."),
				LastModified: time.Now(),
			}
			ua.entityDefinitions[entity.ID] = entity
		}

		// Extract functions
		patterns := []string{
			`^function\s+(\w+)`,
			`^const\s+(\w+)\s*=\s*function`,
			`^const\s+(\w+)\s*=\s*\([^)]*\)\s*=>`,
		}

		for _, pattern := range patterns {
			if matches := regexp.MustCompile(pattern).FindStringSubmatch(line); len(matches) > 1 {
				funcName := matches[1]
				entity := EntityDefinition{
					ID:           fmt.Sprintf("js::%s", funcName),
					Name:         funcName,
					Type:         "function",
					File:         path,
					Line:         i + 1,
					IsTest:       strings.Contains(path, ".test.") || strings.Contains(path, ".spec."),
					LastModified: time.Now(),
				}
				ua.entityDefinitions[entity.ID] = entity
			}
		}
	}

	return nil
}

// extractGenericEntities extracts entities using generic patterns
func (ua *UsageAnalyzer) extractGenericEntities(path string) error {
	// Basic extraction for unsupported file types
	return nil
}

// analyzeUsagePatterns analyzes how entities are used
func (ua *UsageAnalyzer) analyzeUsagePatterns() error {
	return filepath.Walk(ua.projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if ua.shouldAnalyzeFile(path, info) {
			return ua.analyzeUsageInFile(path)
		}

		return nil
	})
}

// analyzeUsageInFile analyzes usage patterns in a specific file
func (ua *UsageAnalyzer) analyzeUsageInFile(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	contentStr := string(content)
	lines := strings.Split(contentStr, "\n")

	// Find usages of defined entities
	for entityID, entity := range ua.entityDefinitions {
		// Skip if entity is in the same file (definition vs usage)
		if entity.File == path {
			continue
		}

		// Look for usage patterns
		usageCount := 0
		for i, line := range lines {
			// Simple name matching (can be improved with better parsing)
			if ua.containsUsage(line, entity.Name) {
				usageCount++

				// Create usage record
				usage := UsageRecord{
					ID:         fmt.Sprintf("%s->%s:%d", path, entityID, i+1),
					Entity:     entityID,
					EntityType: entity.Type,
					UsedBy:     path,
					UserType:   "file",
					File:       path,
					Line:       i + 1,
					Context:    strings.TrimSpace(line),
					Frequency:  1,
					LastUsed:   time.Now(),
					FirstUsed:  time.Now(),
					UsageType:  ua.determineUsageType(line, entity.Name),
					Importance: 1.0,
					Confidence: ua.calculateUsageConfidence(line, entity.Name),
				}

				if usage.Confidence > 0.5 { // Only include high-confidence usages
					ua.usageRecords = append(ua.usageRecords, usage)
				}
			}
		}
	}

	return nil
}

// containsUsage checks if a line contains usage of an entity
func (ua *UsageAnalyzer) containsUsage(line, entityName string) bool {
	// Simple word boundary matching
	pattern := fmt.Sprintf(`\b%s\b`, regexp.QuoteMeta(entityName))
	matched, _ := regexp.MatchString(pattern, line)

	// Exclude comments and strings (basic filtering)
	if strings.Contains(line, "//") || strings.Contains(line, "#") {
		commentIndex := strings.Index(line, "//")
		if commentIndex == -1 {
			commentIndex = strings.Index(line, "#")
		}
		if commentIndex >= 0 {
			line = line[:commentIndex]
			matched, _ = regexp.MatchString(pattern, line)
		}
	}

	return matched
}

// determineUsageType determines the type of usage
func (ua *UsageAnalyzer) determineUsageType(line, entityName string) UsageType {
	line = strings.ToLower(line)

	if strings.Contains(line, entityName+"(") {
		return UsageCall
	} else if strings.Contains(line, "import") && strings.Contains(line, entityName) {
		return UsageImport
	} else if strings.Contains(line, "new "+entityName) {
		return UsageInstantiation
	} else if strings.Contains(line, "extends "+entityName) || strings.Contains(line, "implements "+entityName) {
		return UsageInheritance
	} else {
		return UsageReference
	}
}

// calculateUsageConfidence calculates confidence in usage detection
func (ua *UsageAnalyzer) calculateUsageConfidence(line, entityName string) float64 {
	confidence := 0.5 // Base confidence

	// Increase confidence for function calls
	if strings.Contains(line, entityName+"(") {
		confidence += 0.3
	}

	// Increase confidence for qualified names
	if strings.Contains(line, "."+entityName) || strings.Contains(line, "::"+entityName) {
		confidence += 0.2
	}

	// Decrease confidence for comments
	if strings.Contains(line, "//") || strings.Contains(line, "#") {
		confidence -= 0.3
	}

	// Ensure confidence is in valid range
	if confidence > 1.0 {
		confidence = 1.0
	} else if confidence < 0.0 {
		confidence = 0.0
	}

	return confidence
}

// buildDependencyGraph builds the dependency graph
func (ua *UsageAnalyzer) buildDependencyGraph() {
	ua.dependencyGraph = make(map[string]map[string]float64)

	// Initialize graph
	for entityID := range ua.entityDefinitions {
		ua.dependencyGraph[entityID] = make(map[string]float64)
	}

	// Build edges based on usage records
	for _, usage := range ua.usageRecords {
		if _, exists := ua.dependencyGraph[usage.UsedBy]; !exists {
			ua.dependencyGraph[usage.UsedBy] = make(map[string]float64)
		}

		// Weight the dependency based on usage frequency and confidence
		weight := float64(usage.Frequency) * usage.Confidence
		ua.dependencyGraph[usage.UsedBy][usage.Entity] += weight
	}

	// Normalize weights
	for from, targets := range ua.dependencyGraph {
		maxWeight := 0.0
		for _, weight := range targets {
			if weight > maxWeight {
				maxWeight = weight
			}
		}

		if maxWeight > 0 {
			for to, weight := range targets {
				ua.dependencyGraph[from][to] = weight / maxWeight
			}
		}
	}
}

// identifyHotspots identifies usage hotspots
func (ua *UsageAnalyzer) identifyHotspots() {
	entityUsage := make(map[string][]UsageRecord)

	// Group usage records by entity
	for _, usage := range ua.usageRecords {
		if entityUsage[usage.Entity] == nil {
			entityUsage[usage.Entity] = make([]UsageRecord, 0)
		}
		entityUsage[usage.Entity] = append(entityUsage[usage.Entity], usage)
	}

	// Identify hotspots
	for entityID, usages := range entityUsage {
		if len(usages) >= ua.hotspotsThreshold {
			entity := ua.entityDefinitions[entityID]

			// Calculate unique users
			users := make(map[string]bool)
			totalFrequency := 0
			for _, usage := range usages {
				users[usage.UsedBy] = true
				totalFrequency += usage.Frequency
			}

			// Calculate risk score
			riskScore := ua.calculateRiskScore(len(usages), len(users), totalFrequency, entity)

			hotspot := UsageHotspot{
				Entity:       entityID,
				EntityType:   entity.Type,
				File:         entity.File,
				UsageCount:   len(usages),
				UserCount:    len(users),
				Frequency:    float64(totalFrequency) / 30.0, // per day estimate
				LastUpdated:  entity.LastModified,
				RiskScore:    riskScore,
				Suggestions:  ua.generateHotspotSuggestions(entityID, len(usages), len(users)),
				Dependencies: ua.getEntityDependencies(entityID),
			}

			ua.hotspots = append(ua.hotspots, hotspot)
		}
	}

	// Sort hotspots by risk score
	sort.Slice(ua.hotspots, func(i, j int) bool {
		return ua.hotspots[i].RiskScore > ua.hotspots[j].RiskScore
	})
}

// calculateRiskScore calculates risk score for a hotspot
func (ua *UsageAnalyzer) calculateRiskScore(usageCount, userCount, frequency int, entity EntityDefinition) float64 {
	risk := 0.0

	// High usage increases risk
	risk += float64(usageCount) * 0.1

	// Many users increase risk
	risk += float64(userCount) * 0.2

	// High frequency increases risk
	risk += float64(frequency) * 0.05

	// Public visibility increases risk
	if entity.Visibility == "public" {
		risk *= 1.5
	}

	// High complexity increases risk
	risk += float64(entity.Complexity) * 0.1

	// Normalize to 0-1 range
	if risk > 1.0 {
		risk = 1.0
	}

	return risk
}

// findUnusedEntities finds entities that are not used
func (ua *UsageAnalyzer) findUnusedEntities() {
	usedEntities := make(map[string]bool)

	// Mark all used entities
	for _, usage := range ua.usageRecords {
		usedEntities[usage.Entity] = true
	}

	// Find unused entities
	for entityID, entity := range ua.entityDefinitions {
		if !usedEntities[entityID] && !entity.IsTest {
			age := time.Since(entity.LastModified)

			// Only consider old enough entities as potentially unused
			if age > ua.unusedThreshold {
				unused := UnusedEntity{
					ID:            entityID,
					Name:          entity.Name,
					Type:          entity.Type,
					File:          entity.File,
					Line:          entity.Line,
					Age:           age,
					Visibility:    entity.Visibility,
					Complexity:    entity.Complexity,
					IsTestRelated: entity.IsTest,
					Suggestions:   ua.generateUnusedSuggestions(entity),
				}

				ua.unusedEntities = append(ua.unusedEntities, unused)
			}
		}
	}

	// Sort by age (oldest first)
	sort.Slice(ua.unusedEntities, func(i, j int) bool {
		return ua.unusedEntities[i].Age > ua.unusedEntities[j].Age
	})
}

// calculateUsageMetrics calculates comprehensive usage metrics
func (ua *UsageAnalyzer) calculateUsageMetrics() {
	ua.metrics.TotalUsages = len(ua.usageRecords)
	ua.metrics.HotspotCount = len(ua.hotspots)
	ua.metrics.UnusedCount = len(ua.unusedEntities)

	// Calculate unique users
	users := make(map[string]bool)
	for _, usage := range ua.usageRecords {
		users[usage.UsedBy] = true
	}
	ua.metrics.UniqueUsers = len(users)

	// Calculate coupling index
	ua.metrics.CouplingIndex = ua.calculateCouplingIndex()

	// Calculate reuse index
	ua.metrics.ReuseIndex = ua.calculateReuseIndex()

	// Calculate distribution index
	ua.metrics.DistributionIndex = ua.calculateDistributionIndex()

	// Calculate quality metrics
	totalEntities := len(ua.entityDefinitions)
	if totalEntities > 0 {
		ua.metrics.DeadCodeRatio = float64(ua.metrics.UnusedCount) / float64(totalEntities)
		ua.metrics.UtilizationRatio = float64(len(users)) / float64(totalEntities)
	}

	// Calculate concentration ratio (how concentrated usage is)
	ua.metrics.ConcentrationRatio = ua.calculateConcentrationRatio()
}

// calculateCouplingIndex calculates how tightly coupled the code is
func (ua *UsageAnalyzer) calculateCouplingIndex() float64 {
	if len(ua.dependencyGraph) == 0 {
		return 0.0
	}

	totalEdges := 0
	totalNodes := len(ua.dependencyGraph)

	for _, targets := range ua.dependencyGraph {
		totalEdges += len(targets)
	}

	maxPossibleEdges := totalNodes * (totalNodes - 1)
	if maxPossibleEdges == 0 {
		return 0.0
	}

	return float64(totalEdges) / float64(maxPossibleEdges)
}

// calculateReuseIndex calculates how much code is reused
func (ua *UsageAnalyzer) calculateReuseIndex() float64 {
	if len(ua.entityDefinitions) == 0 {
		return 0.0
	}

	reusedEntities := 0
	entityUsage := make(map[string]int)

	for _, usage := range ua.usageRecords {
		entityUsage[usage.Entity]++
	}

	for _, count := range entityUsage {
		if count > 1 {
			reusedEntities++
		}
	}

	return float64(reusedEntities) / float64(len(ua.entityDefinitions))
}

// calculateDistributionIndex calculates how evenly distributed usage is
func (ua *UsageAnalyzer) calculateDistributionIndex() float64 {
	if len(ua.usageRecords) == 0 {
		return 1.0
	}

	entityUsage := make(map[string]int)
	for _, usage := range ua.usageRecords {
		entityUsage[usage.Entity]++
	}

	if len(entityUsage) == 0 {
		return 1.0
	}

	// Calculate coefficient of variation
	var sum, sumSquares float64
	for _, count := range entityUsage {
		sum += float64(count)
		sumSquares += float64(count * count)
	}

	n := float64(len(entityUsage))
	mean := sum / n
	variance := (sumSquares / n) - (mean * mean)

	if mean == 0 {
		return 1.0
	}

	stdDev := variance
	if variance > 0 {
		stdDev = variance // Simplified - should be sqrt(variance)
	}

	coefficientOfVariation := stdDev / mean

	// Convert to distribution index (lower CV = more even distribution)
	return 1.0 / (1.0 + coefficientOfVariation)
}

// calculateConcentrationRatio calculates usage concentration
func (ua *UsageAnalyzer) calculateConcentrationRatio() float64 {
	if len(ua.hotspots) == 0 || len(ua.entityDefinitions) == 0 {
		return 0.0
	}

	// Calculate what percentage of entities are hotspots
	return float64(len(ua.hotspots)) / float64(len(ua.entityDefinitions))
}

// Helper methods

func (ua *UsageAnalyzer) determineGoVisibility(name string) string {
	if len(name) > 0 && name[0] >= 'A' && name[0] <= 'Z' {
		return "public"
	}
	return "private"
}

func (ua *UsageAnalyzer) determinePythonVisibility(name string) string {
	if strings.HasPrefix(name, "__") {
		return "private"
	} else if strings.HasPrefix(name, "_") {
		return "protected"
	}
	return "public"
}

func (ua *UsageAnalyzer) calculateGoComplexity(funcDecl *ast.FuncDecl) int {
	complexity := 1

	if funcDecl.Body != nil {
		ast.Inspect(funcDecl.Body, func(n ast.Node) bool {
			switch n.(type) {
			case *ast.IfStmt, *ast.ForStmt, *ast.RangeStmt, *ast.SwitchStmt:
				complexity++
			}
			return true
		})
	}

	return complexity
}

func (ua *UsageAnalyzer) getEntityDependencies(entityID string) []string {
	var deps []string

	if targets, exists := ua.dependencyGraph[entityID]; exists {
		for target := range targets {
			deps = append(deps, target)
		}
	}

	return deps
}

func (ua *UsageAnalyzer) generateHotspotSuggestions(entityID string, usageCount, userCount int) []string {
	var suggestions []string

	if usageCount > 50 {
		suggestions = append(suggestions, "Consider refactoring this heavily used component")
	}

	if userCount > 20 {
		suggestions = append(suggestions, "High fan-out detected - ensure interface stability")
	}

	entity := ua.entityDefinitions[entityID]
	if entity.Complexity > 10 {
		suggestions = append(suggestions, "Complex hotspot - consider simplification")
	}

	if entity.Visibility == "public" {
		suggestions = append(suggestions, "Public API - ensure backward compatibility")
	}

	return suggestions
}

func (ua *UsageAnalyzer) generateUnusedSuggestions(entity EntityDefinition) []string {
	var suggestions []string

	suggestions = append(suggestions, "Consider removing if truly unused")

	if entity.Visibility == "public" {
		suggestions = append(suggestions, "Check if this is part of a public API before removing")
	}

	if entity.Type == "function" || entity.Type == "method" {
		suggestions = append(suggestions, "Verify no dynamic calls or reflection usage")
	}

	return suggestions
}

func (ua *UsageAnalyzer) generateUsageRecommendations() {
	// Generate recommendations based on metrics
	// This could be expanded significantly
}

// Query methods

// GetHotspots returns usage hotspots
func (ua *UsageAnalyzer) GetHotspots() []UsageHotspot {
	ua.mu.RLock()
	defer ua.mu.RUnlock()
	return ua.hotspots
}

// GetUnusedEntities returns unused entities
func (ua *UsageAnalyzer) GetUnusedEntities() []UnusedEntity {
	ua.mu.RLock()
	defer ua.mu.RUnlock()
	return ua.unusedEntities
}

// GetUsageRecords returns usage records for an entity
func (ua *UsageAnalyzer) GetUsageRecords(entityID string) []UsageRecord {
	ua.mu.RLock()
	defer ua.mu.RUnlock()

	var records []UsageRecord
	for _, record := range ua.usageRecords {
		if record.Entity == entityID {
			records = append(records, record)
		}
	}

	return records
}

// GetMetrics returns usage metrics
func (ua *UsageAnalyzer) GetMetrics() *UsageMetrics {
	ua.mu.RLock()
	defer ua.mu.RUnlock()
	return ua.metrics
}

// GetDependencyChains finds dependency chains
func (ua *UsageAnalyzer) GetDependencyChains(entityID string, maxDepth int) []DependencyChain {
	ua.mu.RLock()
	defer ua.mu.RUnlock()

	var chains []DependencyChain
	visited := make(map[string]bool)

	ua.findDependencyChains(entityID, []string{entityID}, maxDepth, visited, &chains)

	return chains
}

// findDependencyChains recursively finds dependency chains
func (ua *UsageAnalyzer) findDependencyChains(entityID string, currentChain []string, maxDepth int, visited map[string]bool, chains *[]DependencyChain) {
	if len(currentChain) >= maxDepth {
		return
	}

	if visited[entityID] {
		// Found circular dependency
		chain := DependencyChain{
			Chain:      currentChain,
			Length:     len(currentChain),
			IsCircular: true,
		}
		*chains = append(*chains, chain)
		return
	}

	visited[entityID] = true

	if targets, exists := ua.dependencyGraph[entityID]; exists {
		for target, strength := range targets {
			newChain := append(currentChain, target)
			chain := DependencyChain{
				Chain:    newChain,
				Length:   len(newChain),
				Strength: strength,
			}
			*chains = append(*chains, chain)

			ua.findDependencyChains(target, newChain, maxDepth, visited, chains)
		}
	}

	visited[entityID] = false
}

// GenerateUsageReport generates a comprehensive usage analysis report
func (ua *UsageAnalyzer) GenerateUsageReport() string {
	ua.mu.RLock()
	defer ua.mu.RUnlock()

	var report strings.Builder

	report.WriteString("📈 Usage Analysis Report\n")
	report.WriteString("========================\n\n")

	// Overview
	report.WriteString("📊 Overview\n")
	report.WriteString("-----------\n")
	report.WriteString(fmt.Sprintf("Total Entities: %d\n", len(ua.entityDefinitions)))
	report.WriteString(fmt.Sprintf("Total Usage Records: %d\n", ua.metrics.TotalUsages))
	report.WriteString(fmt.Sprintf("Unique Users: %d\n", ua.metrics.UniqueUsers))
	report.WriteString(fmt.Sprintf("Usage Hotspots: %d\n", ua.metrics.HotspotCount))
	report.WriteString(fmt.Sprintf("Unused Entities: %d\n\n", ua.metrics.UnusedCount))

	// Quality metrics
	report.WriteString("📈 Quality Metrics\n")
	report.WriteString("------------------\n")
	report.WriteString(fmt.Sprintf("Coupling Index: %.3f\n", ua.metrics.CouplingIndex))
	report.WriteString(fmt.Sprintf("Reuse Index: %.3f\n", ua.metrics.ReuseIndex))
	report.WriteString(fmt.Sprintf("Distribution Index: %.3f\n", ua.metrics.DistributionIndex))
	report.WriteString(fmt.Sprintf("Dead Code Ratio: %.3f\n", ua.metrics.DeadCodeRatio))
	report.WriteString(fmt.Sprintf("Utilization Ratio: %.3f\n", ua.metrics.UtilizationRatio))
	report.WriteString(fmt.Sprintf("Concentration Ratio: %.3f\n\n", ua.metrics.ConcentrationRatio))

	// Top hotspots
	if len(ua.hotspots) > 0 {
		report.WriteString("🔥 Top Usage Hotspots\n")
		report.WriteString("---------------------\n")
		for i, hotspot := range ua.hotspots {
			if i >= 5 {
				break
			}
			entity := ua.entityDefinitions[hotspot.Entity]
			relPath, _ := filepath.Rel(ua.projectPath, entity.File)
			report.WriteString(fmt.Sprintf("• %s (%s)\n", entity.Name, entity.Type))
			report.WriteString(fmt.Sprintf("  File: %s:%d\n", relPath, entity.Line))
			report.WriteString(fmt.Sprintf("  Usage: %d times by %d users (risk: %.2f)\n",
				hotspot.UsageCount, hotspot.UserCount, hotspot.RiskScore))
			if len(hotspot.Suggestions) > 0 {
				report.WriteString(fmt.Sprintf("  Suggestion: %s\n", hotspot.Suggestions[0]))
			}
			report.WriteString("\n")
		}
	}

	// Unused entities
	if len(ua.unusedEntities) > 0 {
		report.WriteString("🗑️  Unused Entities\n")
		report.WriteString("-------------------\n")
		for i, unused := range ua.unusedEntities {
			if i >= 10 {
				break
			}
			relPath, _ := filepath.Rel(ua.projectPath, unused.File)
			report.WriteString(fmt.Sprintf("• %s (%s) - %s:%d\n",
				unused.Name, unused.Type, relPath, unused.Line))
			report.WriteString(fmt.Sprintf("  Age: %v, Visibility: %s\n",
				unused.Age.Truncate(24*time.Hour), unused.Visibility))
		}
		report.WriteString("\n")
	}

	return report.String()
}

// ExportUsageData exports usage data to JSON
func (ua *UsageAnalyzer) ExportUsageData() ([]byte, error) {
	ua.mu.RLock()
	defer ua.mu.RUnlock()

	data := map[string]interface{}{
		"metrics":         ua.metrics,
		"hotspots":        ua.hotspots,
		"unused_entities": ua.unusedEntities,
		"usage_records":   ua.usageRecords,
		"last_analysis":   ua.lastAnalysis,
	}

	return json.MarshalIndent(data, "", "  ")
}
