// internal/intelligence/semantic_analyzer.go
package intelligence

import (
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

type SemanticEntity struct {
	ID            string
	Name          string
	Type          EntityType
	Scope         string
	File          string
	Line          int
	Column        int
	Package       string
	Signature     string
	Documentation string
	Visibility    Visibility
	Language      string

	// Relationships
	Uses        []string
	UsedBy      []string
	Calls       []string
	CalledBy    []string
	Implements  []string
	Extends     []string
	Contains    []string
	ContainedBy string

	// Semantic properties
	IsAbstract   bool
	IsInterface  bool
	IsGeneric    bool
	IsStatic     bool
	IsAsync      bool
	IsRecursive  bool
	IsDeprecated bool

	// Metrics
	Complexity   int
	Importance   float64
	ChangeRisk   float64
	LastModified time.Time
}

type EntityType int

const (
	EntityPackage EntityType = iota
	EntityClass
	EntityInterface
	EntityStruct
	EntityFunction
	EntityMethod
	EntityVariable
	EntityConstant
	EntityField
	EntityParameter
	EntityType
	EntityEnum
	EntityTrait
	EntityModule
	EntityNamespace
)

type Visibility int

const (
	VisibilityPublic Visibility = iota
	VisibilityPrivate
	VisibilityProtected
	VisibilityInternal
	VisibilityPackage
)

type SemanticRelationship struct {
	From         string
	To           string
	Type         RelationshipType
	Strength     float64
	Context      string
	File         string
	Line         int
	IsTransitive bool
}

type RelationshipType int

const (
	RelationshipUses RelationshipType = iota
	RelationshipCalls
	RelationshipImplements
	RelationshipExtends
	RelationshipContains
	RelationshipDependsOn
	RelationshipSimilarTo
	RelationshipOverrides
	RelationshipInstantiates
	RelationshipReferences
)

type SemanticGraph struct {
	Entities      map[string]*SemanticEntity
	Relationships []SemanticRelationship
	Clusters      []EntityCluster
	mu            sync.RWMutex
}

type EntityCluster struct {
	ID          string
	Name        string
	Entities    []string
	Cohesion    float64
	Purpose     string
	Suggestions []string
}

type SemanticAnalyzer struct {
	projectPath string
	graph       *SemanticGraph
	fileSet     *token.FileSet
	packages    map[string]*ast.Package

	// Language analyzers
	goAnalyzer     *GoSemanticAnalyzer
	pythonAnalyzer *PythonSemanticAnalyzer
	jsAnalyzer     *JSSemanticAnalyzer

	// Configuration
	maxDepth     int
	includeTests bool
	excludePaths []string

	// Analysis state
	analyzed     map[string]time.Time
	lastAnalysis time.Time
	mu           sync.RWMutex
}

// Language-specific analyzers
type GoSemanticAnalyzer struct {
	fileSet  *token.FileSet
	packages map[string]*ast.Package
}

type PythonSemanticAnalyzer struct {
	// Python-specific analysis state
}

type JSSemanticAnalyzer struct {
	// JavaScript-specific analysis state
}

// NewSemanticAnalyzer creates a new semantic analyzer
func NewSemanticAnalyzer(projectPath string) *SemanticAnalyzer {
	sa := &SemanticAnalyzer{
		projectPath:  projectPath,
		fileSet:      token.NewFileSet(),
		packages:     make(map[string]*ast.Package),
		maxDepth:     10,
		includeTests: true,
		excludePaths: []string{".git", "node_modules", "vendor", "target", "build", "dist"},
		analyzed:     make(map[string]time.Time),
	}

	sa.graph = &SemanticGraph{
		Entities:      make(map[string]*SemanticEntity),
		Relationships: make([]SemanticRelationship, 0),
		Clusters:      make([]EntityCluster, 0),
	}

	// Initialize language-specific analyzers
	sa.goAnalyzer = &GoSemanticAnalyzer{
		fileSet:  sa.fileSet,
		packages: sa.packages,
	}
	sa.pythonAnalyzer = &PythonSemanticAnalyzer{}
	sa.jsAnalyzer = &JSSemanticAnalyzer{}

	return sa
}

// AnalyzeSemantics performs comprehensive semantic analysis
func (sa *SemanticAnalyzer) AnalyzeSemantics() (*SemanticGraph, error) {
	sa.mu.Lock()
	defer sa.mu.Unlock()

	startTime := time.Now()

	// Clear previous analysis
	sa.graph.Entities = make(map[string]*SemanticEntity)
	sa.graph.Relationships = make([]SemanticRelationship, 0)
	sa.graph.Clusters = make([]EntityCluster, 0)

	// Analyze all source files
	err := filepath.Walk(sa.projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if sa.shouldAnalyzeFile(path, info) {
			return sa.analyzeFile(path, info)
		}

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("semantic analysis failed: %w", err)
	}

	// Build relationships
	sa.buildRelationships()

	// Calculate entity importance
	sa.calculateImportance()

	// Identify clusters
	sa.identifyClusters()

	// Calculate complexity metrics
	sa.calculateComplexityMetrics()

	sa.lastAnalysis = time.Now()

	fmt.Printf("Semantic analysis completed in %v\n", time.Since(startTime))
	fmt.Printf("Analyzed %d entities and %d relationships\n", len(sa.graph.Entities), len(sa.graph.Relationships))

	return sa.graph, nil
}

// shouldAnalyzeFile determines if a file should be analyzed
func (sa *SemanticAnalyzer) shouldAnalyzeFile(path string, info os.FileInfo) bool {
	if info.IsDir() {
		return false
	}

	// Skip hidden files
	if strings.HasPrefix(info.Name(), ".") {
		return false
	}

	// Skip excluded paths
	for _, exclude := range sa.excludePaths {
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

// analyzeFile analyzes a single file
func (sa *SemanticAnalyzer) analyzeFile(path string, info os.FileInfo) error {
	ext := filepath.Ext(path)

	switch ext {
	case ".go":
		return sa.analyzeGoFile(path)
	case ".py":
		return sa.analyzePythonFile(path)
	case ".js", ".ts":
		return sa.analyzeJavaScriptFile(path)
	case ".java":
		return sa.analyzeJavaFile(path)
	default:
		return sa.analyzeGenericFile(path)
	}
}

// analyzeGoFile analyzes a Go source file
func (sa *SemanticAnalyzer) analyzeGoFile(path string) error {
	src, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	// Parse the Go file
	file, err := parser.ParseFile(sa.fileSet, path, src, parser.ParseComments)
	if err != nil {
		return fmt.Errorf("failed to parse Go file %s: %w", path, err)
	}

	// Extract package information
	packageName := file.Name.Name
	packageEntity := sa.createPackageEntity(packageName, path)
	sa.graph.Entities[packageEntity.ID] = packageEntity

	// Walk the AST and extract entities
	ast.Inspect(file, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.FuncDecl:
			sa.analyzeFunctionDecl(node, path, packageName)
		case *ast.GenDecl:
			sa.analyzeGenDecl(node, path, packageName)
		case *ast.TypeSpec:
			sa.analyzeTypeSpec(node, path, packageName)
		}
		return true
	})

	return nil
}

// analyzeFunctionDecl analyzes a function declaration
func (sa *SemanticAnalyzer) analyzeFunctionDecl(funcDecl *ast.FuncDecl, file, packageName string) {
	if funcDecl.Name == nil {
		return
	}

	pos := sa.fileSet.Position(funcDecl.Pos())
	funcName := funcDecl.Name.Name

	// Determine if it's a method or function
	var entityType EntityType
	var containedBy string

	if funcDecl.Recv != nil && len(funcDecl.Recv.List) > 0 {
		entityType = EntityMethod
		// Extract receiver type
		if funcDecl.Recv.List[0].Type != nil {
			if ident, ok := funcDecl.Recv.List[0].Type.(*ast.Ident); ok {
				containedBy = ident.Name
			}
		}
	} else {
		entityType = EntityFunction
	}

	entity := &SemanticEntity{
		ID:            fmt.Sprintf("%s::%s", packageName, funcName),
		Name:          funcName,
		Type:          entityType,
		Scope:         packageName,
		File:          file,
		Line:          pos.Line,
		Column:        pos.Column,
		Package:       packageName,
		Signature:     sa.extractFunctionSignature(funcDecl),
		Documentation: sa.extractDocumentation(funcDecl.Doc),
		Visibility:    sa.determineVisibility(funcName),
		Language:      "Go",
		ContainedBy:   containedBy,
		Uses:          make([]string, 0),
		UsedBy:        make([]string, 0),
		Calls:         make([]string, 0),
		CalledBy:      make([]string, 0),
		LastModified:  time.Now(),
	}

	// Analyze function body for calls and usage
	if funcDecl.Body != nil {
		ast.Inspect(funcDecl.Body, func(n ast.Node) bool {
			if callExpr, ok := n.(*ast.CallExpr); ok {
				if ident, ok := callExpr.Fun.(*ast.Ident); ok {
					calledFunc := ident.Name
					entity.Calls = append(entity.Calls, calledFunc)
				}
			}
			return true
		})
	}

	// Calculate complexity
	entity.Complexity = sa.calculateFunctionComplexity(funcDecl)

	sa.graph.Entities[entity.ID] = entity
}

// analyzeGenDecl analyzes general declarations (variables, constants, types)
func (sa *SemanticAnalyzer) analyzeGenDecl(genDecl *ast.GenDecl, file, packageName string) {
	for _, spec := range genDecl.Specs {
		switch s := spec.(type) {
		case *ast.ValueSpec:
			sa.analyzeValueSpec(s, genDecl, file, packageName)
		case *ast.TypeSpec:
			sa.analyzeTypeSpec(s, file, packageName)
		}
	}
}

// analyzeValueSpec analyzes variable/constant specifications
func (sa *SemanticAnalyzer) analyzeValueSpec(valueSpec *ast.ValueSpec, genDecl *ast.GenDecl, file, packageName string) {
	pos := sa.fileSet.Position(valueSpec.Pos())

	var entityType EntityType
	if genDecl.Tok == token.CONST {
		entityType = EntityConstant
	} else {
		entityType = EntityVariable
	}

	for _, name := range valueSpec.Names {
		entity := &SemanticEntity{
			ID:            fmt.Sprintf("%s::%s", packageName, name.Name),
			Name:          name.Name,
			Type:          entityType,
			Scope:         packageName,
			File:          file,
			Line:          pos.Line,
			Column:        pos.Column,
			Package:       packageName,
			Documentation: sa.extractDocumentation(genDecl.Doc),
			Visibility:    sa.determineVisibility(name.Name),
			Language:      "Go",
			Uses:          make([]string, 0),
			UsedBy:        make([]string, 0),
			LastModified:  time.Now(),
		}

		sa.graph.Entities[entity.ID] = entity
	}
}

// analyzeTypeSpec analyzes type specifications
func (sa *SemanticAnalyzer) analyzeTypeSpec(typeSpec *ast.TypeSpec, file, packageName string) {
	pos := sa.fileSet.Position(typeSpec.Pos())
	typeName := typeSpec.Name.Name

	var entityType EntityType
	var isInterface bool

	switch typeSpec.Type.(type) {
	case *ast.StructType:
		entityType = EntityStruct
	case *ast.InterfaceType:
		entityType = EntityInterface
		isInterface = true
	default:
		entityType = EntityType
	}

	entity := &SemanticEntity{
		ID:           fmt.Sprintf("%s::%s", packageName, typeName),
		Name:         typeName,
		Type:         entityType,
		Scope:        packageName,
		File:         file,
		Line:         pos.Line,
		Column:       pos.Column,
		Package:      packageName,
		Visibility:   sa.determineVisibility(typeName),
		Language:     "Go",
		IsInterface:  isInterface,
		Uses:         make([]string, 0),
		UsedBy:       make([]string, 0),
		Contains:     make([]string, 0),
		LastModified: time.Now(),
	}

	// Analyze struct fields or interface methods
	switch t := typeSpec.Type.(type) {
	case *ast.StructType:
		for _, field := range t.Fields.List {
			for _, name := range field.Names {
				fieldEntity := sa.createFieldEntity(name.Name, entity.ID, file, packageName)
				entity.Contains = append(entity.Contains, fieldEntity.ID)
				sa.graph.Entities[fieldEntity.ID] = fieldEntity
			}
		}
	case *ast.InterfaceType:
		for _, method := range t.Methods.List {
			for _, name := range method.Names {
				methodEntity := sa.createMethodEntity(name.Name, entity.ID, file, packageName)
				entity.Contains = append(entity.Contains, methodEntity.ID)
				sa.graph.Entities[methodEntity.ID] = methodEntity
			}
		}
	}

	sa.graph.Entities[entity.ID] = entity
}

// analyzePythonFile analyzes a Python source file
func (sa *SemanticAnalyzer) analyzePythonFile(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	lines := strings.Split(string(content), "\n")

	// Simple Python analysis using regex patterns
	for i, line := range lines {
		line = strings.TrimSpace(line)

		// Analyze classes
		if matches := regexp.MustCompile(`^class\s+(\w+)`).FindStringSubmatch(line); len(matches) > 1 {
			className := matches[1]
			entity := &SemanticEntity{
				ID:           fmt.Sprintf("python::%s", className),
				Name:         className,
				Type:         EntityClass,
				File:         path,
				Line:         i + 1,
				Language:     "Python",
				Visibility:   sa.determinePythonVisibility(className),
				Uses:         make([]string, 0),
				UsedBy:       make([]string, 0),
				Contains:     make([]string, 0),
				LastModified: time.Now(),
			}
			sa.graph.Entities[entity.ID] = entity
		}

		// Analyze functions
		if matches := regexp.MustCompile(`^def\s+(\w+)`).FindStringSubmatch(line); len(matches) > 1 {
			funcName := matches[1]
			entity := &SemanticEntity{
				ID:           fmt.Sprintf("python::%s", funcName),
				Name:         funcName,
				Type:         EntityFunction,
				File:         path,
				Line:         i + 1,
				Language:     "Python",
				Visibility:   sa.determinePythonVisibility(funcName),
				Uses:         make([]string, 0),
				UsedBy:       make([]string, 0),
				Calls:        make([]string, 0),
				CalledBy:     make([]string, 0),
				LastModified: time.Now(),
			}
			sa.graph.Entities[entity.ID] = entity
		}
	}

	return nil
}

// analyzeJavaScriptFile analyzes a JavaScript/TypeScript source file
func (sa *SemanticAnalyzer) analyzeJavaScriptFile(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	lines := strings.Split(string(content), "\n")

	// Simple JavaScript analysis using regex patterns
	for i, line := range lines {
		line = strings.TrimSpace(line)

		// Analyze classes
		if matches := regexp.MustCompile(`^class\s+(\w+)`).FindStringSubmatch(line); len(matches) > 1 {
			className := matches[1]
			entity := &SemanticEntity{
				ID:           fmt.Sprintf("js::%s", className),
				Name:         className,
				Type:         EntityClass,
				File:         path,
				Line:         i + 1,
				Language:     "JavaScript",
				Uses:         make([]string, 0),
				UsedBy:       make([]string, 0),
				Contains:     make([]string, 0),
				LastModified: time.Now(),
			}
			sa.graph.Entities[entity.ID] = entity
		}

		// Analyze functions
		if matches := regexp.MustCompile(`^function\s+(\w+)|^const\s+(\w+)\s*=\s*\(`).FindStringSubmatch(line); len(matches) > 1 {
			funcName := matches[1]
			if funcName == "" {
				funcName = matches[2]
			}
			entity := &SemanticEntity{
				ID:           fmt.Sprintf("js::%s", funcName),
				Name:         funcName,
				Type:         EntityFunction,
				File:         path,
				Line:         i + 1,
				Language:     "JavaScript",
				Uses:         make([]string, 0),
				UsedBy:       make([]string, 0),
				Calls:        make([]string, 0),
				CalledBy:     make([]string, 0),
				LastModified: time.Now(),
			}
			sa.graph.Entities[entity.ID] = entity
		}
	}

	return nil
}

// analyzeJavaFile analyzes a Java source file
func (sa *SemanticAnalyzer) analyzeJavaFile(path string) error {
	// Similar implementation to other languages
	// For brevity, implementing a basic version
	return sa.analyzeGenericFile(path)
}

// analyzeGenericFile analyzes files using generic patterns
func (sa *SemanticAnalyzer) analyzeGenericFile(path string) error {
	// Basic analysis for unsupported languages
	return nil
}

// Helper methods for entity creation

func (sa *SemanticAnalyzer) createPackageEntity(name, file string) *SemanticEntity {
	return &SemanticEntity{
		ID:           fmt.Sprintf("package::%s", name),
		Name:         name,
		Type:         EntityPackage,
		File:         file,
		Language:     "Go",
		Visibility:   VisibilityPublic,
		Uses:         make([]string, 0),
		UsedBy:       make([]string, 0),
		Contains:     make([]string, 0),
		LastModified: time.Now(),
	}
}

func (sa *SemanticAnalyzer) createFieldEntity(name, containerID, file, packageName string) *SemanticEntity {
	return &SemanticEntity{
		ID:           fmt.Sprintf("%s::%s", containerID, name),
		Name:         name,
		Type:         EntityField,
		File:         file,
		Package:      packageName,
		Language:     "Go",
		ContainedBy:  containerID,
		Visibility:   sa.determineVisibility(name),
		Uses:         make([]string, 0),
		UsedBy:       make([]string, 0),
		LastModified: time.Now(),
	}
}

func (sa *SemanticAnalyzer) createMethodEntity(name, containerID, file, packageName string) *SemanticEntity {
	return &SemanticEntity{
		ID:           fmt.Sprintf("%s::%s", containerID, name),
		Name:         name,
		Type:         EntityMethod,
		File:         file,
		Package:      packageName,
		Language:     "Go",
		ContainedBy:  containerID,
		Visibility:   sa.determineVisibility(name),
		Uses:         make([]string, 0),
		UsedBy:       make([]string, 0),
		Calls:        make([]string, 0),
		CalledBy:     make([]string, 0),
		LastModified: time.Now(),
	}
}

// Helper methods for analysis

func (sa *SemanticAnalyzer) extractFunctionSignature(funcDecl *ast.FuncDecl) string {
	// Extract function signature - simplified implementation
	return funcDecl.Name.Name + "()"
}

func (sa *SemanticAnalyzer) extractDocumentation(doc *ast.CommentGroup) string {
	if doc == nil {
		return ""
	}

	var docLines []string
	for _, comment := range doc.List {
		line := strings.TrimPrefix(comment.Text, "//")
		line = strings.TrimPrefix(line, "/*")
		line = strings.TrimSuffix(line, "*/")
		line = strings.TrimSpace(line)
		if line != "" {
			docLines = append(docLines, line)
		}
	}

	return strings.Join(docLines, " ")
}

func (sa *SemanticAnalyzer) determineVisibility(name string) Visibility {
	if name == "" {
		return VisibilityPrivate
	}

	// Go convention: uppercase first letter = public
	if name[0] >= 'A' && name[0] <= 'Z' {
		return VisibilityPublic
	}

	return VisibilityPrivate
}

func (sa *SemanticAnalyzer) determinePythonVisibility(name string) Visibility {
	if strings.HasPrefix(name, "__") {
		return VisibilityPrivate
	} else if strings.HasPrefix(name, "_") {
		return VisibilityProtected
	}
	return VisibilityPublic
}

func (sa *SemanticAnalyzer) calculateFunctionComplexity(funcDecl *ast.FuncDecl) int {
	complexity := 1 // Base complexity

	if funcDecl.Body != nil {
		ast.Inspect(funcDecl.Body, func(n ast.Node) bool {
			switch n.(type) {
			case *ast.IfStmt, *ast.ForStmt, *ast.RangeStmt, *ast.SwitchStmt, *ast.TypeSwitchStmt:
				complexity++
			}
			return true
		})
	}

	return complexity
}

// buildRelationships builds relationships between entities
func (sa *SemanticAnalyzer) buildRelationships() {
	sa.graph.mu.Lock()
	defer sa.graph.mu.Unlock()

	// Build call relationships
	for _, entity := range sa.graph.Entities {
		for _, calledFunc := range entity.Calls {
			// Find the called function entity
			for _, targetEntity := range sa.graph.Entities {
				if targetEntity.Name == calledFunc || strings.HasSuffix(targetEntity.ID, "::"+calledFunc) {
					// Create call relationship
					relationship := SemanticRelationship{
						From:     entity.ID,
						To:       targetEntity.ID,
						Type:     RelationshipCalls,
						Strength: 1.0,
						Context:  "function call",
						File:     entity.File,
						Line:     entity.Line,
					}

					sa.graph.Relationships = append(sa.graph.Relationships, relationship)

					// Update entity relationships
					if !sa.containsString(entity.Calls, targetEntity.ID) {
						entity.Calls = append(entity.Calls, targetEntity.ID)
					}
					if !sa.containsString(targetEntity.CalledBy, entity.ID) {
						targetEntity.CalledBy = append(targetEntity.CalledBy, entity.ID)
					}
					break
				}
			}
		}
	}

	// Build containment relationships
	for _, entity := range sa.graph.Entities {
		if entity.ContainedBy != "" {
			for _, containerEntity := range sa.graph.Entities {
				if containerEntity.Name == entity.ContainedBy || strings.HasSuffix(containerEntity.ID, "::"+entity.ContainedBy) {
					relationship := SemanticRelationship{
						From:     containerEntity.ID,
						To:       entity.ID,
						Type:     RelationshipContains,
						Strength: 1.0,
						Context:  "containment",
						File:     entity.File,
						Line:     entity.Line,
					}

					sa.graph.Relationships = append(sa.graph.Relationships, relationship)
					break
				}
			}
		}
	}
}

// calculateImportance calculates entity importance scores
func (sa *SemanticAnalyzer) calculateImportance() {
	sa.graph.mu.Lock()
	defer sa.graph.mu.Unlock()

	for _, entity := range sa.graph.Entities {
		importance := 0.0

		// Base importance by type
		switch entity.Type {
		case EntityPackage:
			importance = 10.0
		case EntityClass, EntityInterface:
			importance = 8.0
		case EntityFunction, EntityMethod:
			importance = 5.0
		default:
			importance = 2.0
		}

		// Add importance based on usage
		importance += float64(len(entity.UsedBy)) * 2.0
		importance += float64(len(entity.CalledBy)) * 1.5
		importance += float64(len(entity.Contains)) * 1.0

		// Add importance based on visibility
		if entity.Visibility == VisibilityPublic {
			importance *= 1.5
		}

		// Add importance based on complexity
		if entity.Complexity > 10 {
			importance *= 1.2
		}

		entity.Importance = importance
	}
}

// identifyClusters identifies related entity clusters
func (sa *SemanticAnalyzer) identifyClusters() {
	sa.graph.mu.Lock()
	defer sa.graph.mu.Unlock()

	// Group entities by package/namespace
	packageClusters := make(map[string][]string)

	for id, entity := range sa.graph.Entities {
		key := entity.Package
		if key == "" {
			key = "global"
		}

		if packageClusters[key] == nil {
			packageClusters[key] = make([]string, 0)
		}
		packageClusters[key] = append(packageClusters[key], id)
	}

	// Create clusters
	for packageName, entityIDs := range packageClusters {
		if len(entityIDs) > 1 {
			cluster := EntityCluster{
				ID:       fmt.Sprintf("cluster_%s", packageName),
				Name:     fmt.Sprintf("Package: %s", packageName),
				Entities: entityIDs,
				Cohesion: sa.calculateClusterCohesion(entityIDs),
				Purpose:  fmt.Sprintf("Components of %s package", packageName),
			}

			sa.graph.Clusters = append(sa.graph.Clusters, cluster)
		}
	}
}

// calculateClusterCohesion calculates cluster cohesion
func (sa *SemanticAnalyzer) calculateClusterCohesion(entityIDs []string) float64 {
	if len(entityIDs) < 2 {
		return 1.0
	}

	totalRelationships := 0
	internalRelationships := 0

	for _, relationship := range sa.graph.Relationships {
		fromInCluster := sa.containsString(entityIDs, relationship.From)
		toInCluster := sa.containsString(entityIDs, relationship.To)

		if fromInCluster || toInCluster {
			totalRelationships++
			if fromInCluster && toInCluster {
				internalRelationships++
			}
		}
	}

	if totalRelationships == 0 {
		return 0.0
	}

	return float64(internalRelationships) / float64(totalRelationships)
}

// calculateComplexityMetrics calculates various complexity metrics
func (sa *SemanticAnalyzer) calculateComplexityMetrics() {
	sa.graph.mu.Lock()
	defer sa.graph.mu.Unlock()

	for _, entity := range sa.graph.Entities {
		// Calculate change risk based on usage and complexity
		changeRisk := 0.0

		// High usage increases change risk
		usage := float64(len(entity.UsedBy) + len(entity.CalledBy))
		changeRisk += usage * 0.1

		// High complexity increases change risk
		changeRisk += float64(entity.Complexity) * 0.05

		// Public visibility increases change risk
		if entity.Visibility == VisibilityPublic {
			changeRisk *= 1.5
		}

		// Normalize change risk to 0-1 range
		if changeRisk > 1.0 {
			changeRisk = 1.0
		}

		entity.ChangeRisk = changeRisk
	}
}

// containsString checks if a slice contains a string
func (sa *SemanticAnalyzer) containsString(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// Query methods

// GetEntitiesByType returns entities of a specific type
func (sa *SemanticAnalyzer) GetEntitiesByType(entityType EntityType) []*SemanticEntity {
	sa.graph.mu.RLock()
	defer sa.graph.mu.RUnlock()

	var entities []*SemanticEntity
	for _, entity := range sa.graph.Entities {
		if entity.Type == entityType {
			entities = append(entities, entity)
		}
	}

	return entities
}

// GetMostImportantEntities returns entities sorted by importance
func (sa *SemanticAnalyzer) GetMostImportantEntities(limit int) []*SemanticEntity {
	sa.graph.mu.RLock()
	defer sa.graph.mu.RUnlock()

	var entities []*SemanticEntity
	for _, entity := range sa.graph.Entities {
		entities = append(entities, entity)
	}

	sort.Slice(entities, func(i, j int) bool {
		return entities[i].Importance > entities[j].Importance
	})

	if limit > 0 && len(entities) > limit {
		entities = entities[:limit]
	}

	return entities
}

// GetHighRiskEntities returns entities with high change risk
func (sa *SemanticAnalyzer) GetHighRiskEntities(threshold float64) []*SemanticEntity {
	sa.graph.mu.RLock()
	defer sa.graph.mu.RUnlock()

	var entities []*SemanticEntity
	for _, entity := range sa.graph.Entities {
		if entity.ChangeRisk >= threshold {
			entities = append(entities, entity)
		}
	}

	sort.Slice(entities, func(i, j int) bool {
		return entities[i].ChangeRisk > entities[j].ChangeRisk
	})

	return entities
}

// GetRelatedEntities returns entities related to a given entity
func (sa *SemanticAnalyzer) GetRelatedEntities(entityID string, maxDepth int) []*SemanticEntity {
	sa.graph.mu.RLock()
	defer sa.graph.mu.RUnlock()

	visited := make(map[string]bool)
	var related []*SemanticEntity

	sa.findRelatedEntitiesRecursive(entityID, maxDepth, 0, visited, &related)

	return related
}

// findRelatedEntitiesRecursive recursively finds related entities
func (sa *SemanticAnalyzer) findRelatedEntitiesRecursive(entityID string, maxDepth, currentDepth int, visited map[string]bool, related *[]*SemanticEntity) {
	if currentDepth >= maxDepth || visited[entityID] {
		return
	}

	visited[entityID] = true

	if entity, exists := sa.graph.Entities[entityID]; exists {
		*related = append(*related, entity)
	}

	// Find all related entities through relationships
	for _, relationship := range sa.graph.Relationships {
		var nextEntityID string

		if relationship.From == entityID {
			nextEntityID = relationship.To
		} else if relationship.To == entityID {
			nextEntityID = relationship.From
		}

		if nextEntityID != "" {
			sa.findRelatedEntitiesRecursive(nextEntityID, maxDepth, currentDepth+1, visited, related)
		}
	}
}

// GenerateSemanticReport generates a comprehensive semantic analysis report
func (sa *SemanticAnalyzer) GenerateSemanticReport() string {
	sa.graph.mu.RLock()
	defer sa.graph.mu.RUnlock()

	var report strings.Builder

	report.WriteString("🧠 Semantic Analysis Report\n")
	report.WriteString("===========================\n\n")

	// Summary statistics
	report.WriteString("📊 Summary\n")
	report.WriteString("-----------\n")
	report.WriteString(fmt.Sprintf("Total Entities: %d\n", len(sa.graph.Entities)))
	report.WriteString(fmt.Sprintf("Total Relationships: %d\n", len(sa.graph.Relationships)))
	report.WriteString(fmt.Sprintf("Identified Clusters: %d\n\n", len(sa.graph.Clusters)))

	// Entity breakdown by type
	typeCounts := make(map[EntityType]int)
	for _, entity := range sa.graph.Entities {
		typeCounts[entity.Type]++
	}

	report.WriteString("🏗️ Entity Breakdown\n")
	report.WriteString("-------------------\n")
	entityTypeNames := map[EntityType]string{
		EntityPackage:   "Packages",
		EntityClass:     "Classes",
		EntityInterface: "Interfaces",
		EntityStruct:    "Structs",
		EntityFunction:  "Functions",
		EntityMethod:    "Methods",
		EntityVariable:  "Variables",
		EntityConstant:  "Constants",
		EntityField:     "Fields",
	}

	for entityType, count := range typeCounts {
		if name, exists := entityTypeNames[entityType]; exists {
			report.WriteString(fmt.Sprintf("%-12s: %d\n", name, count))
		}
	}
	report.WriteString("\n")

	// Most important entities
	important := sa.GetMostImportantEntities(10)
	if len(important) > 0 {
		report.WriteString("⭐ Most Important Entities\n")
		report.WriteString("--------------------------\n")
		for i, entity := range important {
			if i >= 5 {
				break
			}
			report.WriteString(fmt.Sprintf("• %s (importance: %.1f)\n", entity.Name, entity.Importance))
		}
		report.WriteString("\n")
	}

	// High-risk entities
	highRisk := sa.GetHighRiskEntities(0.5)
	if len(highRisk) > 0 {
		report.WriteString("⚠️  High-Risk Entities\n")
		report.WriteString("---------------------\n")
		for i, entity := range highRisk {
			if i >= 5 {
				break
			}
			report.WriteString(fmt.Sprintf("• %s (risk: %.1f)\n", entity.Name, entity.ChangeRisk))
		}
		report.WriteString("\n")
	}

	// Clusters
	if len(sa.graph.Clusters) > 0 {
		report.WriteString("🔗 Entity Clusters\n")
		report.WriteString("------------------\n")
		for _, cluster := range sa.graph.Clusters {
			report.WriteString(fmt.Sprintf("• %s (%d entities, cohesion: %.2f)\n",
				cluster.Name, len(cluster.Entities), cluster.Cohesion))
		}
		report.WriteString("\n")
	}

	return report.String()
}

// GetGraph returns the semantic graph
func (sa *SemanticAnalyzer) GetGraph() *SemanticGraph {
	sa.graph.mu.RLock()
	defer sa.graph.mu.RUnlock()
	return sa.graph
}
