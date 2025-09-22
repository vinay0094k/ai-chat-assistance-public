package indexer

import (
	"fmt"
	"go/token"
	"sync"
	"time"
)

// GraphBuilder builds and maintains the dependency graph
type GraphBuilder struct {
	graph           *UpdateGraph
	config          *IncrementalConfig
	symbolExtractor *SymbolExtractor
	depAnalyzer     *DependencyAnalyzer
	fileSet         *token.FileSet
}

// SymbolExtractor extracts symbols from parsed code
type SymbolExtractor struct {
	fileSet *token.FileSet
}

// DependencyAnalyzer analyzes dependencies between symbols
type DependencyAnalyzer struct {
	graph   *UpdateGraph
	symbols map[string]*SymbolInfo
	mu      sync.RWMutex
}

// SymbolInfo contains information about a symbol
type SymbolInfo struct {
	Name         string          `json:"name"`
	Type         string          `json:"type"`
	FilePath     string          `json:"file_path"`
	Location     *ChangeLocation `json:"location"`
	Signature    string          `json:"signature"`
	Dependencies []string        `json:"dependencies"`
	Exports      bool            `json:"exports"`
	LastUpdated  time.Time       `json:"last_updated"`
}

// SymbolNode represents a symbol in the dependency graph
type SymbolNode struct {
	*GraphNode
	SymbolInfo *SymbolInfo `json:"symbol_info"`
}

// FileNode represents a file in the dependency graph
type FileNode struct {
	*GraphNode
	Symbols  []*SymbolNode `json:"symbols"`
	Imports  []string      `json:"imports"`
	Exports  []string      `json:"exports"`
	Language string        `json:"language"`
	ParsedAt time.Time     `json:"parsed_at"`
}

// NewGraphBuilder creates a new graph builder
func NewGraphBuilder(graph *UpdateGraph, config *IncrementalConfig) *GraphBuilder {
	return &GraphBuilder{
		graph:   graph,
		config:  config,
		fileSet: token.NewFileSet(),
		symbolExtractor: &SymbolExtractor{
			fileSet: token.NewFileSet(),
		},
		depAnalyzer: &DependencyAnalyzer{
			graph:   graph,
			symbols: make(map[string]*SymbolInfo),
		},
	}
}

// UpdateFileInGraph updates a file's representation in the dependency graph
func (gb *GraphBuilder) UpdateFileInGraph(filePath string, parseResult *ParseResult) error {
	// Remove existing file node if it exists
	gb.RemoveFileFromGraph(filePath)

	// Create new file node
	fileNode := &FileNode{
		GraphNode: &GraphNode{
			ID:          gb.generateNodeID("file", filePath),
			Type:        NodeTypeFile,
			FilePath:    filePath,
			LastUpdated: time.Now(),
		},
		Language: parseResult.Language,
		Imports:  parseResult.Imports,
		Exports:  parseResult.Exports,
		ParsedAt: parseResult.ParsedAt,
	}

	// Extract symbols from chunks
	symbols, err := gb.extractSymbolsFromChunks(parseResult.Chunks, filePath)
	if err != nil {
		return fmt.Errorf("failed to extract symbols: %v", err)
	}

	fileNode.Symbols = symbols

	// Add file node to graph
	gb.graph.AddFileNode(fileNode)

	// Build dependencies
	if err := gb.buildDependencies(fileNode); err != nil {
		return fmt.Errorf("failed to build dependencies: %v", err)
	}

	return nil
}

// RemoveFileFromGraph removes a file from the dependency graph
func (gb *GraphBuilder) RemoveFileFromGraph(filePath string) error {
	gb.graph.mu.Lock()
	defer gb.graph.mu.Unlock()

	// Find and remove file node
	fileNode := gb.graph.files[filePath]
	if fileNode == nil {
		return nil // File not in graph
	}

	// Remove all symbols associated with this file
	for _, symbol := range fileNode.Symbols {
		delete(gb.graph.symbols, symbol.SymbolInfo.Name)
		delete(gb.graph.nodes, symbol.ID)
	}

	// Remove file node
	delete(gb.graph.files, filePath)
	delete(gb.graph.nodes, fileNode.ID)

	// Remove edges
	delete(gb.graph.edges, fileNode.ID)
	for nodeID, edges := range gb.graph.edges {
		var filteredEdges []*GraphEdge
		for _, edge := range edges {
			if edge.From.ID != fileNode.ID && edge.To.ID != fileNode.ID {
				filteredEdges = append(filteredEdges, edge)
			}
		}
		gb.graph.edges[nodeID] = filteredEdges
	}

	return nil
}

// extractSymbolsFromChunks extracts symbols from code chunks
func (gb *GraphBuilder) extractSymbolsFromChunks(chunks []*CodeChunk, filePath string) ([]*SymbolNode, error) {
	var symbols []*SymbolNode

	for _, chunk := range chunks {
		switch chunk.ChunkType {
		case "function":
			symbolNode := gb.createSymbolNode(chunk, "function", filePath)
			symbols = append(symbols, symbolNode)

		case "type", "struct", "interface":
			symbolNode := gb.createSymbolNode(chunk, "type", filePath)
			symbols = append(symbols, symbolNode)

		case "const", "var":
			symbolNode := gb.createSymbolNode(chunk, "variable", filePath)
			symbols = append(symbols, symbolNode)

		case "class":
			symbolNode := gb.createSymbolNode(chunk, "class", filePath)
			symbols = append(symbols, symbolNode)
		}
	}

	return symbols, nil
}

// createSymbolNode creates a symbol node from a code chunk
func (gb *GraphBuilder) createSymbolNode(chunk *CodeChunk, symbolType, filePath string) *SymbolNode {
	symbolInfo := &SymbolInfo{
		Name:     chunk.Name,
		Type:     symbolType,
		FilePath: filePath,
		Location: &ChangeLocation{
			StartLine: chunk.StartLine,
			EndLine:   chunk.EndLine,
		},
		Signature:    chunk.Signature,
		Dependencies: chunk.Dependencies,
		Exports:      len(chunk.Name) > 0 && chunk.Name[0] >= 'A' && chunk.Name[0] <= 'Z', // Simple Go export check
		LastUpdated:  time.Now(),
	}

	graphNode := &GraphNode{
		ID:       gb.generateNodeID("symbol", chunk.Name+"@"+filePath),
		Type:     gb.mapSymbolTypeToNodeType(symbolType),
		FilePath: filePath,
		Symbol:   chunk.Name,
		Location: &ChangeLocation{
			StartLine: chunk.StartLine,
			EndLine:   chunk.EndLine,
		},
		LastUpdated: time.Now(),
		Metadata: map[string]interface{}{
			"chunk_type": chunk.ChunkType,
			"signature":  chunk.Signature,
			"complexity": chunk.Complexity,
			"exports":    symbolInfo.Exports,
		},
	}

	return &SymbolNode{
		GraphNode:  graphNode,
		SymbolInfo: symbolInfo,
	}
}

// buildDependencies builds dependency edges for a file
func (gb *GraphBuilder) buildDependencies(fileNode *FileNode) error {
	// Build import dependencies
	if err := gb.buildImportDependencies(fileNode); err != nil {
		return err
	}

	// Build symbol dependencies
	if err := gb.buildSymbolDependencies(fileNode); err != nil {
		return err
	}

	// Build call dependencies
	if err := gb.buildCallDependencies(fileNode); err != nil {
		return err
	}

	return nil
}

// buildImportDependencies builds dependencies from imports
func (gb *GraphBuilder) buildImportDependencies(fileNode *FileNode) error {
	for _, importPath := range fileNode.Imports {
		// Find the imported file/module
		targetNode := gb.findImportTarget(importPath)
		if targetNode != nil {
			edge := &GraphEdge{
				ID:        gb.generateEdgeID(),
				From:      fileNode.GraphNode,
				To:        targetNode,
				Type:      EdgeTypeImport,
				Weight:    1.0,
				CreatedAt: time.Now(),
			}

			gb.graph.AddEdge(edge)
		}
	}

	return nil
}

// buildSymbolDependencies builds dependencies between symbols
func (gb *GraphBuilder) buildSymbolDependencies(fileNode *FileNode) error {
	for _, symbol := range fileNode.Symbols {
		for _, depName := range symbol.SymbolInfo.Dependencies {
			// Find the dependency symbol
			depSymbol := gb.findSymbolByName(depName)
			if depSymbol != nil {
				edge := &GraphEdge{
					ID:        gb.generateEdgeID(),
					From:      symbol.GraphNode,
					To:        depSymbol.GraphNode,
					Type:      EdgeTypeReference,
					Weight:    1.0,
					CreatedAt: time.Now(),
				}

				gb.graph.AddEdge(edge)
			}
		}
	}

	return nil
}

// buildCallDependencies builds call dependencies by analyzing the AST
func (gb *GraphBuilder) buildCallDependencies(fileNode *FileNode) error {
	// This would perform deeper AST analysis to find function calls
	// For now, we'll use the dependencies already extracted from chunks

	return nil
}

// Graph management

// NewUpdateGraph creates a new update graph
func NewUpdateGraph() *UpdateGraph {
	return &UpdateGraph{
		nodes:   make(map[string]*GraphNode),
		edges:   make(map[string][]*GraphEdge),
		symbols: make(map[string]*SymbolNode),
		files:   make(map[string]*FileNode),
	}
}

// AddFileNode adds a file node to the graph
func (ug *UpdateGraph) AddFileNode(fileNode *FileNode) {
	ug.mu.Lock()
	defer ug.mu.Unlock()

	ug.files[fileNode.FilePath] = fileNode
	ug.nodes[fileNode.ID] = fileNode.GraphNode

	// Add symbol nodes
	for _, symbol := range fileNode.Symbols {
		ug.symbols[symbol.SymbolInfo.Name] = symbol
		ug.nodes[symbol.ID] = symbol.GraphNode
	}
}

// AddEdge adds an edge to the graph
func (ug *UpdateGraph) AddEdge(edge *GraphEdge) {
	ug.mu.Lock()
	defer ug.mu.Unlock()

	fromID := edge.From.ID
	ug.edges[fromID] = append(ug.edges[fromID], edge)

	// Update node dependency lists
	edge.From.Dependencies = append(edge.From.Dependencies, edge.To)
	edge.To.Dependents = append(edge.To.Dependents, edge.From)
}

// GetDependents returns all nodes that depend on the given file
func (ug *UpdateGraph) GetDependents(filePath string) []*GraphNode {
	ug.mu.RLock()
	defer ug.mu.RUnlock()

	var dependents []*GraphNode

	fileNode := ug.files[filePath]
	if fileNode == nil {
		return dependents
	}

	// Get direct dependents of the file
	dependents = append(dependents, fileNode.Dependents...)

	// Get dependents of symbols in the file
	for _, symbol := range fileNode.Symbols {
		dependents = append(dependents, symbol.Dependents...)
	}

	// Remove duplicates
	seen := make(map[string]bool)
	var unique []*GraphNode
	for _, node := range dependents {
		if !seen[node.ID] {
			seen[node.ID] = true
			unique = append(unique, node)
		}
	}

	return unique
}

// GetDependencies returns all dependencies of the given file
func (ug *UpdateGraph) GetDependencies(filePath string) []*GraphNode {
	ug.mu.RLock()
	defer ug.mu.RUnlock()

	var dependencies []*GraphNode

	fileNode := ug.files[filePath]
	if fileNode == nil {
		return dependencies
	}

	// Get direct dependencies of the file
	dependencies = append(dependencies, fileNode.Dependencies...)

	// Get dependencies of symbols in the file
	for _, symbol := range fileNode.Symbols {
		dependencies = append(dependencies, symbol.Dependencies...)
	}

	// Remove duplicates
	seen := make(map[string]bool)
	var unique []*GraphNode
	for _, node := range dependencies {
		if !seen[node.ID] {
			seen[node.ID] = true
			unique = append(unique, node)
		}
	}

	return unique
}

// GetImpactRadius calculates the impact radius of changes to a file
func (ug *UpdateGraph) GetImpactRadius(filePath string, maxDepth int) []*GraphNode {
	ug.mu.RLock()
	defer ug.mu.RUnlock()

	visited := make(map[string]bool)
	var impacted []*GraphNode

	ug.dfsImpact(filePath, 0, maxDepth, visited, &impacted)

	return impacted
}

// dfsImpact performs depth-first search to find impacted nodes
func (ug *UpdateGraph) dfsImpact(filePath string, depth, maxDepth int, visited map[string]bool, impacted *[]*GraphNode) {
	if depth > maxDepth || visited[filePath] {
		return
	}

	visited[filePath] = true

	dependents := ug.GetDependents(filePath)
	for _, dependent := range dependents {
		if !visited[dependent.FilePath] {
			*impacted = append(*impacted, dependent)
			ug.dfsImpact(dependent.FilePath, depth+1, maxDepth, visited, impacted)
		}
	}
}

// Utility methods

func (gb *GraphBuilder) generateNodeID(nodeType, identifier string) string {
	return fmt.Sprintf("%s_%s_%d", nodeType, gb.hashString(identifier), time.Now().UnixNano())
}

func (gb *GraphBuilder) generateEdgeID() string {
	return fmt.Sprintf("edge_%d", time.Now().UnixNano())
}

func (gb *GraphBuilder) hashString(s string) string {
	// Simple hash function for demonstration
	hash := uint32(0)
	for _, c := range s {
		hash = hash*31 + uint32(c)
	}
	return fmt.Sprintf("%x", hash)
}

func (gb *GraphBuilder) mapSymbolTypeToNodeType(symbolType string) NodeType {
	switch symbolType {
	case "function":
		return NodeTypeFunction
	case "class":
		return NodeTypeClass
	case "variable":
		return NodeTypeVariable
	case "type":
		return NodeTypeClass // Treat types as classes for now
	default:
		return NodeTypeFunction
	}
}

func (gb *GraphBuilder) findImportTarget(importPath string) *GraphNode {
	// This would resolve import paths to actual files
	// For now, return nil
	return nil
}

func (gb *GraphBuilder) findSymbolByName(name string) *SymbolNode {
	gb.depAnalyzer.mu.RLock()
	defer gb.depAnalyzer.mu.RUnlock()

	return gb.graph.symbols[name]
}

// Statistics

func (ug *UpdateGraph) GetStatistics() map[string]interface{} {
	ug.mu.RLock()
	defer ug.mu.RUnlock()

	edgeCount := 0
	for _, edges := range ug.edges {
		edgeCount += len(edges)
	}

	return map[string]interface{}{
		"total_nodes":  len(ug.nodes),
		"file_nodes":   len(ug.files),
		"symbol_nodes": len(ug.symbols),
		"total_edges":  edgeCount,
	}
}
