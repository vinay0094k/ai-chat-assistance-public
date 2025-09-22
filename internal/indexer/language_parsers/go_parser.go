package language_parsers

import (
	"crypto/md5"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/indexer"
)

// GoParser implements parsing for Go source code
type GoParser struct {
	fileSet *token.FileSet
}

// NewGoParser creates a new Go parser
func NewGoParser() *GoParser {
	return &GoParser{
		fileSet: token.NewFileSet(),
	}
}

// GetLanguage returns the language name
func (gp *GoParser) GetLanguage() string {
	return "go"
}

// GetFileExtensions returns supported file extensions
func (gp *GoParser) GetFileExtensions() []string {
	return []string{".go"}
}

// SupportsFile checks if the parser supports the given file
func (gp *GoParser) SupportsFile(filePath string) bool {
	ext := strings.ToLower(filepath.Ext(filePath))
	for _, supportedExt := range gp.GetFileExtensions() {
		if ext == supportedExt {
			return true
		}
	}
	return false
}

// ParseFile parses a Go source file into chunks
func (gp *GoParser) ParseFile(filePath string, content []byte) ([]*indexer.CodeChunk, error) {
	// Parse the Go source file
	file, err := parser.ParseFile(gp.fileSet, filePath, content, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Go file: %v", err)
	}

	var chunks []*indexer.CodeChunk

	// Extract package-level chunk
	if file.Name != nil {
		packageChunk := gp.createPackageChunk(file, filePath, content)
		chunks = append(chunks, packageChunk)
	}

	// Extract imports
	for _, imp := range file.Imports {
		importChunk := gp.createImportChunk(imp, filePath, content)
		chunks = append(chunks, importChunk)
	}

	// Walk through declarations
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			functionChunk := gp.createFunctionChunk(d, filePath, content)
			chunks = append(chunks, functionChunk)
		case *ast.GenDecl:
			genChunks := gp.createGenDeclChunks(d, filePath, content)
			chunks = append(chunks, genChunks...)
		}
	}

	// Extract comments as separate chunks
	for _, commentGroup := range file.Comments {
		commentChunk := gp.createCommentChunk(commentGroup, filePath, content)
		if commentChunk != nil {
			chunks = append(chunks, commentChunk)
		}
	}

	return chunks, nil
}

// createPackageChunk creates a chunk for package declaration
func (gp *GoParser) createPackageChunk(file *ast.File, filePath string, content []byte) *indexer.CodeChunk {
	pos := gp.fileSet.Position(file.Package)
	endPos := gp.fileSet.Position(file.Name.End())

	lines := strings.Split(string(content), "\n")
	packageLine := ""
	if pos.Line <= len(lines) {
		packageLine = lines[pos.Line-1]
	}

	return &indexer.CodeChunk{
		ID:        gp.generateChunkID(filePath, "package", pos.Line),
		FilePath:  filePath,
		Language:  "go",
		ChunkType: "package",
		Name:      file.Name.Name,
		Content:   packageLine,
		StartLine: pos.Line,
		EndLine:   endPos.Line,
		Signature: fmt.Sprintf("package %s", file.Name.Name),
		Hash:      gp.calculateHash(packageLine),
		Metadata: map[string]interface{}{
			"package_name": file.Name.Name,
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
}

// createImportChunk creates chunks for import declarations
func (gp *GoParser) createImportChunk(imp *ast.ImportSpec, filePath string, content []byte) *indexer.CodeChunk {
	pos := gp.fileSet.Position(imp.Pos())
	endPos := gp.fileSet.Position(imp.End())

	lines := strings.Split(string(content), "\n")
	importContent := gp.getContentBetweenLines(lines, pos.Line, endPos.Line)

	importPath := ""
	if imp.Path != nil {
		importPath, _ = strconv.Unquote(imp.Path.Value)
	}

	alias := ""
	if imp.Name != nil {
		alias = imp.Name.Name
	}

	return &indexer.CodeChunk{
		ID:        gp.generateChunkID(filePath, "import", pos.Line),
		FilePath:  filePath,
		Language:  "go",
		ChunkType: "import",
		Name:      importPath,
		Content:   importContent,
		StartLine: pos.Line,
		EndLine:   endPos.Line,
		Imports:   []string{importPath},
		Hash:      gp.calculateHash(importContent),
		Metadata: map[string]interface{}{
			"import_path": importPath,
			"alias":       alias,
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
}

// createFunctionChunk creates a chunk for function declarations
func (gp *GoParser) createFunctionChunk(fn *ast.FuncDecl, filePath string, content []byte) *indexer.CodeChunk {
	pos := gp.fileSet.Position(fn.Pos())
	endPos := gp.fileSet.Position(fn.End())

	lines := strings.Split(string(content), "\n")
	functionContent := gp.getContentBetweenLines(lines, pos.Line, endPos.Line)

	// Extract function signature
	signature := gp.extractFunctionSignature(fn)

	// Extract doc string
	docString := ""
	if fn.Doc != nil {
		docString = fn.Doc.Text()
	}

	// Calculate complexity (simplified)
	complexity := gp.calculateCyclomaticComplexity(fn)

	// Extract dependencies (function calls, etc.)
	dependencies := gp.extractFunctionDependencies(fn)

	return &indexer.CodeChunk{
		ID:           gp.generateChunkID(filePath, "function", pos.Line),
		FilePath:     filePath,
		Language:     "go",
		ChunkType:    "function",
		Name:         fn.Name.Name,
		Content:      functionContent,
		StartLine:    pos.Line,
		EndLine:      endPos.Line,
		Signature:    signature,
		DocString:    docString,
		Dependencies: dependencies,
		Complexity:   complexity,
		Hash:         gp.calculateHash(functionContent),
		Metadata: map[string]interface{}{
			"receiver":     gp.extractReceiver(fn),
			"parameters":   gp.extractParameters(fn),
			"return_types": gp.extractReturnTypes(fn),
			"is_exported":  ast.IsExported(fn.Name.Name),
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
}

// createGenDeclChunks creates chunks for general declarations (types, constants, variables)
func (gp *GoParser) createGenDeclChunks(decl *ast.GenDecl, filePath string, content []byte) []*indexer.CodeChunk {
	var chunks []*indexer.CodeChunk

	pos := gp.fileSet.Position(decl.Pos())
	endPos := gp.fileSet.Position(decl.End())

	lines := strings.Split(string(content), "\n")
	declContent := gp.getContentBetweenLines(lines, pos.Line, endPos.Line)

	chunkType := ""
	switch decl.Tok {
	case token.TYPE:
		chunkType = "type"
	case token.CONST:
		chunkType = "const"
	case token.VAR:
		chunkType = "var"
	default:
		chunkType = "declaration"
	}

	// For grouped declarations, create individual chunks for each spec
	if len(decl.Specs) > 1 {
		for _, spec := range decl.Specs {
			specChunk := gp.createSpecChunk(spec, chunkType, filePath, content)
			if specChunk != nil {
				chunks = append(chunks, specChunk)
			}
		}
	} else {
		// Single declaration
		chunk := &indexer.CodeChunk{
			ID:        gp.generateChunkID(filePath, chunkType, pos.Line),
			FilePath:  filePath,
			Language:  "go",
			ChunkType: chunkType,
			Content:   declContent,
			StartLine: pos.Line,
			EndLine:   endPos.Line,
			Hash:      gp.calculateHash(declContent),
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}

		// Extract name from first spec
		if len(decl.Specs) > 0 {
			chunk.Name = gp.extractSpecName(decl.Specs[0])
		}

		chunks = append(chunks, chunk)
	}

	return chunks
}

// createSpecChunk creates a chunk for individual spec within a declaration
func (gp *GoParser) createSpecChunk(spec ast.Spec, chunkType, filePath string, content []byte) *indexer.CodeChunk {
	pos := gp.fileSet.Position(spec.Pos())
	endPos := gp.fileSet.Position(spec.End())

	lines := strings.Split(string(content), "\n")
	specContent := gp.getContentBetweenLines(lines, pos.Line, endPos.Line)

	name := gp.extractSpecName(spec)

	chunk := &indexer.CodeChunk{
		ID:        gp.generateChunkID(filePath, chunkType, pos.Line),
		FilePath:  filePath,
		Language:  "go",
		ChunkType: chunkType,
		Name:      name,
		Content:   specContent,
		StartLine: pos.Line,
		EndLine:   endPos.Line,
		Hash:      gp.calculateHash(specContent),
		Metadata:  make(map[string]interface{}),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	// Add type-specific metadata
	switch s := spec.(type) {
	case *ast.TypeSpec:
		chunk.Metadata["type_name"] = s.Name.Name
		chunk.Metadata["is_exported"] = ast.IsExported(s.Name.Name)
		if structType, ok := s.Type.(*ast.StructType); ok {
			chunk.Metadata["is_struct"] = true
			chunk.Metadata["field_count"] = len(structType.Fields.List)
		}
	case *ast.ValueSpec:
		if len(s.Names) > 0 {
			chunk.Metadata["names"] = gp.extractNames(s.Names)
			chunk.Metadata["is_exported"] = ast.IsExported(s.Names[0].Name)
		}
	}

	return chunk
}

// createCommentChunk creates a chunk for comment groups
func (gp *GoParser) createCommentChunk(commentGroup *ast.CommentGroup, filePath string, content []byte) *indexer.CodeChunk {
	if commentGroup == nil || len(commentGroup.List) == 0 {
		return nil
	}

	pos := gp.fileSet.Position(commentGroup.Pos())
	endPos := gp.fileSet.Position(commentGroup.End())

	lines := strings.Split(string(content), "\n")
	commentContent := gp.getContentBetweenLines(lines, pos.Line, endPos.Line)

	// Skip short comments
	if len(strings.TrimSpace(commentContent)) < 20 {
		return nil
	}

	return &indexer.CodeChunk{
		ID:        gp.generateChunkID(filePath, "comment", pos.Line),
		FilePath:  filePath,
		Language:  "go",
		ChunkType: "comment",
		Content:   commentContent,
		StartLine: pos.Line,
		EndLine:   endPos.Line,
		Hash:      gp.calculateHash(commentContent),
		Metadata: map[string]interface{}{
			"comment_type": gp.getCommentType(commentGroup),
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
}

// Helper methods

func (gp *GoParser) getContentBetweenLines(lines []string, startLine, endLine int) string {
	if startLine < 1 || startLine > len(lines) {
		return ""
	}
	if endLine < startLine || endLine > len(lines) {
		endLine = len(lines)
	}

	return strings.Join(lines[startLine-1:endLine], "\n")
}

func (gp *GoParser) extractFunctionSignature(fn *ast.FuncDecl) string {
	var sig strings.Builder

	sig.WriteString("func ")

	// Add receiver if present
	if fn.Recv != nil {
		sig.WriteString("(")
		for i, field := range fn.Recv.List {
			if i > 0 {
				sig.WriteString(", ")
			}
			if len(field.Names) > 0 {
				sig.WriteString(field.Names[0].Name + " ")
			}
			sig.WriteString(gp.typeToString(field.Type))
		}
		sig.WriteString(") ")
	}

	sig.WriteString(fn.Name.Name)

	// Add parameters
	sig.WriteString("(")
	if fn.Type.Params != nil {
		for i, field := range fn.Type.Params.List {
			if i > 0 {
				sig.WriteString(", ")
			}
			for j, name := range field.Names {
				if j > 0 {
					sig.WriteString(", ")
				}
				sig.WriteString(name.Name)
			}
			if len(field.Names) > 0 {
				sig.WriteString(" ")
			}
			sig.WriteString(gp.typeToString(field.Type))
		}
	}
	sig.WriteString(")")

	// Add return types
	if fn.Type.Results != nil && len(fn.Type.Results.List) > 0 {
		sig.WriteString(" ")
		if len(fn.Type.Results.List) > 1 {
			sig.WriteString("(")
		}
		for i, field := range fn.Type.Results.List {
			if i > 0 {
				sig.WriteString(", ")
			}
			sig.WriteString(gp.typeToString(field.Type))
		}
		if len(fn.Type.Results.List) > 1 {
			sig.WriteString(")")
		}
	}

	return sig.String()
}

func (gp *GoParser) typeToString(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + gp.typeToString(t.X)
	case *ast.ArrayType:
		return "[]" + gp.typeToString(t.Elt)
	case *ast.SelectorExpr:
		return gp.typeToString(t.X) + "." + t.Sel.Name
	default:
		return "interface{}"
	}
}

func (gp *GoParser) calculateCyclomaticComplexity(fn *ast.FuncDecl) int {
	complexity := 1 // Base complexity

	ast.Inspect(fn, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.IfStmt, *ast.RangeStmt, *ast.ForStmt, *ast.TypeSwitchStmt, *ast.SwitchStmt:
			complexity++
		case *ast.CaseClause:
			complexity++
		}
		return true
	})

	return complexity
}

func (gp *GoParser) extractFunctionDependencies(fn *ast.FuncDecl) []string {
	var dependencies []string
	seen := make(map[string]bool)

	ast.Inspect(fn, func(n ast.Node) bool {
		switch call := n.(type) {
		case *ast.CallExpr:
			if ident, ok := call.Fun.(*ast.Ident); ok {
				if !seen[ident.Name] {
					dependencies = append(dependencies, ident.Name)
					seen[ident.Name] = true
				}
			} else if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				funcName := sel.Sel.Name
				if x, ok := sel.X.(*ast.Ident); ok {
					funcName = x.Name + "." + funcName
				}
				if !seen[funcName] {
					dependencies = append(dependencies, funcName)
					seen[funcName] = true
				}
			}
		}
		return true
	})

	return dependencies
}

func (gp *GoParser) extractReceiver(fn *ast.FuncDecl) string {
	if fn.Recv == nil || len(fn.Recv.List) == 0 {
		return ""
	}

	field := fn.Recv.List[0]
	return gp.typeToString(field.Type)
}

func (gp *GoParser) extractParameters(fn *ast.FuncDecl) []string {
	var params []string

	if fn.Type.Params != nil {
		for _, field := range fn.Type.Params.List {
			paramType := gp.typeToString(field.Type)
			for _, name := range field.Names {
				params = append(params, name.Name+" "+paramType)
			}
		}
	}

	return params
}

func (gp *GoParser) extractReturnTypes(fn *ast.FuncDecl) []string {
	var returns []string

	if fn.Type.Results != nil {
		for _, field := range fn.Type.Results.List {
			returns = append(returns, gp.typeToString(field.Type))
		}
	}

	return returns
}

func (gp *GoParser) extractSpecName(spec ast.Spec) string {
	switch s := spec.(type) {
	case *ast.TypeSpec:
		return s.Name.Name
	case *ast.ValueSpec:
		if len(s.Names) > 0 {
			return s.Names[0].Name
		}
	case *ast.ImportSpec:
		if s.Name != nil {
			return s.Name.Name
		}
		if s.Path != nil {
			path, _ := strconv.Unquote(s.Path.Value)
			return path
		}
	}
	return ""
}

func (gp *GoParser) extractNames(names []*ast.Ident) []string {
	var result []string
	for _, name := range names {
		result = append(result, name.Name)
	}
	return result
}

func (gp *GoParser) getCommentType(cg *ast.CommentGroup) string {
	if cg == nil || len(cg.List) == 0 {
		return "unknown"
	}

	first := cg.List[0].Text
	if strings.HasPrefix(first, "//") {
		return "line"
	} else if strings.HasPrefix(first, "/*") {
		return "block"
	}

	return "unknown"
}

func (gp *GoParser) generateChunkID(filePath, chunkType string, line int) string {
	return fmt.Sprintf("%s:%s:%d", gp.calculateHash(filePath), chunkType, line)
}

func (gp *GoParser) calculateHash(content string) string {
	return fmt.Sprintf("%x", md5.Sum([]byte(content)))
}
