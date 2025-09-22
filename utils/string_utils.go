package utils

import (
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// StringProcessor provides string processing utilities
type StringProcessor struct {
	stopWords map[string]bool
}

// NewStringProcessor creates a new string processor
func NewStringProcessor() *StringProcessor {
	stopWords := map[string]bool{
		"a": true, "an": true, "and": true, "are": true, "as": true, "at": true,
		"be": true, "by": true, "for": true, "from": true, "has": true, "he": true,
		"in": true, "is": true, "it": true, "its": true, "of": true, "on": true,
		"that": true, "the": true, "to": true, "was": true, "will": true, "with": true,
		"or": true, "but": true, "not": true, "this": true, "can": true, "have": true,
		"do": true, "if": true, "we": true, "you": true, "all": true, "would": true,
		"there": true, "their": true, "what": true, "so": true, "up": true, "out": true,
		"many": true, "time": true, "them": true, "these": true, "way": true, "could": true,
		"no": true, "make": true, "than": true, "first": true, "been": true, "call": true,
		"who": true, "now": true, "find": true, "long": true, "down": true,
		"day": true, "did": true, "get": true, "come": true, "made": true, "may": true,
		"part": true,
	}

	return &StringProcessor{
		stopWords: stopWords,
	}
}

// CleanText cleans and normalizes text
func (sp *StringProcessor) CleanText(text string) string {
	// Remove extra whitespace
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")

	// Trim whitespace
	text = strings.TrimSpace(text)

	// Convert to lowercase
	text = strings.ToLower(text)

	return text
}

// ExtractKeywords extracts keywords from text
func (sp *StringProcessor) ExtractKeywords(text string, minLength int) []string {
	// Clean and tokenize
	cleaned := sp.CleanText(text)
	words := sp.Tokenize(cleaned)

	var keywords []string
	wordFreq := make(map[string]int)

	for _, word := range words {
		// Skip short words
		if len(word) < minLength {
			continue
		}

		// Skip stop words
		if sp.stopWords[word] {
			continue
		}

		// Skip numbers
		if _, err := strconv.Atoi(word); err == nil {
			continue
		}

		// Count frequency
		wordFreq[word]++
	}

	// Convert to slice and sort by frequency
	type wordCount struct {
		word  string
		count int
	}

	var wordCounts []wordCount
	for word, count := range wordFreq {
		wordCounts = append(wordCounts, wordCount{word, count})
	}

	sort.Slice(wordCounts, func(i, j int) bool {
		return wordCounts[i].count > wordCounts[j].count
	})

	// Extract just the words
	for _, wc := range wordCounts {
		keywords = append(keywords, wc.word)
	}

	return keywords
}

// Tokenize splits text into tokens
func (sp *StringProcessor) Tokenize(text string) []string {
	// Use regex to split on non-alphanumeric characters
	re := regexp.MustCompile(`[^\p{L}\p{N}]+`)
	tokens := re.Split(text, -1)

	var result []string
	for _, token := range tokens {
		token = strings.TrimSpace(token)
		if len(token) > 0 {
			result = append(result, strings.ToLower(token))
		}
	}

	return result
}

// ExtractCamelCaseWords extracts words from camelCase strings
func ExtractCamelCaseWords(s string) []string {
	if s == "" {
		return nil
	}

	var words []string
	var currentWord strings.Builder

	for i, r := range s {
		if unicode.IsUpper(r) && i > 0 {
			// Start new word
			if currentWord.Len() > 0 {
				words = append(words, currentWord.String())
				currentWord.Reset()
			}
		}
		currentWord.WriteRune(unicode.ToLower(r))
	}

	if currentWord.Len() > 0 {
		words = append(words, currentWord.String())
	}

	return words
}

// ExtractSnakeCaseWords extracts words from snake_case strings
func ExtractSnakeCaseWords(s string) []string {
	return strings.Split(s, "_")
}

// ExtractIdentifierWords extracts words from programming identifiers
func ExtractIdentifierWords(identifier string) []string {
	var words []string

	// Handle snake_case
	if strings.Contains(identifier, "_") {
		words = append(words, ExtractSnakeCaseWords(identifier)...)
	} else {
		// Handle camelCase
		words = append(words, ExtractCamelCaseWords(identifier)...)
	}

	// Filter out empty strings
	var result []string
	for _, word := range words {
		word = strings.TrimSpace(word)
		if len(word) > 0 {
			result = append(result, strings.ToLower(word))
		}
	}

	return result
}

// CalculateSimilarity calculates string similarity using Levenshtein distance
func CalculateSimilarity(s1, s2 string) float64 {
	if s1 == s2 {
		return 1.0
	}

	if len(s1) == 0 || len(s2) == 0 {
		return 0.0
	}

	distance := LevenshteinDistance(s1, s2)
	maxLen := len(s1)
	if len(s2) > maxLen {
		maxLen = len(s2)
	}

	return 1.0 - float64(distance)/float64(maxLen)
}

// LevenshteinDistance calculates the Levenshtein distance between two strings
func LevenshteinDistance(s1, s2 string) int {
	if s1 == s2 {
		return 0
	}

	if len(s1) == 0 {
		return len(s2)
	}

	if len(s2) == 0 {
		return len(s1)
	}

	// Create matrix
	matrix := make([][]int, len(s1)+1)
	for i := range matrix {
		matrix[i] = make([]int, len(s2)+1)
	}

	// Initialize first row and column
	for i := 0; i <= len(s1); i++ {
		matrix[i][0] = i
	}
	for j := 0; j <= len(s2); j++ {
		matrix[0][j] = j
	}

	// Fill matrix
	for i := 1; i <= len(s1); i++ {
		for j := 1; j <= len(s2); j++ {
			cost := 1
			if s1[i-1] == s2[j-1] {
				cost = 0
			}

			matrix[i][j] = min(
				matrix[i-1][j]+1,      // deletion
				matrix[i][j-1]+1,      // insertion
				matrix[i-1][j-1]+cost, // substitution
			)
		}
	}

	return matrix[len(s1)][len(s2)]
}

// min returns the minimum of three integers
func min(a, b, c int) int {
	if a < b {
		if a < c {
			return a
		}
		return c
	}
	if b < c {
		return b
	}
	return c
}

// TruncateString truncates a string to a maximum length
func TruncateString(s string, maxLength int) string {
	if len(s) <= maxLength {
		return s
	}

	if maxLength <= 3 {
		return s[:maxLength]
	}

	return s[:maxLength-3] + "..."
}

// TruncateStringByWords truncates string by word boundaries
func TruncateStringByWords(s string, maxLength int) string {
	if len(s) <= maxLength {
		return s
	}

	words := strings.Fields(s)
	var result strings.Builder

	for _, word := range words {
		if result.Len()+len(word)+1 > maxLength-3 {
			break
		}

		if result.Len() > 0 {
			result.WriteString(" ")
		}
		result.WriteString(word)
	}

	if result.Len() < len(s) {
		result.WriteString("...")
	}

	return result.String()
}

// WordCount counts words in a string
func WordCount(s string) int {
	words := strings.Fields(s)
	return len(words)
}

// LineCount counts lines in a string
func LineCount(s string) int {
	if s == "" {
		return 0
	}
	return strings.Count(s, "\n") + 1
}

// ExtractQuotedStrings extracts quoted strings from text
func ExtractQuotedStrings(text string) []string {
	var quotes []string

	// Match double quotes
	doubleQuoteRe := regexp.MustCompile(`"([^"]*)"`)
	matches := doubleQuoteRe.FindAllStringSubmatch(text, -1)
	for _, match := range matches {
		if len(match) > 1 {
			quotes = append(quotes, match[1])
		}
	}

	// Match single quotes
	singleQuoteRe := regexp.MustCompile(`'([^']*)'`)
	matches = singleQuoteRe.FindAllStringSubmatch(text, -1)
	for _, match := range matches {
		if len(match) > 1 {
			quotes = append(quotes, match[1])
		}
	}

	return quotes
}

// RemoveComments removes comments from code
func RemoveComments(code string, language string) string {
	switch strings.ToLower(language) {
	case "go", "java", "javascript", "typescript", "c", "cpp", "csharp", "rust":
		return removeSlashComments(code)
	case "python", "ruby", "shell", "bash":
		return removeHashComments(code)
	case "html", "xml":
		return removeHTMLComments(code)
	case "css":
		return removeCSSComments(code)
	default:
		return code // Unknown language, return as-is
	}
}

// removeSlashComments removes // and /* */ style comments
func removeSlashComments(code string) string {
	lines := strings.Split(code, "\n")
	var result []string
	inBlockComment := false

	for _, line := range lines {
		cleaned := ""
		inString := false
		stringChar := byte(0)

		for i := 0; i < len(line); i++ {
			if inBlockComment {
				if i < len(line)-1 && line[i] == '*' && line[i+1] == '/' {
					inBlockComment = false
					i++ // Skip the '/'
				}
				continue
			}

			if inString {
				cleaned += string(line[i])
				if line[i] == stringChar && (i == 0 || line[i-1] != '\\') {
					inString = false
				}
				continue
			}

			if line[i] == '"' || line[i] == '\'' {
				inString = true
				stringChar = line[i]
				cleaned += string(line[i])
				continue
			}

			if i < len(line)-1 && line[i] == '/' && line[i+1] == '/' {
				break // Rest of line is comment
			}

			if i < len(line)-1 && line[i] == '/' && line[i+1] == '*' {
				inBlockComment = true
				i++ // Skip the '*'
				continue
			}

			cleaned += string(line[i])
		}

		result = append(result, cleaned)
	}

	return strings.Join(result, "\n")
}

// removeHashComments removes # style comments
func removeHashComments(code string) string {
	lines := strings.Split(code, "\n")
	var result []string

	for _, line := range lines {
		cleaned := ""
		inString := false
		stringChar := byte(0)

		for i := 0; i < len(line); i++ {
			if inString {
				cleaned += string(line[i])
				if line[i] == stringChar && (i == 0 || line[i-1] != '\\') {
					inString = false
				}
				continue
			}

			if line[i] == '"' || line[i] == '\'' {
				inString = true
				stringChar = line[i]
				cleaned += string(line[i])
				continue
			}

			if line[i] == '#' {
				break // Rest of line is comment
			}

			cleaned += string(line[i])
		}

		result = append(result, cleaned)
	}

	return strings.Join(result, "\n")
}

// removeHTMLComments removes <!-- --> style comments
func removeHTMLComments(code string) string {
	re := regexp.MustCompile(`<!--.*?-->`)
	return re.ReplaceAllString(code, "")
}

// removeCSSComments removes /* */ style comments
func removeCSSComments(code string) string {
	re := regexp.MustCompile(`/\*.*?\*/`)
	return re.ReplaceAllString(code, "")
}

// ExtractImports extracts import statements from code
func ExtractImports(code string, language string) []string {
	switch strings.ToLower(language) {
	case "go":
		return extractGoImports(code)
	case "python":
		return extractPythonImports(code)
	case "javascript", "typescript":
		return extractJSImports(code)
	case "java":
		return extractJavaImports(code)
	default:
		return nil
	}
}

// extractGoImports extracts Go import statements
func extractGoImports(code string) []string {
	var imports []string

	// Single import: import "package"
	singleRe := regexp.MustCompile(`import\s+"([^"]+)"`)
	matches := singleRe.FindAllStringSubmatch(code, -1)
	for _, match := range matches {
		if len(match) > 1 {
			imports = append(imports, match[1])
		}
	}

	// Multi-import: import ( ... )
	multiRe := regexp.MustCompile(`import\s*\(\s*(.*?)\s*\)`)
	multiMatches := multiRe.FindAllStringSubmatch(code, -1)
	for _, match := range multiMatches {
		if len(match) > 1 {
			importBlock := match[1]
			lineRe := regexp.MustCompile(`"([^"]+)"`)
			lineMatches := lineRe.FindAllStringSubmatch(importBlock, -1)
			for _, lineMatch := range lineMatches {
				if len(lineMatch) > 1 {
					imports = append(imports, lineMatch[1])
				}
			}
		}
	}

	return imports
}

// extractPythonImports extracts Python import statements
func extractPythonImports(code string) []string {
	var imports []string
	lines := strings.Split(code, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)

		// import module
		if strings.HasPrefix(line, "import ") {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				module := parts[1]
				// Handle "import module as alias"
				if len(parts) >= 4 && parts[2] == "as" {
					module = parts[1]
				}
				imports = append(imports, module)
			}
		}

		// from module import ...
		if strings.HasPrefix(line, "from ") {
			re := regexp.MustCompile(`from\s+(\S+)\s+import`)
			matches := re.FindStringSubmatch(line)
			if len(matches) > 1 {
				imports = append(imports, matches[1])
			}
		}
	}

	return imports
}

// extractJSImports extracts JavaScript/TypeScript import statements
func extractJSImports(code string) []string {
	var imports []string

	// import ... from "module"
	re1 := regexp.MustCompile(`import\s+.*?\s+from\s+["']([^"']+)["']`)
	matches := re1.FindAllStringSubmatch(code, -1)
	for _, match := range matches {
		if len(match) > 1 {
			imports = append(imports, match[1])
		}
	}

	// import "module"
	re2 := regexp.MustCompile(`import\s+["']([^"']+)["']`)
	matches = re2.FindAllStringSubmatch(code, -1)
	for _, match := range matches {
		if len(match) > 1 {
			imports = append(imports, match[1])
		}
	}

	// require("module")
	re3 := regexp.MustCompile(`require\s*\(\s*["']([^"']+)["']\s*\)`)
	matches = re3.FindAllStringSubmatch(code, -1)
	for _, match := range matches {
		if len(match) > 1 {
			imports = append(imports, match[1])
		}
	}

	return imports
}

// extractJavaImports extracts Java import statements
func extractJavaImports(code string) []string {
	var imports []string

	re := regexp.MustCompile(`import\s+(?:static\s+)?([a-zA-Z_][a-zA-Z0-9_.]*(?:\.\*)?)\s*;`)
	matches := re.FindAllStringSubmatch(code, -1)
	for _, match := range matches {
		if len(match) > 1 {
			imports = append(imports, match[1])
		}
	}

	return imports
}

// ExtractFunctionNames extracts function names from code
func ExtractFunctionNames(code string, language string) []string {
	switch strings.ToLower(language) {
	case "go":
		return extractGoFunctions(code)
	case "python":
		return extractPythonFunctions(code)
	case "javascript", "typescript":
		return extractJSFunctions(code)
	case "java":
		return extractJavaFunctions(code)
	default:
		return nil
	}
}

// extractGoFunctions extracts Go function names
func extractGoFunctions(code string) []string {
	var functions []string

	// func functionName(...) ...
	re := regexp.MustCompile(`func\s+(?:\([^)]*\)\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(`)
	matches := re.FindAllStringSubmatch(code, -1)
	for _, match := range matches {
		if len(match) > 1 {
			functions = append(functions, match[1])
		}
	}

	return functions
}

// extractPythonFunctions extracts Python function names
func extractPythonFunctions(code string) []string {
	var functions []string

	re := regexp.MustCompile(`def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(`)
	matches := re.FindAllStringSubmatch(code, -1)
	for _, match := range matches {
		if len(match) > 1 {
			functions = append(functions, match[1])
		}
	}

	return functions
}

// extractJSFunctions extracts JavaScript function names
func extractJSFunctions(code string) []string {
	var functions []string

	// function functionName(...) ...
	re1 := regexp.MustCompile(`function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(`)
	matches := re1.FindAllStringSubmatch(code, -1)
	for _, match := range matches {
		if len(match) > 1 {
			functions = append(functions, match[1])
		}
	}

	// const functionName = (...) => ...
	re2 := regexp.MustCompile(`(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>`)
	matches = re2.FindAllStringSubmatch(code, -1)
	for _, match := range matches {
		if len(match) > 1 {
			functions = append(functions, match[1])
		}
	}

	return functions
}

// extractJavaFunctions extracts Java method names
func extractJavaFunctions(code string) []string {
	var functions []string

	// public/private/protected ... methodName(...) ...
	re := regexp.MustCompile(`(?:public|private|protected|static|final|\s)+\s+(?:[a-zA-Z_$][a-zA-Z0-9_$<>[\]]*\s+)?([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{`)
	matches := re.FindAllStringSubmatch(code, -1)
	for _, match := range matches {
		if len(match) > 1 {
			// Skip constructors (methods with capital first letter matching class name)
			if !unicode.IsUpper(rune(match[1][0])) {
				functions = append(functions, match[1])
			}
		}
	}

	return functions
}

// NormalizeWhitespace normalizes whitespace in code
func NormalizeWhitespace(code string) string {
	// Replace tabs with spaces
	code = strings.ReplaceAll(code, "\t", "    ")

	// Remove trailing whitespace from lines
	lines := strings.Split(code, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRight(line, " \t")
	}

	// Remove excessive blank lines (more than 2 consecutive)
	var result []string
	blankCount := 0

	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			blankCount++
			if blankCount <= 2 {
				result = append(result, line)
			}
		} else {
			blankCount = 0
			result = append(result, line)
		}
	}

	return strings.Join(result, "\n")
}

// IndentCode indents code with specified indentation
func IndentCode(code string, indent string) string {
	lines := strings.Split(code, "\n")
	for i, line := range lines {
		if strings.TrimSpace(line) != "" {
			lines[i] = indent + line
		}
	}
	return strings.Join(lines, "\n")
}

// RemoveIndentation removes common indentation from code
func RemoveIndentation(code string) string {
	lines := strings.Split(code, "\n")
	if len(lines) == 0 {
		return code
	}

	// Find minimum indentation (ignoring empty lines)
	minIndent := -1
	for _, line := range lines {
		trimmed := strings.TrimLeft(line, " \t")
		if trimmed == "" {
			continue // Skip empty lines
		}

		indent := len(line) - len(trimmed)
		if minIndent == -1 || indent < minIndent {
			minIndent = indent
		}
	}

	if minIndent <= 0 {
		return code
	}

	// Remove common indentation
	for i, line := range lines {
		if len(line) >= minIndent && strings.TrimSpace(line) != "" {
			lines[i] = line[minIndent:]
		}
	}

	return strings.Join(lines, "\n")
}

// WrapText wraps text to specified line length
func WrapText(text string, lineLength int) string {
	if lineLength <= 0 {
		return text
	}

	words := strings.Fields(text)
	if len(words) == 0 {
		return text
	}

	var result strings.Builder
	currentLength := 0

	for i, word := range words {
		if i > 0 {
			if currentLength+len(word)+1 > lineLength {
				result.WriteString("\n")
				currentLength = 0
			} else {
				result.WriteString(" ")
				currentLength++
			}
		}

		result.WriteString(word)
		currentLength += len(word)
	}

	return result.String()
}

// PadString pads a string to a specific length
func PadString(s string, length int, padChar rune, leftPad bool) string {
	if len(s) >= length {
		return s
	}

	padding := strings.Repeat(string(padChar), length-len(s))
	if leftPad {
		return padding + s
	}
	return s + padding
}

// CenterString centers a string within a given width
func CenterString(s string, width int) string {
	if len(s) >= width {
		return s
	}

	totalPad := width - len(s)
	leftPad := totalPad / 2
	rightPad := totalPad - leftPad

	return strings.Repeat(" ", leftPad) + s + strings.Repeat(" ", rightPad)
}

// IsValidUTF8 checks if a string is valid UTF-8
func IsValidUTF8(s string) bool {
	return utf8.ValidString(s)
}

// CountUTF8Chars counts UTF-8 characters (not bytes)
func CountUTF8Chars(s string) int {
	return utf8.RuneCountInString(s)
}

// ReverseString reverses a string (UTF-8 safe)
func ReverseString(s string) string {
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return string(runes)
}

// FirstNonEmpty returns the first non-empty string from a list
func FirstNonEmpty(values ...string) string {
	for _, s := range values {
		if strings.TrimSpace(s) != "" {
			return s
		}
	}
	return ""
}

// ContainsAny checks if string contains any of the given substrings
func ContainsAny(s string, substrings []string) bool {
	for _, sub := range substrings {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

// StartsWithAny checks if string starts with any of the given prefixes
func StartsWithAny(s string, prefixes []string) bool {
	for _, prefix := range prefixes {
		if strings.HasPrefix(s, prefix) {
			return true
		}
	}
	return false
}

// EndsWithAny checks if string ends with any of the given suffixes
func EndsWithAny(s string, suffixes []string) bool {
	for _, suffix := range suffixes {
		if strings.HasSuffix(s, suffix) {
			return true
		}
	}
	return false
}
