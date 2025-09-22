// display/syntax_highlighter.go
package display

import (
	"regexp"
	"strings"
	"unicode"
)

type TokenType int

const (
	TokenKeyword TokenType = iota
	TokenString
	TokenComment
	TokenNumber
	TokenFunction
	TokenVariable
	TokenType
	TokenOperator
	TokenDelimiter
	TokenWhitespace
	TokenText
)

type Token struct {
	Type      TokenType
	Value     string
	Start     int
	End       int
	ColorType string
}

type Language struct {
	Name              string
	Extensions        []string
	Keywords          []string
	Types             []string
	Operators         []string
	Delimiters        []string
	Patterns          map[TokenType]*regexp.Regexp
	SingleLineComment string
	MultiLineComment  [2]string
	StringDelimiters  []string
	CaseSensitive     bool
}

type SyntaxHighlighter struct {
	languages map[string]*Language
	theme     *ThemeManager
}

// NewSyntaxHighlighter creates a new syntax highlighter
func NewSyntaxHighlighter(theme *ThemeManager) *SyntaxHighlighter {
	if theme == nil {
		theme = GlobalTheme
	}

	sh := &SyntaxHighlighter{
		languages: make(map[string]*Language),
		theme:     theme,
	}

	// Register built-in languages
	sh.registerBuiltinLanguages()

	return sh
}

// registerBuiltinLanguages registers common programming languages
func (sh *SyntaxHighlighter) registerBuiltinLanguages() {
	// Go language
	sh.RegisterLanguage(&Language{
		Name:       "go",
		Extensions: []string{".go"},
		Keywords: []string{
			"break", "case", "chan", "const", "continue", "default", "defer",
			"else", "fallthrough", "for", "func", "go", "goto", "if", "import",
			"interface", "map", "package", "range", "return", "select", "struct",
			"switch", "type", "var",
		},
		Types: []string{
			"bool", "byte", "complex64", "complex128", "error", "float32", "float64",
			"int", "int8", "int16", "int32", "int64", "rune", "string",
			"uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
		},
		Operators:         []string{"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "&^", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "&^=", "&&", "||", "<-", "++", "--", "==", "<", ">", "=", "!", "!=", "<=", ">=", ":=", "...", "(", ")", "[", "]", "{", "}", ",", ";"},
		SingleLineComment: "//",
		MultiLineComment:  [2]string{"/*", "*/"},
		StringDelimiters:  []string{`"`, "`"},
		CaseSensitive:     true,
	})

	// Python language
	sh.RegisterLanguage(&Language{
		Name:       "python",
		Extensions: []string{".py", ".pyw"},
		Keywords: []string{
			"and", "as", "assert", "break", "class", "continue", "def", "del",
			"elif", "else", "except", "exec", "finally", "for", "from", "global",
			"if", "import", "in", "is", "lambda", "not", "or", "pass", "print",
			"raise", "return", "try", "while", "with", "yield", "async", "await",
		},
		Types: []string{
			"bool", "int", "float", "complex", "str", "bytes", "list", "tuple",
			"dict", "set", "frozenset", "None", "True", "False",
		},
		Operators:         []string{"+", "-", "*", "/", "//", "%", "**", "&", "|", "^", "~", "<<", ">>", "+=", "-=", "*=", "/=", "//=", "%=", "**=", "&=", "|=", "^=", "<<=", ">>=", "==", "!=", "<", ">", "<=", ">=", "and", "or", "not", "is", "in"},
		SingleLineComment: "#",
		StringDelimiters:  []string{`"`, `'`, `"""`, `'''`},
		CaseSensitive:     true,
	})

	// JavaScript language
	sh.RegisterLanguage(&Language{
		Name:       "javascript",
		Extensions: []string{".js", ".jsx", ".mjs"},
		Keywords: []string{
			"break", "case", "catch", "class", "const", "continue", "debugger",
			"default", "delete", "do", "else", "export", "extends", "finally",
			"for", "function", "if", "import", "in", "instanceof", "let", "new",
			"return", "super", "switch", "this", "throw", "try", "typeof", "var",
			"void", "while", "with", "yield", "async", "await", "of",
		},
		Types: []string{
			"undefined", "null", "boolean", "number", "string", "symbol", "object",
			"Array", "Object", "String", "Number", "Boolean", "Date", "RegExp",
			"Function", "Promise",
		},
		Operators:         []string{"+", "-", "*", "/", "%", "++", "--", "=", "+=", "-=", "*=", "/=", "%=", "==", "===", "!=", "!==", "<", ">", "<=", ">=", "&&", "||", "!", "&", "|", "^", "~", "<<", ">>", ">>>", "?", ":"},
		SingleLineComment: "//",
		MultiLineComment:  [2]string{"/*", "*/"},
		StringDelimiters:  []string{`"`, `'`, "`"},
		CaseSensitive:     true,
	})

	// Rust language
	sh.RegisterLanguage(&Language{
		Name:       "rust",
		Extensions: []string{".rs"},
		Keywords: []string{
			"as", "break", "const", "continue", "crate", "else", "enum", "extern",
			"false", "fn", "for", "if", "impl", "in", "let", "loop", "match",
			"mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static",
			"struct", "super", "trait", "true", "type", "unsafe", "use", "where",
			"while", "async", "await", "dyn",
		},
		Types: []string{
			"bool", "char", "i8", "i16", "i32", "i64", "i128", "isize",
			"u8", "u16", "u32", "u64", "u128", "usize", "f32", "f64",
			"str", "String", "Vec", "Option", "Result",
		},
		Operators:         []string{"+", "-", "*", "/", "%", "&", "|", "^", "!", "<<", ">>", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "&&", "||", "==", "!=", "<", ">", "<=", ">=", "=", "->", "=>", "..", "..="},
		SingleLineComment: "//",
		MultiLineComment:  [2]string{"/*", "*/"},
		StringDelimiters:  []string{`"`, `'`},
		CaseSensitive:     true,
	})

	// C/C++ language
	sh.RegisterLanguage(&Language{
		Name:       "cpp",
		Extensions: []string{".c", ".cpp", ".cc", ".cxx", ".h", ".hpp"},
		Keywords: []string{
			"auto", "break", "case", "char", "const", "continue", "default", "do",
			"double", "else", "enum", "extern", "float", "for", "goto", "if",
			"int", "long", "register", "return", "short", "signed", "sizeof",
			"static", "struct", "switch", "typedef", "union", "unsigned", "void",
			"volatile", "while", "class", "private", "protected", "public",
			"virtual", "inline", "template", "typename", "namespace", "using",
		},
		Types: []string{
			"bool", "char", "wchar_t", "short", "int", "long", "float", "double",
			"signed", "unsigned", "void", "size_t", "ptrdiff_t", "nullptr_t",
		},
		Operators:         []string{"+", "-", "*", "/", "%", "++", "--", "=", "+=", "-=", "*=", "/=", "%=", "==", "!=", "<", ">", "<=", ">=", "&&", "||", "!", "&", "|", "^", "~", "<<", ">>", "->", "::", ".", "?", ":"},
		SingleLineComment: "//",
		MultiLineComment:  [2]string{"/*", "*/"},
		StringDelimiters:  []string{`"`, `'`},
		CaseSensitive:     true,
	})

	// JSON language (simplified)
	sh.RegisterLanguage(&Language{
		Name:             "json",
		Extensions:       []string{".json"},
		Keywords:         []string{"true", "false", "null"},
		StringDelimiters: []string{`"`},
		CaseSensitive:    true,
	})
}

// RegisterLanguage registers a new language
func (sh *SyntaxHighlighter) RegisterLanguage(lang *Language) {
	sh.languages[lang.Name] = lang

	// Also register by file extensions
	for _, ext := range lang.Extensions {
		sh.languages[ext] = lang
	}
}

// DetectLanguage detects language from filename or content
func (sh *SyntaxHighlighter) DetectLanguage(filename, content string) *Language {
	// Try to detect by file extension
	for ext := range sh.languages {
		if strings.HasSuffix(strings.ToLower(filename), ext) {
			return sh.languages[ext]
		}
	}

	// Try to detect by content patterns
	if strings.Contains(content, "package main") && strings.Contains(content, "func main()") {
		return sh.languages["go"]
	}
	if strings.Contains(content, "def ") && strings.Contains(content, ":") {
		return sh.languages["python"]
	}
	if strings.Contains(content, "function ") || strings.Contains(content, "const ") || strings.Contains(content, "let ") {
		return sh.languages["javascript"]
	}
	if strings.Contains(content, "fn ") && strings.Contains(content, "->") {
		return sh.languages["rust"]
	}

	return nil // Unknown language
}

// Highlight highlights code and returns colored tokens
func (sh *SyntaxHighlighter) Highlight(code, language string) []Token {
	lang := sh.languages[language]
	if lang == nil {
		// Return plain text tokens
		return []Token{{
			Type:      TokenText,
			Value:     code,
			Start:     0,
			End:       len(code),
			ColorType: "code",
		}}
	}

	return sh.tokenize(code, lang)
}

// HighlightToString highlights code and returns a colored string
func (sh *SyntaxHighlighter) HighlightToString(code, language string) string {
	tokens := sh.Highlight(code, language)
	var result strings.Builder

	for _, token := range tokens {
		if token.Type == TokenWhitespace {
			result.WriteString(token.Value)
		} else {
			colored := sh.theme.Sprint(token.ColorType, token.Value)
			result.WriteString(colored)
		}
	}

	return result.String()
}

// HighlightLines highlights code and returns lines with colored tokens
func (sh *SyntaxHighlighter) HighlightLines(code, language string) [][]Token {
	tokens := sh.Highlight(code, language)
	lines := make([][]Token, 0)
	currentLine := make([]Token, 0)

	for _, token := range tokens {
		if strings.Contains(token.Value, "\n") {
			// Split token on newlines
			parts := strings.Split(token.Value, "\n")
			for i, part := range parts {
				if part != "" {
					partToken := token
					partToken.Value = part
					currentLine = append(currentLine, partToken)
				}

				if i < len(parts)-1 {
					// End of line
					lines = append(lines, currentLine)
					currentLine = make([]Token, 0)
				}
			}
		} else {
			currentLine = append(currentLine, token)
		}
	}

	// Add final line if not empty
	if len(currentLine) > 0 {
		lines = append(lines, currentLine)
	}

	return lines
}

// tokenize tokenizes code according to language rules
func (sh *SyntaxHighlighter) tokenize(code string, lang *Language) []Token {
	tokens := make([]Token, 0)
	position := 0

	for position < len(code) {
		// Skip whitespace (but preserve it)
		if unicode.IsSpace(rune(code[position])) {
			start := position
			for position < len(code) && unicode.IsSpace(rune(code[position])) {
				position++
			}
			tokens = append(tokens, Token{
				Type:      TokenWhitespace,
				Value:     code[start:position],
				Start:     start,
				End:       position,
				ColorType: "code",
			})
			continue
		}

		// Try to match multi-line comments
		if lang.MultiLineComment[0] != "" {
			if strings.HasPrefix(code[position:], lang.MultiLineComment[0]) {
				start := position
				position += len(lang.MultiLineComment[0])

				// Find end of comment
				endPos := strings.Index(code[position:], lang.MultiLineComment[1])
				if endPos != -1 {
					position += endPos + len(lang.MultiLineComment[1])
				} else {
					position = len(code) // Comment extends to end of file
				}

				tokens = append(tokens, Token{
					Type:      TokenComment,
					Value:     code[start:position],
					Start:     start,
					End:       position,
					ColorType: "comment",
				})
				continue
			}
		}

		// Try to match single-line comments
		if lang.SingleLineComment != "" {
			if strings.HasPrefix(code[position:], lang.SingleLineComment) {
				start := position

				// Find end of line
				for position < len(code) && code[position] != '\n' {
					position++
				}

				tokens = append(tokens, Token{
					Type:      TokenComment,
					Value:     code[start:position],
					Start:     start,
					End:       position,
					ColorType: "comment",
				})
				continue
			}
		}

		// Try to match strings
		matched := false
		for _, delimiter := range lang.StringDelimiters {
			if strings.HasPrefix(code[position:], delimiter) {
				start := position
				position += len(delimiter)

				// Find end of string (handle escaping)
				for position < len(code) {
					if strings.HasPrefix(code[position:], delimiter) {
						position += len(delimiter)
						break
					}
					if code[position] == '\\' && position+1 < len(code) {
						position += 2 // Skip escaped character
					} else {
						position++
					}
				}

				tokens = append(tokens, Token{
					Type:      TokenString,
					Value:     code[start:position],
					Start:     start,
					End:       position,
					ColorType: "string",
				})
				matched = true
				break
			}
		}
		if matched {
			continue
		}

		// Try to match numbers
		if unicode.IsDigit(rune(code[position])) {
			start := position

			// Match integer part
			for position < len(code) && (unicode.IsDigit(rune(code[position])) || code[position] == '_') {
				position++
			}

			// Match decimal part
			if position < len(code) && code[position] == '.' {
				position++
				for position < len(code) && (unicode.IsDigit(rune(code[position])) || code[position] == '_') {
					position++
				}
			}

			// Match scientific notation
			if position < len(code) && (code[position] == 'e' || code[position] == 'E') {
				position++
				if position < len(code) && (code[position] == '+' || code[position] == '-') {
					position++
				}
				for position < len(code) && unicode.IsDigit(rune(code[position])) {
					position++
				}
			}

			tokens = append(tokens, Token{
				Type:      TokenNumber,
				Value:     code[start:position],
				Start:     start,
				End:       position,
				ColorType: "number",
			})
			continue
		}

		// Try to match identifiers (keywords, functions, variables, types)
		if unicode.IsLetter(rune(code[position])) || code[position] == '_' {
			start := position

			// Match identifier
			for position < len(code) && (unicode.IsLetter(rune(code[position])) || unicode.IsDigit(rune(code[position])) || code[position] == '_') {
				position++
			}

			identifier := code[start:position]
			tokenType := TokenVariable
			colorType := "variable"

			// Check if it's a keyword
			if sh.isKeyword(identifier, lang) {
				tokenType = TokenKeyword
				colorType = "keyword"
			} else if sh.isType(identifier, lang) {
				tokenType = TokenType
				colorType = "type"
			} else if position < len(code) && code[position] == '(' {
				// Likely a function call
				tokenType = TokenFunction
				colorType = "function"
			}

			tokens = append(tokens, Token{
				Type:      tokenType,
				Value:     identifier,
				Start:     start,
				End:       position,
				ColorType: colorType,
			})
			continue
		}

		// Try to match operators
		matched = false
		for _, operator := range lang.Operators {
			if strings.HasPrefix(code[position:], operator) {
				tokens = append(tokens, Token{
					Type:      TokenOperator,
					Value:     operator,
					Start:     position,
					End:       position + len(operator),
					ColorType: "keyword",
				})
				position += len(operator)
				matched = true
				break
			}
		}
		if matched {
			continue
		}

		// Default: treat as delimiter or unknown character
		tokens = append(tokens, Token{
			Type:      TokenDelimiter,
			Value:     string(code[position]),
			Start:     position,
			End:       position + 1,
			ColorType: "code",
		})
		position++
	}

	return tokens
}

// Helper methods

func (sh *SyntaxHighlighter) isKeyword(word string, lang *Language) bool {
	if !lang.CaseSensitive {
		word = strings.ToLower(word)
	}

	for _, keyword := range lang.Keywords {
		compareWord := keyword
		if !lang.CaseSensitive {
			compareWord = strings.ToLower(compareWord)
		}
		if word == compareWord {
			return true
		}
	}
	return false
}

func (sh *SyntaxHighlighter) isType(word string, lang *Language) bool {
	if !lang.CaseSensitive {
		word = strings.ToLower(word)
	}

	for _, typeName := range lang.Types {
		compareWord := typeName
		if !lang.CaseSensitive {
			compareWord = strings.ToLower(compareWord)
		}
		if word == compareWord {
			return true
		}
	}
	return false
}

// GetSupportedLanguages returns list of supported languages
func (sh *SyntaxHighlighter) GetSupportedLanguages() []string {
	var languages []string
	seen := make(map[string]bool)

	for name, lang := range sh.languages {
		if !seen[lang.Name] {
			languages = append(languages, lang.Name)
			seen[lang.Name] = true
		}
	}

	return languages
}

// GetLanguageInfo returns information about a language
func (sh *SyntaxHighlighter) GetLanguageInfo(name string) *Language {
	return sh.languages[name]
}
