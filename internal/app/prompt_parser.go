package app

import (
	"fmt"
	"regexp"
	"strings"
)

type ParsedPrompt struct {
	RawInput    string            `json:"raw_input"`
	Intent      string            `json:"intent"`
	Entities    map[string]string `json:"entities"`
	Parameters  map[string]string `json:"parameters"`
	Context     string            `json:"context"`
	IsSystemCmd bool              `json:"is_system_cmd"`
	Confidence  float64           `json:"confidence"`
}

type PromptParser struct {
	patterns map[string]*PromptPattern
}

type PromptPattern struct {
	Regex      *regexp.Regexp
	Intent     string
	Confidence float64
	Extractor  func(string) map[string]string
}

func NewPromptParser() *PromptParser {
	parser := &PromptParser{
		patterns: make(map[string]*PromptPattern),
	}

	parser.initializePatterns()
	return parser
}

func (pp *PromptParser) Parse(input string) *ParsedPrompt {
	input = strings.TrimSpace(input)

	// Check if it's a system command
	if pp.isSystemCommand(input) {
		return &ParsedPrompt{
			RawInput:    input,
			Intent:      "system_command",
			IsSystemCmd: true,
			Confidence:  1.0,
		}
	}

	// Try to match patterns
	bestMatch := pp.findBestMatch(input)

	return bestMatch
}

func (pp *PromptParser) initializePatterns() {
	patterns := []*PromptPattern{
		// Code Search Patterns
		{
			Regex:      regexp.MustCompile(`(?i)find.*function.*(\w+)|search.*for.*(\w+)|locate.*(\w+)`),
			Intent:     "search_code",
			Confidence: 0.9,
			Extractor:  pp.extractSearchTerms,
		},
		{
			Regex:      regexp.MustCompile(`(?i)show.*dependencies|what.*depends.*on|imports.*of`),
			Intent:     "analyze_dependencies",
			Confidence: 0.85,
			Extractor:  pp.extractDependencyTerms,
		},
		{
			Regex:      regexp.MustCompile(`(?i)how.*does.*work|explain.*(\w+)|what.*is.*(\w+)`),
			Intent:     "explain_code",
			Confidence: 0.8,
			Extractor:  pp.extractExplanationTerms,
		},

		// Code Generation Patterns
		{
			Regex:      regexp.MustCompile(`(?i)create.*function|generate.*(\w+)|implement.*(\w+)|write.*(\w+)`),
			Intent:     "generate_code",
			Confidence: 0.9,
			Extractor:  pp.extractGenerationTerms,
		},
		{
			Regex:      regexp.MustCompile(`(?i)add.*test|create.*test|generate.*test|write.*test`),
			Intent:     "generate_tests",
			Confidence: 0.95,
			Extractor:  pp.extractTestTerms,
		},
		{
			Regex:      regexp.MustCompile(`(?i)refactor.*|improve.*|optimize.*|clean.*up`),
			Intent:     "refactor_code",
			Confidence: 0.85,
			Extractor:  pp.extractRefactorTerms,
		},

		// Documentation Patterns
		{
			Regex:      regexp.MustCompile(`(?i)document.*|add.*comment|generate.*doc|write.*documentation`),
			Intent:     "generate_documentation",
			Confidence: 0.9,
			Extractor:  pp.extractDocumentationTerms,
		},

		// Bug Fixing Patterns
		{
			Regex:      regexp.MustCompile(`(?i)fix.*bug|solve.*problem|debug.*|error.*in`),
			Intent:     "fix_bug",
			Confidence: 0.85,
			Extractor:  pp.extractBugTerms,
		},

		// Review Patterns
		{
			Regex:      regexp.MustCompile(`(?i)review.*code|check.*quality|analyze.*security|validate`),
			Intent:     "review_code",
			Confidence: 0.8,
			Extractor:  pp.extractReviewTerms,
		},
	}

	for i, pattern := range patterns {
		pp.patterns[fmt.Sprintf("pattern_%d", i)] = pattern
	}
}

func (pp *PromptParser) isSystemCommand(input string) bool {
	systemCommands := []string{
		"help", "status", "config", "clear", "version", "quit", "exit",
	}

	inputLower := strings.ToLower(input)

	for _, cmd := range systemCommands {
		if inputLower == cmd {
			return true
		}
	}

	return strings.HasPrefix(inputLower, "set ")
}

func (pp *PromptParser) findBestMatch(input string) *ParsedPrompt {
	bestMatch := &ParsedPrompt{
		RawInput:   input,
		Intent:     "unknown",
		Entities:   make(map[string]string),
		Parameters: make(map[string]string),
		Confidence: 0.0,
	}

	// Normalize to lowercase for matching
	inputLower := strings.ToLower(input)

	for _, pattern := range pp.patterns {
		if pattern.Regex.MatchString(inputLower) {
			confidence := pattern.Confidence

			// Extract entities using pattern's extractor (pass raw input if case matters)
			entities := pattern.Extractor(input)

			if confidence > bestMatch.Confidence {
				bestMatch.Intent = pattern.Intent
				bestMatch.Confidence = confidence
				bestMatch.Entities = entities
				bestMatch.Context = pp.extractContext(inputLower)
			}
		}
	}

	return bestMatch
}

func (pp *PromptParser) extractContext(input string) string {
	// Remove common words and extract meaningful context
	stopWords := []string{"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
	words := strings.Fields(strings.ToLower(input))

	var contextWords []string
	for _, word := range words {
		isStopWord := false
		for _, stopWord := range stopWords {
			if word == stopWord {
				isStopWord = true
				break
			}
		}
		if !isStopWord && len(word) > 2 {
			contextWords = append(contextWords, word)
		}
	}

	return strings.Join(contextWords, " ")
}

// Entity extraction functions
func (pp *PromptParser) extractSearchTerms(input string) map[string]string {
	entities := make(map[string]string)

	// Extract function names, file names, etc.
	functionRegex := regexp.MustCompile(`(?i)function\s+(\w+)|(\w+)\s+function`)
	if matches := functionRegex.FindStringSubmatch(input); len(matches) > 1 {
		for i := 1; i < len(matches); i++ {
			if matches[i] != "" {
				entities["function_name"] = matches[i]
				break
			}
		}
	}

	// Extract file patterns
	fileRegex := regexp.MustCompile(`(\w+\.go|\w+\.py|\w+\.js)`)
	if matches := fileRegex.FindStringSubmatch(input); len(matches) > 1 {
		entities["file_name"] = matches[1]
	}

	return entities
}

func (pp *PromptParser) extractDependencyTerms(input string) map[string]string {
	entities := make(map[string]string)

	// Extract module or package names
	moduleRegex := regexp.MustCompile(`(?i)of\s+(\w+)|(\w+\.go|\w+)`)
	if matches := moduleRegex.FindStringSubmatch(input); len(matches) > 1 {
		for i := 1; i < len(matches); i++ {
			if matches[i] != "" {
				entities["module_name"] = matches[i]
				break
			}
		}
	}

	return entities
}

func (pp *PromptParser) extractExplanationTerms(input string) map[string]string {
	entities := make(map[string]string)

	// Extract what needs to be explained
	explainRegex := regexp.MustCompile(`(?i)explain\s+(\w+)|how\s+(\w+)\s+work|what\s+is\s+(\w+)`)
	if matches := explainRegex.FindStringSubmatch(input); len(matches) > 1 {
		for i := 1; i < len(matches); i++ {
			if matches[i] != "" {
				entities["explain_target"] = matches[i]
				break
			}
		}
	}

	return entities
}

func (pp *PromptParser) extractGenerationTerms(input string) map[string]string {
	entities := make(map[string]string)

	// Extract what type of code to generate
	generateRegex := regexp.MustCompile(`(?i)create\s+(\w+)|generate\s+(\w+)|implement\s+(\w+)|write\s+(\w+)`)
	if matches := generateRegex.FindStringSubmatch(input); len(matches) > 1 {
		for i := 1; i < len(matches); i++ {
			if matches[i] != "" {
				entities["generation_type"] = matches[i]
				break
			}
		}
	}

	// Extract specific patterns like "REST API", "HTTP handler", etc.
	patternRegex := regexp.MustCompile(`(?i)(REST\s+API|HTTP\s+handler|database\s+model|authentication|middleware)`)
	if matches := patternRegex.FindStringSubmatch(input); len(matches) > 1 {
		entities["code_pattern"] = matches[1]
	}

	return entities
}

func (pp *PromptParser) extractTestTerms(input string) map[string]string {
	entities := make(map[string]string)

	// Extract what to test
	testRegex := regexp.MustCompile(`(?i)test\s+for\s+(\w+)|test\s+(\w+)|(\w+)\s+test`)
	if matches := testRegex.FindStringSubmatch(input); len(matches) > 1 {
		for i := 1; i < len(matches); i++ {
			if matches[i] != "" {
				entities["test_target"] = matches[i]
				break
			}
		}
	}

	// Extract test type
	testTypeRegex := regexp.MustCompile(`(?i)(unit|integration|e2e|benchmark)\s+test`)
	if matches := testTypeRegex.FindStringSubmatch(input); len(matches) > 1 {
		entities["test_type"] = matches[1]
	}

	return entities
}

func (pp *PromptParser) extractRefactorTerms(input string) map[string]string {
	entities := make(map[string]string)

	// Extract what to refactor
	refactorRegex := regexp.MustCompile(`(?i)refactor\s+(\w+)|improve\s+(\w+)|optimize\s+(\w+)`)
	if matches := refactorRegex.FindStringSubmatch(input); len(matches) > 1 {
		for i := 1; i < len(matches); i++ {
			if matches[i] != "" {
				entities["refactor_target"] = matches[i]
				break
			}
		}
	}

	return entities
}

func (pp *PromptParser) extractDocumentationTerms(input string) map[string]string {
	entities := make(map[string]string)

	// Extract what to document
	docRegex := regexp.MustCompile(`(?i)document\s+(\w+)|doc\s+for\s+(\w+)|comment\s+(\w+)`)
	if matches := docRegex.FindStringSubmatch(input); len(matches) > 1 {
		for i := 1; i < len(matches); i++ {
			if matches[i] != "" {
				entities["doc_target"] = matches[i]
				break
			}
		}
	}

	return entities
}

func (pp *PromptParser) extractBugTerms(input string) map[string]string {
	entities := make(map[string]string)

	// Extract bug context
	bugRegex := regexp.MustCompile(`(?i)bug\s+in\s+(\w+)|error\s+in\s+(\w+)|fix\s+(\w+)`)
	if matches := bugRegex.FindStringSubmatch(input); len(matches) > 1 {
		for i := 1; i < len(matches); i++ {
			if matches[i] != "" {
				entities["bug_location"] = matches[i]
				break
			}
		}
	}

	return entities
}

func (pp *PromptParser) extractReviewTerms(input string) map[string]string {
	entities := make(map[string]string)

	// Extract review target
	reviewRegex := regexp.MustCompile(`(?i)review\s+(\w+)|check\s+(\w+)|analyze\s+(\w+)`)
	if matches := reviewRegex.FindStringSubmatch(input); len(matches) > 1 {
		for i := 1; i < len(matches); i++ {
			if matches[i] != "" {
				entities["review_target"] = matches[i]
				break
			}
		}
	}

	// Extract review type
	reviewTypeRegex := regexp.MustCompile(`(?i)(security|performance|quality|style)`)
	if matches := reviewTypeRegex.FindStringSubmatch(input); len(matches) > 1 {
		entities["review_type"] = matches[1]
	}

	return entities
}
