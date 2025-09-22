package cli

import (
	"fmt"
	"regexp"
	"strings"
)

// Command represents a parsed command
type Command struct {
	Name    string            `json:"name"`
	Args    []string          `json:"args"`
	Flags   map[string]string `json:"flags"`
	Options map[string]bool   `json:"options"`
	Raw     string            `json:"raw"`
}

// Parser handles command parsing
type Parser struct {
	aliases map[string]string
}

// NewParser creates a new command parser
func NewParser() *Parser {
	return &Parser{
		aliases: map[string]string{
			"?":  "help",
			"q":  "quit",
			"e":  "exit",
			"c":  "clear",
			"s":  "status",
			"h":  "help",
			"v":  "version",
			"ls": "list",
			"cd": "change-dir",
		},
	}
}

// AddAlias adds a command alias
func (p *Parser) AddAlias(alias, command string) {
	p.aliases[alias] = command
}

// Parse parses a command string into a Command struct
func (p *Parser) Parse(input string) (*Command, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return nil, fmt.Errorf("empty command")
	}

	cmd := &Command{
		Flags:   make(map[string]string),
		Options: make(map[string]bool),
		Raw:     input,
	}

	// Split input respecting quotes
	tokens, err := p.tokenize(input)
	if err != nil {
		return nil, err
	}

	if len(tokens) == 0 {
		return nil, fmt.Errorf("no command found")
	}

	// First token is the command name
	cmdName := tokens[0]

	// Check for aliases
	if alias, exists := p.aliases[cmdName]; exists {
		cmdName = alias
	}

	cmd.Name = cmdName

	// Parse remaining tokens for arguments, flags, and options
	i := 1
	for i < len(tokens) {
		token := tokens[i]

		if strings.HasPrefix(token, "--") {
			// Long option or flag
			key := strings.TrimPrefix(token, "--")

			if strings.Contains(key, "=") {
				// Flag with value: --key=value
				parts := strings.SplitN(key, "=", 2)
				cmd.Flags[parts[0]] = parts[1]
			} else if i+1 < len(tokens) && !strings.HasPrefix(tokens[i+1], "-") {
				// Flag with separate value: --key value
				cmd.Flags[key] = tokens[i+1]
				i++
			} else {
				// Boolean option: --option
				cmd.Options[key] = true
			}
		} else if strings.HasPrefix(token, "-") && len(token) > 1 {
			// Short option(s)
			shortOpts := strings.TrimPrefix(token, "-")

			if len(shortOpts) == 1 && i+1 < len(tokens) && !strings.HasPrefix(tokens[i+1], "-") {
				// Single short flag with value: -k value
				cmd.Flags[shortOpts] = tokens[i+1]
				i++
			} else {
				// Multiple short options: -abc
				for _, opt := range shortOpts {
					cmd.Options[string(opt)] = true
				}
			}
		} else {
			// Regular argument
			cmd.Args = append(cmd.Args, token)
		}

		i++
	}

	return cmd, nil
}

// tokenize splits input into tokens respecting quotes
func (p *Parser) tokenize(input string) ([]string, error) {
	var tokens []string
	var current strings.Builder
	var inQuotes bool
	var quoteChar rune

	runes := []rune(input)

	for i := 0; i < len(runes); i++ {
		r := runes[i]

		switch {
		case !inQuotes && (r == '"' || r == '\''):
			// Start of quoted string
			inQuotes = true
			quoteChar = r

		case inQuotes && r == quoteChar:
			// End of quoted string
			inQuotes = false
			quoteChar = 0

		case inQuotes && r == '\\' && i+1 < len(runes):
			// Escaped character in quotes
			next := runes[i+1]
			if next == quoteChar || next == '\\' {
				current.WriteRune(next)
				i++ // Skip the next character
			} else {
				current.WriteRune(r)
			}

		case !inQuotes && (r == ' ' || r == '\t'):
			// Whitespace outside quotes
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}

		default:
			// Regular character
			current.WriteRune(r)
		}
	}

	// Check for unclosed quotes
	if inQuotes {
		return nil, fmt.Errorf("unclosed quote character: %c", quoteChar)
	}

	// Add final token if any
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	return tokens, nil
}

// ParseQuery parses a search or AI query with special syntax
func (p *Parser) ParseQuery(input string) (*QueryCommand, error) {
	query := &QueryCommand{
		Raw:        input,
		Filters:    make(map[string]string),
		Options:    make(map[string]bool),
		Parameters: make(map[string]interface{}),
	}

	// Extract quoted strings as the main query
	quotedRegex := regexp.MustCompile(`"([^"]*)"`)
	quoted := quotedRegex.FindAllStringSubmatch(input, -1)

	if len(quoted) > 0 {
		query.Query = quoted[0][1]
		// Remove quoted part from input for further parsing
		input = quotedRegex.ReplaceAllString(input, "")
	} else {
		// No quotes, take everything as query for now
		words := strings.Fields(input)
		var queryWords []string

		for _, word := range words {
			if !strings.HasPrefix(word, "--") && !strings.HasPrefix(word, "-") {
				queryWords = append(queryWords, word)
			}
		}

		query.Query = strings.Join(queryWords, " ")
	}

	// Parse filters and options from remaining input
	filterRegex := regexp.MustCompile(`--(\w+)(?:=(\S+)|\s+(\S+))?`)
	matches := filterRegex.FindAllStringSubmatch(input, -1)

	for _, match := range matches {
		key := match[1]
		var value string

		if match[2] != "" {
			value = match[2] // --key=value
		} else if match[3] != "" {
			value = match[3] // --key value
		}

		if value != "" {
			query.Filters[key] = value
		} else {
			query.Options[key] = true
		}
	}

	return query, nil
}

// QueryCommand represents a parsed query command
type QueryCommand struct {
	Query      string                 `json:"query"`
	Filters    map[string]string      `json:"filters"`
	Options    map[string]bool        `json:"options"`
	Parameters map[string]interface{} `json:"parameters"`
	Raw        string                 `json:"raw"`
}

// ValidationError represents a command validation error
type ValidationError struct {
	Command string
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("validation error in command '%s', field '%s': %s", e.Command, e.Field, e.Message)
}

// ValidateCommand validates a parsed command
func (p *Parser) ValidateCommand(cmd *Command) error {
	switch cmd.Name {
	case "search":
		if len(cmd.Args) == 0 && cmd.Raw == "search" {
			return &ValidationError{cmd.Name, "query", "search query is required"}
		}
	case "generate":
		if len(cmd.Args) == 0 {
			return &ValidationError{cmd.Name, "description", "code description is required"}
		}
	case "explain":
		if len(cmd.Args) == 0 {
			return &ValidationError{cmd.Name, "code", "code to explain is required"}
		}
	case "config":
		if len(cmd.Args) > 0 && cmd.Args[0] == "set" && len(cmd.Args) < 3 {
			return &ValidationError{cmd.Name, "arguments", "config set requires key and value"}
		}
	}

	return nil
}

// GetCommandHelp returns help text for a command
func (p *Parser) GetCommandHelp(commandName string) string {
	helpTexts := map[string]string{
		"help":     "Show available commands or help for specific command\nUsage: help [command]",
		"search":   "Search through indexed code\nUsage: search <query> [--type function] [--file *.go]",
		"generate": "Generate code from description\nUsage: generate <description> [--lang go] [--tests]",
		"explain":  "Explain how code works\nUsage: explain <code> [--detail brief]",
		"index":    "Index a project directory\nUsage: index [path]",
		"analyze":  "Analyze code quality and patterns\nUsage: analyze [file]",
		"config":   "Show or modify configuration\nUsage: config [show|set|get] [key] [value]",
		"status":   "Show current system status\nUsage: status",
		"version":  "Show version information\nUsage: version",
		"history":  "Show command history\nUsage: history [--clear]",
		"quit":     "Exit the application\nUsage: quit",
		"clear":    "Clear the screen\nUsage: clear",
	}

	if help, exists := helpTexts[commandName]; exists {
		return help
	}

	return fmt.Sprintf("No help available for command: %s", commandName)
}
