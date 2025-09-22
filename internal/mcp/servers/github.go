package servers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/mcp"
)

// GitHubServer provides GitHub API operations for MCP
type GitHubServer struct {
	token      string
	baseURL    string
	httpClient *http.Client
	rateLimit  *RateLimit
}

// GitHubConfig contains configuration for GitHub operations
type GitHubConfig struct {
	Token   string `json:"token"`
	BaseURL string `json:"base_url,omitempty"` // For GitHub Enterprise
}

// RateLimit tracks GitHub API rate limit
type RateLimit struct {
	Limit     int       `json:"limit"`
	Remaining int       `json:"remaining"`
	Reset     time.Time `json:"reset"`
}

// Repository represents a GitHub repository
type Repository struct {
	ID              int       `json:"id"`
	Name            string    `json:"name"`
	FullName        string    `json:"full_name"`
	Owner           User      `json:"owner"`
	Private         bool      `json:"private"`
	Description     string    `json:"description"`
	Fork            bool      `json:"fork"`
	Language        string    `json:"language"`
	StargazersCount int       `json:"stargazers_count"`
	WatchersCount   int       `json:"watchers_count"`
	ForksCount      int       `json:"forks_count"`
	OpenIssuesCount int       `json:"open_issues_count"`
	DefaultBranch   string    `json:"default_branch"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
	PushedAt        time.Time `json:"pushed_at"`
	CloneURL        string    `json:"clone_url"`
	SSHURL          string    `json:"ssh_url"`
	HTMLURL         string    `json:"html_url"`
}

// Issue represents a GitHub issue
type Issue struct {
	ID          int          `json:"id"`
	Number      int          `json:"number"`
	Title       string       `json:"title"`
	Body        string       `json:"body"`
	User        User         `json:"user"`
	Labels      []Label      `json:"labels"`
	Assignees   []User       `json:"assignees"`
	Milestone   *Milestone   `json:"milestone"`
	State       string       `json:"state"`
	Locked      bool         `json:"locked"`
	Comments    int          `json:"comments"`
	CreatedAt   time.Time    `json:"created_at"`
	UpdatedAt   time.Time    `json:"updated_at"`
	ClosedAt    *time.Time   `json:"closed_at"`
	HTMLURL     string       `json:"html_url"`
	PullRequest *PullRequest `json:"pull_request,omitempty"`
}

// PullRequest represents a GitHub pull request
type PullRequest struct {
	ID             int        `json:"id"`
	Number         int        `json:"number"`
	Title          string     `json:"title"`
	Body           string     `json:"body"`
	User           User       `json:"user"`
	State          string     `json:"state"`
	Head           Branch     `json:"head"`
	Base           Branch     `json:"base"`
	Merged         bool       `json:"merged"`
	Mergeable      *bool      `json:"mergeable"`
	MergeableState string     `json:"mergeable_state"`
	MergedBy       *User      `json:"merged_by"`
	Comments       int        `json:"comments"`
	ReviewComments int        `json:"review_comments"`
	Commits        int        `json:"commits"`
	Additions      int        `json:"additions"`
	Deletions      int        `json:"deletions"`
	ChangedFiles   int        `json:"changed_files"`
	CreatedAt      time.Time  `json:"created_at"`
	UpdatedAt      time.Time  `json:"updated_at"`
	ClosedAt       *time.Time `json:"closed_at"`
	MergedAt       *time.Time `json:"merged_at"`
	HTMLURL        string     `json:"html_url"`
	DiffURL        string     `json:"diff_url"`
	PatchURL       string     `json:"patch_url"`
}

// User represents a GitHub user
type User struct {
	ID        int    `json:"id"`
	Login     string `json:"login"`
	AvatarURL string `json:"avatar_url"`
	HTMLURL   string `json:"html_url"`
	Type      string `json:"type"`
}

// Label represents a GitHub label
type Label struct {
	ID          int    `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Color       string `json:"color"`
}

// Milestone represents a GitHub milestone
type Milestone struct {
	ID           int        `json:"id"`
	Number       int        `json:"number"`
	Title        string     `json:"title"`
	Description  string     `json:"description"`
	State        string     `json:"state"`
	OpenIssues   int        `json:"open_issues"`
	ClosedIssues int        `json:"closed_issues"`
	CreatedAt    time.Time  `json:"created_at"`
	UpdatedAt    time.Time  `json:"updated_at"`
	DueOn        *time.Time `json:"due_on"`
}

// Branch represents a GitHub branch
type Branch struct {
	Label string     `json:"label"`
	Ref   string     `json:"ref"`
	SHA   string     `json:"sha"`
	Repo  Repository `json:"repo"`
}

// Release represents a GitHub release
type Release struct {
	ID              int       `json:"id"`
	TagName         string    `json:"tag_name"`
	TargetCommitish string    `json:"target_commitish"`
	Name            string    `json:"name"`
	Body            string    `json:"body"`
	Draft           bool      `json:"draft"`
	Prerelease      bool      `json:"prerelease"`
	CreatedAt       time.Time `json:"created_at"`
	PublishedAt     time.Time `json:"published_at"`
	Author          User      `json:"author"`
	HTMLURL         string    `json:"html_url"`
	TarballURL      string    `json:"tarball_url"`
	ZipballURL      string    `json:"zipball_url"`
}

// SearchResult represents search results
type SearchResult struct {
	TotalCount        int           `json:"total_count"`
	IncompleteResults bool          `json:"incomplete_results"`
	Items             []interface{} `json:"items"`
}

// FileContent represents file content from GitHub
type FileContent struct {
	Name        string `json:"name"`
	Path        string `json:"path"`
	SHA         string `json:"sha"`
	Size        int    `json:"size"`
	URL         string `json:"url"`
	HTMLURL     string `json:"html_url"`
	GitURL      string `json:"git_url"`
	DownloadURL string `json:"download_url"`
	Type        string `json:"type"`
	Content     string `json:"content"`
	Encoding    string `json:"encoding"`
}

// NewGitHubServer creates a new GitHub server
func NewGitHubServer(config GitHubConfig) *GitHubServer {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "https://api.github.com"
	}

	return &GitHubServer{
		token:   config.Token,
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		rateLimit: &RateLimit{},
	}
}

// GetTools returns available GitHub tools
func (gs *GitHubServer) GetTools() []mcp.MCPTool {
	return []mcp.MCPTool{
		{
			Name:        "github_get_repo",
			Description: "Get information about a GitHub repository",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "Repository owner (username or organization)",
					},
					"repo": map[string]interface{}{
						"type":        "string",
						"description": "Repository name",
					},
				},
				"required": []string{"owner", "repo"},
			},
		},
		{
			Name:        "github_list_repos",
			Description: "List repositories for a user or organization",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "Username or organization name",
					},
					"type": map[string]interface{}{
						"type":        "string",
						"description": "Repository type (all, owner, public, private, member)",
						"default":     "all",
					},
					"sort": map[string]interface{}{
						"type":        "string",
						"description": "Sort order (created, updated, pushed, full_name)",
						"default":     "updated",
					},
					"direction": map[string]interface{}{
						"type":        "string",
						"description": "Sort direction (asc, desc)",
						"default":     "desc",
					},
					"per_page": map[string]interface{}{
						"type":        "integer",
						"description": "Results per page (1-100)",
						"default":     30,
					},
				},
				"required": []string{"owner"},
			},
		},
		{
			Name:        "github_create_issue",
			Description: "Create a new issue in a repository",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "Repository owner",
					},
					"repo": map[string]interface{}{
						"type":        "string",
						"description": "Repository name",
					},
					"title": map[string]interface{}{
						"type":        "string",
						"description": "Issue title",
					},
					"body": map[string]interface{}{
						"type":        "string",
						"description": "Issue body/description",
					},
					"labels": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Labels to apply to the issue",
					},
					"assignees": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Users to assign to the issue",
					},
					"milestone": map[string]interface{}{
						"type":        "integer",
						"description": "Milestone number",
					},
				},
				"required": []string{"owner", "repo", "title"},
			},
		},
		{
			Name:        "github_list_issues",
			Description: "List issues in a repository",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "Repository owner",
					},
					"repo": map[string]interface{}{
						"type":        "string",
						"description": "Repository name",
					},
					"state": map[string]interface{}{
						"type":        "string",
						"description": "Issue state (open, closed, all)",
						"default":     "open",
					},
					"labels": map[string]interface{}{
						"type":        "string",
						"description": "Comma-separated list of labels",
					},
					"assignee": map[string]interface{}{
						"type":        "string",
						"description": "Username of assignee",
					},
					"sort": map[string]interface{}{
						"type":        "string",
						"description": "Sort by (created, updated, comments)",
						"default":     "created",
					},
					"direction": map[string]interface{}{
						"type":        "string",
						"description": "Sort direction (asc, desc)",
						"default":     "desc",
					},
					"per_page": map[string]interface{}{
						"type":        "integer",
						"description": "Results per page (1-100)",
						"default":     30,
					},
				},
				"required": []string{"owner", "repo"},
			},
		},
		{
			Name:        "github_get_issue",
			Description: "Get details of a specific issue",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "Repository owner",
					},
					"repo": map[string]interface{}{
						"type":        "string",
						"description": "Repository name",
					},
					"issue_number": map[string]interface{}{
						"type":        "integer",
						"description": "Issue number",
					},
				},
				"required": []string{"owner", "repo", "issue_number"},
			},
		},
		{
			Name:        "github_update_issue",
			Description: "Update an existing issue",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "Repository owner",
					},
					"repo": map[string]interface{}{
						"type":        "string",
						"description": "Repository name",
					},
					"issue_number": map[string]interface{}{
						"type":        "integer",
						"description": "Issue number",
					},
					"title": map[string]interface{}{
						"type":        "string",
						"description": "Issue title",
					},
					"body": map[string]interface{}{
						"type":        "string",
						"description": "Issue body",
					},
					"state": map[string]interface{}{
						"type":        "string",
						"description": "Issue state (open, closed)",
					},
					"labels": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Labels",
					},
				},
				"required": []string{"owner", "repo", "issue_number"},
			},
		},
		{
			Name:        "github_create_pr",
			Description: "Create a new pull request",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "Repository owner",
					},
					"repo": map[string]interface{}{
						"type":        "string",
						"description": "Repository name",
					},
					"title": map[string]interface{}{
						"type":        "string",
						"description": "Pull request title",
					},
					"body": map[string]interface{}{
						"type":        "string",
						"description": "Pull request description",
					},
					"head": map[string]interface{}{
						"type":        "string",
						"description": "Source branch",
					},
					"base": map[string]interface{}{
						"type":        "string",
						"description": "Target branch",
					},
					"draft": map[string]interface{}{
						"type":        "boolean",
						"description": "Create as draft PR",
						"default":     false,
					},
				},
				"required": []string{"owner", "repo", "title", "head", "base"},
			},
		},
		{
			Name:        "github_list_prs",
			Description: "List pull requests in a repository",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "Repository owner",
					},
					"repo": map[string]interface{}{
						"type":        "string",
						"description": "Repository name",
					},
					"state": map[string]interface{}{
						"type":        "string",
						"description": "PR state (open, closed, all)",
						"default":     "open",
					},
					"base": map[string]interface{}{
						"type":        "string",
						"description": "Base branch name",
					},
					"head": map[string]interface{}{
						"type":        "string",
						"description": "Head branch name",
					},
					"sort": map[string]interface{}{
						"type":        "string",
						"description": "Sort by (created, updated, popularity)",
						"default":     "created",
					},
					"direction": map[string]interface{}{
						"type":        "string",
						"description": "Sort direction (asc, desc)",
						"default":     "desc",
					},
				},
				"required": []string{"owner", "repo"},
			},
		},
		{
			Name:        "github_search_repos",
			Description: "Search for repositories on GitHub",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Search query",
					},
					"sort": map[string]interface{}{
						"type":        "string",
						"description": "Sort by (stars, forks, help-wanted-issues, updated)",
					},
					"order": map[string]interface{}{
						"type":        "string",
						"description": "Sort order (asc, desc)",
						"default":     "desc",
					},
					"per_page": map[string]interface{}{
						"type":        "integer",
						"description": "Results per page (1-100)",
						"default":     30,
					},
				},
				"required": []string{"query"},
			},
		},
		{
			Name:        "github_search_issues",
			Description: "Search for issues and pull requests",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Search query",
					},
					"sort": map[string]interface{}{
						"type":        "string",
						"description": "Sort by (comments, reactions, author-date, committer-date, created, updated)",
					},
					"order": map[string]interface{}{
						"type":        "string",
						"description": "Sort order (asc, desc)",
						"default":     "desc",
					},
					"per_page": map[string]interface{}{
						"type":        "integer",
						"description": "Results per page (1-100)",
						"default":     30,
					},
				},
				"required": []string{"query"},
			},
		},
		{
			Name:        "github_search_code",
			Description: "Search for code in repositories",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Search query",
					},
					"sort": map[string]interface{}{
						"type":        "string",
						"description": "Sort by (indexed)",
					},
					"order": map[string]interface{}{
						"type":        "string",
						"description": "Sort order (asc, desc)",
						"default":     "desc",
					},
					"per_page": map[string]interface{}{
						"type":        "integer",
						"description": "Results per page (1-100)",
						"default":     30,
					},
				},
				"required": []string{"query"},
			},
		},
		{
			Name:        "github_get_file",
			Description: "Get file content from a repository",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "Repository owner",
					},
					"repo": map[string]interface{}{
						"type":        "string",
						"description": "Repository name",
					},
					"path": map[string]interface{}{
						"type":        "string",
						"description": "File path",
					},
					"ref": map[string]interface{}{
						"type":        "string",
						"description": "Branch, tag, or commit SHA",
						"default":     "main",
					},
				},
				"required": []string{"owner", "repo", "path"},
			},
		},
		{
			Name:        "github_list_branches",
			Description: "List branches in a repository",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "Repository owner",
					},
					"repo": map[string]interface{}{
						"type":        "string",
						"description": "Repository name",
					},
					"protected": map[string]interface{}{
						"type":        "boolean",
						"description": "Only protected branches",
					},
					"per_page": map[string]interface{}{
						"type":        "integer",
						"description": "Results per page (1-100)",
						"default":     30,
					},
				},
				"required": []string{"owner", "repo"},
			},
		},
		{
			Name:        "github_list_releases",
			Description: "List releases for a repository",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "Repository owner",
					},
					"repo": map[string]interface{}{
						"type":        "string",
						"description": "Repository name",
					},
					"per_page": map[string]interface{}{
						"type":        "integer",
						"description": "Results per page (1-100)",
						"default":     30,
					},
				},
				"required": []string{"owner", "repo"},
			},
		},
	}
}

// ExecuteTool executes a GitHub tool
func (gs *GitHubServer) ExecuteTool(ctx context.Context, toolName string, input map[string]interface{}) (interface{}, error) {
	if gs.token == "" {
		return nil, fmt.Errorf("GitHub token is required")
	}

	switch toolName {
	case "github_get_repo":
		return gs.getRepository(input)
	case "github_list_repos":
		return gs.listRepositories(input)
	case "github_create_issue":
		return gs.createIssue(input)
	case "github_list_issues":
		return gs.listIssues(input)
	case "github_get_issue":
		return gs.getIssue(input)
	case "github_update_issue":
		return gs.updateIssue(input)
	case "github_create_pr":
		return gs.createPullRequest(input)
	case "github_list_prs":
		return gs.listPullRequests(input)
	case "github_search_repos":
		return gs.searchRepositories(input)
	case "github_search_issues":
		return gs.searchIssues(input)
	case "github_search_code":
		return gs.searchCode(input)
	case "github_get_file":
		return gs.getFileContent(input)
	case "github_list_branches":
		return gs.listBranches(input)
	case "github_list_releases":
		return gs.listReleases(input)
	default:
		return nil, fmt.Errorf("unknown tool: %s", toolName)
	}
}

// getRepository gets repository information
func (gs *GitHubServer) getRepository(input map[string]interface{}) (interface{}, error) {
	owner, _ := input["owner"].(string)
	repo, _ := input["repo"].(string)

	if owner == "" || repo == "" {
		return nil, fmt.Errorf("owner and repo are required")
	}

	url := fmt.Sprintf("%s/repos/%s/%s", gs.baseURL, owner, repo)

	var repository Repository
	if err := gs.makeRequest("GET", url, nil, &repository); err != nil {
		return nil, err
	}

	return repository, nil
}

// listRepositories lists repositories for a user/organization
func (gs *GitHubServer) listRepositories(input map[string]interface{}) (interface{}, error) {
	owner, _ := input["owner"].(string)
	if owner == "" {
		return nil, fmt.Errorf("owner is required")
	}

	repoType, _ := input["type"].(string)
	if repoType == "" {
		repoType = "all"
	}

	sort, _ := input["sort"].(string)
	if sort == "" {
		sort = "updated"
	}

	direction, _ := input["direction"].(string)
	if direction == "" {
		direction = "desc"
	}

	perPage := 30
	if pp, ok := input["per_page"].(float64); ok {
		perPage = int(pp)
	}

	url := fmt.Sprintf("%s/users/%s/repos?type=%s&sort=%s&direction=%s&per_page=%d",
		gs.baseURL, owner, repoType, sort, direction, perPage)

	var repositories []Repository
	if err := gs.makeRequest("GET", url, nil, &repositories); err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"repositories": repositories,
		"count":        len(repositories),
	}, nil
}

// createIssue creates a new issue
func (gs *GitHubServer) createIssue(input map[string]interface{}) (interface{}, error) {
	owner, _ := input["owner"].(string)
	repo, _ := input["repo"].(string)
	title, _ := input["title"].(string)

	if owner == "" || repo == "" || title == "" {
		return nil, fmt.Errorf("owner, repo, and title are required")
	}

	payload := map[string]interface{}{
		"title": title,
	}

	if body, ok := input["body"].(string); ok {
		payload["body"] = body
	}

	if labels, ok := input["labels"].([]interface{}); ok {
		labelStrings := make([]string, len(labels))
		for i, label := range labels {
			if labelStr, ok := label.(string); ok {
				labelStrings[i] = labelStr
			}
		}
		payload["labels"] = labelStrings
	}

	if assignees, ok := input["assignees"].([]interface{}); ok {
		assigneeStrings := make([]string, len(assignees))
		for i, assignee := range assignees {
			if assigneeStr, ok := assignee.(string); ok {
				assigneeStrings[i] = assigneeStr
			}
		}
		payload["assignees"] = assigneeStrings
	}

	if milestone, ok := input["milestone"].(float64); ok {
		payload["milestone"] = int(milestone)
	}

	url := fmt.Sprintf("%s/repos/%s/%s/issues", gs.baseURL, owner, repo)

	var issue Issue
	if err := gs.makeRequest("POST", url, payload, &issue); err != nil {
		return nil, err
	}

	return issue, nil
}

// listIssues lists issues in a repository
func (gs *GitHubServer) listIssues(input map[string]interface{}) (interface{}, error) {
	owner, _ := input["owner"].(string)
	repo, _ := input["repo"].(string)

	if owner == "" || repo == "" {
		return nil, fmt.Errorf("owner and repo are required")
	}

	state, _ := input["state"].(string)
	if state == "" {
		state = "open"
	}

	sort, _ := input["sort"].(string)
	if sort == "" {
		sort = "created"
	}

	direction, _ := input["direction"].(string)
	if direction == "" {
		direction = "desc"
	}

	perPage := 30
	if pp, ok := input["per_page"].(float64); ok {
		perPage = int(pp)
	}

	url := fmt.Sprintf("%s/repos/%s/%s/issues?state=%s&sort=%s&direction=%s&per_page=%d",
		gs.baseURL, owner, repo, state, sort, direction, perPage)

	if labels, ok := input["labels"].(string); ok && labels != "" {
		url += fmt.Sprintf("&labels=%s", labels)
	}

	if assignee, ok := input["assignee"].(string); ok && assignee != "" {
		url += fmt.Sprintf("&assignee=%s", assignee)
	}

	var issues []Issue
	if err := gs.makeRequest("GET", url, nil, &issues); err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"issues": issues,
		"count":  len(issues),
	}, nil
}

// getIssue gets a specific issue
func (gs *GitHubServer) getIssue(input map[string]interface{}) (interface{}, error) {
	owner, _ := input["owner"].(string)
	repo, _ := input["repo"].(string)
	issueNumber, _ := input["issue_number"].(float64)

	if owner == "" || repo == "" || issueNumber == 0 {
		return nil, fmt.Errorf("owner, repo, and issue_number are required")
	}

	url := fmt.Sprintf("%s/repos/%s/%s/issues/%d", gs.baseURL, owner, repo, int(issueNumber))

	var issue Issue
	if err := gs.makeRequest("GET", url, nil, &issue); err != nil {
		return nil, err
	}

	return issue, nil
}

// updateIssue updates an existing issue
func (gs *GitHubServer) updateIssue(input map[string]interface{}) (interface{}, error) {
	owner, _ := input["owner"].(string)
	repo, _ := input["repo"].(string)
	issueNumber, _ := input["issue_number"].(float64)

	if owner == "" || repo == "" || issueNumber == 0 {
		return nil, fmt.Errorf("owner, repo, and issue_number are required")
	}

	payload := make(map[string]interface{})

	if title, ok := input["title"].(string); ok {
		payload["title"] = title
	}

	if body, ok := input["body"].(string); ok {
		payload["body"] = body
	}

	if state, ok := input["state"].(string); ok {
		payload["state"] = state
	}

	if labels, ok := input["labels"].([]interface{}); ok {
		labelStrings := make([]string, len(labels))
		for i, label := range labels {
			if labelStr, ok := label.(string); ok {
				labelStrings[i] = labelStr
			}
		}
		payload["labels"] = labelStrings
	}

	if len(payload) == 0 {
		return nil, fmt.Errorf("at least one field to update is required")
	}

	url := fmt.Sprintf("%s/repos/%s/%s/issues/%d", gs.baseURL, owner, repo, int(issueNumber))

	var issue Issue
	if err := gs.makeRequest("PATCH", url, payload, &issue); err != nil {
		return nil, err
	}

	return issue, nil
}

// createPullRequest creates a new pull request
func (gs *GitHubServer) createPullRequest(input map[string]interface{}) (interface{}, error) {
	owner, _ := input["owner"].(string)
	repo, _ := input["repo"].(string)
	title, _ := input["title"].(string)
	head, _ := input["head"].(string)
	base, _ := input["base"].(string)

	if owner == "" || repo == "" || title == "" || head == "" || base == "" {
		return nil, fmt.Errorf("owner, repo, title, head, and base are required")
	}

	payload := map[string]interface{}{
		"title": title,
		"head":  head,
		"base":  base,
	}

	if body, ok := input["body"].(string); ok {
		payload["body"] = body
	}

	if draft, ok := input["draft"].(bool); ok {
		payload["draft"] = draft
	}

	url := fmt.Sprintf("%s/repos/%s/%s/pulls", gs.baseURL, owner, repo)

	var pr PullRequest
	if err := gs.makeRequest("POST", url, payload, &pr); err != nil {
		return nil, err
	}

	return pr, nil
}

// listPullRequests lists pull requests
func (gs *GitHubServer) listPullRequests(input map[string]interface{}) (interface{}, error) {
	owner, _ := input["owner"].(string)
	repo, _ := input["repo"].(string)

	if owner == "" || repo == "" {
		return nil, fmt.Errorf("owner and repo are required")
	}

	state, _ := input["state"].(string)
	if state == "" {
		state = "open"
	}

	sort, _ := input["sort"].(string)
	if sort == "" {
		sort = "created"
	}

	direction, _ := input["direction"].(string)
	if direction == "" {
		direction = "desc"
	}

	url := fmt.Sprintf("%s/repos/%s/%s/pulls?state=%s&sort=%s&direction=%s",
		gs.baseURL, owner, repo, state, sort, direction)

	if base, ok := input["base"].(string); ok && base != "" {
		url += fmt.Sprintf("&base=%s", base)
	}

	if head, ok := input["head"].(string); ok && head != "" {
		url += fmt.Sprintf("&head=%s", head)
	}

	var prs []PullRequest
	if err := gs.makeRequest("GET", url, nil, &prs); err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"pull_requests": prs,
		"count":         len(prs),
	}, nil
}

// searchRepositories searches for repositories
func (gs *GitHubServer) searchRepositories(input map[string]interface{}) (interface{}, error) {
	query, _ := input["query"].(string)
	if query == "" {
		return nil, fmt.Errorf("query is required")
	}

	url := fmt.Sprintf("%s/search/repositories?q=%s", gs.baseURL, query)

	if sort, ok := input["sort"].(string); ok && sort != "" {
		url += fmt.Sprintf("&sort=%s", sort)
	}

	order, _ := input["order"].(string)
	if order == "" {
		order = "desc"
	}
	url += fmt.Sprintf("&order=%s", order)

	perPage := 30
	if pp, ok := input["per_page"].(float64); ok {
		perPage = int(pp)
	}
	url += fmt.Sprintf("&per_page=%d", perPage)

	var result SearchResult
	if err := gs.makeRequest("GET", url, nil, &result); err != nil {
		return nil, err
	}

	return result, nil
}

// searchIssues searches for issues and pull requests
func (gs *GitHubServer) searchIssues(input map[string]interface{}) (interface{}, error) {
	query, _ := input["query"].(string)
	if query == "" {
		return nil, fmt.Errorf("query is required")
	}

	url := fmt.Sprintf("%s/search/issues?q=%s", gs.baseURL, query)

	if sort, ok := input["sort"].(string); ok && sort != "" {
		url += fmt.Sprintf("&sort=%s", sort)
	}

	order, _ := input["order"].(string)
	if order == "" {
		order = "desc"
	}
	url += fmt.Sprintf("&order=%s", order)

	perPage := 30
	if pp, ok := input["per_page"].(float64); ok {
		perPage = int(pp)
	}
	url += fmt.Sprintf("&per_page=%d", perPage)

	var result SearchResult
	if err := gs.makeRequest("GET", url, nil, &result); err != nil {
		return nil, err
	}

	return result, nil
}

// searchCode searches for code
func (gs *GitHubServer) searchCode(input map[string]interface{}) (interface{}, error) {
	query, _ := input["query"].(string)
	if query == "" {
		return nil, fmt.Errorf("query is required")
	}

	url := fmt.Sprintf("%s/search/code?q=%s", gs.baseURL, query)

	if sort, ok := input["sort"].(string); ok && sort != "" {
		url += fmt.Sprintf("&sort=%s", sort)
	}

	order, _ := input["order"].(string)
	if order == "" {
		order = "desc"
	}
	url += fmt.Sprintf("&order=%s", order)

	perPage := 30
	if pp, ok := input["per_page"].(float64); ok {
		perPage = int(pp)
	}
	url += fmt.Sprintf("&per_page=%d", perPage)

	var result SearchResult
	if err := gs.makeRequest("GET", url, nil, &result); err != nil {
		return nil, err
	}

	return result, nil
}

// getFileContent gets file content from repository
func (gs *GitHubServer) getFileContent(input map[string]interface{}) (interface{}, error) {
	owner, _ := input["owner"].(string)
	repo, _ := input["repo"].(string)
	path, _ := input["path"].(string)

	if owner == "" || repo == "" || path == "" {
		return nil, fmt.Errorf("owner, repo, and path are required")
	}

	url := fmt.Sprintf("%s/repos/%s/%s/contents/%s", gs.baseURL, owner, repo, path)

	if ref, ok := input["ref"].(string); ok && ref != "" {
		url += fmt.Sprintf("?ref=%s", ref)
	}

	var content FileContent
	if err := gs.makeRequest("GET", url, nil, &content); err != nil {
		return nil, err
	}

	return content, nil
}

// listBranches lists repository branches
func (gs *GitHubServer) listBranches(input map[string]interface{}) (interface{}, error) {
	owner, _ := input["owner"].(string)
	repo, _ := input["repo"].(string)

	if owner == "" || repo == "" {
		return nil, fmt.Errorf("owner and repo are required")
	}

	url := fmt.Sprintf("%s/repos/%s/%s/branches", gs.baseURL, owner, repo)

	if protected, ok := input["protected"].(bool); ok {
		url += fmt.Sprintf("?protected=%t", protected)
	}

	perPage := 30
	if pp, ok := input["per_page"].(float64); ok {
		perPage = int(pp)
	}
	url += fmt.Sprintf("&per_page=%d", perPage)

	var branches []Branch
	if err := gs.makeRequest("GET", url, nil, &branches); err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"branches": branches,
		"count":    len(branches),
	}, nil
}

// listReleases lists repository releases
func (gs *GitHubServer) listReleases(input map[string]interface{}) (interface{}, error) {
	owner, _ := input["owner"].(string)
	repo, _ := input["repo"].(string)

	if owner == "" || repo == "" {
		return nil, fmt.Errorf("owner and repo are required")
	}

	perPage := 30
	if pp, ok := input["per_page"].(float64); ok {
		perPage = int(pp)
	}

	url := fmt.Sprintf("%s/repos/%s/%s/releases?per_page=%d", gs.baseURL, owner, repo, perPage)

	var releases []Release
	if err := gs.makeRequest("GET", url, nil, &releases); err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"releases": releases,
		"count":    len(releases),
	}, nil
}

// makeRequest makes an HTTP request to the GitHub API
func (gs *GitHubServer) makeRequest(method, url string, payload interface{}, result interface{}) error {
	var body io.Reader

	if payload != nil {
		jsonData, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("failed to marshal payload: %v", err)
		}
		body = bytes.NewBuffer(jsonData)
	}

	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Authorization", "token "+gs.token)
	req.Header.Set("Accept", "application/vnd.github.v3+json")
	req.Header.Set("User-Agent", "AI-Code-Assistant/1.0")

	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := gs.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("HTTP request failed: %v", err)
	}
	defer resp.Body.Close()

	// Update rate limit info
	gs.updateRateLimit(resp)

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %v", err)
	}

	if resp.StatusCode >= 400 {
		var apiError map[string]interface{}
		if json.Unmarshal(respBody, &apiError) == nil {
			if message, ok := apiError["message"].(string); ok {
				return fmt.Errorf("GitHub API error (%d): %s", resp.StatusCode, message)
			}
		}
		return fmt.Errorf("GitHub API error: %d %s", resp.StatusCode, resp.Status)
	}

	if result != nil {
		if err := json.Unmarshal(respBody, result); err != nil {
			return fmt.Errorf("failed to unmarshal response: %v", err)
		}
	}

	return nil
}

// updateRateLimit updates rate limit information from response headers
func (gs *GitHubServer) updateRateLimit(resp *http.Response) {
	if limit := resp.Header.Get("X-RateLimit-Limit"); limit != "" {
		if l, err := strconv.Atoi(limit); err == nil {
			gs.rateLimit.Limit = l
		}
	}

	if remaining := resp.Header.Get("X-RateLimit-Remaining"); remaining != "" {
		if r, err := strconv.Atoi(remaining); err == nil {
			gs.rateLimit.Remaining = r
		}
	}

	if reset := resp.Header.Get("X-RateLimit-Reset"); reset != "" {
		if r, err := strconv.ParseInt(reset, 10, 64); err == nil {
			gs.rateLimit.Reset = time.Unix(r, 0)
		}
	}
}

// GetRateLimit returns current rate limit status
func (gs *GitHubServer) GetRateLimit() *RateLimit {
	return gs.rateLimit
}
