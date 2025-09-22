package servers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/yourusername/ai-code-assistant/internal/mcp"
)

// DockerServer provides Docker operations for MCP
type DockerServer struct {
	dockerCmd     string
	allowedOps    []string
	restricted    bool
	maxContainers int
}

// DockerConfig contains configuration for Docker operations
type DockerConfig struct {
	DockerCmd     string   `json:"docker_cmd"`     // Path to docker binary
	AllowedOps    []string `json:"allowed_ops"`    // Allowed operations in restricted mode
	Restricted    bool     `json:"restricted"`     // Enable restricted mode
	MaxContainers int      `json:"max_containers"` // Maximum containers to list
}

// Container represents a Docker container
type Container struct {
	ID          string            `json:"id"`
	Names       []string          `json:"names"`
	Image       string            `json:"image"`
	ImageID     string            `json:"image_id"`
	Command     string            `json:"command"`
	Created     time.Time         `json:"created"`
	Status      string            `json:"status"`
	State       string            `json:"state"`
	Ports       []PortBinding     `json:"ports"`
	Labels      map[string]string `json:"labels"`
	Mounts      []Mount           `json:"mounts"`
	NetworkMode string            `json:"network_mode"`
	Size        *ContainerSize    `json:"size,omitempty"`
}

// Image represents a Docker image
type Image struct {
	ID          string            `json:"id"`
	ParentID    string            `json:"parent_id"`
	RepoTags    []string          `json:"repo_tags"`
	RepoDigests []string          `json:"repo_digests"`
	Created     time.Time         `json:"created"`
	Size        int64             `json:"size"`
	VirtualSize int64             `json:"virtual_size"`
	Labels      map[string]string `json:"labels"`
	Containers  int               `json:"containers"`
}

// Network represents a Docker network
type Network struct {
	ID         string                      `json:"id"`
	Name       string                      `json:"name"`
	Created    time.Time                   `json:"created"`
	Scope      string                      `json:"scope"`
	Driver     string                      `json:"driver"`
	EnableIPv6 bool                        `json:"enable_ipv6"`
	Internal   bool                        `json:"internal"`
	Attachable bool                        `json:"attachable"`
	Ingress    bool                        `json:"ingress"`
	ConfigFrom *NetworkConfigFrom          `json:"config_from,omitempty"`
	ConfigOnly bool                        `json:"config_only"`
	Containers map[string]NetworkContainer `json:"containers"`
	Options    map[string]string           `json:"options"`
	Labels     map[string]string           `json:"labels"`
}

// Volume represents a Docker volume
type Volume struct {
	CreatedAt  time.Time              `json:"created_at"`
	Driver     string                 `json:"driver"`
	Labels     map[string]string      `json:"labels"`
	Mountpoint string                 `json:"mountpoint"`
	Name       string                 `json:"name"`
	Options    map[string]string      `json:"options"`
	Scope      string                 `json:"scope"`
	Status     map[string]interface{} `json:"status,omitempty"`
	UsageData  *VolumeUsageData       `json:"usage_data,omitempty"`
}

// PortBinding represents container port bindings
type PortBinding struct {
	PrivatePort int    `json:"private_port"`
	PublicPort  int    `json:"public_port,omitempty"`
	Type        string `json:"type"`
	IP          string `json:"ip,omitempty"`
}

// Mount represents container mounts
type Mount struct {
	Type        string `json:"type"`
	Source      string `json:"source"`
	Destination string `json:"destination"`
	Mode        string `json:"mode,omitempty"`
	RW          bool   `json:"rw"`
	Propagation string `json:"propagation,omitempty"`
}

// ContainerSize represents container size information
type ContainerSize struct {
	RootFsSize int64 `json:"root_fs_size"`
	RwSize     int64 `json:"rw_size"`
}

// NetworkConfigFrom represents network config source
type NetworkConfigFrom struct {
	Network string `json:"network"`
}

// NetworkContainer represents a container in a network
type NetworkContainer struct {
	Name        string `json:"name"`
	EndpointID  string `json:"endpoint_id"`
	MacAddress  string `json:"mac_address"`
	IPv4Address string `json:"ipv4_address"`
	IPv6Address string `json:"ipv6_address"`
}

// VolumeUsageData represents volume usage information
type VolumeUsageData struct {
	Size     int64 `json:"size"`
	RefCount int64 `json:"ref_count"`
}

// DockerStats represents container statistics
type DockerStats struct {
	ContainerID string  `json:"container_id"`
	Name        string  `json:"name"`
	CPUPercent  float64 `json:"cpu_percent"`
	MemUsage    string  `json:"mem_usage"`
	MemPercent  float64 `json:"mem_percent"`
	NetIO       string  `json:"net_io"`
	BlockIO     string  `json:"block_io"`
	PIDs        int     `json:"pids"`
}

// DockerSystemInfo represents Docker system information
type DockerSystemInfo struct {
	ID                 string                 `json:"id"`
	Containers         int                    `json:"containers"`
	ContainersRunning  int                    `json:"containers_running"`
	ContainersPaused   int                    `json:"containers_paused"`
	ContainersStopped  int                    `json:"containers_stopped"`
	Images             int                    `json:"images"`
	Driver             string                 `json:"driver"`
	DriverStatus       [][]string             `json:"driver_status"`
	SystemStatus       [][]string             `json:"system_status,omitempty"`
	Plugins            DockerPlugins          `json:"plugins"`
	MemoryLimit        bool                   `json:"memory_limit"`
	SwapLimit          bool                   `json:"swap_limit"`
	KernelMemory       bool                   `json:"kernel_memory"`
	CPUCfsQuota        bool                   `json:"cpu_cfs_quota"`
	CPUCfsPeriod       bool                   `json:"cpu_cfs_period"`
	CPUShares          bool                   `json:"cpu_shares"`
	CPUSet             bool                   `json:"cpu_set"`
	IPv4Forwarding     bool                   `json:"ipv4_forwarding"`
	BridgeNfIptables   bool                   `json:"bridge_nf_iptables"`
	BridgeNfIP6tables  bool                   `json:"bridge_nf_ip6tables"`
	Debug              bool                   `json:"debug"`
	NFd                int                    `json:"nfd"`
	OomKillDisable     bool                   `json:"oom_kill_disable"`
	NGoroutines        int                    `json:"ngoroutines"`
	SystemTime         time.Time              `json:"system_time"`
	LoggingDriver      string                 `json:"logging_driver"`
	CgroupDriver       string                 `json:"cgroup_driver"`
	NEventsListener    int                    `json:"nevents_listener"`
	KernelVersion      string                 `json:"kernel_version"`
	OperatingSystem    string                 `json:"operating_system"`
	OSType             string                 `json:"os_type"`
	Architecture       string                 `json:"architecture"`
	IndexServerAddress string                 `json:"index_server_address"`
	RegistryConfig     *RegistryServiceConfig `json:"registry_config,omitempty"`
	NCPU               int                    `json:"ncpu"`
	MemTotal           int64                  `json:"mem_total"`
	DockerRootDir      string                 `json:"docker_root_dir"`
	HTTPProxy          string                 `json:"http_proxy"`
	HTTPSProxy         string                 `json:"https_proxy"`
	NoProxy            string                 `json:"no_proxy"`
	Name               string                 `json:"name"`
	Labels             []string               `json:"labels"`
	ExperimentalBuild  bool                   `json:"experimental_build"`
	ServerVersion      string                 `json:"server_version"`
	ClusterStore       string                 `json:"cluster_store,omitempty"`
	ClusterAdvertise   string                 `json:"cluster_advertise,omitempty"`
	Runtimes           map[string]Runtime     `json:"runtimes"`
	DefaultRuntime     string                 `json:"default_runtime"`
	Swarm              SwarmInfo              `json:"swarm"`
	LiveRestoreEnabled bool                   `json:"live_restore_enabled"`
	Isolation          string                 `json:"isolation"`
	InitBinary         string                 `json:"init_binary"`
	ContainerdCommit   Commit                 `json:"containerd_commit"`
	RuncCommit         Commit                 `json:"runc_commit"`
	InitCommit         Commit                 `json:"init_commit"`
	SecurityOptions    []string               `json:"security_options"`
}

// DockerPlugins represents Docker plugins information
type DockerPlugins struct {
	Volume        []string `json:"volume"`
	Network       []string `json:"network"`
	Authorization []string `json:"authorization"`
	Log           []string `json:"log"`
}

// RegistryServiceConfig represents registry service configuration
type RegistryServiceConfig struct {
	AllowNondistributableArtifactsCIDRs     []string              `json:"allow_nondistributable_artifacts_cidrs"`
	AllowNondistributableArtifactsHostnames []string              `json:"allow_nondistributable_artifacts_hostnames"`
	InsecureRegistryCIDRs                   []string              `json:"insecure_registry_cidrs"`
	IndexConfigs                            map[string]*IndexInfo `json:"index_configs"`
	Mirrors                                 []string              `json:"mirrors"`
}

// IndexInfo represents registry index information
type IndexInfo struct {
	Name     string   `json:"name"`
	Mirrors  []string `json:"mirrors"`
	Secure   bool     `json:"secure"`
	Official bool     `json:"official"`
}

// Runtime represents container runtime information
type Runtime struct {
	Path string   `json:"path"`
	Args []string `json:"runtimeArgs"`
}

// SwarmInfo represents Docker Swarm information
type SwarmInfo struct {
	NodeID           string       `json:"node_id"`
	NodeAddr         string       `json:"node_addr"`
	LocalNodeState   string       `json:"local_node_state"`
	ControlAvailable bool         `json:"control_available"`
	Error            string       `json:"error"`
	RemoteManagers   []Peer       `json:"remote_managers"`
	Nodes            int          `json:"nodes"`
	Managers         int          `json:"managers"`
	Cluster          *ClusterInfo `json:"cluster,omitempty"`
}

// Peer represents a swarm peer
type Peer struct {
	NodeID string `json:"node_id"`
	Addr   string `json:"addr"`
}

// ClusterInfo represents swarm cluster information
type ClusterInfo struct {
	ID        string      `json:"id"`
	Version   VersionInfo `json:"version"`
	CreatedAt time.Time   `json:"created_at"`
	UpdatedAt time.Time   `json:"updated_at"`
	Spec      SwarmSpec   `json:"spec"`
}

// VersionInfo represents version information
type VersionInfo struct {
	Index int `json:"index"`
}

// SwarmSpec represents swarm specification
type SwarmSpec struct {
	Name             string              `json:"name"`
	Labels           map[string]string   `json:"labels"`
	Orchestration    OrchestrationConfig `json:"orchestration"`
	Raft             RaftConfig          `json:"raft"`
	Dispatcher       DispatcherConfig    `json:"dispatcher"`
	CAConfig         CAConfig            `json:"ca_config"`
	TaskDefaults     TaskDefaults        `json:"task_defaults"`
	EncryptionConfig *EncryptionConfig   `json:"encryption_config,omitempty"`
}

// OrchestrationConfig represents orchestration configuration
type OrchestrationConfig struct {
	TaskHistoryRetentionLimit int `json:"task_history_retention_limit"`
}

// RaftConfig represents Raft configuration
type RaftConfig struct {
	SnapshotInterval           int `json:"snapshot_interval"`
	KeepOldSnapshots           int `json:"keep_old_snapshots"`
	LogEntriesForSlowFollowers int `json:"log_entries_for_slow_followers"`
	ElectionTick               int `json:"election_tick"`
	HeartbeatTick              int `json:"heartbeat_tick"`
}

// DispatcherConfig represents dispatcher configuration
type DispatcherConfig struct {
	HeartbeatPeriod int `json:"heartbeat_period"`
}

// CAConfig represents CA configuration
type CAConfig struct {
	NodeCertExpiry time.Duration `json:"node_cert_expiry"`
}

// TaskDefaults represents task defaults
type TaskDefaults struct {
	LogDriver *Driver `json:"log_driver,omitempty"`
}

// Driver represents a generic driver
type Driver struct {
	Name    string            `json:"name"`
	Options map[string]string `json:"options"`
}

// EncryptionConfig represents encryption configuration
type EncryptionConfig struct {
	AutoLockManagers bool `json:"auto_lock_managers"`
}

// Commit represents commit information
type Commit struct {
	ID       string `json:"id"`
	Expected string `json:"expected"`
}

// NewDockerServer creates a new Docker server
func NewDockerServer(config DockerConfig) *DockerServer {
	dockerCmd := config.DockerCmd
	if dockerCmd == "" {
		dockerCmd = "docker"
	}

	maxContainers := config.MaxContainers
	if maxContainers == 0 {
		maxContainers = 100
	}

	return &DockerServer{
		dockerCmd:     dockerCmd,
		allowedOps:    config.AllowedOps,
		restricted:    config.Restricted,
		maxContainers: maxContainers,
	}
}

// GetTools returns available Docker tools
func (ds *DockerServer) GetTools() []mcp.MCPTool {
	tools := []mcp.MCPTool{
		{
			Name:        "docker_ps",
			Description: "List Docker containers",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"all": map[string]interface{}{
						"type":        "boolean",
						"description": "Show all containers (including stopped)",
						"default":     false,
					},
					"filter": map[string]interface{}{
						"type":        "string",
						"description": "Filter containers (e.g., status=running)",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"description": "Format output using Go template",
					},
					"size": map[string]interface{}{
						"type":        "boolean",
						"description": "Display container sizes",
						"default":     false,
					},
					"limit": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum number of containers to return",
					},
				},
			},
		},
		{
			Name:        "docker_images",
			Description: "List Docker images",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"all": map[string]interface{}{
						"type":        "boolean",
						"description": "Show all images (including intermediate)",
						"default":     false,
					},
					"filter": map[string]interface{}{
						"type":        "string",
						"description": "Filter images",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"description": "Format output using Go template",
					},
					"digests": map[string]interface{}{
						"type":        "boolean",
						"description": "Show digests",
						"default":     false,
					},
				},
			},
		},
		{
			Name:        "docker_inspect",
			Description: "Inspect Docker container or image",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"name_or_id": map[string]interface{}{
						"type":        "string",
						"description": "Container or image name/ID",
					},
					"type": map[string]interface{}{
						"type":        "string",
						"description": "Type to inspect (container, image)",
						"default":     "container",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"description": "Format output using Go template",
					},
				},
				"required": []string{"name_or_id"},
			},
		},
		{
			Name:        "docker_logs",
			Description: "Get container logs",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"container": map[string]interface{}{
						"type":        "string",
						"description": "Container name or ID",
					},
					"follow": map[string]interface{}{
						"type":        "boolean",
						"description": "Follow log output",
						"default":     false,
					},
					"tail": map[string]interface{}{
						"type":        "integer",
						"description": "Number of lines to show from end of logs",
						"default":     100,
					},
					"since": map[string]interface{}{
						"type":        "string",
						"description": "Show logs since timestamp or relative (e.g. 1h)",
					},
					"timestamps": map[string]interface{}{
						"type":        "boolean",
						"description": "Show timestamps",
						"default":     true,
					},
				},
				"required": []string{"container"},
			},
		},
		{
			Name:        "docker_exec",
			Description: "Execute a command in a running container",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"container": map[string]interface{}{
						"type":        "string",
						"description": "Container name or ID",
					},
					"command": map[string]interface{}{
						"type":        "string",
						"description": "Command to execute",
					},
					"interactive": map[string]interface{}{
						"type":        "boolean",
						"description": "Keep STDIN open",
						"default":     false,
					},
					"tty": map[string]interface{}{
						"type":        "boolean",
						"description": "Allocate a pseudo-TTY",
						"default":     false,
					},
					"user": map[string]interface{}{
						"type":        "string",
						"description": "User to run command as",
					},
					"workdir": map[string]interface{}{
						"type":        "string",
						"description": "Working directory",
					},
				},
				"required": []string{"container", "command"},
			},
		},
		{
			Name:        "docker_stats",
			Description: "Get container resource usage statistics",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"containers": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Container names or IDs (empty for all)",
					},
					"no_stream": map[string]interface{}{
						"type":        "boolean",
						"description": "Disable streaming stats and only pull first result",
						"default":     true,
					},
					"format": map[string]interface{}{
						"type":        "string",
						"description": "Format output using Go template",
					},
				},
			},
		},
		{
			Name:        "docker_networks",
			Description: "List Docker networks",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"filter": map[string]interface{}{
						"type":        "string",
						"description": "Filter networks",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"description": "Format output using Go template",
					},
					"quiet": map[string]interface{}{
						"type":        "boolean",
						"description": "Only show network IDs",
						"default":     false,
					},
				},
			},
		},
		{
			Name:        "docker_volumes",
			Description: "List Docker volumes",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"filter": map[string]interface{}{
						"type":        "string",
						"description": "Filter volumes",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"description": "Format output using Go template",
					},
					"quiet": map[string]interface{}{
						"type":        "boolean",
						"description": "Only show volume names",
						"default":     false,
					},
				},
			},
		},
		{
			Name:        "docker_system_info",
			Description: "Get Docker system information",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"format": map[string]interface{}{
						"type":        "string",
						"description": "Format output using Go template",
					},
				},
			},
		},
		{
			Name:        "docker_system_df",
			Description: "Show Docker disk usage",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"verbose": map[string]interface{}{
						"type":        "boolean",
						"description": "Show detailed information",
						"default":     false,
					},
					"format": map[string]interface{}{
						"type":        "string",
						"description": "Format output using Go template",
					},
				},
			},
		},
	}

	// Add management tools if not restricted or if allowed
	managementTools := []mcp.MCPTool{
		{
			Name:        "docker_start",
			Description: "Start one or more stopped containers",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"containers": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Container names or IDs",
					},
					"attach": map[string]interface{}{
						"type":        "boolean",
						"description": "Attach container's STDOUT and STDERR",
						"default":     false,
					},
				},
				"required": []string{"containers"},
			},
		},
		{
			Name:        "docker_stop",
			Description: "Stop one or more running containers",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"containers": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Container names or IDs",
					},
					"time": map[string]interface{}{
						"type":        "integer",
						"description": "Seconds to wait for stop before killing",
						"default":     10,
					},
				},
				"required": []string{"containers"},
			},
		},
		{
			Name:        "docker_restart",
			Description: "Restart one or more containers",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"containers": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Container names or IDs",
					},
					"time": map[string]interface{}{
						"type":        "integer",
						"description": "Seconds to wait for stop before killing",
						"default":     10,
					},
				},
				"required": []string{"containers"},
			},
		},
		{
			Name:        "docker_remove",
			Description: "Remove one or more containers",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"containers": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Container names or IDs",
					},
					"force": map[string]interface{}{
						"type":        "boolean",
						"description": "Force removal of running containers",
						"default":     false,
					},
					"volumes": map[string]interface{}{
						"type":        "boolean",
						"description": "Remove associated volumes",
						"default":     false,
					},
				},
				"required": []string{"containers"},
			},
		},
		{
			Name:        "docker_run",
			Description: "Run a new container",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"image": map[string]interface{}{
						"type":        "string",
						"description": "Image name or ID",
					},
					"command": map[string]interface{}{
						"type":        "string",
						"description": "Command to run in container",
					},
					"name": map[string]interface{}{
						"type":        "string",
						"description": "Container name",
					},
					"detach": map[string]interface{}{
						"type":        "boolean",
						"description": "Run container in background",
						"default":     true,
					},
					"ports": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Port mappings (e.g. 8080:80)",
					},
					"volumes": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Volume mounts (e.g. /host:/container)",
					},
					"env": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Environment variables (e.g. VAR=value)",
					},
					"rm": map[string]interface{}{
						"type":        "boolean",
						"description": "Remove container when it exits",
						"default":     false,
					},
				},
				"required": []string{"image"},
			},
		},
		{
			Name:        "docker_pull",
			Description: "Pull an image from a registry",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"image": map[string]interface{}{
						"type":        "string",
						"description": "Image name and tag",
					},
					"all_tags": map[string]interface{}{
						"type":        "boolean",
						"description": "Download all tagged images",
						"default":     false,
					},
				},
				"required": []string{"image"},
			},
		},
		{
			Name:        "docker_build",
			Description: "Build an image from a Dockerfile",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Build context path",
						"default":     ".",
					},
					"dockerfile": map[string]interface{}{
						"type":        "string",
						"description": "Dockerfile name",
						"default":     "Dockerfile",
					},
					"tag": map[string]interface{}{
						"type":        "string",
						"description": "Image tag",
					},
					"build_args": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Build arguments (e.g. ARG=value)",
					},
					"no_cache": map[string]interface{}{
						"type":        "boolean",
						"description": "Don't use cache when building",
						"default":     false,
					},
				},
			},
		},
	}

	// Add management tools if allowed
	if !ds.restricted || ds.isOperationAllowed("management") {
		tools = append(tools, managementTools...)
	}

	return tools
}

// ExecuteTool executes a Docker tool
func (ds *DockerServer) ExecuteTool(ctx context.Context, toolName string, input map[string]interface{}) (interface{}, error) {
	// Check if Docker is available
	if err := ds.checkDockerAvailable(); err != nil {
		return nil, err
	}

	// Check operation permissions
	if ds.restricted && !ds.isOperationAllowed(toolName) {
		return nil, fmt.Errorf("operation %s not allowed in restricted mode", toolName)
	}

	switch toolName {
	case "docker_ps":
		return ds.listContainers(input)
	case "docker_images":
		return ds.listImages(input)
	case "docker_inspect":
		return ds.inspectObject(input)
	case "docker_logs":
		return ds.getContainerLogs(input)
	case "docker_exec":
		return ds.execCommand(input)
	case "docker_stats":
		return ds.getContainerStats(input)
	case "docker_networks":
		return ds.listNetworks(input)
	case "docker_volumes":
		return ds.listVolumes(input)
	case "docker_system_info":
		return ds.getSystemInfo(input)
	case "docker_system_df":
		return ds.getSystemUsage(input)
	case "docker_start":
		return ds.startContainers(input)
	case "docker_stop":
		return ds.stopContainers(input)
	case "docker_restart":
		return ds.restartContainers(input)
	case "docker_remove":
		return ds.removeContainers(input)
	case "docker_run":
		return ds.runContainer(input)
	case "docker_pull":
		return ds.pullImage(input)
	case "docker_build":
		return ds.buildImage(input)
	default:
		return nil, fmt.Errorf("unknown tool: %s", toolName)
	}
}

// listContainers lists Docker containers
func (ds *DockerServer) listContainers(input map[string]interface{}) (interface{}, error) {
	args := []string{"ps", "--format", "json", "--no-trunc"}

	all, _ := input["all"].(bool)
	if all {
		args = append(args, "-a")
	}

	if filter, ok := input["filter"].(string); ok && filter != "" {
		args = append(args, "--filter", filter)
	}

	if size, ok := input["size"].(bool); ok && size {
		args = append(args, "-s")
	}

	if limit, ok := input["limit"].(float64); ok && limit > 0 {
		args = append(args, "--last", strconv.Itoa(int(limit)))
	}

	output, err := ds.runDockerCommand(args...)
	if err != nil {
		return nil, err
	}

	containers, err := ds.parseContainersOutput(string(output))
	if err != nil {
		return nil, err
	}

	// Apply our own limit if set
	if limit, ok := input["limit"].(float64); ok && limit > 0 && len(containers) > int(limit) {
		containers = containers[:int(limit)]
	}

	if len(containers) > ds.maxContainers {
		containers = containers[:ds.maxContainers]
	}

	return map[string]interface{}{
		"containers": containers,
		"count":      len(containers),
	}, nil
}

// listImages lists Docker images
func (ds *DockerServer) listImages(input map[string]interface{}) (interface{}, error) {
	args := []string{"images", "--format", "json", "--no-trunc"}

	all, _ := input["all"].(bool)
	if all {
		args = append(args, "-a")
	}

	if filter, ok := input["filter"].(string); ok && filter != "" {
		args = append(args, "--filter", filter)
	}

	if digests, ok := input["digests"].(bool); ok && digests {
		args = append(args, "--digests")
	}

	output, err := ds.runDockerCommand(args...)
	if err != nil {
		return nil, err
	}

	images, err := ds.parseImagesOutput(string(output))
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"images": images,
		"count":  len(images),
	}, nil
}

// inspectObject inspects a container or image
func (ds *DockerServer) inspectObject(input map[string]interface{}) (interface{}, error) {
	nameOrID, _ := input["name_or_id"].(string)
	if nameOrID == "" {
		return nil, fmt.Errorf("name_or_id is required")
	}

	objectType, _ := input["type"].(string)
	if objectType == "" {
		objectType = "container"
	}

	args := []string{"inspect"}

	if format, ok := input["format"].(string); ok && format != "" {
		args = append(args, "--format", format)
	}

	args = append(args, nameOrID)

	output, err := ds.runDockerCommand(args...)
	if err != nil {
		return nil, err
	}

	// Try to parse as JSON
	var result interface{}
	if err := json.Unmarshal(output, &result); err == nil {
		return result, nil
	}

	// Return raw output if JSON parsing fails
	return map[string]interface{}{
		"raw_output": string(output),
	}, nil
}

// getContainerLogs gets container logs
func (ds *DockerServer) getContainerLogs(input map[string]interface{}) (interface{}, error) {
	container, _ := input["container"].(string)
	if container == "" {
		return nil, fmt.Errorf("container is required")
	}

	args := []string{"logs"}

	if timestamps, ok := input["timestamps"].(bool); ok && timestamps {
		args = append(args, "--timestamps")
	}

	if tail, ok := input["tail"].(float64); ok && tail > 0 {
		args = append(args, "--tail", strconv.Itoa(int(tail)))
	}

	if since, ok := input["since"].(string); ok && since != "" {
		args = append(args, "--since", since)
	}

	follow, _ := input["follow"].(bool)
	if follow {
		args = append(args, "--follow")
	}

	args = append(args, container)

	output, err := ds.runDockerCommand(args...)
	if err != nil {
		return nil, err
	}

	logs := strings.Split(string(output), "\n")
	// Remove empty last line
	if len(logs) > 0 && logs[len(logs)-1] == "" {
		logs = logs[:len(logs)-1]
	}

	return map[string]interface{}{
		"container": container,
		"logs":      logs,
		"count":     len(logs),
	}, nil
}

// execCommand executes a command in a container
func (ds *DockerServer) execCommand(input map[string]interface{}) (interface{}, error) {
	container, _ := input["container"].(string)
	command, _ := input["command"].(string)

	if container == "" || command == "" {
		return nil, fmt.Errorf("container and command are required")
	}

	args := []string{"exec"}

	if interactive, ok := input["interactive"].(bool); ok && interactive {
		args = append(args, "-i")
	}

	if tty, ok := input["tty"].(bool); ok && tty {
		args = append(args, "-t")
	}

	if user, ok := input["user"].(string); ok && user != "" {
		args = append(args, "--user", user)
	}

	if workdir, ok := input["workdir"].(string); ok && workdir != "" {
		args = append(args, "--workdir", workdir)
	}

	args = append(args, container)

	// Split command into parts
	commandParts := strings.Fields(command)
	args = append(args, commandParts...)

	start := time.Now()
	output, err := ds.runDockerCommand(args...)

	result := map[string]interface{}{
		"container": container,
		"command":   command,
		"output":    string(output),
		"success":   err == nil,
		"took":      time.Since(start),
	}

	if err != nil {
		result["error"] = err.Error()
	}

	return result, nil
}

// getContainerStats gets container statistics
func (ds *DockerServer) getContainerStats(input map[string]interface{}) (interface{}, error) {
	args := []string{"stats", "--format", "json"}

	noStream, _ := input["no_stream"].(bool)
	if noStream {
		args = append(args, "--no-stream")
	}

	if containers, ok := input["containers"].([]interface{}); ok && len(containers) > 0 {
		for _, container := range containers {
			if containerStr, ok := container.(string); ok {
				args = append(args, containerStr)
			}
		}
	}

	output, err := ds.runDockerCommand(args...)
	if err != nil {
		return nil, err
	}

	stats, err := ds.parseStatsOutput(string(output))
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"stats": stats,
		"count": len(stats),
	}, nil
}

// listNetworks lists Docker networks
func (ds *DockerServer) listNetworks(input map[string]interface{}) (interface{}, error) {
	args := []string{"network", "ls", "--format", "json"}

	if filter, ok := input["filter"].(string); ok && filter != "" {
		args = append(args, "--filter", filter)
	}

	if quiet, ok := input["quiet"].(bool); ok && quiet {
		args = append(args, "--quiet")
	}

	output, err := ds.runDockerCommand(args...)
	if err != nil {
		return nil, err
	}

	networks, err := ds.parseNetworksOutput(string(output))
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"networks": networks,
		"count":    len(networks),
	}, nil
}

// listVolumes lists Docker volumes
func (ds *DockerServer) listVolumes(input map[string]interface{}) (interface{}, error) {
	args := []string{"volume", "ls", "--format", "json"}

	if filter, ok := input["filter"].(string); ok && filter != "" {
		args = append(args, "--filter", filter)
	}

	if quiet, ok := input["quiet"].(bool); ok && quiet {
		args = append(args, "--quiet")
	}

	output, err := ds.runDockerCommand(args...)
	if err != nil {
		return nil, err
	}

	volumes, err := ds.parseVolumesOutput(string(output))
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"volumes": volumes,
		"count":   len(volumes),
	}, nil
}

// getSystemInfo gets Docker system information
func (ds *DockerServer) getSystemInfo(input map[string]interface{}) (interface{}, error) {
	args := []string{"system", "info", "--format", "json"}

	output, err := ds.runDockerCommand(args...)
	if err != nil {
		return nil, err
	}

	var systemInfo DockerSystemInfo
	if err := json.Unmarshal(output, &systemInfo); err != nil {
		return nil, fmt.Errorf("failed to parse system info: %v", err)
	}

	return systemInfo, nil
}

// getSystemUsage gets Docker system disk usage
func (ds *DockerServer) getSystemUsage(input map[string]interface{}) (interface{}, error) {
	args := []string{"system", "df", "--format", "json"}

	if verbose, ok := input["verbose"].(bool); ok && verbose {
		args = append(args, "--verbose")
	}

	output, err := ds.runDockerCommand(args...)
	if err != nil {
		return nil, err
	}

	// Try to parse as JSON
	var result interface{}
	if err := json.Unmarshal(output, &result); err == nil {
		return result, nil
	}

	return map[string]interface{}{
		"raw_output": string(output),
	}, nil
}

// Container management operations

// startContainers starts containers
func (ds *DockerServer) startContainers(input map[string]interface{}) (interface{}, error) {
	containers, ok := input["containers"].([]interface{})
	if !ok || len(containers) == 0 {
		return nil, fmt.Errorf("containers list is required")
	}

	args := []string{"start"}

	if attach, ok := input["attach"].(bool); ok && attach {
		args = append(args, "--attach")
	}

	for _, container := range containers {
		if containerStr, ok := container.(string); ok {
			args = append(args, containerStr)
		}
	}

	output, err := ds.runDockerCommand(args...)

	return map[string]interface{}{
		"success":    err == nil,
		"output":     string(output),
		"containers": containers,
		"error":      errorToString(err),
	}, nil
}

// stopContainers stops containers
func (ds *DockerServer) stopContainers(input map[string]interface{}) (interface{}, error) {
	containers, ok := input["containers"].([]interface{})
	if !ok || len(containers) == 0 {
		return nil, fmt.Errorf("containers list is required")
	}

	args := []string{"stop"}

	if timeout, ok := input["time"].(float64); ok && timeout > 0 {
		args = append(args, "--time", strconv.Itoa(int(timeout)))
	}

	for _, container := range containers {
		if containerStr, ok := container.(string); ok {
			args = append(args, containerStr)
		}
	}

	output, err := ds.runDockerCommand(args...)

	return map[string]interface{}{
		"success":    err == nil,
		"output":     string(output),
		"containers": containers,
		"error":      errorToString(err),
	}, nil
}

// restartContainers restarts containers
func (ds *DockerServer) restartContainers(input map[string]interface{}) (interface{}, error) {
	containers, ok := input["containers"].([]interface{})
	if !ok || len(containers) == 0 {
		return nil, fmt.Errorf("containers list is required")
	}

	args := []string{"restart"}

	if timeout, ok := input["time"].(float64); ok && timeout > 0 {
		args = append(args, "--time", strconv.Itoa(int(timeout)))
	}

	for _, container := range containers {
		if containerStr, ok := container.(string); ok {
			args = append(args, containerStr)
		}
	}

	output, err := ds.runDockerCommand(args...)

	return map[string]interface{}{
		"success":    err == nil,
		"output":     string(output),
		"containers": containers,
		"error":      errorToString(err),
	}, nil
}

// removeContainers removes containers
func (ds *DockerServer) removeContainers(input map[string]interface{}) (interface{}, error) {
	containers, ok := input["containers"].([]interface{})
	if !ok || len(containers) == 0 {
		return nil, fmt.Errorf("containers list is required")
	}

	args := []string{"rm"}

	if force, ok := input["force"].(bool); ok && force {
		args = append(args, "--force")
	}

	if volumes, ok := input["volumes"].(bool); ok && volumes {
		args = append(args, "--volumes")
	}

	for _, container := range containers {
		if containerStr, ok := container.(string); ok {
			args = append(args, containerStr)
		}
	}

	output, err := ds.runDockerCommand(args...)

	return map[string]interface{}{
		"success":    err == nil,
		"output":     string(output),
		"containers": containers,
		"error":      errorToString(err),
	}, nil
}

// runContainer runs a new container
func (ds *DockerServer) runContainer(input map[string]interface{}) (interface{}, error) {
	image, _ := input["image"].(string)
	if image == "" {
		return nil, fmt.Errorf("image is required")
	}

	args := []string{"run"}

	if detach, ok := input["detach"].(bool); ok && detach {
		args = append(args, "--detach")
	}

	if name, ok := input["name"].(string); ok && name != "" {
		args = append(args, "--name", name)
	}

	if rm, ok := input["rm"].(bool); ok && rm {
		args = append(args, "--rm")
	}

	// Add port mappings
	if ports, ok := input["ports"].([]interface{}); ok {
		for _, port := range ports {
			if portStr, ok := port.(string); ok {
				args = append(args, "-p", portStr)
			}
		}
	}

	// Add volume mounts
	if volumes, ok := input["volumes"].([]interface{}); ok {
		for _, volume := range volumes {
			if volumeStr, ok := volume.(string); ok {
				args = append(args, "-v", volumeStr)
			}
		}
	}

	// Add environment variables
	if env, ok := input["env"].([]interface{}); ok {
		for _, envVar := range env {
			if envStr, ok := envVar.(string); ok {
				args = append(args, "-e", envStr)
			}
		}
	}

	args = append(args, image)

	// Add command if specified
	if command, ok := input["command"].(string); ok && command != "" {
		commandParts := strings.Fields(command)
		args = append(args, commandParts...)
	}

	start := time.Now()
	output, err := ds.runDockerCommand(args...)

	result := map[string]interface{}{
		"success": err == nil,
		"output":  strings.TrimSpace(string(output)),
		"image":   image,
		"took":    time.Since(start),
	}

	if err != nil {
		result["error"] = err.Error()
	} else {
		// Output should be container ID if detached
		containerID := strings.TrimSpace(string(output))
		if containerID != "" {
			result["container_id"] = containerID
		}
	}

	return result, nil
}

// pullImage pulls an image
func (ds *DockerServer) pullImage(input map[string]interface{}) (interface{}, error) {
	image, _ := input["image"].(string)
	if image == "" {
		return nil, fmt.Errorf("image is required")
	}

	args := []string{"pull"}

	if allTags, ok := input["all_tags"].(bool); ok && allTags {
		args = append(args, "--all-tags")
	}

	args = append(args, image)

	start := time.Now()
	output, err := ds.runDockerCommand(args...)

	return map[string]interface{}{
		"success": err == nil,
		"output":  string(output),
		"image":   image,
		"took":    time.Since(start),
		"error":   errorToString(err),
	}, nil
}

// buildImage builds an image
func (ds *DockerServer) buildImage(input map[string]interface{}) (interface{}, error) {
	path, _ := input["path"].(string)
	if path == "" {
		path = "."
	}

	args := []string{"build"}

	if dockerfile, ok := input["dockerfile"].(string); ok && dockerfile != "" {
		args = append(args, "--file", dockerfile)
	}

	if tag, ok := input["tag"].(string); ok && tag != "" {
		args = append(args, "--tag", tag)
	}

	if noCache, ok := input["no_cache"].(bool); ok && noCache {
		args = append(args, "--no-cache")
	}

	// Add build arguments
	if buildArgs, ok := input["build_args"].([]interface{}); ok {
		for _, arg := range buildArgs {
			if argStr, ok := arg.(string); ok {
				args = append(args, "--build-arg", argStr)
			}
		}
	}

	args = append(args, path)

	start := time.Now()
	output, err := ds.runDockerCommand(args...)

	return map[string]interface{}{
		"success": err == nil,
		"output":  string(output),
		"path":    path,
		"took":    time.Since(start),
		"error":   errorToString(err),
	}, nil
}

// Helper methods

// checkDockerAvailable checks if Docker is available
func (ds *DockerServer) checkDockerAvailable() error {
	cmd := exec.Command(ds.dockerCmd, "version", "--format", "json")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("Docker is not available: %v", err)
	}
	return nil
}

// isOperationAllowed checks if an operation is allowed in restricted mode
func (ds *DockerServer) isOperationAllowed(operation string) bool {
	if !ds.restricted {
		return true
	}

	for _, allowed := range ds.allowedOps {
		if allowed == operation || allowed == "all" {
			return true
		}
	}

	return false
}

// runDockerCommand runs a Docker command and returns output
func (ds *DockerServer) runDockerCommand(args ...string) ([]byte, error) {
	cmd := exec.Command(ds.dockerCmd, args...)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("docker command failed: %v, stderr: %s", err, stderr.String())
	}

	return stdout.Bytes(), nil
}

// Parse output methods

// parseContainersOutput parses docker ps JSON output
func (ds *DockerServer) parseContainersOutput(output string) ([]Container, error) {
	lines := strings.Split(strings.TrimSpace(output), "\n")
	var containers []Container

	for _, line := range lines {
		if line == "" {
			continue
		}

		var container Container
		if err := json.Unmarshal([]byte(line), &container); err != nil {
			continue // Skip invalid lines
		}

		containers = append(containers, container)
	}

	return containers, nil
}

// parseImagesOutput parses docker images JSON output
func (ds *DockerServer) parseImagesOutput(output string) ([]Image, error) {
	lines := strings.Split(strings.TrimSpace(output), "\n")
	var images []Image

	for _, line := range lines {
		if line == "" {
			continue
		}

		var image Image
		if err := json.Unmarshal([]byte(line), &image); err != nil {
			continue // Skip invalid lines
		}

		images = append(images, image)
	}

	return images, nil
}

// parseStatsOutput parses docker stats JSON output
func (ds *DockerServer) parseStatsOutput(output string) ([]DockerStats, error) {
	lines := strings.Split(strings.TrimSpace(output), "\n")
	var stats []DockerStats

	for _, line := range lines {
		if line == "" {
			continue
		}

		var stat DockerStats
		if err := json.Unmarshal([]byte(line), &stat); err != nil {
			continue // Skip invalid lines
		}

		stats = append(stats, stat)
	}

	return stats, nil
}

// parseNetworksOutput parses docker network ls JSON output
func (ds *DockerServer) parseNetworksOutput(output string) ([]Network, error) {
	lines := strings.Split(strings.TrimSpace(output), "\n")
	var networks []Network

	for _, line := range lines {
		if line == "" {
			continue
		}

		var network Network
		if err := json.Unmarshal([]byte(line), &network); err != nil {
			continue // Skip invalid lines
		}

		networks = append(networks, network)
	}

	return networks, nil
}

// parseVolumesOutput parses docker volume ls JSON output
func (ds *DockerServer) parseVolumesOutput(output string) ([]Volume, error) {
	lines := strings.Split(strings.TrimSpace(output), "\n")
	var volumes []Volume

	for _, line := range lines {
		if line == "" {
			continue
		}

		var volume Volume
		if err := json.Unmarshal([]byte(line), &volume); err != nil {
			continue // Skip invalid lines
		}

		volumes = append(volumes, volume)
	}

	return volumes, nil
}

// errorToString converts error to string, returning empty string for nil
func errorToString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
