/**
 * DevCrew Agents Web Interface JavaScript
 * Handles real-time communication, UI interactions, and data management
 */

class DevCrewWebInterface {
    constructor() {
        this.ws = null;
        this.wsReconnectInterval = null;
        this.isProcessing = false;
        this.currentTab = 'overview';
        this.currentRightTab = 'files';

        // Initialize the interface
        this.init();
    }

    init () {
        this.setupEventListeners();
        this.connectWebSocket();
        this.loadInitialData();
        this.updateSystemTime();
        this.setupAutoRefresh();
    }

    setupEventListeners () {
        // Chat form submission
        const chatForm = document.getElementById('chatForm');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');

        chatForm.addEventListener('submit', (e) => this.handleChatSubmit(e));
        messageInput.addEventListener('input', () => this.updateCharCounter());
        messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));

        // Quick action buttons
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.handleQuickAction(e));
        });

        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e));
        });

        document.querySelectorAll('.right-tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchRightTab(e));
        });

        // Button clicks
        document.getElementById('clearChatBtn').addEventListener('click', () => this.clearChat());
        document.getElementById('refreshFilesBtn').addEventListener('click', () => this.loadFiles());
        document.getElementById('clearLogsBtn').addEventListener('click', () => this.clearLogs());
        document.getElementById('refreshTasksBtn').addEventListener('click', () => this.loadAgentHistory());

        // Modal handling
        document.getElementById('fileModalClose').addEventListener('click', () => this.closeModal());
        document.getElementById('fileModal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('fileModal')) {
                this.closeModal();
            }
        });

        // Sandbox handling - updated for new structure
        document.getElementById('runCodeBtn').addEventListener('click', () => this.runSandboxCode());
        document.getElementById('clearSandboxBtn').addEventListener('click', () => this.clearSandbox());
        document.getElementById('maximizeSandboxBtn').addEventListener('click', () => this.toggleSandboxMaximize());
        document.getElementById('clearOutputBtn').addEventListener('click', () => this.clearSandboxOutput());

        // Language tabs - updated for new structure
        document.querySelectorAll('.lang-tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchSandboxLanguage(e));
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleGlobalKeydown(e));
    }

    connectWebSocket () {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${ protocol }//${ window.location.host }/ws`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
                if (this.wsReconnectInterval) {
                    clearInterval(this.wsReconnectInterval);
                    this.wsReconnectInterval = null;
                }
            };

            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                this.scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };

        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.updateConnectionStatus(false);
            this.scheduleReconnect();
        }
    }

    scheduleReconnect () {
        if (!this.wsReconnectInterval) {
            this.wsReconnectInterval = setInterval(() => {
                console.log('Attempting to reconnect...');
                this.connectWebSocket();
            }, 5000);
        }
    }

    handleWebSocketMessage (event) {
        try {
            const message = JSON.parse(event.data);

            switch (message.type) {
                case 'user_message':
                    this.addChatMessage('user', 'You', message.content, message.timestamp);
                    break;

                case 'agent_response':
                    this.addAgentResponse(message.content, message.timestamp, message.current_agent);
                    this.updateProjectPhase(message.phase);
                    this.updateActiveAgent(message.current_agent);
                    break;

                case 'error':
                    this.addChatMessage('system', 'System', `Error: ${ message.content }`, message.timestamp);
                    break;

                case 'status':
                    this.updateStatus(message.data);
                    break;

                case 'pong':
                    // Handle ping/pong for connection health
                    break;

                default:
                    console.log('Unknown message type:', message.type);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }

    updateConnectionStatus (connected) {
        const statusIndicator = document.getElementById('connectionStatus');
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('.status-text');

        if (connected) {
            statusDot.className = 'status-dot online';
            statusText.textContent = 'Connected';
        } else {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'Disconnected';
        }
    }

    async handleChatSubmit (e) {
        e.preventDefault();

        if (this.isProcessing) return;

        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();

        if (!message) return;

        this.isProcessing = true;
        messageInput.value = '';
        this.updateCharCounter();

        try {
            const response = await fetch('/api/send-message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${ response.status }`);
            }

            const data = await response.json();
            console.log('Message sent:', data);

        } catch (error) {
            console.error('Error sending message:', error);
            this.addChatMessage('system', 'System', `Error sending message: ${ error.message }`, new Date().toISOString());
        } finally {
            this.isProcessing = false;
        }
    }

    handleKeyDown (e) {
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            this.handleChatSubmit(e);
        }
    }

    handleGlobalKeydown (e) {
        // Escape key to close modal
        if (e.key === 'Escape') {
            this.closeModal();
        }

        // Focus message input with '/'
        if (e.key === '/' && !e.target.matches('input, textarea')) {
            e.preventDefault();
            document.getElementById('messageInput').focus();
        }

        // Ctrl/Cmd + Enter to run code in sandbox
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
            const sandboxEditor = document.getElementById('sandboxEditor');
            if (sandboxEditor && document.activeElement === sandboxEditor) {
                e.preventDefault();
                this.runSandboxCode();
            }
        }
    }

    handleQuickAction (e) {
        const message = e.currentTarget.dataset.message;
        if (message) {
            document.getElementById('messageInput').value = message;
            this.updateCharCounter();
            document.getElementById('messageInput').focus();
        }
    }

    updateCharCounter () {
        const input = document.getElementById('messageInput');
        const counter = document.querySelector('.char-counter');
        const length = input.value.length;
        counter.textContent = `${ length } / 2000`;

        if (length > 1800) {
            counter.style.color = 'var(--warning-color)';
        } else if (length > 1950) {
            counter.style.color = 'var(--error-color)';
        } else {
            counter.style.color = 'var(--text-secondary)';
        }
    }

    addChatMessage (type, sender, content, timestamp) {
        const container = document.getElementById('conversationContainer');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message';

        const timeFormatted = new Date(timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });

        messageDiv.innerHTML = `
            <div class="agent-avatar ${ type }">
                ${ this.getAvatarIcon(type) }
            </div>
            <div class="message-content">
                <div class="message-header">
                    <span class="agent-name">${ sender }</span>
                    <span class="message-time">${ timeFormatted }</span>
                </div>
                <div class="message-text">${ this.formatMessageContent(content) }</div>
            </div>
        `;

        container.appendChild(messageDiv);
        this.scrollToBottom(container);
    }

    addAgentResponse (content, timestamp, agentType) {
        // Parse the response to extract different sections
        const sections = this.parseAgentResponse(content);

        sections.forEach(section => {
            this.addChatMessage(
                section.agentType || agentType || 'system',
                this.getAgentDisplayName(section.agentType || agentType),
                section.content,
                timestamp
            );
        });
    }

    parseAgentResponse (content) {
        // Split response by agent sections (## Agent Name)
        const sections = [];
        const lines = content.split('\n');
        let currentSection = { content: '', agentType: null };

        for (const line of lines) {
            const agentMatch = line.match(/^##\s+(.*?)\s+Response?/i);
            if (agentMatch) {
                if (currentSection.content.trim()) {
                    sections.push(currentSection);
                }
                const agentName = agentMatch[1].toLowerCase().replace(/\s+/g, '_');
                currentSection = { content: '', agentType: agentName };
            } else {
                currentSection.content += line + '\n';
            }
        }

        if (currentSection.content.trim()) {
            sections.push(currentSection);
        }

        return sections.length > 0 ? sections : [{ content, agentType: null }];
    }

    getAvatarIcon (type) {
        const icons = {
            system: '<i class="fas fa-robot"></i>',
            user: '<i class="fas fa-user"></i>',
            project_manager: '<i class="fas fa-project-diagram"></i>',
            designer: '<i class="fas fa-palette"></i>',
            coder: '<i class="fas fa-code"></i>',
            tester: '<i class="fas fa-bug"></i>'
        };
        return icons[type] || '<i class="fas fa-robot"></i>';
    }

    getAgentDisplayName (type) {
        const names = {
            project_manager: 'Project Manager',
            designer: 'Designer',
            coder: 'Coder',
            tester: 'Tester',
            system: 'System'
        };
        return names[type] || 'Agent';
    }

    formatMessageContent (content) {
        // Convert markdown-like formatting to HTML
        let formatted = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');

        // Handle code blocks
        formatted = formatted.replace(/```(\w+)?\n?([\s\S]*?)```/g,
            '<pre><code class="language-$1">$2</code></pre>');

        return formatted;
    }

    scrollToBottom (container) {
        setTimeout(() => {
            container.scrollTop = container.scrollHeight;
        }, 100);
    }

    switchTab (e) {
        const targetTab = e.currentTarget.dataset.tab;

        // Update tab buttons
        document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
        e.currentTarget.classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
        document.getElementById(targetTab).classList.add('active');

        this.currentTab = targetTab;

        // Load tab-specific data
        this.loadTabData(targetTab);
    }

    switchRightTab (e) {
        const targetTab = e.currentTarget.dataset.tab;

        // Update tab buttons
        document.querySelectorAll('.right-tab').forEach(tab => tab.classList.remove('active'));
        e.currentTarget.classList.add('active');

        // Update tab content
        document.querySelectorAll('.right-tab-content').forEach(content => content.classList.remove('active'));
        document.getElementById(targetTab).classList.add('active');

        this.currentRightTab = targetTab;

        // Load tab-specific data
        if (targetTab === 'files') {
            this.loadFiles();
        } else if (targetTab === 'logs') {
            this.loadLogs();
        }
    }

    async loadTabData (tab) {
        switch (tab) {
            case 'overview':
                await this.loadStatus();
                break;
            case 'agents':
                await this.loadAgents();
                break;
            case 'tasks':
                await this.loadAgentHistory();
                break;
            case 'architecture':
                this.loadArchitecture();
                break;
        }
    }

    async loadInitialData () {
        await Promise.all([
            this.loadStatus(),
            this.loadConversationHistory(),
            this.loadFiles(),
            this.loadLogs()
        ]);
    }

    async loadStatus () {
        try {
            const response = await fetch('/api/status');
            if (!response.ok) throw new Error(`HTTP error! status: ${ response.status }`);

            const data = await response.json();
            this.updateStatus(data);
        } catch (error) {
            console.error('Error loading status:', error);
        }
    }

    updateStatus (data) {
        // Update metrics
        document.getElementById('completedTasks').textContent = data.completed_tasks || 0;
        document.getElementById('failedTasks').textContent = data.failed_tasks || 0;
        document.getElementById('activeQueries').textContent = data.active_queries || 0;
        document.getElementById('activeAgentsCount').textContent = data.agents?.length || 0;

        // Update project phase
        this.updateProjectPhase(data.current_phase);
        this.updatePhaseProgress(data.phase_completion || 0);

        // Update active agent
        this.updateActiveAgent(data.last_agent);

        // Update project phases display
        this.updateProjectPhases(data.current_phase);
    }

    updateProjectPhase (phase) {
        const phaseElement = document.getElementById('currentPhase');
        if (phaseElement && phase) {
            phaseElement.textContent = phase.charAt(0).toUpperCase() + phase.slice(1);
        }
    }

    updatePhaseProgress (completion) {
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.width = `${ Math.round(completion * 100) }%`;
        }
    }

    updateActiveAgent (agent) {
        const agentElement = document.getElementById('activeAgent');
        if (agentElement) {
            agentElement.textContent = agent ? this.getAgentDisplayName(agent) : 'No active agent';
        }
    }

    updateProjectPhases (currentPhase) {
        const container = document.getElementById('phasesContainer');
        if (!container) return;

        const phases = [
            'initialization', 'planning', 'design',
            'implementation', 'testing', 'review', 'complete'
        ];

        container.innerHTML = phases.map(phase => {
            const isActive = phase === currentPhase;
            const isCompleted = phases.indexOf(phase) < phases.indexOf(currentPhase);

            let className = 'phase-item';
            if (isActive) className += ' active';
            if (isCompleted) className += ' completed';

            return `<div class="${ className }">${ phase.charAt(0).toUpperCase() + phase.slice(1) }</div>`;
        }).join('');
    }

    async loadAgents () {
        try {
            const [statusRes, performanceRes] = await Promise.all([
                fetch('/api/status'),
                fetch('/api/agent-performance')
            ]);

            const statusData = await statusRes.json();
            const performanceData = await performanceRes.json();

            this.displayAgents(statusData, performanceData);
        } catch (error) {
            console.error('Error loading agents:', error);
        }
    }

    displayAgents (statusData, performanceData) {
        const container = document.getElementById('agentsGrid');
        if (!container) return;

        const agents = statusData.agents || [];
        const models = statusData.models || {};
        const performance = performanceData.performance || {};

        container.innerHTML = agents.map(agentType => {
            const agentPerf = performance[agentType] || {};
            const model = models[agentType] || 'Unknown';

            return `
                <div class="agent-card">
                    <div class="agent-header">
                        <div class="agent-avatar-large ${ agentType }">
                            ${ this.getAvatarIcon(agentType) }
                        </div>
                        <div class="agent-info">
                            <h4>${ this.getAgentDisplayName(agentType) }</h4>
                            <div class="agent-model">${ model }</div>
                        </div>
                    </div>
                    <div class="agent-stats">
                        <div class="agent-stat">
                            <div class="agent-stat-value">${ agentPerf.tasks_completed || 0 }</div>
                            <div class="agent-stat-label">Tasks</div>
                        </div>
                        <div class="agent-stat">
                            <div class="agent-stat-value">${ (agentPerf.avg_execution_time || 0).toFixed(1) }s</div>
                            <div class="agent-stat-label">Avg Time</div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    async loadAgentHistory () {
        try {
            const response = await fetch('/api/agent-history');
            if (!response.ok) throw new Error(`HTTP error! status: ${ response.status }`);

            const data = await response.json();
            this.displayAgentHistory(data);
        } catch (error) {
            console.error('Error loading agent history:', error);
        }
    }

    displayAgentHistory (data) {
        const container = document.getElementById('tasksList');
        if (!container) return;

        const turns = data.agent_turns || [];

        if (turns.length === 0) {
            container.innerHTML = '<p class="placeholder">No recent agent activity</p>';
            return;
        }

        container.innerHTML = turns.slice(-20).reverse().map(turn => {
            const time = new Date(turn.timestamp).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit'
            });

            return `
                <div class="task-item">
                    <div class="agent-avatar ${ turn.agent_type }">
                        ${ this.getAvatarIcon(turn.agent_type) }
                    </div>
                    <div class="task-info">
                        <div class="task-header">
                            <span class="agent-name">${ this.getAgentDisplayName(turn.agent_type) }</span>
                            <span class="task-time">${ time }</span>
                        </div>
                        <div class="task-details">
                            <span class="task-reason">${ turn.reason.replace(/_/g, ' ') }</span>
                            <span class="task-id">${ turn.task_id }</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    async loadConversationHistory () {
        try {
            const response = await fetch('/api/conversation-history');
            if (!response.ok) throw new Error(`HTTP error! status: ${ response.status }`);

            const data = await response.json();
            this.displayConversationHistory(data.conversations || []);
        } catch (error) {
            console.error('Error loading conversation history:', error);
        }
    }

    displayConversationHistory (conversations) {
        const container = document.getElementById('conversationContainer');
        const welcomeMessage = container.querySelector('.welcome-message');

        // Clear existing messages except welcome
        const existingMessages = container.querySelectorAll('.chat-message');
        existingMessages.forEach(msg => msg.remove());

        // Add conversation history
        conversations.forEach(conv => {
            this.addChatMessage('user', 'You', conv.query, conv.timestamp);
            if (conv.response) {
                this.addAgentResponse(conv.response, conv.timestamp);
            }
        });
    }

    async loadFiles () {
        try {
            const response = await fetch('/api/files');
            if (!response.ok) throw new Error(`HTTP error! status: ${ response.status }`);

            const data = await response.json();
            this.displayFiles(data.files || []);
        } catch (error) {
            console.error('Error loading files:', error);
        }
    }

    displayFiles (files) {
        const container = document.getElementById('filesList');
        if (!container) return;

        if (files.length === 0) {
            container.innerHTML = '<p class="placeholder">No files found</p>';
            return;
        }

        container.innerHTML = files.map(file => {
            const icon = this.getFileIcon(file.type);
            const size = this.formatFileSize(file.size);
            const modified = new Date(file.modified).toLocaleDateString();

            return `
                <div class="file-item" data-path="${ file.path }">
                    <div class="file-icon">${ icon }</div>
                    <div class="file-info">
                        <div class="file-name" title="${ file.path }">${ file.name }</div>
                        <div class="file-details">
                            <span>${ size }</span>
                            <span>${ modified }</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Add click handlers
        container.querySelectorAll('.file-item').forEach(item => {
            item.addEventListener('click', () => {
                const path = item.dataset.path;
                this.openFile(path);
            });
        });
    }

    getFileIcon (extension) {
        const icons = {
            '.py': '<i class="fab fa-python"></i>',
            '.js': '<i class="fab fa-js-square"></i>',
            '.html': '<i class="fab fa-html5"></i>',
            '.css': '<i class="fab fa-css3-alt"></i>',
            '.md': '<i class="fab fa-markdown"></i>',
            '.json': '<i class="fas fa-brackets-curly"></i>',
            '.yaml': '<i class="fas fa-file-code"></i>',
            '.yml': '<i class="fas fa-file-code"></i>',
            '.txt': '<i class="fas fa-file-text"></i>'
        };
        return icons[extension] || '<i class="fas fa-file"></i>';
    }

    formatFileSize (bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    async openFile (path) {
        try {
            const response = await fetch(`/api/file/${ encodeURIComponent(path) }`);
            if (!response.ok) throw new Error(`HTTP error! status: ${ response.status }`);

            const data = await response.json();
            this.showFileModal(data);
        } catch (error) {
            console.error('Error opening file:', error);
            alert('Error opening file: ' + error.message);
        }
    }

    showFileModal (fileData) {
        const modal = document.getElementById('fileModal');
        const title = document.getElementById('fileModalTitle');
        const content = document.getElementById('fileModalContent');

        title.textContent = fileData.path;
        content.textContent = fileData.content;
        content.className = `language-${ this.getLanguageFromExtension(fileData.type) }`;

        modal.classList.add('active');

        // Re-highlight syntax
        if (window.Prism) {
            Prism.highlightElement(content);
        }
    }

    getLanguageFromExtension (ext) {
        const languages = {
            '.py': 'python',
            '.js': 'javascript',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml'
        };
        return languages[ext] || 'text';
    }

    closeModal () {
        document.getElementById('fileModal').classList.remove('active');
    }

    async loadLogs () {
        try {
            const response = await fetch('/api/logs');
            if (!response.ok) throw new Error(`HTTP error! status: ${ response.status }`);

            const data = await response.json();
            this.displayLogs(data.logs || []);
        } catch (error) {
            console.error('Error loading logs:', error);
        }
    }

    displayLogs (logs) {
        const container = document.getElementById('logsContainer');
        if (!container) return;

        if (logs.length === 0) {
            container.innerHTML = '<p class="placeholder">No logs available</p>';
            return;
        }

        container.innerHTML = logs.map(log => {
            let logClass = 'log-entry';
            if (log.content.toLowerCase().includes('error')) logClass += ' error';
            else if (log.content.toLowerCase().includes('warning')) logClass += ' warning';
            else if (log.content.toLowerCase().includes('info')) logClass += ' info';

            return `<div class="${ logClass }">[${ log.source }] ${ log.content }</div>`;
        }).join('');

        // Auto-scroll to bottom
        container.scrollTop = container.scrollHeight;
    }

    loadArchitecture () {
        // Placeholder for architecture diagram
        const container = document.getElementById('architectureDiagram');
        if (container) {
            // This would integrate with a diagramming library like D3.js or mermaid
            container.innerHTML = `
                <div class="placeholder">
                    <i class="fas fa-sitemap"></i>
                    <p>Architecture diagram will be displayed here</p>
                    <p>Integration with diagramming tools coming soon...</p>
                </div>
            `;
        }
    }

    clearChat () {
        const container = document.getElementById('conversationContainer');
        const messages = container.querySelectorAll('.chat-message');
        messages.forEach(msg => msg.remove());
    }

    clearLogs () {
        const container = document.getElementById('logsContainer');
        if (container) {
            container.innerHTML = '<p class="placeholder">Logs cleared</p>';
        }
    }

    updateSystemTime () {
        const systemTime = document.getElementById('systemTime');
        if (systemTime) {
            systemTime.textContent = new Date().toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit'
            });
        }
    }

    // Sandbox functionality
    async runSandboxCode () {
        const editor = document.getElementById('sandboxEditor');
        const output = document.getElementById('sandboxOutput');
        const activeTab = document.querySelector('.lang-tab.active');

        if (!editor || !output) {
            console.error('Sandbox elements not found');
            return;
        }

        const code = editor.value.trim();
        if (!code) {
            this.displaySandboxOutput('No code to execute', 'error');
            return;
        }

        const language = activeTab ? activeTab.dataset.lang : 'python';

        // Show running state
        const previewContainer = document.querySelector('.preview-container');
        if (previewContainer) {
            previewContainer.classList.add('preview-running');
        }

        // Update run button to show loading state
        const runBtn = document.getElementById('runCodeBtn');
        if (runBtn) {
            const originalText = runBtn.innerHTML;
            runBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running...';
            runBtn.disabled = true;
        }

        try {
            console.log(`Executing ${ language } code:`, code.substring(0, 50) + '...');

            const response = await fetch('/api/execute-code', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    code: code,
                    language: language
                })
            });

            console.log('Response status:', response.status);

            const result = await response.json();
            console.log('Execution result:', result);

            if (response.ok && result.status === 'success') {
                const output = result.output || result.message || '(No output)';
                this.displaySandboxOutput(output, 'success');
            } else {
                const error = result.error || result.message || 'Execution failed';
                this.displaySandboxOutput(error, 'error');
            }
        } catch (error) {
            console.error('Network error:', error);
            this.displaySandboxOutput(`Network error: ${ error.message }`, 'error');
        } finally {
            // Remove running state and restore button
            if (previewContainer) {
                previewContainer.classList.remove('preview-running');
            }

            if (runBtn) {
                runBtn.innerHTML = '<i class="fas fa-play"></i> <span>Run</span>';
                runBtn.disabled = false;
            }
        }
    }

    displaySandboxOutput (content, type = 'info') {
        const output = document.getElementById('sandboxOutput');
        const outputContainer = output.closest('.output-section');

        // Clear placeholder
        const placeholder = output.querySelector('.output-placeholder');
        if (placeholder) {
            placeholder.remove();
        }

        // Add timestamp
        const timestamp = new Date().toLocaleTimeString();
        const timestampDiv = document.createElement('div');
        timestampDiv.className = `output-${ type }`;
        timestampDiv.style.borderBottom = '1px solid #475569';
        timestampDiv.style.paddingBottom = '8px';
        timestampDiv.style.marginBottom = '12px';
        timestampDiv.style.fontSize = '11px';
        timestampDiv.style.opacity = '0.8';
        timestampDiv.textContent = `[${ timestamp }] Execution ${ type }`;

        // Add output content
        const contentDiv = document.createElement('div');
        contentDiv.className = `output-${ type }`;
        contentDiv.style.whiteSpace = 'pre-wrap';
        contentDiv.style.fontFamily = 'Monaco, Menlo, Ubuntu Mono, monospace';
        contentDiv.style.lineHeight = '1.4';
        contentDiv.textContent = content;

        // Update output container styling
        outputContainer.className = `output-section ${ type }`;

        // Append to output
        output.appendChild(timestampDiv);
        output.appendChild(contentDiv);

        // Auto-scroll to bottom
        output.scrollTop = output.scrollHeight;
    }

    clearSandbox () {
        const activeTab = document.querySelector('.lang-tab.active');
        const language = activeTab ? activeTab.dataset.lang : 'python';
        document.getElementById('sandboxEditor').value = this.getDefaultCode(language);
        this.clearSandboxOutput();
    }

    clearSandboxOutput () {
        const output = document.getElementById('sandboxOutput');
        const outputContainer = output.closest('.output-section');

        output.innerHTML = `
            <div class="output-placeholder">
                <div class="placeholder-icon">
                    <i class="fas fa-play-circle"></i>
                </div>
                <div class="placeholder-text">
                    <h4>Ready to Execute</h4>
                    <p>Click the <strong>Run</strong> button or press <kbd>Ctrl + Enter</kbd> to execute your code</p>
                    <div class="execution-tips">
                        <div class="tip">
                            <i class="fas fa-lightbulb"></i>
                            <span>Supports Python, JavaScript, and HTML</span>
                        </div>
                        <div class="tip">
                            <i class="fas fa-clock"></i>
                            <span>30-second execution timeout</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Reset container styling
        outputContainer.className = 'output-section';
    }

    toggleSandboxMaximize () {
        const sandboxSection = document.querySelector('.sandbox-section');
        const maximizeBtn = document.getElementById('maximizeSandboxBtn');
        const editor = document.getElementById('sandboxEditor');

        // Get current editor content and selected language
        const currentCode = editor.value;
        const activeTab = document.querySelector('.lang-tab.active');
        const currentLang = activeTab ? activeTab.dataset.lang : 'python';

        // Create sandbox HTML for new window
        const sandboxHTML = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DevCrew Sandbox - ${ currentLang.charAt(0).toUpperCase() + currentLang.slice(1) }</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary: #2563eb;
            --success-color: #10b981;
            --error-color: #ef4444;
            --warning-color: #f59e0b;
            --border: #e2e8f0;
            --bg-secondary: #f8fafc;
            --text-primary: #0f172a;
            --text-secondary: #64748b;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            background: #ffffff;
        }
        
        .sandbox-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            background: var(--bg-secondary);
        }
        
        .sandbox-header h1 {
            margin: 0;
            color: var(--text-primary);
            font-size: 18px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .sandbox-controls {
            display: flex;
            gap: 12px;
        }
        
        .btn-small {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 8px 12px;
            border: 1px solid var(--border);
            background: white;
            color: var(--text-secondary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s ease;
        }
        
        .btn-small:hover {
            background: var(--bg-secondary);
            color: var(--text-primary);
        }
        
        #runCodeBtn {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }
        
        #runCodeBtn:hover {
            background: #1d4ed8;
        }
        
        .sandbox-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            flex: 1;
            overflow: hidden;
        }
        
        .sandbox-editor {
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        
        .sandbox-tabs {
            display: flex;
            border-bottom: 1px solid var(--border);
            background: var(--bg-secondary);
        }
        
        .sandbox-tab {
            padding: 8px 16px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .sandbox-tab:hover {
            background: #f1f5f9;
            color: var(--text-primary);
        }
        
        .sandbox-tab.active {
            background: var(--primary);
            color: white;
        }
        
        #sandboxEditor {
            flex: 1;
            border: none;
            padding: 16px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 14px;
            background: white;
            color: var(--text-primary);
            resize: none;
            outline: none;
            line-height: 1.5;
        }
        
        .sandbox-output {
            display: flex;
            flex-direction: column;
            background: #1e293b;
        }
        
        .sandbox-output-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 16px;
            border-bottom: 1px solid var(--border);
            background: var(--bg-secondary);
            font-size: 12px;
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        .sandbox-output-content {
            flex: 1;
            padding: 16px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 13px;
            color: #e2e8f0;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        
        .output-placeholder {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #64748b;
            font-style: italic;
        }
        
        .output-placeholder i {
            font-size: 24px;
            margin-bottom: 8px;
            opacity: 0.5;
        }
        
        .btn-tiny {
            padding: 4px 6px;
            font-size: 10px;
            border: 1px solid var(--border);
            background: transparent;
            color: var(--text-secondary);
            border-radius: 3px;
            cursor: pointer;
        }
        
        .output-success { color: var(--success-color); }
        .output-error { color: var(--error-color); }
        .output-info { color: #64748b; }
        
        .sandbox-running #runCodeBtn {
            background: var(--warning-color);
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="sandbox-header">
        <h1><i class="fas fa-play-circle"></i> DevCrew Code Sandbox</h1>
        <div class="sandbox-controls">
            <button class="btn-small" id="runCodeBtn">
                <i class="fas fa-play"></i> Run Code
            </button>
            <button class="btn-small" id="clearSandboxBtn">
                <i class="fas fa-broom"></i> Clear
            </button>
            <button class="btn-small" id="clearOutputBtn">
                <i class="fas fa-trash"></i> Clear Output
            </button>
        </div>
    </div>
    
    <div class="sandbox-container">
        <div class="sandbox-editor">
            <div class="sandbox-tabs">
                <button class="sandbox-tab ${ currentLang === 'python' ? 'active' : '' }" data-lang="python">
                    <i class="fab fa-python"></i> Python
                </button>
                <button class="sandbox-tab ${ currentLang === 'javascript' ? 'active' : '' }" data-lang="javascript">
                    <i class="fab fa-js"></i> JavaScript
                </button>
                <button class="sandbox-tab ${ currentLang === 'html' ? 'active' : '' }" data-lang="html">
                    <i class="fab fa-html5"></i> HTML
                </button>
            </div>
            <textarea id="sandboxEditor" placeholder="Write your code here...">${ currentCode }</textarea>
        </div>
        
        <div class="sandbox-output">
            <div class="sandbox-output-header">
                <span><i class="fas fa-terminal"></i> Output</span>
                <button class="btn-tiny" id="clearOutputOnly">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="sandbox-output-content" id="sandboxOutput">
                <div class="output-placeholder">
                    <i class="fas fa-play-circle"></i>
                    <p>Run code to see output here</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Get current origin for API calls
        const API_BASE = window.location.origin;
        
        // Language switching
        document.querySelectorAll('.sandbox-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const targetLang = e.currentTarget.dataset.lang;
                
                // Update tab buttons
                document.querySelectorAll('.sandbox-tab').forEach(t => t.classList.remove('active'));
                e.currentTarget.classList.add('active');
                
                // Update editor content
                const editor = document.getElementById('sandboxEditor');
                editor.value = getDefaultCode(targetLang);
                
                // Clear output
                clearOutput();
            });
        });
        
        // Run code
        document.getElementById('runCodeBtn').addEventListener('click', async () => {
            const editor = document.getElementById('sandboxEditor');
            const activeTab = document.querySelector('.sandbox-tab.active');
            const code = editor.value.trim();
            const language = activeTab ? activeTab.dataset.lang : 'python';
            
            if (!code) {
                displayOutput('No code to execute', 'error');
                return;
            }
            
            document.body.classList.add('sandbox-running');
            
            try {
                const response = await fetch(API_BASE + '/api/execute-code', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code, language })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    displayOutput(result.output || result.message, 'success');
                } else {
                    displayOutput(result.error || 'Execution failed', 'error');
                }
            } catch (error) {
                displayOutput('Network error: ' + error.message, 'error');
            } finally {
                document.body.classList.remove('sandbox-running');
            }
        });
        
        // Clear sandbox
        document.getElementById('clearSandboxBtn').addEventListener('click', () => {
            const activeTab = document.querySelector('.sandbox-tab.active');
            const language = activeTab ? activeTab.dataset.lang : 'python';
            document.getElementById('sandboxEditor').value = getDefaultCode(language);
            clearOutput();
        });
        
        // Clear output only
        document.getElementById('clearOutputBtn').addEventListener('click', clearOutput);
        document.getElementById('clearOutputOnly').addEventListener('click', clearOutput);
        
        function displayOutput(content, type = 'info') {
            const output = document.getElementById('sandboxOutput');
            const placeholder = output.querySelector('.output-placeholder');
            if (placeholder) placeholder.remove();
            
            const timestamp = new Date().toLocaleTimeString();
            const timestampDiv = document.createElement('div');
            timestampDiv.className = 'output-' + type;
            timestampDiv.style.borderBottom = '1px solid #334155';
            timestampDiv.style.paddingBottom = '8px';
            timestampDiv.style.marginBottom = '12px';
            timestampDiv.style.fontSize = '11px';
            timestampDiv.textContent = '[' + timestamp + '] Execution ' + type;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'output-' + type;
            contentDiv.style.whiteSpace = 'pre-wrap';
            contentDiv.textContent = content;
            
            output.appendChild(timestampDiv);
            output.appendChild(contentDiv);
            output.scrollTop = output.scrollHeight;
        }
        
        function clearOutput() {
            document.getElementById('sandboxOutput').innerHTML = 
                '<div class="output-placeholder"><i class="fas fa-play-circle"></i><p>Run code to see output here</p></div>';
        }
        
        function getDefaultCode(language) {
            const codes = {
                python: \`# Example Python code
print("Hello from DevCrew Sandbox!")

# You can test code snippets here
x = 42
y = 8
result = x + y
print(f"The answer is: {result}")

# Import libraries and run experiments
import datetime
print(f"Current time: {datetime.datetime.now()}")\`,
                
                javascript: \`// Example JavaScript code
console.log("Hello from DevCrew Sandbox!");

// You can test JavaScript here
const x = 42;
const y = 8;
const result = x + y;
console.log(\\\`The answer is: \\\${result}\\\`);

// Work with objects and arrays
const data = { name: "DevCrew", type: "AI Agents" };
console.log("Project:", data);\`,
                
                html: \`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DevCrew Sandbox</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container { 
            max-width: 600px; 
            margin: 0 auto; 
            text-align: center; 
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 DevCrew Sandbox</h1>
        <p>This is a sample HTML page running in the sandbox!</p>
        <button onclick="alert('Hello from DevCrew!')">Click Me</button>
    </div>
</body>
</html>\`
            };
            return codes[language] || codes.python;
        }
        
        // Keyboard shortcut - Ctrl/Cmd + Enter to run
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                document.getElementById('runCodeBtn').click();
            }
        });
    </script>
</body>
</html>`;

        // Open sandbox in new tab
        const newTab = window.open('', '_blank');
        newTab.document.write(sandboxHTML);
        newTab.document.close();
    }

    switchSandboxLanguage (e) {
        const targetLang = e.currentTarget.dataset.lang;

        // Update tab buttons
        document.querySelectorAll('.lang-tab').forEach(tab => tab.classList.remove('active'));
        e.currentTarget.classList.add('active');

        // Update editor content with default code for the language
        const editor = document.getElementById('sandboxEditor');
        editor.value = this.getDefaultCode(targetLang);

        // Clear output
        this.clearSandboxOutput();
    }

    getDefaultCode (language) {
        const defaultCodes = {
            python: `# Example Python code
print("Hello from DevCrew Sandbox!")

# You can test code snippets here
x = 42
y = 8
result = x + y
print(f"The answer is: {result}")

# Import libraries and run experiments
import datetime
print(f"Current time: {datetime.datetime.now()}")`,

            javascript: `// Example JavaScript code
console.log("Hello from DevCrew Sandbox!");

// You can test JavaScript here
const x = 42;
const y = 8;
const result = x + y;
console.log(\`The answer is: \${result}\`);

// Work with objects and arrays
const data = { name: "DevCrew", type: "AI Agents" };
console.log("Project:", data);`,

            html: `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DevCrew Sandbox</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container { 
            max-width: 600px; 
            margin: 0 auto; 
            text-align: center; 
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 DevCrew Sandbox</h1>
        <p>This is a sample HTML page running in the sandbox!</p>
        <button onclick="alert('Hello from DevCrew!')">Click Me</button>
    </div>
</body>
</html>`
        };

        return defaultCodes[language] || defaultCodes.python;
    }

    setupAutoRefresh () {
        // Refresh status every 30 seconds
        setInterval(() => {
            if (this.currentTab === 'overview') {
                this.loadStatus();
            }
        }, 30000);

        // Refresh logs every 10 seconds if visible
        setInterval(() => {
            if (this.currentRightTab === 'logs') {
                this.loadLogs();
            }
        }, 10000);

        // Check for new generated code every 5 seconds
        setInterval(() => {
            this.checkForGeneratedCode();
        }, 5000);

        // Update system time every minute
        setInterval(() => {
            this.updateSystemTime();
        }, 60000);

        // Ping WebSocket every 30 seconds
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    async checkForGeneratedCode () {
        try {
            const response = await fetch('/api/generated-code');
            if (!response.ok) return;

            const data = await response.json();

            if (data.status === 'success' && data.code_data) {
                const codeData = data.code_data;

                // Check if this is new code (different from what's currently in editor)
                const editor = document.getElementById('sandboxEditor');
                const currentCode = editor.value.trim();
                const newCode = codeData.code.trim();

                // Check if we should force clear the sandbox before displaying new code
                const forceClear = codeData.force_clear === true;

                // Enhanced detection: Check if the code is different from current content
                // Prioritize any generated code over default templates
                const isNewCode = newCode && (
                    forceClear ||
                    newCode !== currentCode || // Different from current editor content
                    (currentCode === this.getDefaultCode(codeData.language)) // Current content is just template
                );

                // Additional check: Only filter out actual template examples (not user-generated content)
                const isActualTemplate = newCode === this.getDefaultCode(codeData.language) ||
                    newCode.includes('# Example Python code\nprint("Hello from DevCrew Sandbox!")');

                if (isNewCode && !isActualTemplate) {
                    this.loadGeneratedCodeIntoSandbox(codeData);

                    // Auto-focus sandbox if code was generated from user input
                    if (codeData.description && (
                        codeData.description.includes('user') ||
                        codeData.description.includes('request') ||
                        codeData.description.includes('task') ||
                        codeData.agent === 'coder' ||
                        codeData.agent === 'debug_script'
                    )) {
                        // Switch to overview tab to show the sandbox
                        if (this.currentTab !== 'overview') {
                            document.querySelector('[data-tab="overview"]').click();
                        }
                    }
                }
            }
        } catch (error) {
            // Silently fail - don't spam console for normal polling
            console.debug('Generated code check failed:', error);
        }
    }

    loadGeneratedCodeIntoSandbox (codeData) {
        try {
            const editor = document.getElementById('sandboxEditor');
            const languageTabs = document.querySelectorAll('.lang-tab');

            // Switch to the appropriate language tab
            languageTabs.forEach(tab => {
                tab.classList.remove('active');
                if (tab.dataset.lang === codeData.language) {
                    tab.classList.add('active');
                }
            });

            // Load the generated code into the editor
            editor.value = codeData.code;

            // Clear any existing output
            this.clearSandboxOutput();

            // Show notification about new code
            this.showCodeLoadedNotification(codeData);

            // Switch to the preview tab if not already there
            if (this.currentTab !== 'overview') {
                document.querySelector('[data-tab="overview"]').click();
            }

            console.log('Loaded generated code into sandbox:', codeData.filename || 'Generated Code');

        } catch (error) {
            console.error('Error loading generated code:', error);
        }
    }

    showCodeLoadedNotification (codeData) {
        // Create a temporary notification
        const notification = document.createElement('div');
        notification.className = 'code-notification';
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-code"></i>
                <div class="notification-text">
                    <strong>New Code Generated!</strong>
                    <div class="notification-details">
                        ${ codeData.filename || 'Generated Code' } (${ codeData.language })
                        ${ codeData.description ? `<br><small>${ codeData.description }</small>` : '' }
                    </div>
                </div>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        // Add styles if not already present
        if (!document.getElementById('notification-styles')) {
            const styles = document.createElement('style');
            styles.id = 'notification-styles';
            styles.textContent = `
                .code-notification {
                    position: fixed;
                    top: 80px;
                    right: 20px;
                    z-index: 1000;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                    animation: slideInRight 0.3s ease-out;
                    max-width: 350px;
                }
                
                .notification-content {
                    display: flex;
                    align-items: center;
                    padding: 16px;
                    gap: 12px;
                }
                
                .notification-content i.fas {
                    font-size: 24px;
                    opacity: 0.9;
                }
                
                .notification-text {
                    flex: 1;
                }
                
                .notification-text strong {
                    display: block;
                    margin-bottom: 4px;
                }
                
                .notification-details {
                    font-size: 13px;
                    opacity: 0.9;
                }
                
                .notification-close {
                    background: none;
                    border: none;
                    color: white;
                    cursor: pointer;
                    padding: 4px;
                    opacity: 0.7;
                    transition: opacity 0.2s;
                }
                
                .notification-close:hover {
                    opacity: 1;
                }
                
                @keyframes slideInRight {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
            `;
            document.head.appendChild(styles);
        }

        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideInRight 0.3s ease-out reverse';
                setTimeout(() => notification.remove(), 300);
            }
        }, 5000);
    }
}

// Initialize the interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.devCrewInterface = new DevCrewWebInterface();
});

// Add some additional CSS for task items that wasn't in the main CSS
const additionalStyles = `
<style>
.task-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 8px;
    transition: background-color 0.2s ease;
}

.task-item:hover {
    background: #f8fafc;
}

.task-info {
    flex: 1;
    min-width: 0;
}

.task-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}

.task-details {
    display: flex;
    gap: 8px;
    font-size: 12px;
    color: var(--text-secondary);
}

.task-reason {
    text-transform: capitalize;
}

.task-id {
    font-family: monospace;
    background: var(--border);
    padding: 1px 4px;
    border-radius: 2px;
}

.task-time {
    font-size: 12px;
    color: var(--text-secondary);
}

.placeholder {
    text-align: center;
    color: var(--text-secondary);
    padding: 40px 20px;
    font-style: italic;
}
</style>
`;

// Inject additional styles
document.head.insertAdjacentHTML('beforeend', additionalStyles);