import { useState, useCallback, useEffect, useRef } from "react";
import Editor, { type OnMount, loader } from "@monaco-editor/react";
import type { editor as MonacoEditor } from "monaco-editor";
import { open } from "@tauri-apps/plugin-dialog";
import { readDir, readTextFile } from "@tauri-apps/plugin-fs";
import { LazyStore } from "@tauri-apps/plugin-store";
import { invoke } from "@tauri-apps/api/core";
import * as monaco from 'monaco-editor';
import "./App.css";

// Configure Monaco to use local package instead of CDN
loader.config({ monaco });

interface FileNode {
  name: string;
  path: string;
  isDir: boolean;
  children?: FileNode[];
  expanded?: boolean;
}

interface Tab {
  path: string;
  name: string;
  content: string;
  language: string;
}

type ProviderType = "ollama" | "nvidia" | "custom";

interface ProviderSettings {
  type: ProviderType;
  ollamaBaseUrl: string;
  ollamaModel: string;
  nvidiaApiKey: string;
  nvidiaModel: string;
  customBaseUrl: string;
  customApiKey: string;
  customModel: string;
}

interface EditorSettings {
  fontSize: number;
  tabSize: 2 | 4;
  wordWrap: boolean;
}

interface ChatMessage {
  role: "user" | "assistant" | "error";
  content: string;
}

const DEFAULT_PROVIDER: ProviderSettings = {
  type: "ollama",
  ollamaBaseUrl: "http://localhost:11434",
  ollamaModel: "qwen2.5-coder:7b",
  nvidiaApiKey: "",
  nvidiaModel: "meta/llama-3.1-8b-instruct",
  customBaseUrl: "",
  customApiKey: "",
  customModel: "",
};

const DEFAULT_EDITOR: EditorSettings = {
  fontSize: 14,
  tabSize: 2,
  wordWrap: false,
};

let storeInstance: LazyStore | null = null;
async function getStore(): Promise<LazyStore> {
  if (!storeInstance) {
    storeInstance = new LazyStore("settings.json");
  }
  return storeInstance;
}

function App() {
  const [rootPath, setRootPath] = useState<string>("");
  const [fileTree, setFileTree] = useState<FileNode[]>([]);
  const [tabs, setTabs] = useState<Tab[]>([]);
  const [activeTabPath, setActiveTabPath] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [providerSettings, setProviderSettings] = useState<ProviderSettings>(DEFAULT_PROVIDER);
  const [editorSettings, setEditorSettings] = useState<EditorSettings>(DEFAULT_EDITOR);
  const [draftProvider, setDraftProvider] = useState<ProviderSettings>(DEFAULT_PROVIDER);
  const [draftEditor, setDraftEditor] = useState<EditorSettings>(DEFAULT_EDITOR);
  const [embeddingApiKey, setEmbeddingApiKey] = useState<string>("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [contextFiles, setContextFiles] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<MonacoEditor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<typeof import('monaco-editor') | null>(null);
  const completionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingCompletionRef = useRef<string | null>(null);
  const completionAbortRef = useRef(false);
  const completionProviderRef = useRef<any>(null);
  const [isCompleting, setIsCompleting] = useState(false);
  const [terminalOpen, setTerminalOpen] = useState(false);
  const [terminalOutput, setTerminalOutput] = useState<string>("");
  const [terminalInput, setTerminalInput] = useState<string>("");
  const [terminalHeight, setTerminalHeight] = useState(200);
  const [terminalId, setTerminalId] = useState<string | null>(null);
  const terminalOutputRef = useRef<HTMLDivElement>(null);
  const terminalResizingRef = useRef(false);
  const [indexStatus, setIndexStatus] = useState<{ type: 'idle' | 'indexing' | 'ready' | 'error'; message: string }>({ type: 'idle', message: '' });

  // Load settings on startup
  useEffect(() => {
    (async () => {
      try {
        const store = await getStore();
        const savedProvider = await store.get<ProviderSettings>("provider");
        const savedEditor = await store.get<EditorSettings>("editor");
        const savedEmbeddingKey = await store.get<string>("embeddingApiKey");
        
        if (savedProvider) {
          setProviderSettings(savedProvider);
          setDraftProvider(savedProvider);
        }
        if (savedEditor) {
          setEditorSettings(savedEditor);
          setDraftEditor(savedEditor);
        }
        
        // Set embedding API key, defaulting to existing NIM key if not set
        if (savedEmbeddingKey) {
          setEmbeddingApiKey(savedEmbeddingKey);
        } else if (savedProvider?.nvidiaApiKey) {
          setEmbeddingApiKey(savedProvider.nvidiaApiKey);
        }
      } catch (error) {
        console.error("Error loading settings:", error);
      }
    })();
  }, []);

  // Background index workspace when folder opens
  useEffect(() => {
    if (!rootPath || !embeddingApiKey) return;

    const runIndex = async () => {
      setIndexStatus({ type: 'indexing', message: 'Indexing...' });
      try {
        const result = await invoke<{ indexed: number; total: number }>('index_workspace', {
          folderPath: rootPath,
          apiKey: embeddingApiKey
        });
        setIndexStatus({ type: 'ready', message: `Index ready (${result.indexed} files)` });
      } catch (error) {
        const errMsg = typeof error === 'string' ? error : (error as Error).message || 'Unknown error';
        setIndexStatus({ type: 'error', message: `Index failed: ${errMsg}` });
        console.error('Indexing error:', error);
      }
    };

    runIndex();
  }, [rootPath, embeddingApiKey]);

  const saveSettings = useCallback(async () => {
    try {
      const store = await getStore();
      await store.set("provider", draftProvider);
      await store.set("editor", draftEditor);
      await store.set("embeddingApiKey", embeddingApiKey);
      await store.save();
      setProviderSettings(draftProvider);
      setEditorSettings(draftEditor);
      setSettingsOpen(false);
    } catch (error) {
      console.error("Error saving settings:", error);
    }
  }, [draftProvider, draftEditor, embeddingApiKey]);

  const getLanguageFromExtension = useCallback((filename: string): string => {
    const ext = filename.split('.').pop()?.toLowerCase();
    const languageMap: Record<string, string> = {
      'ts': 'typescript', 'tsx': 'typescript', 'js': 'javascript', 'jsx': 'javascript',
      'py': 'python', 'rs': 'rust', 'go': 'go', 'java': 'java', 'c': 'c', 'cpp': 'cpp',
      'cs': 'csharp', 'php': 'php', 'rb': 'ruby', 'swift': 'swift', 'kt': 'kotlin',
      'scala': 'scala', 'html': 'html', 'css': 'css', 'scss': 'scss', 'sass': 'sass',
      'json': 'json', 'xml': 'xml', 'yaml': 'yaml', 'yml': 'yaml', 'md': 'markdown',
      'sql': 'sql', 'sh': 'shell', 'bash': 'shell', 'zsh': 'shell', 'ps1': 'powershell',
      'dockerfile': 'dockerfile', 'toml': 'toml', 'ini': 'ini', 'conf': 'ini',
    };
    return languageMap[ext || ''] || 'plaintext';
  }, []);

  const buildFileTree = useCallback(async (dirPath: string): Promise<FileNode[]> => {
    try {
      const entries = await readDir(dirPath);
      const nodes: FileNode[] = [];
      for (const entry of entries) {
        const fullPath = `${dirPath}/${entry.name}`;
        const isDir = entry.isDirectory;
        const node: FileNode = { name: entry.name, path: fullPath, isDir: isDir };
        if (isDir) {
          node.children = await buildFileTree(fullPath);
        }
        nodes.push(node);
      }
      nodes.sort((a, b) => {
        if (a.isDir && !b.isDir) return -1;
        if (!a.isDir && b.isDir) return 1;
        return a.name.localeCompare(b.name);
      });
      return nodes;
    } catch (error) {
      console.error('Error reading directory:', error);
      return [];
    }
  }, []);

  const handleOpenFolder = useCallback(async () => {
    try {
      const selected = await open({ directory: true, multiple: false, title: "Select a folder" });
      if (selected && typeof selected === 'string') {
        setRootPath(selected);
        const tree = await buildFileTree(selected);
        setFileTree(tree);
      }
    } catch (error) {
      console.error('Error opening folder:', error);
      alert('Failed to open folder: ' + (error as Error).message);
    }
  }, [buildFileTree]);

  const handleFileClick = useCallback(async (node: FileNode) => {
    if (node.isDir) return;
    const existingTab = tabs.find(t => t.path === node.path);
    if (existingTab) { setActiveTabPath(node.path); return; }
    try {
      const content = await readTextFile(node.path);
      const language = getLanguageFromExtension(node.name);
      const newTab: Tab = { path: node.path, name: node.name, content, language };
      setTabs(prev => [...prev, newTab]);
      setActiveTabPath(node.path);
    } catch (error) {
      console.error('Error reading file:', error);
    }
  }, [getLanguageFromExtension, tabs]);

  const closeTab = useCallback((tabPath: string) => {
    setTabs(prev => {
      const idx = prev.findIndex(t => t.path === tabPath);
      const newTabs = prev.filter(t => t.path !== tabPath);
      if (tabPath === activeTabPath && newTabs.length > 0) {
        const newIdx = Math.min(idx, newTabs.length - 1);
        setActiveTabPath(newTabs[newIdx].path);
      } else if (newTabs.length === 0) {
        setActiveTabPath(null);
      }
      return newTabs;
    });
  }, [activeTabPath]);

  const toggleFolder = useCallback((node: FileNode) => {
    const toggleRecursive = (nodes: FileNode[]): FileNode[] => {
      return nodes.map(n => {
        if (n.path === node.path) return { ...n, expanded: !n.expanded };
        if (n.children) return { ...n, children: toggleRecursive(n.children) };
        return n;
      });
    };
    setFileTree(toggleRecursive(fileTree));
  }, [fileTree]);

  const renderFileTree = useCallback((nodes: FileNode[], depth: number = 0): React.ReactNode[] => {
    return nodes.map((node) => (
      <div key={node.path}>
        <div
          className={`file-tree-item ${depth > 0 ? 'nested' : ''}`}
          style={{ paddingLeft: `${depth * 16 + 8}px` }}
          onClick={() => node.isDir ? toggleFolder(node) : handleFileClick(node)}
        >
          <span className="file-tree-icon">
            {node.isDir ? (node.expanded ? '▼' : '▶') : '📄'}
          </span>
          <span className="file-tree-name">{node.name}</span>
        </div>
        {node.isDir && node.expanded && node.children && (
          <div className="file-tree-children">
            {renderFileTree(node.children, depth + 1)}
          </div>
        )}
      </div>
    ));
  }, [toggleFolder, handleFileClick]);

  const activeTab = tabs.find(t => t.path === activeTabPath);

  const getProviderConfig = useCallback(() => {
    switch (providerSettings.type) {
      case "ollama":
        return { baseUrl: providerSettings.ollamaBaseUrl, model: providerSettings.ollamaModel, apiKey: "" };
      case "nvidia":
        return { baseUrl: "https://integrate.api.nvidia.com/v1", model: providerSettings.nvidiaModel, apiKey: providerSettings.nvidiaApiKey };
      case "custom":
        return { baseUrl: providerSettings.customBaseUrl, model: providerSettings.customModel, apiKey: providerSettings.customApiKey };
    }
  }, [providerSettings]);

  const triggerCompletion = useCallback(async () => {
    if (!editorRef.current || !activeTab) return;

    const position = editorRef.current.getPosition();
    if (!position) return;

    const model = editorRef.current.getModel();
    if (!model) return;

    // Get text on current line
    const lineContent = model.getLineContent(position.lineNumber);
    const textBeforeCursor = lineContent.substring(0, position.column - 1);

    // Only trigger if at least 3 characters typed on current line
    if (textBeforeCursor.trim().length < 3) return;

    completionAbortRef.current = false;
    setIsCompleting(true);

    const config = getProviderConfig();
    const systemPrompt = `You are a code completion engine. Complete the code at the cursor. Return ONLY the completion text, no explanation, no markdown, no backticks. Just the raw code that comes next.`;

    // Get last 2000 chars of file content before cursor for context
    const fullContentBeforeCursor = model.getValueInRange({
      startLineNumber: 1,
      startColumn: 1,
      endLineNumber: position.lineNumber,
      endColumn: position.column,
    });
    const contextLength = Math.min(fullContentBeforeCursor.length, 2000);
    const context = fullContentBeforeCursor.slice(-contextLength);

    try {
      const url = `${config.baseUrl.replace(/\/+$/, "")}/chat/completions`;
      const result = await invoke<string>('ai_chat', {
        url,
        apiKey: config.apiKey,
        model: config.model,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: context },
        ],
      });

      if (completionAbortRef.current) return;

      const parsed = JSON.parse(result);
      const completionText = parsed.choices?.[0]?.message?.content?.trim() || '';

      if (!completionText || completionAbortRef.current) {
        setIsCompleting(false);
        return;
      }

      // Register inline completion provider
      if (monacoRef.current) {
        const monaco = monacoRef.current;
        
        // Dispose previous provider if it exists
        if (completionProviderRef.current) {
          completionProviderRef.current.dispose();
        }

        const provider = {
          provideInlineCompletions: () => ({
            items: [
              {
                insertText: completionText,
                range: new monaco.Range(
                  position.lineNumber,
                  position.column,
                  position.lineNumber,
                  position.column
                ),
              },
            ],
          }),
        } as any;

        completionProviderRef.current = monaco.languages.registerInlineCompletionsProvider(
          activeTab.language,
          provider
        );

        pendingCompletionRef.current = completionText;
      }
    } catch (error) {
      console.error('Completion request failed:', error);
      setIsCompleting(false);
    }
  }, [activeTab, getProviderConfig]);

  const handleEditorMount: OnMount = useCallback((editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;

    // Register keyboard shortcuts for Tab (accept) and Escape (dismiss)
    editor.addCommand(monaco.KeyCode.Tab, () => {
      if (pendingCompletionRef.current) {
        const position = editor.getPosition();
        if (position) {
          editor.executeEdits('complete', [
            {
              range: new monaco.Range(
                position.lineNumber,
                position.column,
                position.lineNumber,
                position.column
              ),
              text: pendingCompletionRef.current,
            },
          ]);
          pendingCompletionRef.current = null;
          setIsCompleting(false);
        }
      }
    });

    editor.addCommand(monaco.KeyCode.Escape, () => {
      if (pendingCompletionRef.current) {
        pendingCompletionRef.current = null;
        setIsCompleting(false);
        completionAbortRef.current = true;
      }
    });
  }, []);

  const handleEditorChangeWithCompletion = useCallback((value: string | undefined) => {
    if (!activeTabPath || value === undefined) return;
    
    // Update tab content
    setTabs(prev => prev.map(t => t.path === activeTabPath ? { ...t, content: value } : t));

    // Reset completion abort flag on keystroke
    completionAbortRef.current = true;
    pendingCompletionRef.current = null;
    setIsCompleting(false);

    // Clear existing timer
    if (completionTimerRef.current) {
      clearTimeout(completionTimerRef.current);
    }

    // Set new debounce timer for completion trigger (600ms)
    completionTimerRef.current = setTimeout(() => {
      triggerCompletion();
    }, 600);
  }, [activeTabPath, triggerCompletion]);

  const clearTerminal = useCallback(() => {
    setTerminalOutput("");
  }, []);

  // Spawn terminal when first opened
  useEffect(() => {
    if (terminalOpen && !terminalId) {
      const spawnTerm = async () => {
        try {
          const cwd = rootPath || "";
          const id = await invoke<string>('spawn_terminal', { cwd });
          setTerminalId(id);
          setTerminalOutput("Terminal started\n");
        } catch (error) {
          setTerminalOutput(`Failed to start terminal: ${error}\n`);
        }
      };
      spawnTerm();
    }
  }, [terminalOpen, terminalId, rootPath]);

  // Kill terminal when closed
  useEffect(() => {
    if (!terminalOpen && terminalId) {
      invoke('kill_terminal', { terminalId }).catch(console.error);
      setTerminalId(null);
    }
  }, [terminalOpen, terminalId]);

  // Poll for terminal output
  useEffect(() => {
    if (!terminalId || !terminalOpen) return;
    
    const interval = setInterval(async () => {
      try {
        const output = await invoke<[string, string][]>('read_terminal_output', { terminalId });
        if (output.length > 0) {
          const text = output.map(([type, line]) => {
            if (type === 'stderr') return `<span class="terminal-error">${line}</span>`;
            return line;
          }).join('\n');
          setTerminalOutput(prev => prev + (prev ? '\n' : '') + text);
        }
      } catch (error) {
        console.error('Failed to read terminal output:', error);
      }
    }, 100);

    return () => clearInterval(interval);
  }, [terminalId, terminalOpen]);

  const handleTerminalInput = useCallback(async () => {
    const input = terminalInput.trim();
    if (!input || !terminalId) return;

    setTerminalOutput(prev => prev + `\n> ${input}\n`);
    
    try {
      await invoke('write_terminal_input', { terminalId, input });
    } catch (error) {
      setTerminalOutput(prev => prev + `Error: ${error}\n`);
    }
    
    setTerminalInput("");
  }, [terminalInput, terminalId]);

  const handleTerminalResize = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    terminalResizingRef.current = true;
    const startY = e.clientY;
    const startHeight = terminalHeight;

    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!terminalResizingRef.current) return;
      const deltaY = startY - moveEvent.clientY;
      const newHeight = Math.max(100, Math.min(600, startHeight + deltaY));
      setTerminalHeight(newHeight);
    };

    const handleMouseUp = () => {
      terminalResizingRef.current = false;
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  }, [terminalHeight]);

  useEffect(() => {
    if (terminalOutputRef.current) {
      terminalOutputRef.current.scrollTop = terminalOutputRef.current.scrollHeight;
    }
  }, [terminalOutput]);

  const sendMessage = useCallback(async () => {
    const text = chatInput.trim();
    if (!text || isStreaming) return;

    setChatInput("");
    const userMsg: ChatMessage = { role: "user", content: text };
    setMessages(prev => [...prev, userMsg]);
    setIsStreaming(true);

    const config = getProviderConfig();
    const fileContent = activeTab?.content || "";

    // Search for relevant codebase context
    let relevantChunks: any[] = [];
    let usedContextFiles: string[] = [];
    
    console.log('=== SEARCH DEBUG ===');
    console.log('Searching index for:', text);
    console.log('Current folder path:', rootPath);
    console.log('Embedding API key set:', embeddingApiKey ? 'yes' : 'no');
    console.log('Condition check:');
    console.log('  - rootPath exists:', !!rootPath);
    console.log('  - embedding api key exists:', !!embeddingApiKey);
    console.log('  - all conditions met:', !!(rootPath && embeddingApiKey));
    
    if (rootPath && embeddingApiKey) {
      try {
        const searchResults = await invoke<any[]>('search_index', {
          query: text,
          apiKey: embeddingApiKey,
          folderPath: rootPath
        });
        
        console.log('Search results:', searchResults);
        console.log('Raw search results:', JSON.stringify(searchResults, null, 2));
        
        // Filter by similarity score > 0.25 and take top 5
        relevantChunks = searchResults
          .filter((result: any) => result.similarity_score > 0.25)
          .slice(0, 5);
          
        console.log('Filtered chunks:', relevantChunks.length);
        console.log('Similarity scores:', searchResults.map((r: any) => r.similarity_score));
          
        usedContextFiles = relevantChunks.map((chunk: any) => chunk.file_path);
      } catch (error) {
        console.error('Failed to search index:', error);
      }
    }

    // Build system prompt with context
    const contextSection = relevantChunks.length > 0 
      ? `RELEVANT PROJECT CODE:\n\n${relevantChunks.map((chunk: any) => 
          `File: ${chunk.file_path}\n\`\`\`\n${chunk.content}\n\`\`\``
        ).join('\n\n')}`
      : 'No additional context available.';

    const systemPrompt = `You are an expert coding assistant. 
The user has opened a software project. Answer questions 
about the project using the context provided below.

${contextSection}

${activeTabPath ? `CURRENTLY OPEN FILE:\nFile: ${activeTabPath}\n\`\`\`\n${fileContent}\n\`\`\`` : ''}

Answer based on the actual project code shown above.`;

    setContextFiles(usedContextFiles);

    const conversationHistory = [...messages, userMsg]
      .filter(m => m.role !== "error")
      .map(m => ({ role: m.role, content: m.content }));

    console.log('=== FINAL SYSTEM PROMPT ===');
    console.log(systemPrompt);

    const requestMessages = [
      { role: "system" as const, content: systemPrompt },
      ...conversationHistory,
    ];

    console.log('AI Config:', {
      provider: providerSettings.type,
      baseUrl: config.baseUrl,
      apiKey: config.apiKey ? 'set' : 'MISSING',
      model: config.model,
      url: `${config.baseUrl.replace(/\/+$/, "")}/chat/completions`,
    });

    try {
      const url = `${config.baseUrl.replace(/\/+$/, "")}/chat/completions`;
      const result = await invoke<string>('ai_chat', {
        url,
        apiKey: config.apiKey,
        model: config.model,
        messages: requestMessages,
      });

      const parsed = JSON.parse(result);
      const assistantContent = parsed.choices?.[0]?.message?.content || 'No response';
      setMessages(prev => [...prev, { role: "assistant", content: assistantContent }]);
    } catch (error) {
      const errMsg = typeof error === 'string' ? error : (error as Error).message || 'Unknown error';
      setMessages(prev => [...prev, { role: "error", content: `Connection error: ${errMsg}` }]);
    }
    setIsStreaming(false);
  }, [chatInput, isStreaming, getProviderConfig, activeTab, activeTabPath, messages]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="app-container">
      {/* Left Sidebar - File Explorer */}
      <aside className="sidebar-left">
        <div className="sidebar-header">
          <button className="open-folder-btn" onClick={handleOpenFolder}>Open Folder</button>
        </div>
        <div className="file-explorer">
          {rootPath ? (
            <div className="file-tree">{renderFileTree(fileTree)}</div>
          ) : (
            <div className="placeholder">
              <p className="placeholder-text">File Explorer</p>
              <p className="placeholder-subtext">Open a folder to begin</p>
            </div>
          )}
        </div>
      </aside>

      {/* Center - Monaco Editor */}
      <main className="editor-container">
        <div className="tab-bar">
          {tabs.map(tab => (
            <div
              key={tab.path}
              className={`tab ${tab.path === activeTabPath ? 'active' : ''}`}
              onClick={() => setActiveTabPath(tab.path)}
            >
              <span className="tab-name">{tab.name}</span>
              <button className="tab-close" onClick={(e) => { e.stopPropagation(); closeTab(tab.path); }}>✕</button>
            </div>
          ))}
          <button className="settings-btn" onClick={() => { setDraftProvider(providerSettings); setDraftEditor(editorSettings); setSettingsOpen(true); }}>⚙</button>
          <button className="terminal-btn" onClick={() => setTerminalOpen(!terminalOpen)} title="Toggle Terminal">⌨</button>
        </div>
        <div className="editor-container-inner">
          <div className="editor-area">
            <Editor
              height="100%"
              language={activeTab?.language || 'typescript'}
              theme="vs-dark"
              value={activeTab?.content || ''}
              onChange={handleEditorChangeWithCompletion}
              onMount={handleEditorMount}
              options={{
                minimap: { enabled: false },
                fontSize: editorSettings.fontSize,
                lineNumbers: "on",
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: editorSettings.tabSize,
                wordWrap: editorSettings.wordWrap ? "on" : "off",
              }}
            />
            {isCompleting && <div className="completion-loading">⟳</div>}
          </div>

          {terminalOpen && (
            <>
              <div className="terminal-divider" onMouseDown={handleTerminalResize} />
              <div className="terminal-panel" style={{ height: `${terminalHeight}px` }}>
                <div className="terminal-header">
                  <span className="terminal-title">Terminal</span>
                  <button className="terminal-clear" onClick={clearTerminal}>Clear</button>
                </div>
                <div className="terminal-output" ref={terminalOutputRef} dangerouslySetInnerHTML={{ __html: terminalOutput || '<span class="terminal-prompt">Ready</span>' }} />
                <div className="terminal-input-area">
                  <span className="terminal-prompt-char">❯</span>
                  <input
                    type="text"
                    className="terminal-input"
                    placeholder="Enter command..."
                    value={terminalInput}
                    onChange={(e) => setTerminalInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") handleTerminalInput(); }}
                  />
                </div>
              </div>
            </>
          )}
        </div>
      </main>

      {/* Right Sidebar - AI Chat */}
      <aside className="sidebar-right">
        <div className="chat-container">
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="chat-placeholder">
                <p className="placeholder-text">AI Chat</p>
                <p className="placeholder-subtext">Messages will appear here</p>
              </div>
            ) : (
              messages.map((msg, i) => (
                <div key={i} className={`chat-bubble chat-bubble-${msg.role}`}>
                  <div className="chat-bubble-role">{msg.role === "user" ? "You" : msg.role === "assistant" ? "AI" : "Error"}</div>
                  <div className="chat-bubble-content">{msg.content}</div>
                  {msg.role === "assistant" && i === messages.length - 1 && (
                    <div className="chat-context">
                      <div className="chat-context-header" onClick={() => {
                        const el = document.querySelector(`.chat-context-files-${i}`);
                        if (el) {
                          el.classList.toggle('chat-context-files-expanded');
                        }
                      }}>
                        Context used: {contextFiles.length > 0 ? `${contextFiles.length} files` : 'Active file only'}
                        <span className="chat-context-toggle">▼</span>
                      </div>
                      <div className={`chat-context-files chat-context-files-${i}`}>
                        {contextFiles.map((file, j) => (
                          <div key={j} className="chat-context-file">{file}</div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
            {isStreaming && <div className="chat-streaming-indicator">●●●</div>}
            <div ref={messagesEndRef} />
          </div>
          <div className="chat-input-area">
            <input
              type="text"
              className="chat-input"
              placeholder="Ask AI..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
              disabled={isStreaming}
            />
          </div>
        </div>
      </aside>

      {/* Settings Modal */}
      {settingsOpen && (
        <div className="modal-overlay" onClick={() => setSettingsOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Settings</h2>
              <button className="modal-close" onClick={() => setSettingsOpen(false)}>✕</button>
            </div>
            <div className="modal-body">
              <h3>AI Provider</h3>
              <div className="settings-group">
                <label className="settings-label">Active Provider</label>
                <select className="settings-select" value={draftProvider.type} onChange={(e) => setDraftProvider({ ...draftProvider, type: e.target.value as ProviderType })}>
                  <option value="ollama">Ollama (Local)</option>
                  <option value="nvidia">NVIDIA NIM</option>
                  <option value="custom">Custom BYOK</option>
                </select>
              </div>

              {draftProvider.type === "ollama" && (
                <>
                  <div className="settings-group">
                    <label className="settings-label">Base URL</label>
                    <input className="settings-input" type="text" value={draftProvider.ollamaBaseUrl} onChange={(e) => setDraftProvider({ ...draftProvider, ollamaBaseUrl: e.target.value })} />
                  </div>
                  <div className="settings-group">
                    <label className="settings-label">Model</label>
                    <input className="settings-input" type="text" value={draftProvider.ollamaModel} onChange={(e) => setDraftProvider({ ...draftProvider, ollamaModel: e.target.value })} />
                  </div>
                </>
              )}

              {draftProvider.type === "nvidia" && (
                <>
                  <div className="settings-group">
                    <label className="settings-label">API Key</label>
                    <input className="settings-input" type="password" value={draftProvider.nvidiaApiKey} onChange={(e) => setDraftProvider({ ...draftProvider, nvidiaApiKey: e.target.value })} placeholder="nvapi-..." />
                  </div>
                  <div className="settings-group">
                    <label className="settings-label">Base URL</label>
                    <input className="settings-input" type="text" value="https://integrate.api.nvidia.com/v1" readOnly />
                  </div>
                  <div className="settings-group">
                    <label className="settings-label">Model</label>
                    <input className="settings-input" type="text" value={draftProvider.nvidiaModel} onChange={(e) => setDraftProvider({ ...draftProvider, nvidiaModel: e.target.value })} />
                  </div>
                </>
              )}

              {draftProvider.type === "custom" && (
                <>
                  <div className="settings-group">
                    <label className="settings-label">Base URL</label>
                    <input className="settings-input" type="text" value={draftProvider.customBaseUrl} onChange={(e) => setDraftProvider({ ...draftProvider, customBaseUrl: e.target.value })} placeholder="https://api.example.com/v1" />
                  </div>
                  <div className="settings-group">
                    <label className="settings-label">API Key</label>
                    <input className="settings-input" type="password" value={draftProvider.customApiKey} onChange={(e) => setDraftProvider({ ...draftProvider, customApiKey: e.target.value })} />
                  </div>
                  <div className="settings-group">
                    <label className="settings-label">Model</label>
                    <input className="settings-input" type="text" value={draftProvider.customModel} onChange={(e) => setDraftProvider({ ...draftProvider, customModel: e.target.value })} />
                  </div>
                </>
              )}

              <h3>Workspace Indexing</h3>
              <div className="settings-group">
                <label className="settings-label">NIM Embedding API Key (for workspace indexing)</label>
                <input 
                  className="settings-input" 
                  type="password" 
                  value={embeddingApiKey} 
                  onChange={(e) => setEmbeddingApiKey(e.target.value)} 
                  placeholder="nvapi-..." 
                />
              </div>

              <h3>Editor</h3>
              <div className="settings-group">
                <label className="settings-label">Font Size ({draftEditor.fontSize})</label>
                <input className="settings-input" type="range" min={12} max={24} value={draftEditor.fontSize} onChange={(e) => setDraftEditor({ ...draftEditor, fontSize: Number(e.target.value) })} />
              </div>
              <div className="settings-group">
                <label className="settings-label">Tab Size</label>
                <select className="settings-select" value={draftEditor.tabSize} onChange={(e) => setDraftEditor({ ...draftEditor, tabSize: Number(e.target.value) as 2 | 4 })}>
                  <option value={2}>2 spaces</option>
                  <option value={4}>4 spaces</option>
                </select>
              </div>
              <div className="settings-group">
                <label className="settings-label">Word Wrap</label>
                <label className="settings-toggle">
                  <input type="checkbox" checked={draftEditor.wordWrap} onChange={(e) => setDraftEditor({ ...draftEditor, wordWrap: e.target.checked })} />
                  <span>{draftEditor.wordWrap ? 'On' : 'Off'}</span>
                </label>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn-secondary" onClick={() => setSettingsOpen(false)}>Cancel</button>
              <button className="btn-primary" onClick={saveSettings}>Save</button>
            </div>
          </div>
        </div>
      )}

      {/* Status Bar */}
      <div className="status-bar">
        <span className={`index-status index-status-${indexStatus.type}`}>
          {indexStatus.message || 'Ready'}
        </span>
      </div>
    </div>
  );
}

export default App;
