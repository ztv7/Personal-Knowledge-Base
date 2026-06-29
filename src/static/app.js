(function() {
  'use strict';

  // ---- DOM refs ----
  const $ = (s) => document.querySelector(s);
  const chatMessages = $('#chat-messages');
  const chatInput = $('#chat-input');
  const btnSend = $('#btn-send');
  const uploadDrop = $('#upload-drop');
  const fileInput = $('#file-input');
  const uploadStatus = $('#upload-status');
  const fileList = $('#file-list');
  const btnExport = $('#btn-export');
  const btnClear = $('#btn-clear');
  const sidebarToggle = $('#sidebar-toggle');
  const sidebar = $('#sidebar');
  const btnSaveConfig = $('#btn-save-config');

  // ---- Config sliders with live value display ----
  const cfgKeys = ['chunk_size','chunk_overlap','top_k','rerank_top_n','temperature','relevance_threshold'];
  cfgKeys.forEach(k => {
    const el = $(`#cfg-${k}`);
    if (el) {
      el.addEventListener('input', () => {
        const valEl = $(`#val-${k}`);
        if (valEl) valEl.textContent = el.value;
      });
    }
  });

  // ---- State ----
  let isStreaming = false;

  // ---- Init ----
  loadFiles();
  loadConfig();

  // ---- Sidebar toggle ----
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    sidebarToggle.textContent = sidebar.classList.contains('collapsed') ? '>' : '<';
  });

  // ---- File upload ----
  uploadDrop.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) uploadFile(fileInput.files[0]);
  });

  uploadDrop.addEventListener('dragover', (e) => { e.preventDefault(); uploadDrop.classList.add('drag-over'); });
  uploadDrop.addEventListener('dragleave', () => uploadDrop.classList.remove('drag-over'));
  uploadDrop.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadDrop.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  async function uploadFile(file) {
    const form = new FormData();
    form.append('file', file);
    showUploadStatus('uploading', '上传中...');
    try {
      const resp = await fetch('/v1/files', { method: 'POST', body: form });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || err.error?.message || '上传失败');
      }
      showUploadStatus('success', `${file.name} 上传成功`);
      loadFiles();
    } catch (e) {
      showUploadStatus('error', e.message);
    } finally {
      fileInput.value = '';
    }
  }

  function showUploadStatus(type, msg) {
    uploadStatus.className = `upload-status ${type}`;
    uploadStatus.textContent = msg;
    uploadStatus.classList.remove('hidden');
    setTimeout(() => uploadStatus.classList.add('hidden'), 4000);
  }

  async function loadFiles() {
    try {
      const resp = await fetch('/v1/files');
      const data = await resp.json();
      fileList.innerHTML = '';
      if (data.data && data.data.length) {
        data.data.forEach(f => {
          const div = document.createElement('div');
          div.className = 'file-item';
          div.innerHTML = `<span>${escHtml(f.filename)}</span><button class="file-del" data-id="${escHtml(f.id)}" title="删除">&times;</button>`;
          fileList.appendChild(div);
        });
        // bind delete
        fileList.querySelectorAll('.file-del').forEach(btn => {
          btn.addEventListener('click', async () => {
            await fetch(`/v1/files/${encodeURIComponent(btn.dataset.id)}`, { method: 'DELETE' });
            loadFiles();
          });
        });
      } else {
        fileList.innerHTML = '<p class="placeholder">暂无文档</p>';
      }
    } catch (e) {
      // silent
    }
  }

  async function loadConfig() {
    try {
      const resp = await fetch('/v1/config');
      const data = await resp.json();
      if (data.config) {
        Object.entries(data.config).forEach(([k, v]) => {
          const el = $(`#cfg-${k}`);
          if (el) { el.value = v; const valEl = $(`#val-${k}`); if (valEl) valEl.textContent = v; }
        });
      }
    } catch (e) { /* silent */ }
  }

  btnSaveConfig.addEventListener('click', async () => {
    const config = {};
    cfgKeys.forEach(k => { const el = $(`#cfg-${k}`); if (el) { let v = el.value; if (k.includes('chunk_size') || k.includes('overlap') || k.includes('top_k') || k.includes('rerank_top_n')) v = parseInt(v); else v = parseFloat(v); config[k] = v; } });
    try {
      await fetch('/v1/config', { method: 'PUT', headers: {'Content-Type':'application/json'}, body: JSON.stringify(config) });
      toast('配置已更新', 'success');
    } catch (e) { toast('配置更新失败', 'error'); }
  });

  // ---- Chat ----

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  btnSend.addEventListener('click', sendMessage);

  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || isStreaming) return;
    chatInput.value = '';
    chatInput.style.height = 'auto';
    isStreaming = true;
    btnSend.disabled = true;

    appendMessage('user', text);
    const assistantDiv = appendMessage('assistant', '', true);

    // Build OpenAI-format messages from visible chat
    const messages = collectMessages();
    messages.push({ role: 'user', content: text });

    try {
      const resp = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages, stream: true }),
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || err.error?.message || '请求失败');
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let content = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            try {
              const chunk = JSON.parse(data);
              const delta = chunk.choices?.[0]?.delta?.content;
              if (delta) {
                content += delta;
                assistantDiv.innerHTML = renderMarkdown(content) + '<span class="streaming-cursor"></span>';
                chatMessages.scrollTop = chatMessages.scrollHeight;
              }
            } catch (e) { /* skip malformed chunk */ }
          }
        }
      }
      assistantDiv.innerHTML = renderMarkdown(content);
    } catch (e) {
      assistantDiv.innerHTML = renderMarkdown(`**[错误]** ${escHtml(e.message)}`);
    } finally {
      isStreaming = false;
      btnSend.disabled = false;
      chatInput.focus();
    }
  }

  function collectMessages() {
    const messages = [];
    const kids = chatMessages.children;
    for (const el of kids) {
      if (el.classList.contains('message')) {
        const role = el.dataset.role;
        if (role === 'user' || role === 'assistant') {
          messages.push({
            role: role,
            content: el.dataset.raw || el.textContent.trim()
          });
        }
      }
    }
    return messages;
  }

  function appendMessage(role, content, isStreaming) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.dataset.role = role;
    div.dataset.raw = content;
    if (!isStreaming) {
      div.innerHTML = renderMarkdown(content);
    }
    // remove welcome
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
  }

  // ---- Simple markdown renderer ----
  function renderMarkdown(text) {
    if (!text) return '';
    let html = escHtml(text);
    // code blocks
    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    // inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // newlines to <br> or <p>
    const lines = html.split('\n');
    let result = '';
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line) result += `<p>${line}</p>`;
    }
    return result || html;
  }

  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  // ---- Export ----
  btnExport.addEventListener('click', async () => {
    try {
      const resp = await fetch('/v1/chat/history/export');
      if (!resp.ok) { const err = await resp.json(); throw new Error(err.detail || '导出失败'); }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `chat_export_${Date.now()}.md`; a.click();
      URL.revokeObjectURL(url);
      toast('导出成功', 'success');
    } catch (e) { toast(e.message, 'error'); }
  });

  // ---- Clear chat ----
  btnClear.addEventListener('click', () => {
    chatMessages.innerHTML = `<div class="welcome-message">
      <p>对话已清空，上传文档后开始提问</p>
      <p class="welcome-hint">例："A是什么，它的特点是什么"</p>
    </div>`;
    toast('对话已清空', 'success');
  });

  // ---- Toast ----
  function toast(msg, type) {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3000);
  }

  // ---- Auto-resize textarea ----
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
  });

})();
