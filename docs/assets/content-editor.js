/**
 * Lightweight WYSIWYG for [data-editable] blocks (native contenteditable).
 * Enable with ?edit=1 or the "Edit text" link.
 */
(function () {
  const STORAGE_PREFIX = "lce-content:";
  const pageId = location.pathname.split("/").pop() || "index.html";

  const TOOLBAR_ACTIONS = [
    { cmd: "bold", label: "B", title: "Bold" },
    { cmd: "italic", label: "I", title: "Italic" },
    { cmd: "underline", label: "U", title: "Underline" },
    { cmd: "insertUnorderedList", label: "• List", title: "Bullet list" },
    { cmd: "insertOrderedList", label: "1. List", title: "Numbered list" },
    { cmd: "createLink", label: "Link", title: "Select text, then add a link" },
    { cmd: "insertImage", label: "Image", title: "Insert image from URL or upload a file" },
    { cmd: "removeFormat", label: "Clear", title: "Clear formatting" },
  ];

  function $(sel, root) {
    return (root || document).querySelector(sel);
  }

  function storageKey(id) {
    return `${STORAGE_PREFIX}${pageId}:${id}`;
  }

  function isEditMode() {
    return new URLSearchParams(location.search).get("edit") === "1";
  }

  function blocks() {
    return [...document.querySelectorAll("[data-editable]")];
  }

  function blockId(el) {
    return el.getAttribute("data-editable") || "block";
  }

  function loadDraft(id) {
    try {
      return localStorage.getItem(storageKey(id));
    } catch (_) {
      return null;
    }
  }

  function saveDraft(id, html) {
    try {
      localStorage.setItem(storageKey(id), html);
    } catch (_) {
      /* ignore */
    }
  }

  function clearDrafts() {
    blocks().forEach((el) => {
      try {
        localStorage.removeItem(storageKey(blockId(el)));
      } catch (_) {
        /* ignore */
      }
    });
  }

  function applyDrafts() {
    blocks().forEach((el) => {
      const id = blockId(el);
      const draft = loadDraft(id);
      if (draft != null && draft.trim() !== el.innerHTML.trim()) {
        el.innerHTML = draft;
        markDraftBadge(el);
      }
    });
  }

  function markDraftBadge(el) {
    if (el.querySelector(":scope > .content-draft-badge")) return;
    const badge = document.createElement("span");
    badge.className = "content-draft-badge";
    badge.textContent = "draft";
    el.insertAdjacentElement("afterbegin", badge);
  }

  function escapeAttr(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/</g, "&lt;");
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function saveSelection(container) {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return null;
    const range = sel.getRangeAt(0);
    if (!container.contains(range.commonAncestorContainer)) return null;
    return range.cloneRange();
  }

  function restoreSelection(range) {
    if (!range) return;
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  }

  function anchorForRange(range) {
    let node = range.commonAncestorContainer;
    if (node.nodeType === Node.TEXT_NODE) node = node.parentElement;
    return node?.closest?.("a") || null;
  }

  function insertImageHtml(el, savedRange, src, alt) {
    el.focus();
    if (savedRange) restoreSelection(savedRange);
    const altAttr = alt ? ` alt="${escapeAttr(alt)}"` : ' alt=""';
    document.execCommand(
      "insertHTML",
      false,
      `<img src="${escapeAttr(src)}"${altAttr} class="content-image" />`
    );
  }

  function pickImageFile(el, savedRange) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.style.display = "none";
    document.body.appendChild(input);
    input.addEventListener("change", () => {
      const file = input.files?.[0];
      input.remove();
      if (!file) return;
      if (file.size > 2 * 1024 * 1024) {
        const ok = window.confirm(
          "This image is large and will be embedded as base64 in the HTML. Continue?"
        );
        if (!ok) return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        const defaultAlt = file.name.replace(/\.[^.]+$/, "");
        const alt = window.prompt("Alt text (optional):", defaultAlt);
        if (alt == null) return;
        insertImageHtml(el, savedRange, reader.result, alt.trim());
      };
      reader.readAsDataURL(file);
    });
    input.click();
  }

  function insertImage(el, savedRange) {
    const url = window.prompt("Image URL (leave empty to upload a file):");
    if (url === null) return;
    if (url.trim()) {
      const alt = window.prompt("Alt text (optional):", "");
      if (alt == null) return;
      insertImageHtml(el, savedRange, url.trim(), alt.trim());
      return;
    }
    pickImageFile(el, savedRange);
  }

  function bindImagePaste(el) {
    el.addEventListener("paste", (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (!item.type.startsWith("image/")) continue;
        e.preventDefault();
        const file = item.getAsFile();
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
          insertImageHtml(el, saveSelection(el), reader.result, "");
        };
        reader.readAsDataURL(file);
        return;
      }
    });
  }

  function insertLink(el, savedRange) {
    const range = savedRange || saveSelection(el);
    if (!range) {
      window.alert("Select the text you want to link, then click Link.");
      return;
    }

    const selectedText = range.collapsed ? "" : range.toString();
    const existing = anchorForRange(range);
    const currentHref = existing?.getAttribute("href") || "https://";
    const url = window.prompt("Link URL:", currentHref);
    if (url == null || url.trim() === "") return;

    const href = url.trim();
    el.focus();
    restoreSelection(range);

    if (existing && (range.collapsed || selectedText)) {
      existing.setAttribute("href", href);
      if (selectedText && existing.textContent !== selectedText) {
        existing.textContent = selectedText;
      }
      return;
    }

    if (selectedText) {
      document.execCommand("createLink", false, href);
      return;
    }

    const label = window.prompt("Link text:", href);
    if (label == null || label.trim() === "") return;
    document.execCommand(
      "insertHTML",
      false,
      `<a href="${escapeAttr(href)}">${escapeHtml(label.trim())}</a>`
    );
  }

  function runCommand(action, el, savedRange) {
    if (action.cmd === "createLink") {
      insertLink(el, savedRange);
      return;
    }
    if (action.cmd === "insertImage") {
      insertImage(el, savedRange);
      return;
    }
    if (savedRange) restoreSelection(savedRange);
    document.execCommand(action.cmd, false, null);
  }

  function mountToolbar(el) {
    const bar = document.createElement("div");
    bar.className = "editable-toolbar";
    bar.setAttribute("role", "toolbar");

    TOOLBAR_ACTIONS.forEach((action) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = action.label;
      btn.title = action.title || action.label;
      let savedRange = null;
      btn.addEventListener("mousedown", (e) => {
        e.preventDefault();
        savedRange = saveSelection(el);
      });
      btn.addEventListener("click", () => {
        el.focus();
        runCommand(action, el, savedRange);
        savedRange = null;
      });
      bar.appendChild(btn);
    });

    el.parentNode.insertBefore(bar, el);
    return bar;
  }

  function enableBlock(el) {
    const id = blockId(el);
    const draft = loadDraft(id);
    if (draft != null) el.innerHTML = draft;

    const toolbar = mountToolbar(el);
    el.contentEditable = "true";
    el.classList.add("editable-active");
    el.dataset.editing = "true";
    bindImagePaste(el);

    return { el, toolbar, id };
  }

  function currentBlocks() {
    return blocks().map((el) => ({ el, id: blockId(el) }));
  }

  function persistAll() {
    currentBlocks().forEach(({ el, id }) => {
      saveDraft(id, el.innerHTML);
    });
  }

  function exportJson() {
    const out = { page: pageId, blocks: {} };
    currentBlocks().forEach(({ el, id }) => {
      out.blocks[id] = el.innerHTML;
    });
    return JSON.stringify(out, null, 2);
  }

  function canSaveToFile() {
    return (
      location.hostname === "localhost" ||
      location.hostname === "127.0.0.1" ||
      location.hostname === "[::1]"
    );
  }

  async function saveToFile() {
    persistAll();
    const json = exportJson();
    const res = await fetch("/api/save-page-content", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: json,
    });
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        const payload = await res.json();
        if (payload.error) detail = payload.error;
      } catch (_) {
        /* ignore */
      }
      throw new Error(detail);
    }
    return res.json();
  }

  function leaveEditMode() {
    const url = new URL(location.href);
    url.searchParams.delete("edit");
    location.href = `${url.pathname}${url.search}${url.hash}`;
  }

  function editUrl(on) {
    const url = new URL(location.href);
    if (on) url.searchParams.set("edit", "1");
    else url.searchParams.delete("edit");
    return url.toString();
  }

  function setEditEntryVisible(visible) {
    document.querySelectorAll(".edit-entry").forEach((el) => {
      el.classList.toggle("hidden", !visible);
    });
  }

  function wireEditLinks() {
    document.querySelectorAll(".edit-page-link[data-edit-on]").forEach((a) => {
      if (a.dataset.wired === "1") return;
      a.dataset.wired = "1";
      a.addEventListener("click", (e) => {
        e.preventDefault();
        location.href = editUrl(true);
      });
    });
  }

  function injectEditLink() {
    wireEditLinks();
  }

  function injectEditorBar(active) {
    if ($(".content-editor-bar")) return;

    const bar = document.createElement("div");
    bar.className = "content-editor-bar";
    bar.innerHTML = `
      <strong>Editing page text</strong>
      <span class="hint">Done saves all blocks in this browser only. Use Save to file (dev server) to write docs/index.html.</span>
      <div class="editor-actions">
        <button type="button" data-action="save">Save draft</button>
        <button type="button" class="primary" data-action="save-file">Save to file</button>
        <button type="button" data-action="export">Export HTML</button>
        <button type="button" data-action="reset">Reset drafts</button>
        <button type="button" data-action="done">Done</button>
      </div>
      <div class="content-export-panel hidden" id="content-export-panel">
        <textarea readonly id="content-export-text" aria-label="Exported HTML JSON"></textarea>
      </div>
    `;

    $("main").insertBefore(bar, $("main").firstChild);

    bar.querySelector('[data-action="save"]').addEventListener("click", () => {
      persistAll();
      alert("Draft saved in this browser.");
    });

    const saveFileBtn = bar.querySelector('[data-action="save-file"]');
    if (!canSaveToFile()) {
      saveFileBtn.disabled = true;
      saveFileBtn.title = "Use scripts/dev_site.py on localhost to save into docs/index.html";
    } else {
      saveFileBtn.addEventListener("click", async () => {
        saveFileBtn.disabled = true;
        try {
          const result = await saveToFile();
          alert(`Saved ${result.path}`);
        } catch (err) {
          alert(
            `Save failed: ${err.message}\n\nStart the dev server with:\npython scripts/dev_site.py`
          );
        } finally {
          saveFileBtn.disabled = false;
        }
      });
    }

    bar.querySelector('[data-action="export"]').addEventListener("click", () => {
      persistAll();
      const json = exportJson();
      const panel = $("#content-export-panel");
      const ta = $("#content-export-text");
      panel.classList.remove("hidden");
      ta.value = json;
      ta.focus();
      ta.select();
      navigator.clipboard?.writeText(json).catch(() => {});
    });

    bar.querySelector('[data-action="reset"]').addEventListener("click", () => {
      if (!confirm("Clear all local drafts for this page?")) return;
      clearDrafts();
      leaveEditMode();
    });

    bar.querySelector('[data-action="done"]').addEventListener("click", () => {
      persistAll();
      leaveEditMode();
    });
  }

  function enableEditMode() {
    document.body.classList.add("content-editing");
    const active = blocks().map(enableBlock);
    injectEditorBar(active);
    setEditEntryVisible(false);
  }

  function isProductionSite() {
    return /github\.io$/i.test(location.hostname);
  }

  function init() {
    if (!blocks().length) return;

    if (isProductionSite()) {
      document.querySelectorAll(".edit-entry").forEach((el) => el.remove());
    }

    if (isEditMode()) {
      enableEditMode();
    } else {
      applyDrafts();
      injectEditLink();
      setEditEntryVisible(true);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
