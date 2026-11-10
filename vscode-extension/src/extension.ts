import * as vscode from "vscode";

type PatchResult = {
  file: string;
  diff: string;
  replacement: string;
  tests: { status: string; passed: boolean; reason?: string };
};

type PendingPatch = {
  document: vscode.TextDocument;
  version: number;
  replacement: string;
};

const pending = new Map<string, PendingPatch>();
let mostRecentPending: string | undefined;

async function getToken(context: vscode.ExtensionContext): Promise<string | undefined> {
  let token = await context.secrets.get("ace.serviceToken");
  if (!token) {
    token = await vscode.window.showInputBox({
      prompt: "Paste the token printed by ace-serve",
      password: true,
      ignoreFocusOut: true,
    });
    if (token) await context.secrets.store("ace.serviceToken", token);
  }
  return token || undefined;
}

async function post<T>(
  context: vscode.ExtensionContext,
  path: string,
  body: Record<string, unknown>,
): Promise<T> {
  const token = await getToken(context);
  if (!token) throw new Error("ACE service token is required");
  const baseUrl = vscode.workspace.getConfiguration("ace").get<string>("baseUrl", "http://127.0.0.1:8765");
  const endpoint = new URL(path, `${baseUrl.replace(/\/$/, "")}/`);
  if (endpoint.hostname !== "127.0.0.1" && endpoint.hostname !== "localhost" && endpoint.hostname !== "::1") {
    throw new Error("ACE extension only connects to a loopback service");
  }
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(900_000),
  });
  const result = await response.json() as T & { error?: string };
  if (!response.ok) throw new Error(result.error || `ACE service returned HTTP ${response.status}`);
  return result;
}

function workspaceRelativePath(document: vscode.TextDocument): string {
  const folder = vscode.workspace.getWorkspaceFolder(document.uri);
  if (!folder || document.uri.scheme !== "file") throw new Error("Open a file inside the current workspace");
  const relative = vscode.workspace.asRelativePath(document.uri, false).replace(/\\/g, "/");
  if (relative.startsWith("../") || relative === "..") throw new Error("File is outside the workspace");
  return relative;
}

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(vscode.commands.registerCommand("ace.setToken", async () => {
    await context.secrets.delete("ace.serviceToken");
    const token = await getToken(context);
    if (token) vscode.window.showInformationMessage("ACE service token saved in VS Code secret storage.");
  }));

  context.subscriptions.push(vscode.languages.registerInlineCompletionItemProvider(
    { scheme: "file" },
    {
      async provideInlineCompletionItems(document, position, _context, cancellation) {
        if (cancellation.isCancellationRequested || !vscode.workspace.getWorkspaceFolder(document.uri)) return [];
        const start = Math.max(0, position.line - 120);
        const end = Math.min(document.lineCount - 1, position.line + 120);
        const range = new vscode.Range(start, 0, end, document.lineAt(end).text.length);
        const text = document.getText(range);
        const offset = document.offsetAt(position) - document.offsetAt(new vscode.Position(start, 0));
        const prefix = text.slice(0, offset).slice(-8_000);
        const suffix = text.slice(offset).slice(0, 4_000);
        const prompt = `<|fim_prefix|>${prefix}<|fim_suffix|>${suffix}<|fim_middle|>`;
        try {
          const result = await post<{ text: string }>(context, "completion", {
            prompt,
            max_new_tokens: vscode.workspace.getConfiguration("ace").get<number>("maxNewTokens", 128),
          });
          if (cancellation.isCancellationRequested || !result.text) return [];
          return [new vscode.InlineCompletionItem(result.text, new vscode.Range(position, position))];
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          vscode.window.setStatusBarMessage(`ACE: ${message}`, 5000);
          return [];
        }
      },
    },
  ));

  context.subscriptions.push(vscode.commands.registerCommand("ace.proposePatch", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    try {
      const instructions = await vscode.window.showInputBox({ prompt: "Describe the code change", ignoreFocusOut: true });
      if (!instructions?.trim()) return;
      const file = workspaceRelativePath(editor.document);
      const result = await post<PatchResult>(context, "patch", { file, instructions });
      const proposal = await vscode.workspace.openTextDocument({
        content: result.replacement,
        language: editor.document.languageId,
      });
      pending.set(editor.document.uri.toString(), {
        document: editor.document,
        version: editor.document.version,
        replacement: result.replacement,
      });
      mostRecentPending = editor.document.uri.toString();
      await vscode.commands.executeCommand("vscode.diff", editor.document.uri, proposal.uri, `ACE Proposal: ${file}`, { preview: true });
      const choice = await vscode.window.showInformationMessage(
        `Review the ACE diff. Tests: ${result.tests.status}${result.tests.reason ? ` (${result.tests.reason})` : ""}. Apply only after checking the proposed changes.`,
        "Apply Reviewed Patch",
        "Discard",
      );
      if (choice === "Apply Reviewed Patch") await vscode.commands.executeCommand("ace.applyPatch");
      else if (choice === "Discard") await vscode.commands.executeCommand("ace.discardPatch");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      vscode.window.showErrorMessage(`ACE: ${message}`);
    }
  }));

  context.subscriptions.push(vscode.commands.registerCommand("ace.applyPatch", async () => {
    const key = mostRecentPending;
    const proposal = key ? pending.get(key) : undefined;
    if (!proposal) {
      vscode.window.showWarningMessage("There is no ACE proposal for this document.");
      return;
    }
    if (proposal.document.version !== proposal.version) {
      pending.delete(key!);
      mostRecentPending = undefined;
      vscode.window.showWarningMessage("The source changed after ACE generated this proposal. Request a new patch.");
      return;
    }
    const edit = new vscode.WorkspaceEdit();
    edit.replace(proposal.document.uri, new vscode.Range(
      proposal.document.positionAt(0),
      proposal.document.positionAt(proposal.document.getText().length),
    ), proposal.replacement);
    if (await vscode.workspace.applyEdit(edit)) {
      pending.delete(key!);
      mostRecentPending = undefined;
      vscode.window.showInformationMessage("Reviewed ACE patch applied.");
    }
  }));

  context.subscriptions.push(vscode.commands.registerCommand("ace.discardPatch", async () => {
    if (mostRecentPending) pending.delete(mostRecentPending);
    mostRecentPending = undefined;
  }));
}

export function deactivate(): void {}
