import * as fs from "fs";
import * as path from "path";
import { randomUUID } from "crypto";

export interface SavedPrompt {
  id: string;
  name: string;
  prompt: string;
  negative_prompt: string;
  timestamp: number;
  params?: Record<string, unknown>;
}

export class PromptStore {
  private entries: SavedPrompt[] = [];

  constructor(private filePath: string) {
    this.load();
  }

  private load() {
    try {
      if (fs.existsSync(this.filePath)) {
        this.entries = JSON.parse(fs.readFileSync(this.filePath, "utf-8"));
      }
    } catch {
      this.entries = [];
    }
  }

  private persist() {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.writeFileSync(
      this.filePath,
      JSON.stringify(this.entries, null, 2),
      "utf-8",
    );
  }

  getAll(): SavedPrompt[] {
    return this.entries;
  }

  save(entry: Omit<SavedPrompt, "id" | "timestamp">): SavedPrompt {
    const full: SavedPrompt = {
      ...entry,
      id: randomUUID(),
      timestamp: Date.now(),
    };
    this.entries.unshift(full);
    this.persist();
    return full;
  }

  update(id: string, patch: Partial<Omit<SavedPrompt, "id">>): SavedPrompt | null {
    const idx = this.entries.findIndex((e) => e.id === id);
    if (idx < 0) return null;
    this.entries[idx] = { ...this.entries[idx], ...patch };
    this.persist();
    return this.entries[idx];
  }

  delete(id: string): boolean {
    const before = this.entries.length;
    this.entries = this.entries.filter((e) => e.id !== id);
    if (this.entries.length !== before) {
      this.persist();
      return true;
    }
    return false;
  }
}
