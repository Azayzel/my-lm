import * as fs from "fs";
import * as path from "path";
import { randomUUID } from "crypto";

export interface HistoryEntry {
  id: string;
  type: "chat" | "image";
  timestamp: number;
  // chat
  messages?: { role: string; content: string }[];
  // image
  prompt?: string;
  negative_prompt?: string;
  imagePath?: string;
  params?: Record<string, unknown>;
}

export class HistoryStore {
  private entries: HistoryEntry[] = [];

  constructor(private filePath: string) {
    this.load();
  }

  private load() {
    try {
      if (fs.existsSync(this.filePath)) {
        const raw = fs.readFileSync(this.filePath, "utf-8");
        this.entries = JSON.parse(raw);
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

  getAll(): HistoryEntry[] {
    return this.entries;
  }

  save(entry: Omit<HistoryEntry, "id" | "timestamp">): HistoryEntry {
    const full: HistoryEntry = {
      ...(entry as HistoryEntry),
      id: randomUUID(),
      timestamp: Date.now(),
    };
    this.entries.unshift(full);
    // keep last 200 entries
    if (this.entries.length > 200) this.entries = this.entries.slice(0, 200);
    this.persist();
    return full;
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

  clear() {
    this.entries = [];
    this.persist();
  }
}
