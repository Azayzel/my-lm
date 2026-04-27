import * as fs from "fs";
import * as path from "path";

export interface AppConfig {
  // MongoDB (BookMind)
  mongoUri: string;
  mongoDb: string;
  embedModel: string;
  vectorIndex: string;

  // Defaults
  defaultLlmModel: string;
  defaultImageModel: string;
  goodreadsUser: string;
}

const DEFAULTS: AppConfig = {
  mongoUri: "",
  mongoDb: "bookmind",
  embedModel: "sentence-transformers/all-MiniLM-L6-v2",
  vectorIndex: "vs_books_embedding",
  defaultLlmModel: "",
  defaultImageModel: "",
  goodreadsUser: "",
};

export class ConfigStore {
  private config: AppConfig;

  constructor(private filePath: string) {
    this.config = { ...DEFAULTS };
    this.load();
  }

  private load() {
    try {
      if (fs.existsSync(this.filePath)) {
        const raw = JSON.parse(fs.readFileSync(this.filePath, "utf-8"));
        this.config = { ...DEFAULTS, ...raw };
      }
    } catch {
      this.config = { ...DEFAULTS };
    }
  }

  private persist() {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.writeFileSync(
      this.filePath,
      JSON.stringify(this.config, null, 2),
      "utf-8",
    );
  }

  get(): AppConfig {
    return { ...this.config };
  }

  set(patch: Partial<AppConfig>): AppConfig {
    this.config = { ...this.config, ...patch };
    this.persist();
    return { ...this.config };
  }

  /** Build env vars for Python subprocess spawning. */
  toEnv(): Record<string, string> {
    const env: Record<string, string> = {};
    if (this.config.mongoUri) env["MONGODB_URI"] = this.config.mongoUri;
    if (this.config.mongoDb) env["MONGODB_DB"] = this.config.mongoDb;
    if (this.config.embedModel)
      env["BOOKMIND_EMBED_MODEL"] = this.config.embedModel;
    if (this.config.vectorIndex)
      env["BOOKMIND_VECTOR_INDEX"] = this.config.vectorIndex;
    return env;
  }

  isMongoConfigured(): boolean {
    return !!this.config.mongoUri;
  }
}
