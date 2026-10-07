export const MACHINE_KEYS_ENV_VAR = "POLICY_ARENA_MACHINE_KEYS_JSON";

export type MachineScope = "ingest" | "curate" | "admin";

export type MachineKeyRecord = {
  sha256: string;
  scopes: MachineScope[];
};

export type MachineKeyRegistry = Record<string, MachineKeyRecord>;

export type MachineIdentity = {
  keyId: string;
  scopes: MachineScope[];
};

export class MachineAuthError extends Error {
  readonly status: 401 | 403;

  constructor(message: string, status: 401 | 403) {
    super(message);
    this.name = "MachineAuthError";
    this.status = status;
  }
}

declare const process: {
  env: Record<string, string | undefined>;
};

function isMachineScope(value: unknown): value is MachineScope {
  return value === "ingest" || value === "curate" || value === "admin";
}

export function parseMachineKeyRegistry(
  serialized: string,
): MachineKeyRegistry {
  const parsed: unknown = JSON.parse(serialized);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error(`${MACHINE_KEYS_ENV_VAR} must contain a JSON object`);
  }

  const registry: MachineKeyRegistry = {};
  for (const [keyId, rawRecord] of Object.entries(parsed)) {
    if (!/^[A-Za-z0-9_-]{3,100}$/.test(keyId)) {
      throw new Error(`Machine key ID ${JSON.stringify(keyId)} is invalid`);
    }
    if (
      typeof rawRecord !== "object" ||
      rawRecord === null ||
      Array.isArray(rawRecord)
    ) {
      throw new Error(`Machine key ${keyId} must contain a JSON object`);
    }

    const sha256 = Reflect.get(rawRecord, "sha256");
    const scopes = Reflect.get(rawRecord, "scopes");
    if (typeof sha256 !== "string" || !/^[0-9a-f]{64}$/.test(sha256)) {
      throw new Error(`Machine key ${keyId} has an invalid SHA-256 digest`);
    }
    if (
      !Array.isArray(scopes) ||
      scopes.length === 0 ||
      !scopes.every(isMachineScope)
    ) {
      throw new Error(`Machine key ${keyId} has invalid scopes`);
    }
    registry[keyId] = { sha256, scopes };
  }
  return registry;
}

export async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

function constantTimeEqual(left: string, right: string): boolean {
  if (left.length !== right.length) return false;
  let difference = 0;
  for (let index = 0; index < left.length; index++) {
    difference |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return difference === 0;
}

export async function authenticateMachine(
  authorizationHeader: string | null,
  requiredScope: MachineScope | null,
  registry: MachineKeyRegistry,
): Promise<MachineIdentity> {
  if (!authorizationHeader?.startsWith("Bearer ")) {
    throw new MachineAuthError("Missing bearer credential", 401);
  }

  const credential = authorizationHeader.slice("Bearer ".length);
  const separatorIndex = credential.indexOf(".");
  if (separatorIndex <= 0 || separatorIndex === credential.length - 1) {
    throw new MachineAuthError("Malformed bearer credential", 401);
  }

  const keyId = credential.slice(0, separatorIndex);
  const record = registry[keyId];
  if (record === undefined) {
    throw new MachineAuthError("Unknown machine credential", 401);
  }

  const actualDigest = await sha256Hex(credential);
  if (!constantTimeEqual(actualDigest, record.sha256)) {
    throw new MachineAuthError("Invalid machine credential", 401);
  }
  if (requiredScope !== null && !record.scopes.includes(requiredScope)) {
    throw new MachineAuthError(
      `Machine credential lacks ${requiredScope} scope`,
      403,
    );
  }

  return { keyId, scopes: record.scopes };
}

export async function authenticateMachineRequest(
  request: Request,
  requiredScope: MachineScope | null,
): Promise<MachineIdentity> {
  const serializedRegistry = process.env[MACHINE_KEYS_ENV_VAR];
  if (serializedRegistry === undefined) {
    throw new Error(`${MACHINE_KEYS_ENV_VAR} is not configured`);
  }
  return await authenticateMachine(
    request.headers.get("Authorization"),
    requiredScope,
    parseMachineKeyRegistry(serializedRegistry),
  );
}
