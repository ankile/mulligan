import {
  ConvexError,
  jsonToConvex,
  type JSONValue,
  type Value,
} from "convex/values";

export type MachineRequestErrorCode =
  | "invalid_json"
  | "invalid_body"
  | "invalid_idempotency_key";

export class MachineRequestError extends Error {
  readonly status: 400;
  readonly code: MachineRequestErrorCode;

  constructor(message: string, code: MachineRequestErrorCode) {
    super(message);
    this.name = "MachineRequestError";
    this.status = 400;
    this.code = code;
  }
}

export function decodeMachineArguments(
  bodyText: string,
): Record<string, Value> {
  let encoded: JSONValue;
  try {
    encoded = JSON.parse(bodyText) as JSONValue;
  } catch {
    throw new MachineRequestError(
      "Request body must be valid JSON",
      "invalid_json",
    );
  }
  let decoded: Value;
  try {
    decoded = jsonToConvex(encoded);
  } catch {
    throw new MachineRequestError(
      "Request body must use valid Convex JSON encoding",
      "invalid_body",
    );
  }
  if (
    typeof decoded !== "object" ||
    decoded === null ||
    Array.isArray(decoded)
  ) {
    throw new MachineRequestError(
      "Request body must encode an object",
      "invalid_body",
    );
  }
  return decoded as Record<string, Value>;
}

export type PublicMutationFailure = {
  status: 409 | 500;
  code: "idempotency_conflict" | "mutation_failed";
  message: string;
};

export function publicMutationFailure(error: unknown): PublicMutationFailure {
  if (
    error instanceof ConvexError &&
    typeof error.data === "object" &&
    error.data !== null &&
    !Array.isArray(error.data) &&
    Reflect.get(error.data, "code") === "idempotency_conflict"
  ) {
    return {
      status: 409,
      code: "idempotency_conflict",
      message: "Idempotency-Key was already used with a different request body",
    };
  }
  return {
    status: 500,
    code: "mutation_failed",
    message: "Machine mutation failed",
  };
}
