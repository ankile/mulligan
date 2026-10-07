import { describe, expect, test } from "bun:test";
import { ConvexError, convexToJson } from "convex/values";

import {
  decodeMachineArguments,
  MachineRequestError,
  publicMutationFailure,
} from "../convex/machineHttpErrors";

describe("machine HTTP request errors", () => {
  test("reports malformed JSON as a client error", () => {
    expect(() => decodeMachineArguments("{")).toThrow(
      expect.objectContaining<Partial<MachineRequestError>>({
        status: 400,
        code: "invalid_json",
      }),
    );
  });

  test("requires the decoded body to be an object", () => {
    expect(() => decodeMachineArguments("[]")).toThrow(
      expect.objectContaining<Partial<MachineRequestError>>({
        status: 400,
        code: "invalid_body",
      }),
    );
  });

  test("rejects malformed Convex JSON values as a client error", () => {
    expect(() =>
      decodeMachineArguments('{"count":{"$integer":"not-base64"}}'),
    ).toThrow(
      expect.objectContaining<Partial<MachineRequestError>>({
        status: 400,
        code: "invalid_body",
      }),
    );
  });

  test("decodes valid Convex JSON arguments", () => {
    expect(
      decodeMachineArguments(
        JSON.stringify(convexToJson({ count: 12n, label: "ready" })),
      ),
    ).toEqual({ count: 12n, label: "ready" });
  });
});

describe("machine mutation error disclosure", () => {
  test("exposes the allowlisted idempotency conflict contract", () => {
    expect(
      publicMutationFailure(new ConvexError({ code: "idempotency_conflict" })),
    ).toEqual({
      status: 409,
      code: "idempotency_conflict",
      message: "Idempotency-Key was already used with a different request body",
    });
  });

  test("redacts unexpected errors and their stack details", () => {
    const failure = publicMutationFailure(
      new Error(
        "Dataset not found: secret/repo\n    at handler (../convex/datasets.ts:70:29)",
      ),
    );
    expect(failure).toEqual({
      status: 500,
      code: "mutation_failed",
      message: "Machine mutation failed",
    });
    expect(JSON.stringify(failure)).not.toContain("secret/repo");
    expect(JSON.stringify(failure)).not.toContain("datasets.ts");
  });

  test("does not recognize conflicts by matching an arbitrary error string", () => {
    expect(
      publicMutationFailure(
        new Error("Idempotency conflict from unrelated code"),
      ),
    ).toEqual({
      status: 500,
      code: "mutation_failed",
      message: "Machine mutation failed",
    });
  });

  test("redacts unrecognized ConvexError data", () => {
    expect(
      publicMutationFailure(
        new ConvexError({
          code: "private_backend_error",
          secret: "do-not-leak",
        }),
      ),
    ).toEqual({
      status: 500,
      code: "mutation_failed",
      message: "Machine mutation failed",
    });
  });
});
