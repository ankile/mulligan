import { describe, expect, test } from "bun:test";

import {
  MachineAuthError,
  authenticateMachine,
  parseMachineKeyRegistry,
  sha256Hex,
} from "../convex/machineAuth";

describe("machine API authentication", () => {
  test("accepts a valid credential with the required scope", async () => {
    const credential = "pa_robot_test.correct-horse-battery-staple";
    const registry = {
      pa_robot_test: {
        sha256: await sha256Hex(credential),
        scopes: ["ingest" as const],
      },
    };

    await expect(
      authenticateMachine(`Bearer ${credential}`, "ingest", registry),
    ).resolves.toEqual({
      keyId: "pa_robot_test",
      scopes: ["ingest"],
    });
  });

  test("rejects an invalid secret", async () => {
    const registry = {
      pa_robot_test: {
        sha256: await sha256Hex("pa_robot_test.real-secret"),
        scopes: ["ingest" as const],
      },
    };

    await expect(
      authenticateMachine(
        "Bearer pa_robot_test.wrong-secret",
        "ingest",
        registry,
      ),
    ).rejects.toEqual(
      expect.objectContaining<Partial<MachineAuthError>>({ status: 401 }),
    );
  });

  test("rejects a valid credential without the required scope", async () => {
    const credential = "pa_robot_test.real-secret";
    const registry = {
      pa_robot_test: {
        sha256: await sha256Hex(credential),
        scopes: ["ingest" as const],
      },
    };

    await expect(
      authenticateMachine(`Bearer ${credential}`, "admin", registry),
    ).rejects.toEqual(
      expect.objectContaining<Partial<MachineAuthError>>({ status: 403 }),
    );
  });

  test("fails loudly on malformed registry configuration", () => {
    expect(() =>
      parseMachineKeyRegistry(
        JSON.stringify({ machine: { sha256: "short", scopes: ["ingest"] } }),
      ),
    ).toThrow("invalid SHA-256 digest");
  });
});
