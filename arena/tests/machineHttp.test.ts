import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  mock,
  spyOn,
  test,
} from "bun:test";
import { getFunctionName, type PublicHttpAction } from "convex/server";
import { ConvexError, type Value } from "convex/values";

import { api } from "../convex/_generated/api";
import http from "../convex/http";
import {
  MACHINE_KEYS_ENV_VAR,
  sha256Hex,
  type MachineScope,
} from "../convex/machineAuth";

type MachineResponse = {
  ok: boolean;
  actor?: string;
  scopes?: MachineScope[];
  value?: Value;
  code?: string;
  error?: string;
  error_id?: string;
};

type MutationCall = {
  reference: unknown;
  args: Record<string, Value>;
};

type TestActionContext = {
  runMutation: (
    reference: unknown,
    args: Record<string, Value>,
  ) => Promise<unknown>;
};

type TestableHttpAction = PublicHttpAction & {
  _handler: (ctx: TestActionContext, request: Request) => Promise<Response>;
};

const credentials = {
  ingest: "pa_ingest_test.ingest-secret-with-enough-entropy",
  curate: "pa_curate_test.curate-secret-with-enough-entropy",
  admin: "pa_admin_test.admin-secret-with-enough-entropy",
} satisfies Record<MachineScope, string>;

type OperationExpectation = {
  operation: string;
  mutationName: string;
};

const operationsByScope = {
  ingest: [
    {
      operation: "evalSessions/submit",
      mutationName: getFunctionName(api.evalSessions.submit),
    },
    {
      operation: "evalSessions/addRounds",
      mutationName: getFunctionName(api.evalSessions.addRounds),
    },
    {
      operation: "datasets/register",
      mutationName: getFunctionName(api.datasets.register),
    },
    {
      operation: "datasets/refreshStats",
      mutationName: getFunctionName(api.datasets.refreshStats),
    },
    {
      operation: "datasets/updateStats",
      mutationName: getFunctionName(api.datasets.updateStats),
    },
    {
      operation: "policies/register",
      mutationName: getFunctionName(api.policies.register),
    },
    {
      operation: "taskSpecs/upsert",
      mutationName: getFunctionName(api.taskSpecs.upsert),
    },
    {
      operation: "stageTaskSpecs/upsert",
      mutationName: getFunctionName(api.stageTaskSpecs.upsert),
    },
    {
      operation: "stagePredictions/begin",
      mutationName: getFunctionName(api.stagePredictions.begin),
    },
    {
      operation: "stagePredictions/appendBatch",
      mutationName: getFunctionName(api.stagePredictions.appendBatch),
    },
    {
      operation: "stagePredictions/publish",
      mutationName: getFunctionName(api.stagePredictions.publish),
    },
    {
      operation: "stagePredictions/activate",
      mutationName: getFunctionName(api.stagePredictions.activate),
    },
  ],
  curate: [
    {
      operation: "policies/setTags",
      mutationName: getFunctionName(api.policies.setTags),
    },
    {
      operation: "evalSessions/setStatus",
      mutationName: getFunctionName(api.evalSessions.setStatus),
    },
    {
      operation: "evalSessions/setOperator",
      mutationName: getFunctionName(api.evalSessions.setOperator),
    },
    {
      operation: "evalSessions/updateNotes",
      mutationName: getFunctionName(api.evalSessions.updateNotes),
    },
    {
      operation: "operators/add",
      mutationName: getFunctionName(api.operators.add),
    },
    {
      operation: "datasets/updateTask",
      mutationName: getFunctionName(api.datasets.updateTask),
    },
    {
      operation: "datasets/updateClassification",
      mutationName: getFunctionName(api.datasets.updateClassification),
    },
    {
      operation: "datasets/setStatus",
      mutationName: getFunctionName(api.datasets.setStatus),
    },
    {
      operation: "policies/setStatus",
      mutationName: getFunctionName(api.policies.setStatus),
    },
    {
      operation: "policies/updateEnvironment",
      mutationName: getFunctionName(api.policies.updateEnvironment),
    },
    {
      operation: "statuses/setTaskStatus",
      mutationName: getFunctionName(api.statuses.setTaskStatus),
    },
    {
      operation: "reviews/save",
      mutationName: getFunctionName(api.reviews.save),
    },
    {
      operation: "stageReviews/save",
      mutationName: getFunctionName(api.stageReviews.save),
    },
    {
      operation: "applyJobs/enqueue",
      mutationName: getFunctionName(api.applyJobs.enqueue),
    },
    {
      operation: "applyJobs/cancel",
      mutationName: getFunctionName(api.applyJobs.cancel),
    },
  ],
  admin: [
    {
      operation: "evalSessions/deleteSession",
      mutationName: getFunctionName(api.evalSessions.deleteSession),
    },
    {
      operation: "evalSessions/removePolicyFromSession",
      mutationName: getFunctionName(api.evalSessions.removePolicyFromSession),
    },
    {
      operation: "datasets/deleteByRepo",
      mutationName: getFunctionName(api.datasets.deleteByRepo),
    },
    {
      operation: "policies/deletePolicy",
      mutationName: getFunctionName(api.policies.deletePolicy),
    },
    {
      operation: "maintenance/repairSessionDerivedData",
      mutationName: getFunctionName(api.maintenance.repairSessionDerivedData),
    },
  ],
} satisfies Record<MachineScope, OperationExpectation[]>;

const originalRegistry = process.env[MACHINE_KEYS_ENV_VAR];
const originalServiceToken = process.env.ARENA_SERVICE_TOKEN;
let testRegistry: string;
let serialTail = Promise.resolve();
let releaseSerialTest: (() => void) | null = null;
let consoleSpies: Array<{ mockRestore: () => void }> = [];

function restoreEnvironment(name: string, value: string | undefined): void {
  if (value === undefined) {
    delete process.env[name];
  } else {
    process.env[name] = value;
  }
}

function routeHandler(
  path: string,
  method: "GET" | "POST",
): TestableHttpAction {
  const match = http.lookup(path, method);
  if (match === null) throw new Error(`Missing ${method} route ${path}`);
  return match[0] as TestableHttpAction;
}

function mutationRequest(
  operation: string,
  credential: string | null,
  body = "{}",
  idempotencyKey = "integration-test-request",
): Request {
  const headers = new Headers({
    "Content-Type": "application/json",
    "Idempotency-Key": idempotencyKey,
  });
  if (credential !== null) headers.set("Authorization", `Bearer ${credential}`);
  return new Request(`https://arena.example/api/v1/mutate/${operation}`, {
    method: "POST",
    headers,
    body,
  });
}

function actionContext(
  implementation: TestActionContext["runMutation"] = async () => null,
): TestActionContext {
  return { runMutation: implementation };
}

function mutationName(call: MutationCall): string {
  return getFunctionName(
    call.reference as Parameters<typeof getFunctionName>[0],
  );
}

async function responseBody(response: Response): Promise<MachineResponse> {
  return (await response.json()) as MachineResponse;
}

async function invokeMutation(
  operation: string,
  credential: string | null,
  ctx = actionContext(),
  body = "{}",
  idempotencyKey = "integration-test-request",
): Promise<Response> {
  const action = routeHandler(`/api/v1/mutate/${operation}`, "POST");
  return await action._handler(
    ctx,
    mutationRequest(operation, credential, body, idempotencyKey),
  );
}

async function invokeWhoami(credential: string | null): Promise<Response> {
  const headers = new Headers();
  if (credential !== null) headers.set("Authorization", `Bearer ${credential}`);
  const action = routeHandler("/api/v1/auth/whoami", "GET");
  return await action._handler(
    actionContext(),
    new Request("https://arena.example/api/v1/auth/whoami", { headers }),
  );
}

beforeAll(async () => {
  testRegistry = JSON.stringify({
    pa_ingest_test: {
      sha256: await sha256Hex(credentials.ingest),
      scopes: ["ingest"],
    },
    pa_curate_test: {
      sha256: await sha256Hex(credentials.curate),
      scopes: ["curate"],
    },
    pa_admin_test: {
      sha256: await sha256Hex(credentials.admin),
      scopes: ["admin"],
    },
  });
});

beforeEach(async () => {
  // These integration tests exercise production handlers that intentionally
  // read process.env. Serialize the file even when Bun is invoked with
  // --concurrent so one request cannot observe another test's credentials.
  let release!: () => void;
  const previous = serialTail;
  serialTail = new Promise<void>((resolve) => {
    release = resolve;
  });
  await previous;
  releaseSerialTest = release;

  process.env[MACHINE_KEYS_ENV_VAR] = testRegistry;
  process.env.ARENA_SERVICE_TOKEN = "server-only-service-token";
  consoleSpies = [
    spyOn(console, "info").mockImplementation(() => undefined),
    spyOn(console, "warn").mockImplementation(() => undefined),
    spyOn(console, "error").mockImplementation(() => undefined),
  ];
});

afterEach(() => {
  for (const spy of consoleSpies) spy.mockRestore();
  consoleSpies = [];
  if (releaseSerialTest === null) {
    throw new Error("Machine HTTP test serial lock was not acquired");
  }
  releaseSerialTest();
  releaseSerialTest = null;
});

afterAll(() => {
  restoreEnvironment(MACHINE_KEYS_ENV_VAR, originalRegistry);
  restoreEnvironment("ARENA_SERVICE_TOKEN", originalServiceToken);
});

describe("machine mutation HTTP routes", () => {
  test("registers the expected mutation route inventory", () => {
    const actual = http
      .getRoutes()
      .filter(([path]) => path.startsWith("/api/v1/mutate/"))
      .map(([path, method]) => `${method} ${path}`)
      .sort();
    const expected = Object.values(operationsByScope)
      .flat()
      .map(({ operation }) => `POST /api/v1/mutate/${operation}`)
      .sort();
    expect(actual).toEqual(expected);
  });

  test("enforces the declared scope on every mutation route", async () => {
    for (const scope of ["ingest", "curate", "admin"] as const) {
      for (const expectation of operationsByScope[scope]) {
        const acceptedCalls: MutationCall[] = [];
        const accepted = await invokeMutation(
          expectation.operation,
          credentials[scope],
          actionContext(async (reference, args) => {
            acceptedCalls.push({ reference, args });
            return null;
          }),
        );
        expect(accepted.status).toBe(200);
        expect(await responseBody(accepted)).toEqual({
          ok: true,
          actor: `pa_${scope}_test`,
          value: null,
        });
        expect(acceptedCalls).toHaveLength(1);
        expect(mutationName(acceptedCalls[0])).toBe(expectation.mutationName);

        for (const wrongScope of ["ingest", "curate", "admin"] as const) {
          if (wrongScope === scope) continue;
          const runMutation = mock(async () => null);
          const rejected = await invokeMutation(
            expectation.operation,
            credentials[wrongScope],
            actionContext(runMutation),
          );
          expect(rejected.status).toBe(403);
          expect(await responseBody(rejected)).toEqual({
            ok: false,
            code: "insufficient_scope",
            error: `Machine credential lacks ${scope} scope`,
          });
          expect(runMutation).not.toHaveBeenCalled();
        }
      }
    }
  });

  test("does not reveal whether a machine key ID is registered", async () => {
    const runMutation = mock(async () => null);
    const ctx = actionContext(runMutation);
    const missing = await invokeMutation("datasets/register", null, ctx);
    const unknown = await invokeMutation(
      "datasets/register",
      "pa_unknown_test.unknown-secret-with-enough-entropy",
      ctx,
    );
    const invalid = await invokeMutation(
      "datasets/register",
      "pa_ingest_test.wrong-secret-with-enough-entropy",
      ctx,
    );

    expect(missing.status).toBe(401);
    expect(unknown.status).toBe(401);
    expect(invalid.status).toBe(401);
    expect(await responseBody(missing)).toEqual({
      ok: false,
      code: "invalid_credentials",
      error: "Invalid machine credential",
    });
    expect(await responseBody(unknown)).toEqual(await responseBody(invalid));
    expect(runMutation).not.toHaveBeenCalled();
  });

  test("injects the server token without exposing it in the response", async () => {
    const calls: MutationCall[] = [];
    const ctx = actionContext(async (reference, args) => {
      calls.push({ reference, args });
      return "dataset-id";
    });
    const response = await invokeMutation(
      "datasets/register",
      credentials.ingest,
      ctx,
      '{"repo_id":"example-org/example","serviceToken":"client-controlled"}',
    );

    expect(response.status).toBe(200);
    expect(calls).toHaveLength(1);
    expect(calls[0].args).toEqual({
      repo_id: "example-org/example",
      serviceToken: "server-only-service-token",
    });
    expect(mutationName(calls[0])).toBe("datasets:register");
    const serialized = JSON.stringify(await responseBody(response));
    expect(serialized).not.toContain("server-only-service-token");
    expect(serialized).not.toContain(credentials.ingest);
  });

  test("derives identity-scoped idempotency arguments from the raw body", async () => {
    const calls: MutationCall[] = [];
    const body =
      '{"task":"square_d2","serviceToken":"client-controlled",' +
      '"submission_id":"client-controlled","submission_fingerprint":"client-controlled"}';
    const ctx = actionContext(async (reference, args) => {
      calls.push({ reference, args });
      return "session-id";
    });
    const response = await invokeMutation(
      "evalSessions/submit",
      credentials.ingest,
      ctx,
      body,
      "submit-square-round-4",
    );

    expect(response.status).toBe(200);
    expect(calls).toHaveLength(1);
    expect(calls[0].args).toEqual({
      task: "square_d2",
      serviceToken: "server-only-service-token",
      submission_id: "pa_ingest_test:submit-square-round-4",
      submission_fingerprint: await sha256Hex(body),
    });
    expect(mutationName(calls[0])).toBe("evalSessions:submit");
  });

  test("rejects invalid requests before running a mutation", async () => {
    const runMutation = mock(async () => null);
    const ctx = actionContext(runMutation);
    const invalidJson = await invokeMutation(
      "datasets/register",
      credentials.ingest,
      ctx,
      "{",
    );
    const invalidKey = await invokeMutation(
      "evalSessions/submit",
      credentials.ingest,
      ctx,
      "{}",
      "short",
    );

    expect(invalidJson.status).toBe(400);
    expect(await responseBody(invalidJson)).toEqual({
      ok: false,
      code: "invalid_json",
      error: "Request body must be valid JSON",
    });
    expect(invalidKey.status).toBe(400);
    expect(await responseBody(invalidKey)).toEqual({
      ok: false,
      code: "invalid_idempotency_key",
      error: "Idempotency-Key must contain 8 to 200 URL-safe characters",
    });
    expect(runMutation).not.toHaveBeenCalled();
  });

  test("redacts unexpected mutation failures and returns a correlation ID", async () => {
    const ctx = actionContext(async () => {
      throw new Error(
        "private dataset example-org/secret at ../convex/datasets.ts:70:29",
      );
    });
    const response = await invokeMutation(
      "datasets/register",
      credentials.ingest,
      ctx,
    );
    const body = await responseBody(response);

    expect(response.status).toBe(500);
    expect(body).toEqual({
      ok: false,
      code: "mutation_failed",
      error: "Machine mutation failed",
      error_id: expect.stringMatching(
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
      ),
    });
    expect(JSON.stringify(body)).not.toContain("example-org/secret");
    expect(JSON.stringify(body)).not.toContain("datasets.ts");
  });

  test("redacts a missing server credential before running a mutation", async () => {
    delete process.env.ARENA_SERVICE_TOKEN;
    const runMutation = mock(async () => null);
    const response = await invokeMutation(
      "datasets/register",
      credentials.ingest,
      actionContext(runMutation),
    );
    const body = await responseBody(response);

    expect(response.status).toBe(500);
    expect(body).toEqual({
      ok: false,
      code: "mutation_failed",
      error: "Machine mutation failed",
      error_id: expect.any(String),
    });
    expect(JSON.stringify(body)).not.toContain("ARENA_SERVICE_TOKEN");
    expect(runMutation).not.toHaveBeenCalled();
  });

  test("returns the public idempotency conflict contract", async () => {
    const ctx = actionContext(async () => {
      throw new ConvexError({ code: "idempotency_conflict" });
    });
    const response = await invokeMutation(
      "evalSessions/submit",
      credentials.ingest,
      ctx,
    );

    expect(response.status).toBe(409);
    expect(await responseBody(response)).toEqual({
      ok: false,
      code: "idempotency_conflict",
      error: "Idempotency-Key was already used with a different request body",
    });
  });
});

describe("machine whoami HTTP route", () => {
  test("returns the authenticated machine identity", async () => {
    const response = await invokeWhoami(credentials.curate);

    expect(response.status).toBe(200);
    expect(await responseBody(response)).toEqual({
      ok: true,
      actor: "pa_curate_test",
      scopes: ["curate"],
    });
  });

  test("uses one public contract for missing and invalid credentials", async () => {
    const missing = await invokeWhoami(null);
    const unknown = await invokeWhoami(
      "pa_unknown_test.unknown-secret-with-enough-entropy",
    );
    const invalid = await invokeWhoami(
      "pa_curate_test.wrong-secret-with-enough-entropy",
    );
    const expected = {
      ok: false,
      code: "invalid_credentials",
      error: "Invalid machine credential",
    };

    expect(missing.status).toBe(401);
    expect(unknown.status).toBe(401);
    expect(invalid.status).toBe(401);
    expect(await responseBody(missing)).toEqual(expected);
    expect(await responseBody(unknown)).toEqual(expected);
    expect(await responseBody(invalid)).toEqual(expected);
  });

  test("redacts authentication infrastructure failures", async () => {
    process.env[MACHINE_KEYS_ENV_VAR] = "not-json";
    const response = await invokeWhoami(null);
    const body = await responseBody(response);

    expect(response.status).toBe(500);
    expect(body).toEqual({
      ok: false,
      code: "auth_unavailable",
      error: "Machine authentication is unavailable",
      error_id: expect.any(String),
    });
    expect(JSON.stringify(body)).not.toContain("not-json");
  });
});
