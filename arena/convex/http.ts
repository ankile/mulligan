import { httpRouter, type FunctionReference } from "convex/server";
import { convexToJson, type Value } from "convex/values";

import { api } from "./_generated/api";
import { httpAction } from "./_generated/server";
import { auth } from "./auth";
import {
  MachineAuthError,
  authenticateMachineRequest,
  sha256Hex,
  type MachineScope,
} from "./machineAuth";
import {
  decodeMachineArguments,
  MachineRequestError,
  publicMutationFailure,
} from "./machineHttpErrors";

declare const process: {
  env: Record<string, string | undefined>;
};

const http = httpRouter();

auth.addHttpRoutes(http);

type PublicMutationReference = FunctionReference<
  "mutation",
  "public",
  Record<string, Value>,
  unknown
>;

type MachineRouteOptions = {
  operation: string;
  scope: MachineScope;
  mutation: PublicMutationReference;
  idempotentSubmission?: boolean;
};

function jsonResponse(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function submissionKey(request: Request, machineId: string): string {
  const idempotencyKey = request.headers.get("Idempotency-Key");
  if (
    idempotencyKey === null ||
    !/^[A-Za-z0-9._:-]{8,200}$/.test(idempotencyKey)
  ) {
    throw new MachineRequestError(
      "Idempotency-Key must contain 8 to 200 URL-safe characters",
      "invalid_idempotency_key",
    );
  }
  return `${machineId}:${idempotencyKey}`;
}

function registerMachineRoute(options: MachineRouteOptions): void {
  http.route({
    path: `/api/v1/mutate/${options.operation}`,
    method: "POST",
    handler: httpAction(async (ctx, request) => {
      try {
        const identity = await authenticateMachineRequest(
          request,
          options.scope,
        );
        const bodyText = await request.text();
        const args = decodeMachineArguments(bodyText);
        const serviceToken = process.env.ARENA_SERVICE_TOKEN;
        if (serviceToken === undefined) {
          throw new Error("ARENA_SERVICE_TOKEN is not configured");
        }
        args.serviceToken = serviceToken;

        if (options.idempotentSubmission) {
          args.submission_id = submissionKey(request, identity.keyId);
          args.submission_fingerprint = await sha256Hex(bodyText);
        }

        const result = await ctx.runMutation(options.mutation, args);
        console.info(
          `[machine-api] ${identity.keyId} completed ${options.operation}`,
        );
        return jsonResponse(
          {
            ok: true,
            actor: identity.keyId,
            value: convexToJson((result ?? null) as Value),
          },
          200,
        );
      } catch (error) {
        if (error instanceof MachineAuthError) {
          console.warn(
            `[machine-api] rejected ${options.operation}: ${error.message}`,
          );
          return jsonResponse(
            {
              ok: false,
              code:
                error.status === 401
                  ? "invalid_credentials"
                  : "insufficient_scope",
              error:
                error.status === 401
                  ? "Invalid machine credential"
                  : error.message,
            },
            error.status,
          );
        }
        if (error instanceof MachineRequestError) {
          console.warn(
            `[machine-api] invalid ${options.operation}: ${error.message}`,
          );
          return jsonResponse(
            { ok: false, code: error.code, error: error.message },
            error.status,
          );
        }

        const failure = publicMutationFailure(error);
        if (failure.status === 409) {
          console.warn(
            `[machine-api] conflict in ${options.operation}: ${failure.code}`,
          );
          return jsonResponse(
            { ok: false, code: failure.code, error: failure.message },
            failure.status,
          );
        }
        const errorId = crypto.randomUUID();
        console.error(
          `[machine-api] error_id=${errorId} ${options.operation} failed`,
          error,
        );
        return jsonResponse(
          {
            ok: false,
            code: failure.code,
            error: failure.message,
            error_id: errorId,
          },
          failure.status,
        );
      }
    }),
  });
}

http.route({
  path: "/api/v1/auth/whoami",
  method: "GET",
  handler: httpAction(async (_ctx, request) => {
    try {
      const identity = await authenticateMachineRequest(request, null);
      return jsonResponse(
        { ok: true, actor: identity.keyId, scopes: identity.scopes },
        200,
      );
    } catch (error) {
      if (error instanceof MachineAuthError) {
        return jsonResponse(
          {
            ok: false,
            code:
              error.status === 401
                ? "invalid_credentials"
                : "insufficient_scope",
            error:
              error.status === 401
                ? "Invalid machine credential"
                : error.message,
          },
          error.status,
        );
      }
      const errorId = crypto.randomUUID();
      console.error(
        `[machine-api] error_id=${errorId} auth check failed`,
        error,
      );
      return jsonResponse(
        {
          ok: false,
          code: "auth_unavailable",
          error: "Machine authentication is unavailable",
          error_id: errorId,
        },
        500,
      );
    }
  }),
});

const routes: MachineRouteOptions[] = [
  {
    operation: "evalSessions/submit",
    scope: "ingest",
    mutation: api.evalSessions.submit,
    idempotentSubmission: true,
  },
  {
    operation: "evalSessions/addRounds",
    scope: "ingest",
    mutation: api.evalSessions.addRounds,
  },
  {
    operation: "datasets/register",
    scope: "ingest",
    mutation: api.datasets.register,
  },
  {
    operation: "datasets/refreshStats",
    scope: "ingest",
    mutation: api.datasets.refreshStats,
  },
  {
    operation: "datasets/updateStats",
    scope: "ingest",
    mutation: api.datasets.updateStats,
  },
  {
    operation: "policies/register",
    scope: "ingest",
    mutation: api.policies.register,
  },
  {
    operation: "taskSpecs/upsert",
    scope: "ingest",
    mutation: api.taskSpecs.upsert,
  },
  {
    operation: "stageTaskSpecs/upsert",
    scope: "ingest",
    mutation: api.stageTaskSpecs.upsert,
  },
  {
    operation: "stagePredictions/begin",
    scope: "ingest",
    mutation: api.stagePredictions.begin,
  },
  {
    operation: "stagePredictions/appendBatch",
    scope: "ingest",
    mutation: api.stagePredictions.appendBatch,
  },
  {
    operation: "stagePredictions/publish",
    scope: "ingest",
    mutation: api.stagePredictions.publish,
  },
  {
    operation: "stagePredictions/activate",
    scope: "ingest",
    mutation: api.stagePredictions.activate,
  },
  {
    operation: "evalSessions/setStatus",
    scope: "curate",
    mutation: api.evalSessions.setStatus,
  },
  {
    operation: "evalSessions/setOperator",
    scope: "curate",
    mutation: api.evalSessions.setOperator,
  },
  {
    operation: "evalSessions/updateNotes",
    scope: "curate",
    mutation: api.evalSessions.updateNotes,
  },
  { operation: "operators/add", scope: "curate", mutation: api.operators.add },
  {
    operation: "datasets/updateTask",
    scope: "curate",
    mutation: api.datasets.updateTask,
  },
  {
    operation: "datasets/updateClassification",
    scope: "curate",
    mutation: api.datasets.updateClassification,
  },
  {
    operation: "datasets/setStatus",
    scope: "curate",
    mutation: api.datasets.setStatus,
  },
  {
    operation: "policies/setStatus",
    scope: "curate",
    mutation: api.policies.setStatus,
  },
  {
    operation: "policies/setTags",
    scope: "curate",
    mutation: api.policies.setTags,
  },
  {
    operation: "policies/updateEnvironment",
    scope: "curate",
    mutation: api.policies.updateEnvironment,
  },
  {
    operation: "statuses/setTaskStatus",
    scope: "curate",
    mutation: api.statuses.setTaskStatus,
  },
  { operation: "reviews/save", scope: "curate", mutation: api.reviews.save },
  {
    operation: "stageReviews/save",
    scope: "curate",
    mutation: api.stageReviews.save,
  },
  {
    operation: "applyJobs/enqueue",
    scope: "curate",
    mutation: api.applyJobs.enqueue,
  },
  {
    operation: "applyJobs/cancel",
    scope: "curate",
    mutation: api.applyJobs.cancel,
  },
  {
    operation: "evalSessions/deleteSession",
    scope: "admin",
    mutation: api.evalSessions.deleteSession,
  },
  {
    operation: "evalSessions/removePolicyFromSession",
    scope: "admin",
    mutation: api.evalSessions.removePolicyFromSession,
  },
  {
    operation: "datasets/deleteByRepo",
    scope: "admin",
    mutation: api.datasets.deleteByRepo,
  },
  {
    operation: "policies/deletePolicy",
    scope: "admin",
    mutation: api.policies.deletePolicy,
  },
  {
    operation: "maintenance/repairSessionDerivedData",
    scope: "admin",
    mutation: api.maintenance.repairSessionDerivedData,
  },
];

for (const route of routes) registerMachineRoute(route);

export default http;
