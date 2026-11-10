import assert from "node:assert/strict";
import { test } from "node:test";
import { substitute, substituteEnv } from "../src/lib/cluster-endpoints.js";

test("endpoint values keep replacement metacharacters literally", () => {
  const endpoint = "http://user:pa$$word$&@host:8080";
  assert.equal(substitute("curl $ENDPOINT", { ENDPOINT: endpoint }), `curl ${endpoint}`);
  assert.deepEqual(substituteEnv({ API_URL: "$ENDPOINT", COUNT: 2 }, { ENDPOINT: endpoint }), { API_URL: endpoint, COUNT: 2 });
});

test("unset endpoints and longer identifiers remain unchanged", () => {
  assert.equal(substitute("$HEAD $HEAD_PORT $EMPTY", { HEAD: "10.0.0.1", EMPTY: "" }), "10.0.0.1 $HEAD_PORT $EMPTY");
});
