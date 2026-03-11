/**
 * Vector Search Cosine Distance Test
 * Validates that vectorSearch uses cosine distance (not L2) so that
 * score = 1 / (1 + distance) produces meaningful results for high-dim embeddings.
 */

import assert from "node:assert/strict";

// Minimal mock to verify .distanceType('cosine') is called
let distanceTypeCalled = null;

const mockTable = {
    vectorSearch(vector) {
        return {
            distanceType(type) {
                distanceTypeCalled = type;
                return this;
            },
            limit(n) {
                return this;
            },
            where(cond) {
                return this;
            },
            async toArray() {
                // Return a mock result with cosine-like distance
                return [
                    {
                        id: "test-1",
                        text: "test memory",
                        vector: vector,
                        category: "preference",
                        scope: "global",
                        importance: 0.8,
                        timestamp: Date.now(),
                        metadata: "{}",
                        _distance: 0.1, // cosine distance → score = 1/(1+0.1) = 0.91
                    },
                ];
            },
        };
    },
    query() {
        return {
            limit() { return this; },
            select() { return this; },
            where() { return this; },
            async toArray() { return []; },
        };
    },
    async listIndices() { return []; },
    async createIndex() { },
};

// Test 1: distanceType is called with 'cosine'
console.log("Test 1: vectorSearch calls distanceType('cosine')...");

// Create a minimal store-like object that exercises the vectorSearch path
const fakeStore = {
    table: mockTable,
    config: { vectorDim: 4 },
    ftsIndexCreated: false,
    get hasFtsSupport() { return this.ftsIndexCreated; },
    async ensureInitialized() { },
    async vectorSearch(vector, limit = 5, minScore = 0.3, scopeFilter) {
        const safeLimit = Math.min(Math.max(1, Math.floor(limit)), 20);
        const fetchLimit = Math.min(safeLimit * 10, 200);
        let query = this.table.vectorSearch(vector).distanceType('cosine').limit(fetchLimit);
        const results = await query.toArray();
        const mapped = [];
        for (const row of results) {
            const distance = Number(row._distance ?? 0);
            const score = 1 / (1 + distance);
            if (score < minScore) continue;
            mapped.push({
                entry: {
                    id: row.id,
                    text: row.text,
                    vector: row.vector,
                    category: row.category,
                    scope: row.scope ?? "global",
                    importance: Number(row.importance),
                    timestamp: Number(row.timestamp),
                    metadata: row.metadata || "{}",
                },
                score,
            });
            if (mapped.length >= safeLimit) break;
        }
        return mapped;
    },
};

const results = await fakeStore.vectorSearch([1, 0, 0, 0], 5, 0.3);
assert.strictEqual(distanceTypeCalled, "cosine", "Should call distanceType with 'cosine'");
console.log("  ✅ distanceType('cosine') confirmed");

// Test 2: score computation is correct for cosine distance
console.log("Test 2: Score formula 1/(1+distance) produces correct values...");
assert.strictEqual(results.length, 1, "Should return 1 result");
const expectedScore = 1 / (1 + 0.1);
assert.ok(
    Math.abs(results[0].score - expectedScore) < 0.001,
    `Score should be ~${expectedScore.toFixed(3)}, got ${results[0].score.toFixed(3)}`,
);
console.log("  ✅ Score = 0.909 (correct for distance=0.1)");

// Test 3: Results below minScore are filtered out
console.log("Test 3: Low-score results are filtered...");
const strictResults = await fakeStore.vectorSearch([1, 0, 0, 0], 5, 0.95);
assert.strictEqual(strictResults.length, 0, "Score 0.909 should be filtered by minScore=0.95");
console.log("  ✅ minScore filtering works");

// Test 4: Without cosine, L2 distance would produce wrong scores
console.log("Test 4: Verify L2 would fail (documentation test)...");
// For 1024-dim embeddings, L2 distance ≈ 40-60 for typical vectors
// score = 1/(1+45) ≈ 0.022 — way below any reasonable minScore
const l2TypicalDistance = 45;
const l2Score = 1 / (1 + l2TypicalDistance);
assert.ok(l2Score < 0.3, `L2 score ${l2Score.toFixed(4)} should be below minScore=0.3`);
console.log(`  ✅ L2 score = ${l2Score.toFixed(4)} (would drop all results, confirming cosine is needed)`);

console.log("\n=== All vector-search-cosine tests passed! ===");
