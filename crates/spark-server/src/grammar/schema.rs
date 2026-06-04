// SPDX-License-Identifier: AGPL-3.0-only

//! JSON Schema → xgrammar-encodable schema, in strict mode.
//!
//! The plan (Stage 1 A2): "Replace `sanitize_schema_for_grammar`'s
//! loosening with **reject at request time**: if the tool schema can't
//! be expressed in EBNF, return HTTP 400 with a clear error naming the
//! offending tool + the structural reason."
//!
//! Public entry points:
//! - [`validate_tools_for_grammar`] — called early in
//!   `chat_completions_inner`; returns `Err(Vec<ToolSchemaError>)` if
//!   any tool schema is unencodable, so the dispatcher can return HTTP
//!   400.
//! - [`compile_schema_strict`] — used by `grammar/compile_tools.rs`'s
//!   per-tool grammar builders. Returns `Err` for the same conditions.
//! - [`enforce_min_length_on_required_strings`] — constraint
//!   TIGHTENING (adds `minLength: 1` to required string properties).
//!   The plan endorses this as legitimate.
//! - [`augment_schema_with_tafc_think`] — adds an optional `_think`
//!   scratchpad field at the head of the properties (CRANE/TAFC).
//!
//! What counts as "loosening" (and therefore an error):
//! - `enum: []` — no valid values; cannot be encoded structurally.
//! - `false` boolean schema — accepts nothing; cannot be encoded.
//! - `type: "object"` with no `properties` AND no structural keys
//!   (`additionalProperties` / `patternProperties` /
//!   `unevaluatedProperties` / `propertyNames`) — inserting
//!   `additionalProperties: true` here would change "no properties
//!   allowed" into "any properties allowed".
//! - A required property whose schema is itself unencodable — dropping
//!   the property and pruning `required` would silently weaken the
//!   contract.
//!
//! What counts as "normalization" (legitimate, still applied silently):
//! - `$ref` inlining (xgrammar does not resolve refs).
//! - Empty `anyOf` / `oneOf` removal (logically vacuous; safe to drop).
//! - Single-element `anyOf` / `oneOf` flattening (equivalent to the
//!   element).
//! - `allOf` merging (xgrammar doesn't support `allOf` natively).
//! - Recursion into `properties`, `items`, `additionalProperties`.
//!
//! Test policy: tests for the loosenings ASSERT they now return `Err`.
//! Tests for the normalizations still assert `Ok` with the expected
//! normalized shape.

use crate::tool_parser::ToolDefinition;

const MAX_RECURSION_DEPTH: usize = 32;

// ── Public API ─────────────────────────────────────────────────────────

/// A single schema problem found in a tool definition. Surfaces all
/// the way up to the HTTP 400 body so the operator can fix it at
/// deploy time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSchemaError {
    pub tool_name: String,
    pub path: String,
    pub reason: SchemaReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaReason {
    /// `enum: []` — no valid values.
    EmptyEnum,
    /// `type: "object"` with no `properties` and no structural keys.
    EmptyObjectNoStructuralKey,
    /// JSON Schema `false` accepts nothing.
    BoolFalseSchema,
    /// Recursion exceeded [`MAX_RECURSION_DEPTH`] — likely an
    /// unresolvable `$ref` cycle.
    DepthExceeded,
}

impl std::fmt::Display for SchemaReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyEnum => f.write_str("empty `enum: []` — no valid values"),
            Self::EmptyObjectNoStructuralKey => f.write_str(
                "object schema has no `properties` and no `additionalProperties` / \
                 `patternProperties` / `unevaluatedProperties` / `propertyNames` — \
                 cannot be encoded without loosening",
            ),
            Self::BoolFalseSchema => f.write_str("`false` schema accepts no values"),
            Self::DepthExceeded => write!(
                f,
                "schema recursion exceeded {MAX_RECURSION_DEPTH} levels (probable `$ref` cycle)",
            ),
        }
    }
}

impl std::fmt::Display for ToolSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tool `{}` at `{}`: {}", self.tool_name, self.path, self.reason)
    }
}

/// Walk every tool's schema and return all unencodable errors. Called
/// from `chat_completions_inner` BEFORE the request is dispatched to
/// the scheduler. If `Err`, the caller returns HTTP 400.
pub fn validate_tools_for_grammar(tools: &[ToolDefinition]) -> Result<(), Vec<ToolSchemaError>> {
    let mut errors = Vec::new();
    for tool in tools {
        let tool_name = &tool.function.name;
        let raw_schema = tool
            .function
            .parameters
            .as_ref()
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
        if let Err(e) = compile_schema_strict(&raw_schema) {
            errors.push(ToolSchemaError {
                tool_name: tool_name.clone(),
                path: e.path,
                reason: e.reason,
            });
        }
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// Normalize a JSON schema for xgrammar in strict mode. Returns the
/// normalized schema on success; returns `SchemaValidationError`
/// (path + reason) on the first loosening that would have been
/// required.
///
/// Internal callers in `compile_tools.rs` can `.expect("validated upstream")`
/// when they know `validate_tools_for_grammar` ran first.
pub(super) fn compile_schema_strict(
    schema: &serde_json::Value,
) -> Result<serde_json::Value, SchemaValidationError> {
    compile_recursive(schema, schema, 0, "#".to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SchemaValidationError {
    pub path: String,
    pub reason: SchemaReason,
}

// ── Constraint tightening (unchanged) ──────────────────────────────────

/// Add `"minLength": 1` to required string properties that don't
/// already specify a `minLength`. xgrammar then generates EBNF `{1,}`
/// repetition, physically preventing the model from emitting `""` for
/// a required string parameter.
///
/// CRANE / TAFC schema augmentation (A.4, 2026-04-25). Constraint
/// TIGHTENING, not loosening; plan A2 endorses this.
pub(super) fn enforce_min_length_on_required_strings(
    schema: &serde_json::Value,
) -> serde_json::Value {
    let mut schema = schema.clone();
    let obj = match schema.as_object_mut() {
        Some(o) => o,
        None => return schema,
    };

    let required: Vec<String> = obj
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if required.is_empty() {
        return schema;
    }

    if let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
        for key in &required {
            if let Some(prop) = props.get_mut(key).and_then(|p| p.as_object_mut()) {
                let is_string = prop.get("type").and_then(|t| t.as_str()) == Some("string");
                if is_string && !prop.contains_key("minLength") {
                    prop.insert("minLength".to_string(), serde_json::Value::Number(1.into()));
                }
            }
        }
    }

    schema
}

/// Optionally inject an `_think` scratchpad string property at the
/// head of an object schema's `properties`. Per Think-Augmented
/// Function Calling (TAFC, arXiv:2601.18282) and CRANE (ICML 2025);
/// +12 pp Pass-Rate on Qwen2.5-72B. Safe no-op when the schema isn't
/// an object, `_think` already exists, or `properties` is missing.
pub fn augment_schema_with_tafc_think(schema: &serde_json::Value) -> serde_json::Value {
    let mut schema = schema.clone();
    let Some(obj) = schema.as_object_mut() else {
        return schema;
    };
    let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) else {
        return schema;
    };
    if props.contains_key("_think") {
        return schema;
    }
    let mut new_props = serde_json::Map::with_capacity(props.len() + 1);
    new_props.insert(
        "_think".to_string(),
        serde_json::json!({
            "type": "string",
            "description": "Optional scratchpad: brief rationale for selecting this tool and these arguments. Server-side only — not forwarded to the tool implementation.",
        }),
    );
    for (k, v) in props.iter() {
        new_props.insert(k.clone(), v.clone());
    }
    obj.insert(
        "properties".to_string(),
        serde_json::Value::Object(new_props),
    );
    schema
}

// ── Strict compilation (the core change) ───────────────────────────────

fn compile_recursive(
    schema: &serde_json::Value,
    root: &serde_json::Value,
    depth: usize,
    path: String,
) -> Result<serde_json::Value, SchemaValidationError> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(SchemaValidationError {
            path,
            reason: SchemaReason::DepthExceeded,
        });
    }

    // Boolean schemas: true = any, false = rejected.
    if let Some(b) = schema.as_bool() {
        return if b {
            Ok(serde_json::json!({}))
        } else {
            Err(SchemaValidationError {
                path,
                reason: SchemaReason::BoolFalseSchema,
            })
        };
    }

    let obj = match schema.as_object() {
        Some(o) => o,
        // Non-object, non-bool schema (e.g. number, null): pass through.
        None => return Ok(schema.clone()),
    };
    let mut result = obj.clone();

    // ── $ref inlining (normalization, silent) ──
    if let Some(ref_str) = result
        .get("$ref")
        .and_then(|v| v.as_str())
        .map(String::from)
        && let Some(resolved) = resolve_local_ref(&ref_str, root)
    {
        result.remove("$ref");
        if let Some(resolved_obj) = resolved.as_object() {
            for (k, v) in resolved_obj {
                if !result.contains_key(k) {
                    result.insert(k.clone(), v.clone());
                }
            }
        }
    }

    // ── enum: [] — STRICT REJECT ──
    if let Some(arr) = result.get("enum").and_then(|v| v.as_array())
        && arr.is_empty()
    {
        return Err(SchemaValidationError {
            path: format!("{path}/enum"),
            reason: SchemaReason::EmptyEnum,
        });
    }

    // ── anyOf / oneOf — empty removed, single-element flattened
    //    (normalization, silent) ──
    for key in ["anyOf", "oneOf"] {
        if let Some(arr) = result.get(key).and_then(|v| v.as_array()).cloned() {
            if arr.is_empty() {
                result.remove(key);
            } else if arr.len() == 1 {
                let inner = compile_recursive(
                    &arr[0],
                    root,
                    depth + 1,
                    format!("{path}/{key}/0"),
                )?;
                result.remove(key);
                if let Some(inner_obj) = inner.as_object() {
                    for (k, v) in inner_obj {
                        if !result.contains_key(k) {
                            result.insert(k.clone(), v.clone());
                        }
                    }
                }
            } else {
                let mut sanitized: Vec<serde_json::Value> = Vec::with_capacity(arr.len());
                for (i, el) in arr.iter().enumerate() {
                    sanitized.push(compile_recursive(
                        el,
                        root,
                        depth + 1,
                        format!("{path}/{key}/{i}"),
                    )?);
                }
                result.insert(key.to_string(), serde_json::Value::Array(sanitized));
            }
        }
    }

    // ── allOf merging (normalization, silent) ──
    if let Some(arr) = result.get("allOf").and_then(|v| v.as_array()).cloned() {
        if arr.is_empty() {
            result.remove("allOf");
        } else if arr.len() == 1 {
            let inner =
                compile_recursive(&arr[0], root, depth + 1, format!("{path}/allOf/0"))?;
            result.remove("allOf");
            if let Some(inner_obj) = inner.as_object() {
                for (k, v) in inner_obj {
                    if !result.contains_key(k) {
                        result.insert(k.clone(), v.clone());
                    }
                }
            }
        } else {
            let mut merged_props = serde_json::Map::new();
            let mut merged_required: Vec<serde_json::Value> = Vec::new();
            let mut merged_type: Option<serde_json::Value> = None;
            for (i, sub) in arr.iter().enumerate() {
                let s =
                    compile_recursive(sub, root, depth + 1, format!("{path}/allOf/{i}"))?;
                if let Some(o) = s.as_object() {
                    if let Some(t) = o.get("type") {
                        merged_type.get_or_insert_with(|| t.clone());
                    }
                    if let Some(p) = o.get("properties").and_then(|p| p.as_object()) {
                        for (k, v) in p {
                            merged_props.insert(k.clone(), v.clone());
                        }
                    }
                    if let Some(r) = o.get("required").and_then(|r| r.as_array()) {
                        for item in r {
                            if !merged_required.contains(item) {
                                merged_required.push(item.clone());
                            }
                        }
                    }
                }
            }
            result.remove("allOf");
            if let Some(t) = merged_type {
                result.entry("type").or_insert(t);
            }
            if !merged_props.is_empty() {
                result
                    .entry("properties")
                    .or_insert(serde_json::Value::Object(merged_props));
            }
            if !merged_required.is_empty() {
                result
                    .entry("required")
                    .or_insert(serde_json::Value::Array(merged_required));
            }
        }
    }

    // ── Empty object with no structural key — STRICT REJECT ──
    let is_object = result.get("type").and_then(|t| t.as_str()) == Some("object");
    let has_props = result
        .get("properties")
        .and_then(|p| p.as_object())
        .is_some_and(|p| !p.is_empty());
    let has_structural_keys = result.contains_key("patternProperties")
        || result.contains_key("additionalProperties")
        || result.contains_key("unevaluatedProperties")
        || result.contains_key("propertyNames");

    if is_object && !has_props && !has_structural_keys {
        return Err(SchemaValidationError {
            path,
            reason: SchemaReason::EmptyObjectNoStructuralKey,
        });
    }

    // ── Recurse into property schemas. ANY unencodable property
    //    schema (required or optional) makes the whole object
    //    unencodable; surface the inner error verbatim so the
    //    operator sees the precise reason + JSON Pointer.
    if let Some(props) = result.get("properties").cloned()
        && let Some(props_obj) = props.as_object()
    {
        let mut new_props = serde_json::Map::new();
        for (k, v) in props_obj {
            let sanitized =
                compile_recursive(v, root, depth + 1, format!("{path}/properties/{k}"))?;
            new_props.insert(k.clone(), sanitized);
        }
        result.insert(
            "properties".to_string(),
            serde_json::Value::Object(new_props),
        );
    }

    // ── Recurse into items ──
    if let Some(items) = result.get("items").cloned() {
        let sanitized =
            compile_recursive(&items, root, depth + 1, format!("{path}/items"))?;
        result.insert("items".to_string(), sanitized);
    }

    // ── Recurse into additionalProperties when it's a schema ──
    if let Some(addl) = result.get("additionalProperties").cloned()
        && addl.is_object()
    {
        let sanitized = compile_recursive(
            &addl,
            root,
            depth + 1,
            format!("{path}/additionalProperties"),
        )?;
        result.insert("additionalProperties".to_string(), sanitized);
    }

    Ok(serde_json::Value::Object(result))
}

/// Resolve a local JSON Pointer `$ref` (e.g. `#/$defs/Foo`).
fn resolve_local_ref(ref_str: &str, root: &serde_json::Value) -> Option<serde_json::Value> {
    let path = ref_str.strip_prefix("#/")?;
    let mut current = root;
    for segment in path.split('/') {
        let decoded = segment.replace("~1", "/").replace("~0", "~");
        current = current.get(&decoded)?;
    }
    Some(current.clone())
}
