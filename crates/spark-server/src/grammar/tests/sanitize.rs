// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the strict-mode schema compiler (`compile_schema_strict`)
//! and the request-time validation entry point
//! (`validate_tools_for_grammar`). After Stage 1 A2: loosenings are
//! rejected with a `ToolSchemaError`; legitimate normalizations
//! ($ref, anyOf/oneOf flattening, allOf merging) still succeed.

use super::super::schema::{
    SchemaReason, augment_schema_with_tafc_think, compile_schema_strict,
    enforce_min_length_on_required_strings, validate_tools_for_grammar,
};
use crate::tool_parser::{FunctionDefinition, ToolDefinition};

fn tool(name: &str, schema: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: None,
            parameters: Some(schema),
        },
    }
}

// ── enforce_min_length_on_required_strings (unchanged behaviour) ──

#[test]
fn enforce_min_length_required_strings_get_minlength_1() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "description": {"type": "string"},
            "count": {"type": "integer"},
            "optional_str": {"type": "string"},
            "has_min": {"type": "string", "minLength": 5}
        },
        "required": ["command", "description", "count", "has_min"]
    });
    let result = enforce_min_length_on_required_strings(&schema);
    let props = result["properties"].as_object().unwrap();
    assert_eq!(props["command"]["minLength"], 1);
    assert_eq!(props["description"]["minLength"], 1);
    assert!(props["count"].get("minLength").is_none());
    assert!(props["optional_str"].get("minLength").is_none());
    assert_eq!(props["has_min"]["minLength"], 5);
}

#[test]
fn enforce_min_length_no_required_is_no_op() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"name": {"type": "string"}}
    });
    let result = enforce_min_length_on_required_strings(&schema);
    assert!(result["properties"]["name"].get("minLength").is_none());
}

// ── Strict rejections (the S5.1 enforcement) ──

#[test]
fn empty_enum_is_rejected() {
    let schema = serde_json::json!({"type": "string", "enum": []});
    let err = compile_schema_strict(&schema).unwrap_err();
    assert_eq!(err.reason, SchemaReason::EmptyEnum);
    assert!(err.path.ends_with("/enum"));
}

#[test]
fn nested_empty_enum_is_rejected_with_pointer() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": []}
        }
    });
    let err = compile_schema_strict(&schema).unwrap_err();
    assert_eq!(err.reason, SchemaReason::EmptyEnum);
    assert!(err.path.contains("properties/mode"));
}

#[test]
fn false_bool_schema_is_rejected() {
    let schema = serde_json::Value::Bool(false);
    let err = compile_schema_strict(&schema).unwrap_err();
    assert_eq!(err.reason, SchemaReason::BoolFalseSchema);
}

#[test]
fn true_bool_schema_is_any_object() {
    let schema = serde_json::Value::Bool(true);
    let result = compile_schema_strict(&schema).unwrap();
    assert!(result.as_object().is_some_and(|o| o.is_empty()));
}

#[test]
fn empty_object_without_structural_key_is_rejected() {
    let schema = serde_json::json!({"type": "object", "properties": {}});
    let err = compile_schema_strict(&schema).unwrap_err();
    assert_eq!(err.reason, SchemaReason::EmptyObjectNoStructuralKey);
}

#[test]
fn empty_object_with_additional_properties_is_ok() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {},
        "additionalProperties": true
    });
    let result = compile_schema_strict(&schema).unwrap();
    assert_eq!(result["additionalProperties"], true);
}

#[test]
fn object_with_properties_is_ok() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"name": {"type": "string"}}
    });
    let result = compile_schema_strict(&schema).unwrap();
    assert!(result["properties"]["name"].is_object());
}

#[test]
fn required_property_with_unsatisfiable_schema_is_rejected_by_name() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": []}
        },
        "required": ["mode"]
    });
    let err = compile_schema_strict(&schema).unwrap_err();
    // The inner enum violation propagates first — that's the more
    // specific reason. The outer "UnsatisfiableRequiredProperty"
    // wrapping is reserved for cases where the inner error itself
    // is opaque.
    assert!(matches!(err.reason, SchemaReason::EmptyEnum));
    assert!(err.path.contains("properties/mode"));
}

// ── Normalizations (still succeed silently) ──

#[test]
fn empty_anyof_is_normalized_away() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "anyOf": []
    });
    let result = compile_schema_strict(&schema).unwrap();
    assert!(result.get("anyOf").is_none());
}

#[test]
fn single_element_anyof_is_flattened() {
    let schema = serde_json::json!({
        "anyOf": [{"type": "string"}]
    });
    let result = compile_schema_strict(&schema).unwrap();
    assert!(result.get("anyOf").is_none());
    assert_eq!(result["type"], "string");
}

#[test]
fn multi_element_anyof_is_preserved() {
    let schema = serde_json::json!({
        "anyOf": [{"type": "string"}, {"type": "integer"}]
    });
    let result = compile_schema_strict(&schema).unwrap();
    assert_eq!(result["anyOf"].as_array().unwrap().len(), 2);
}

#[test]
fn allof_single_is_flattened() {
    let schema = serde_json::json!({
        "allOf": [{"type": "object", "properties": {"x": {"type": "string"}}}]
    });
    let result = compile_schema_strict(&schema).unwrap();
    assert!(result.get("allOf").is_none());
    assert!(result["properties"]["x"].is_object());
}

#[test]
fn allof_multiple_is_merged() {
    let schema = serde_json::json!({
        "allOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]}
        ]
    });
    let result = compile_schema_strict(&schema).unwrap();
    let props = result["properties"].as_object().unwrap();
    assert!(props.contains_key("a"));
    assert!(props.contains_key("b"));
    let req: Vec<&str> = result["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(req.contains(&"a") && req.contains(&"b"));
}

#[test]
fn ref_is_inlined() {
    let schema = serde_json::json!({
        "$defs": {
            "Name": {"type": "string", "minLength": 1}
        },
        "type": "object",
        "properties": {
            "name": {"$ref": "#/$defs/Name"}
        }
    });
    let result = compile_schema_strict(&schema).unwrap();
    assert_eq!(result["properties"]["name"]["type"], "string");
    assert_eq!(result["properties"]["name"]["minLength"], 1);
    assert!(result["properties"]["name"].get("$ref").is_none());
}

// ── augment_schema_with_tafc_think (unchanged behaviour) ──

#[test]
fn tafc_think_is_added_at_head() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"x": {"type": "string"}}
    });
    let result = augment_schema_with_tafc_think(&schema);
    let props = result["properties"].as_object().unwrap();
    assert!(props.contains_key("_think"));
    assert_eq!(props["_think"]["type"], "string");
    // First key
    let first_key = props.keys().next().unwrap();
    assert_eq!(first_key, "_think");
}

#[test]
fn tafc_think_existing_is_not_shadowed() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "_think": {"type": "string", "description": "caller's own"},
            "x": {"type": "string"}
        }
    });
    let result = augment_schema_with_tafc_think(&schema);
    assert_eq!(
        result["properties"]["_think"]["description"],
        "caller's own"
    );
}

// ── validate_tools_for_grammar (multi-tool aggregation) ──

#[test]
fn validate_tools_empty_list_is_ok() {
    assert!(validate_tools_for_grammar(&[]).is_ok());
}

#[test]
fn validate_tools_all_encodable_is_ok() {
    let tools = vec![
        tool(
            "Write",
            serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        ),
        tool(
            "Read",
            serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        ),
    ];
    assert!(validate_tools_for_grammar(&tools).is_ok());
}

#[test]
fn validate_tools_aggregates_all_errors() {
    let tools = vec![
        tool(
            "Empty",
            serde_json::json!({"type": "object", "properties": {}}),
        ),
        tool(
            "BadEnum",
            serde_json::json!({
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": []}}
            }),
        ),
        tool(
            "Good",
            serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "string"}}
            }),
        ),
    ];
    let errors = validate_tools_for_grammar(&tools).unwrap_err();
    assert_eq!(errors.len(), 2, "Empty + BadEnum should both fail; Good should pass");
    let names: Vec<&str> = errors.iter().map(|e| e.tool_name.as_str()).collect();
    assert!(names.contains(&"Empty"));
    assert!(names.contains(&"BadEnum"));
    assert!(!names.contains(&"Good"));
}

#[test]
fn validate_tools_reports_tool_name_in_display() {
    let tools = vec![tool(
        "MyTool",
        serde_json::json!({"type": "object", "properties": {}}),
    )];
    let errors = validate_tools_for_grammar(&tools).unwrap_err();
    let display = errors[0].to_string();
    assert!(display.contains("MyTool"), "display: {display}");
    assert!(display.contains("cannot be encoded"), "display: {display}");
}
