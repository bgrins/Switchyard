// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codex MCP namespace preservation across a Responses/Chat translation.
//!
//! Codex groups each MCP server's tools in a non-standard ``namespace`` tool
//! container and expects the namespace back on the function call it receives:
//! `{"type": "function_call", "name": "search", "namespace": "mcp__docs__"}`.
//! OpenAI-compatible upstreams accept only flat `function` tools, so the request
//! codec flattens the container and the namespace is lost.
//!
//! Capturing the child-to-namespace mapping from the request lets the response
//! path put it back.

use std::collections::HashMap;

use serde_json::Value;

use crate::WireFormat;

/// Map every unambiguous Responses function tool to its Codex MCP namespace.
///
/// Returns an empty map for any wire format other than
/// [`WireFormat::OpenAiResponses`], for a request without `tools`, and for any
/// name that cannot be attributed to exactly one namespace. Pair with
/// [`restore_responses_tool_namespaces`] on the response.
pub fn responses_tool_namespaces(body: &Value, wire_format: WireFormat) -> HashMap<String, String> {
    if wire_format != WireFormat::OpenAiResponses {
        return HashMap::new();
    }
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return HashMap::new();
    };
    let mut namespaces = HashMap::new();
    collect_responses_tool_namespaces(tools, None, &mut namespaces);
    namespaces
        .into_iter()
        .filter_map(|(name, namespace)| namespace.map(|namespace| (name, namespace)))
        .collect()
}

// A None entry marks a name that cannot be attributed to one namespace, which
// leaves the corresponding call flat.
fn collect_responses_tool_namespaces(
    tools: &[Value],
    parent_namespace: Option<&str>,
    namespaces: &mut HashMap<String, Option<String>>,
) {
    for tool in tools {
        let Some(tool) = tool.as_object() else {
            continue;
        };
        if tool.get("type").and_then(Value::as_str) == Some("namespace") {
            // An unnamed container carries nothing to dispatch on.
            let namespace = tool
                .get("name")
                .and_then(Value::as_str)
                .filter(|namespace| !namespace.is_empty());
            if let Some(children) = tool.get("tools").and_then(Value::as_array) {
                collect_responses_tool_namespaces(children, namespace, namespaces);
            }
            continue;
        }
        let name = tool
            .get("function")
            .and_then(Value::as_object)
            .and_then(|function| function.get("name"))
            .or_else(|| tool.get("name"))
            .or_else(|| tool.get("id"))
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty());
        let Some(name) = name else {
            continue;
        };
        let Some(namespace) = parent_namespace else {
            // A tool outside any namespace shares one flat name space with the
            // flattened children, so a child of the same name cannot be told
            // apart from it.
            namespaces.insert(name.to_string(), None);
            continue;
        };
        match namespaces.get(name) {
            None => {
                namespaces.insert(name.to_string(), Some(namespace.to_string()));
            }
            Some(Some(existing)) if existing != namespace => {
                namespaces.insert(name.to_string(), None);
            }
            Some(_) => {}
        }
    }
}

/// Re-attach Codex MCP namespaces to every `function_call` in a response.
///
/// Walks the whole value, covering a buffered body and each streaming event,
/// where the item is nested under `item` (`response.output_item.added` /
/// `.done`) or `response.output` (`response.completed`). An existing
/// `namespace` is never overwritten; a name absent from `namespaces` stays flat.
pub fn restore_responses_tool_namespaces(body: &mut Value, namespaces: &HashMap<String, String>) {
    if namespaces.is_empty() {
        return;
    }
    match body {
        Value::Array(values) => {
            for value in values {
                restore_responses_tool_namespaces(value, namespaces);
            }
        }
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("function_call")
                && let Some(name) = object.get("name").and_then(Value::as_str)
                && let Some(namespace) = namespaces.get(name)
            {
                object
                    .entry("namespace".to_string())
                    .or_insert_with(|| Value::String(namespace.clone()));
            }
            for value in object.values_mut() {
                restore_responses_tool_namespaces(value, namespaces);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{WireFormat, responses_tool_namespaces, restore_responses_tool_namespaces};

    fn websearch_request() -> serde_json::Value {
        json!({
            "tools": [{
                "type": "namespace",
                "name": "mcp__open_websearch__",
                "tools": [{
                    "type": "function",
                    "name": "search",
                    "parameters": {"type": "object"}
                }]
            }]
        })
    }

    #[test]
    fn restores_codex_mcp_namespace_on_responses_function_call() {
        let namespaces =
            responses_tool_namespaces(&websearch_request(), WireFormat::OpenAiResponses);
        let mut response = json!({
            "output": [{
                "type": "function_call",
                "name": "search",
                "arguments": "{\"q\":\"Rust\"}"
            }]
        });

        restore_responses_tool_namespaces(&mut response, &namespaces);

        assert_eq!(response["output"][0]["namespace"], "mcp__open_websearch__");
    }

    // Streaming events nest the item one level deeper than a buffered body.
    #[test]
    fn restores_codex_mcp_namespace_on_nested_streaming_items() {
        let namespaces =
            responses_tool_namespaces(&websearch_request(), WireFormat::OpenAiResponses);
        let mut added = json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "name": "search", "arguments": ""}
        });
        let mut completed = json!({
            "type": "response.completed",
            "response": {
                "output": [{"type": "function_call", "name": "search", "arguments": "{}"}]
            }
        });

        restore_responses_tool_namespaces(&mut added, &namespaces);
        restore_responses_tool_namespaces(&mut completed, &namespaces);

        assert_eq!(added["item"]["namespace"], "mcp__open_websearch__");
        assert_eq!(
            completed["response"]["output"][0]["namespace"],
            "mcp__open_websearch__"
        );
    }

    // A leaf name colliding with a top-level tool is equally ambiguous: stamping
    // it would send that tool's calls to the MCP server.
    #[test]
    fn skips_names_shared_with_a_top_level_tool() {
        let namespaced_first = json!({
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__fs__",
                    "tools": [{"type": "function", "name": "read_file"}]
                },
                {"type": "function", "name": "read_file", "parameters": {}}
            ]
        });
        let top_level_first = json!({
            "tools": [
                {"type": "function", "name": "read_file", "parameters": {}},
                {
                    "type": "namespace",
                    "name": "mcp__fs__",
                    "tools": [{"type": "function", "name": "read_file"}]
                }
            ]
        });

        assert!(
            responses_tool_namespaces(&namespaced_first, WireFormat::OpenAiResponses).is_empty()
        );
        assert!(
            responses_tool_namespaces(&top_level_first, WireFormat::OpenAiResponses).is_empty()
        );
    }

    #[test]
    fn skips_an_empty_namespace_name() {
        let request = json!({
            "tools": [{
                "type": "namespace",
                "name": "",
                "tools": [{"type": "function", "name": "search"}]
            }]
        });

        assert!(responses_tool_namespaces(&request, WireFormat::OpenAiResponses).is_empty());
    }

    // Both the collector and the flattener recurse, so a leaf is attributed to
    // the innermost container that names it.
    #[test]
    fn attributes_a_nested_leaf_to_its_innermost_namespace() {
        let request = json!({
            "tools": [{
                "type": "namespace",
                "name": "mcp__outer__",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "mcp__inner__",
                        "tools": [{"type": "function", "name": "stat_file"}]
                    },
                    {"type": "function", "name": "list_files"}
                ]
            }]
        });

        let namespaces = responses_tool_namespaces(&request, WireFormat::OpenAiResponses);

        assert_eq!(
            namespaces.get("stat_file").map(String::as_str),
            Some("mcp__inner__")
        );
        assert_eq!(
            namespaces.get("list_files").map(String::as_str),
            Some("mcp__outer__")
        );
    }

    // The same name under two namespaces cannot be told apart once flat.
    #[test]
    fn skips_ambiguous_codex_mcp_tool_names() {
        let request = json!({
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__first__",
                    "tools": [{"type": "function", "name": "search"}]
                },
                {
                    "type": "namespace",
                    "name": "mcp__second__",
                    "tools": [{"type": "function", "name": "search"}]
                }
            ]
        });

        assert!(responses_tool_namespaces(&request, WireFormat::OpenAiResponses).is_empty());
    }

    #[test]
    fn preserves_an_upstream_supplied_namespace() {
        let namespaces =
            responses_tool_namespaces(&websearch_request(), WireFormat::OpenAiResponses);
        let mut response = json!({
            "output": [{
                "type": "function_call",
                "name": "search",
                "namespace": "mcp__upstream__"
            }]
        });

        restore_responses_tool_namespaces(&mut response, &namespaces);

        assert_eq!(response["output"][0]["namespace"], "mcp__upstream__");
    }

    #[test]
    fn ignores_non_responses_wire_formats() {
        assert!(responses_tool_namespaces(&websearch_request(), WireFormat::OpenAiChat).is_empty());
    }
}
